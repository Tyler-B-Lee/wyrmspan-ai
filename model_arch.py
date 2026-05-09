import torch
import torch.nn as nn

class WyrmspanActionScorer(nn.Module):
    def __init__(self, state_dim, action_dim=128, fusion_dim=256):
        super().__init__()
        
        # Projection MLP to fuse scalar features + pooled action embeddings before scoring
        # action_dim = action_scalar (128) + pooled_action_cards (16) = 144
        # fusion_dim = 256 (learned fusion of features)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(action_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Project fused action embedding to match state dimension for dot-product scoring
        self.action_encoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
    def forward(self, state_embedding, action_vectors):
        """
        state_embedding: [batch, state_dim] (The 'thought' vector from the Transformer)
        action_vectors: [batch, num_legal_actions, action_dim] (Scalar features + pooled action embeddings)
        """
        # 1. Fuse scalar features + embeddings via MLP
        action_fused = self.fusion_mlp(action_vectors)  # [batch, N, fusion_dim]
        
        # 2. Project fused actions into the same 'semantic space' as the state
        action_embeddings = self.action_encoder(action_fused)  # [batch, N, state_dim]
        
        # 3. Compute a score for every action via Dot Product
        # batch matrix multiplication: [batch, 1, state_dim] @ [batch, state_dim, N] -> [batch, 1, N] -> [batch, N]
        scores = torch.bmm(state_embedding.unsqueeze(1), action_embeddings.transpose(1, 2)).squeeze(1)
        
        return scores


class WyrmspanAgent(nn.Module):
    def __init__(
        self,
        main_emb_dim=512,
        state_dim=256,
        fusion_dim=256,          # New: fusion MLP output dimension
        action_vocab_size=512,  # should match env.action_token_vocab_size
        action_token_embed_dim=64,
        action_pad_id=0,
        max_action_tokens=64,
    ):
        super().__init__()
        # linear layers to create a token per global info input
        self.timing_encoder = nn.Linear(29, main_emb_dim)
        self.guild_status_encoder = nn.Linear(67, main_emb_dim)
        self.deck_status_encoder = nn.Linear(260, main_emb_dim)
        self.player_resources_encoder = nn.Linear(16, main_emb_dim)
        self.automa_status_encoder = nn.Linear(29, main_emb_dim)

        # card embedding layers - part of the action encoding process
        self.action_token_embed = nn.Embedding(action_vocab_size, action_token_embed_dim, padding_idx=action_pad_id)
        self.action_position_embed = nn.Embedding(max_action_tokens, action_token_embed_dim)  # positional embedding for action tokens
        self.slot_type_embed = nn.Embedding(3, action_token_embed_dim)  # 3 slot types: empty, excavated, occupied
        self.slot_details_encoder = nn.Linear(17, action_token_embed_dim)  # 17 slot details as per game_env.py
        self.card_location_embed = nn.Embedding(13, action_token_embed_dim)  # 13 locations: hand or board space 1-12

        # other items
        self.guild_embed = nn.Embedding(4, main_emb_dim)  # 4 guilds
        self.objective_embed = nn.Embedding(20, main_emb_dim)  # 20 objectives
        self.objective_round_embed = nn.Embedding(4, main_emb_dim)  # objective round 0-4

        # Main state encoder that produces the 'thought' vector
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=main_emb_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        # Main action encoder that produces action encodings for scoring
        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=action_token_embed_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=2
        )

        self.actor_scorer = WyrmspanActionScorer(state_dim, action_token_embed_dim, fusion_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        observations: dict
    ):
        """        
        Args:
            observations: dict containing all the necessary tensors from the environment, including:
                - timing: [batch, 29]
                - guild_status: [batch, 67]
                - deck_status: [batch, 260]
                - player_resources: [batch, 16]
                - automa_status: [batch, 29]
                - hand_card_ids: [batch, max_hand_size]
                - slot_types: [batch, 12]
                - dragons_on_slots: [batch, 12]
                - slot_details: [batch, 12, 17]
                - other_indices: [batch, 5]
                Actions information
                - action_token_ids: [batch, num_legal_actions, max_action_tokens]
                - action_token_mask: [batch, num_legal_actions, max_action_tokens] (1 for valid tokens, 0 for padding)
                - action_mask: [batch, num_legal_actions] (1 for legal actions, 0 for illegal)
        Returns:
            - action_scores: [batch, num_legal_actions] (raw scores for each legal action)
            - state_value: [batch, 1] (value estimate for the current state)
        """
        batch_size = observations["timing"].shape[0]

        # Encode global scalar features into token embeddings
        timing_emb = self.timing_encoder(observations["timing"])  # [batch, main_emb_dim]
        guild_status_emb = self.guild_status_encoder(observations["guild_status"])
        deck_status_emb = self.deck_status_encoder(observations["deck_status"])
        player_resources_emb = self.player_resources_encoder(observations["player_resources"])
        automa_status_emb = self.automa_status_encoder(observations["automa_status"])
        guild_emb = self.guild_embed(observations["other_indices"][:, 0].long())  # [batch, main_emb_dim]

        state_emb_input = [timing_emb, guild_status_emb, deck_status_emb, player_resources_emb, automa_status_emb, guild_emb]

        # objectives
        for round_i in range(4):
            objective_ids = observations["other_indices"][:, 1 + round_i].long()  # [batch]
            objective_emb = self.objective_embed(objective_ids)  # [batch, main_emb_dim]
            objective_round_emb = self.objective_round_embed(torch.full_like(objective_ids, round_i))  # [batch, main_emb_dim]
            state_emb_input.append(objective_emb + objective_round_emb)  # combine objective and round info

        state_emb_input = torch.stack(state_emb_input, dim=1)  # [batch, num_tokens, main_emb_dim]

        # Encode the global state into a 'thought' vector using the Transformer
        state_embedding = self.state_encoder(state_emb_input)  # [batch, num_tokens, main_emb_dim]

        # Process action information to create action vectors for scoring
        hand_card_ids = observations["hand_card_ids"]  # [batch, max_hand_size]
        slot_types = observations["slot_types"]  # [batch, 12]
        slot_details = observations["slot_details"]  # [batch, 12, 17]

        # Embed hand cards
        hand_card_embeddings = self.action_token_embed(hand_card_ids)  # [batch, max_hand_size, action_token_embed_dim]
        hand_card_embeddings = hand_card_embeddings.mean(dim=1)  # simple pooling: [batch, action_token_embed_dim]

        # Embed slot types and details for each of the 12 board spaces
        slot_type_embeddings = self.slot_type_embed(slot_types)  # [batch, 12, action_token_embed_dim]
        slot_details_embeddings = self.slot_details_encoder(slot_details)  # [batch, 12, action_token_embed_dim]

        # Combine slot type and details embeddings (e.g., by summation)
        slot_embeddings = slot_type_embeddings + slot_details_embeddings  # [batch, 12, action_token_embed_dim]
        slot_embeddings_pooled = slot_embeddings.mean(dim=1)  # pool over the 12 slots: [batch, action_token_embed_dim]

        # Combine all action-related embeddings into a single vector per action
        action_vectors = hand_card_embeddings + slot_embeddings_pooled.unsqueeze(1)  # [batch, 1, action_token_embed_dim] -> broadcast to [batch, num_legal_actions, action_token_embed_dim]