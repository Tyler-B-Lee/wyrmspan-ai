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
        embedding_dim=768,
        hidden_dim=512,
        state_dim=256,
        action_scalar_dim=128,  # Reduced from 192 (uses 93/128 dims, was 93/192)
        dragon_embed_dim=16,     # Optimized from 32
        cave_embed_dim=12,       # Optimized from 24
        fusion_dim=256,          # New: fusion MLP output dimension
        num_dragons=184,
        num_caves=76,
    ):
        super().__init__()
        
        # NEW STATE ENCODER INPUT DIMENSIONALITY:
        # The env now sends card IDs instead of pre-embedded vectors.
        # State encoder input: global_stats (20) + hand_pooled (16) + board_pooled (16) = 52 dims
        # This is drastically reduced from the previous 15*768 + 12*768 = 20736 when embeddings were stubbed!
        flattened_dim = 20 + dragon_embed_dim + dragon_embed_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Card embeddings: handled locally inside forward() to make env representation-agnostic
        self.dragon_embed = nn.Embedding(num_dragons, dragon_embed_dim, padding_idx=0)
        self.cave_embed = nn.Embedding(num_caves, cave_embed_dim, padding_idx=0)

        # Total action dimension after local pooling: scalar (128) + pooled_action_cards (16) = 144
        action_dim = action_scalar_dim + dragon_embed_dim
        self.actor_scorer = WyrmspanActionScorer(state_dim, action_dim, fusion_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_stats, hand_card_ids, board_slot_card_ids, action_candidates, action_cards, action_mask):
        """
        NEW ARCHITECTURE: Model owns all embedding and pooling logic.
        
        Args:
            global_stats: [batch, 20] - Global game info
            hand_card_ids: [batch, 15] - Dragon IDs in hand (int64)
            board_slot_card_ids: [batch, 12] - Dragon IDs on board (int64)
            action_candidates: [batch, max_actions, 128] - Scalar action features
            action_cards: [batch, max_actions, max_cards_per_action, 2] - Card references per action
            action_mask: [batch, max_actions] - Which actions are valid
        """
        batch_size = global_stats.shape[0]
        
        # ========== STATE ENCODING WITH LOCAL EMBEDDING POOLING ==========
        # 1. Embed and pool hand cards
        hand_embeds = self.dragon_embed(hand_card_ids.to(torch.long))  # [batch, 15, 16]
        hand_pooled = hand_embeds.mean(dim=1)  # [batch, 16] - average pooling
        
        # 2. Embed and pool board cards
        board_embeds = self.dragon_embed(board_slot_card_ids.to(torch.long))  # [batch, 12, 16]
        board_pooled = board_embeds.mean(dim=1)  # [batch, 16] - average pooling
        
        # 3. Combine global stats + pooled embeddings for state representation
        state_repr = torch.cat([global_stats, hand_pooled, board_pooled], dim=-1)  # [batch, 52]
        
        # 4. Encode state -> state_embedding
        state_embedding = self.state_encoder(state_repr)  # [batch, state_dim]
        
        # 5. Critic evaluates state value
        value = self.critic(state_embedding).squeeze(-1)  # [batch]
        
        # ========== ACTION ENCODING WITH LOCAL CARD POOLING ==========
        # 1. Pool card references for each action
        # action_cards: [batch, max_actions, max_cards_per_action, 2]
        # We need to extract embeddings and pool them per action
        max_actions = action_cards.shape[1]
        max_cards = action_cards.shape[2]
        
        # Reshape for embedding lookup: [batch * max_actions * max_cards, 2]
        action_cards_flat = action_cards.view(-1, 2).to(torch.long)
        card_kinds = action_cards_flat[:, 0]  # Dragon (0) or Cave (1)
        card_ids = action_cards_flat[:, 1]
        
        # Embed cards (only dragons for now; caves don't have action embeddings in current design)
        # For dragons: use dragon_embed
        # For caves (kind=1): use cave_embed (which is 12-dim; we'll pad to 16 for consistency)
        action_embeds = torch.zeros(card_kinds.shape[0], self.dragon_embed.embedding_dim, device=global_stats.device)
        
        # Embed dragons (kind == 0)
        dragon_mask = (card_kinds == 0)
        if dragon_mask.any():
            dragon_ids_clamped = torch.clamp(card_ids[dragon_mask], min=0, max=self.dragon_embed.num_embeddings - 1)
            action_embeds[dragon_mask] = self.dragon_embed(dragon_ids_clamped)
        
        # Embed caves (kind == 1) - pad cave embeddings to match dragon embedding dim
        cave_mask = (card_kinds == 1)
        if cave_mask.any():
            cave_ids_clamped = torch.clamp(card_ids[cave_mask], min=0, max=self.cave_embed.num_embeddings - 1)
            cave_vecs = self.cave_embed(cave_ids_clamped)  # [N, 12]
            # Pad to 16 dims to match dragon embed dim
            padded_cave_vecs = torch.cat([
                cave_vecs,
                torch.zeros(cave_vecs.shape[0], self.dragon_embed.embedding_dim - cave_vecs.shape[1], device=global_stats.device)
            ], dim=-1)
            action_embeds[cave_mask] = padded_cave_vecs
        
        # Reshape back: [batch, max_actions, max_cards, 16]
        action_embeds = action_embeds.view(batch_size, max_actions, max_cards, -1)
        
        # 2. Pool embeddings per action (mean pooling across cards, ignoring zero-padded entries)
        # Create a mask for non-zero card references: [batch, max_actions, max_cards]
        card_ref_mask = (action_cards[..., 1] > 0).unsqueeze(-1).float()  # [batch, max_actions, max_cards, 1]
        
        # Apply mask before pooling
        masked_embeds = action_embeds * card_ref_mask  # [batch, max_actions, max_cards, 16]
        pooled_action_embeds = masked_embeds.sum(dim=2) / (card_ref_mask.sum(dim=2) + 1e-8)  # [batch, max_actions, 16]
        
        # 3. Concatenate scalar features + pooled action embeddings
        action_features = torch.cat([action_candidates, pooled_action_embeds], dim=-1)  # [batch, max_actions, 208]
        
        # ========== ACTION SCORING ==========
        scores = self.actor_scorer(state_embedding, action_features)  # [batch, max_actions]
        
        # Mask invalid actions so they can't be chosen
        scores = scores.masked_fill(action_mask == 0, float('-inf'))
        
        return scores, value
 
