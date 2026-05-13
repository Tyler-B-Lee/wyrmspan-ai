import torch
import torch.nn as nn
import math

class WyrmspanActionScorer(nn.Module):
    def __init__(self, state_dim, action_dim, fusion_dim=256, use_attention=False):
        """
        Action Scorer that ranks legal actions based on state embedding.
        
        This module takes a state embedding (the "thought" vector from the state encoder)
        and action embeddings (produced by ActionSequenceEncoder) and produces scores
        for each action via learned comparison in a shared semantic space.
        
        Args:
            state_dim: Dimension of the state embedding (typically main_emb_dim, e.g., 256)
            action_dim: Dimension of each action embedding (typically main_emb_dim, e.g., 256)
            fusion_dim: Dimension of the learned fusion space (default 256)
            use_attention: If True, use multi-head attention; if False, use MLP-based scoring (default)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention
        
        # Projection MLPs to refine action embeddings before scoring
        # Input: action embeddings [batch, num_actions, action_dim]
        # Output: action embeddings in fusion space [batch, num_actions, fusion_dim]
        self.action_refiner = nn.Sequential(
            nn.Linear(action_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # Project refined actions into the state's semantic space for dot-product scoring
        self.action_projector = nn.Linear(fusion_dim, state_dim)
        
        if use_attention:
            # Alternative scoring via cross-attention
            # Projects state into query space and computes attention over actions
            self.attention = nn.MultiheadAttention(
                embed_dim=state_dim,
                num_heads=4,
                batch_first=True
            )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, state_embedding, action_embeddings, action_mask=None) -> torch.Tensor:
        """
        Score legal actions based on state embedding.
        
        Args:
            state_embedding: [batch, state_dim] 
                The "thought" vector from the state encoder (WyrmspanAgent's state_encoder)
            action_embeddings: [batch, num_legal_actions, action_dim]
                Action embeddings from ActionSequenceEncoder
            action_mask: [batch, num_legal_actions] optional
                Boolean mask where True indicates valid actions (will be converted to attention mask)
        
        Returns:
            scores: [batch, num_legal_actions]
                Unnormalized scores for each action (higher = better)
        """
        batch_size, num_actions, _ = action_embeddings.shape
        if action_mask is not None:
            action_mask = action_mask.to(torch.bool)
        
        # 1. Refine action embeddings through learned representation
        refined_actions = self.action_refiner(action_embeddings)  # [batch, num_actions, fusion_dim]
        
        # 2. Project refined actions into state's semantic space
        projected_actions = self.action_projector(refined_actions)  # [batch, num_actions, state_dim]
        
        if self.use_attention:
            # 3a. Score via cross-attention between state and actions
            # state_embedding [batch, state_dim] -> [batch, 1, state_dim] (query)
            # projected_actions [batch, num_actions, state_dim] (key/value)
            state_query = state_embedding.unsqueeze(1)  # [batch, 1, state_dim]
            
            # Use the legal-action mask inside attention so invalid actions do not
            # influence the query representation.
            key_padding_mask = None
            if action_mask is not None:
                key_padding_mask = ~action_mask  # MultiheadAttention expects True for masked positions
            
            attn_output, attn_weights = self.attention(
                state_query, 
                projected_actions, 
                projected_actions,
                key_padding_mask=key_padding_mask
            )
            
            # Use attention weights as scores
            scores = attn_weights.squeeze(1)  # [batch, num_actions]
        else:
            # 3b. Score via dot-product similarity in state space
            # Compute: state @ action^T for each action
            # [batch, 1, state_dim] @ [batch, state_dim, num_actions] -> [batch, 1, num_actions]
            scores = torch.bmm(
                state_embedding.unsqueeze(1), 
                projected_actions.transpose(1, 2)
            ).squeeze(1)  # [batch, num_actions]
            scores = scores * (1.0 / math.sqrt(self.state_dim))
        
        # Apply mask to scores if provided (set invalid actions to large negative value)
        if action_mask is not None:
            scores = scores.masked_fill(~action_mask, -1e9)
        
        return scores


class ActionSequenceEncoder(nn.Module):
    def __init__(self, action_vocab_size=512, action_emb_dim=128, max_action_tokens=64, padding_idx=0, dropout: float = 0.0):
        super().__init__()
        self.action_token_embed = nn.Embedding(action_vocab_size, action_emb_dim, padding_idx=padding_idx)
        self.position_embed = nn.Embedding(max_action_tokens, action_emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=action_emb_dim, nhead=8, dim_feedforward=512, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_proj = nn.Linear(action_emb_dim, action_emb_dim)  # Project to desired output dimension if needed

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, action_token_ids, action_token_mask) -> torch.Tensor:
        """
        action_token_ids: [batch, num_legal_actions, max_action_tokens]
        action_token_mask: [batch, num_legal_actions, max_action_tokens] (1 for valid tokens, 0 for padding)
        """
        batch_size, num_actions, max_tokens = action_token_ids.shape
        
        # Embed tokens and positions
        flat_seqs = action_token_ids.view(-1, max_tokens)  # [batch*num_actions, max_tokens]
        flat_mask = action_token_mask.view(-1, max_tokens)  # [batch*num_actions, max_tokens]
        positions = torch.arange(max_tokens, device=action_token_ids.device).unsqueeze(0) # [1, max_tokens]

        # get embeddings
        token_emb = self.action_token_embed(flat_seqs)  # [batch*num_actions, max_tokens, action_emb_dim]
        position_emb = self.position_embed(positions)  # [1, max_tokens, action_emb_dim]
        token_emb = token_emb + position_emb   # [batch*num_actions, max_tokens, action_emb_dim]

        valid_rows = flat_mask.any(dim=1)
        output = torch.zeros(
            (flat_seqs.shape[0], token_emb.shape[-1]),
            device=token_emb.device,
            dtype=token_emb.dtype,
        )

        if valid_rows.any():
            valid_token_emb = token_emb[valid_rows]
            valid_flat_mask = flat_mask[valid_rows]

            # read the sequence
            encoded_seqs = self.transformer(valid_token_emb, src_key_padding_mask=(valid_flat_mask == 0))

            # Pool only over valid tokens to avoid padding dilution.
            valid_token_weights = valid_flat_mask.unsqueeze(-1).to(encoded_seqs.dtype)
            pooled = (encoded_seqs * valid_token_weights).sum(dim=1)
            pooled = pooled / valid_token_weights.sum(dim=1).clamp_min(1.0)

            # Project if needed
            valid_output = self.output_proj(pooled)
            output[valid_rows] = valid_output

        return output.view(batch_size, num_actions, -1)  # [batch, num_actions, action_emb_dim]


class WyrmspanAgent(nn.Module):
    def __init__(
        self,
        main_emb_dim=256,
        fusion_dim=256,          # New: fusion MLP output dimension
        action_vocab_size=512,  # should match env.action_token_vocab_size
        action_pad_id=0,
        max_action_tokens=64,
        max_queue_size=5,
        max_hand_size=15,
        dropout: float = 0.0,
    ):
        super().__init__()
        # linear layers to create a token per global info input
        self.timing_encoder = nn.Linear(30, main_emb_dim)
        self.guild_status_encoder = nn.Linear(67, main_emb_dim)
        self.deck_status_encoder = nn.Linear(260, main_emb_dim)
        self.player_resources_encoder = nn.Linear(16, main_emb_dim)
        self.automa_status_encoder = nn.Linear(29, main_emb_dim)
        self.state_feature_norm = nn.LayerNorm(main_emb_dim)

        # card embedding layers - part of both state and action encoding processes
        self.slot_type_embed = nn.Embedding(3, main_emb_dim)  # 3 slot types: empty, excavated, occupied
        self.slot_details_encoder = nn.Linear(18, main_emb_dim)  # 18 slot details as per game_env.py
        # global, guild, objective round 1-4, event queue, hand, display 1-3, or board space 1-12
        # hand cards have no order, so they will share one position embedding
        self.max_hand_size = max_hand_size
        self.input_position_embed = nn.Embedding(
            5 + 1 + 4 + max_queue_size + 1 + 6 + 12,
            main_emb_dim)

        # other items
        self.guild_embed = nn.Embedding(4, main_emb_dim)  # 4 guilds
        self.objective_embed = nn.Embedding(20, main_emb_dim)  # 20 objectives

        # Main state encoder that produces the 'thought' vector
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=main_emb_dim, nhead=8, dim_feedforward=512, batch_first=True, dropout=0.1),
            num_layers=3
        )

        self.action_sequence_encoder = ActionSequenceEncoder(action_vocab_size, main_emb_dim, max_action_tokens, action_pad_id, dropout=dropout)
        self.actor_scorer = WyrmspanActionScorer(
            main_emb_dim, main_emb_dim, fusion_dim,
            use_attention=False
        )
        self.critic = nn.Sequential(
            nn.Linear(main_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0.0 else 0.0),
            nn.Linear(128, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

        # Keep critic output head conservative to avoid early value spikes.
        last_linear = self.critic[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.orthogonal_(last_linear.weight, gain=1.0)

    def forward(
        self,
        observations: dict,
        debug=False
    ):
        """        
        Args:
            observations: dict containing all the necessary tensors from the environment, including:
                - timing: [batch, 30]
                - guild_status: [batch, 67]
                - deck_status: [batch, 260]
                - player_resources: [batch, 16]
                - automa_status: [batch, 29]
                - card_display_dragons: [batch, 3]
                - card_display_caves: [batch, 3]
                - hand_card_ids: [batch, max_hand_size]
                - hand_card_mask: [batch, max_hand_size] (1 for valid cards, 0 for padding)
                - slot_types: [batch, 12]
                - dragons_on_slots: [batch, 12]
                - slot_details: [batch, 12, 18]
                - other_indices: [batch, 5]
                Queue information
                - queue_tokens: [batch, max_queue_size, max_queue_token_length]
                - queue_pad_mask: [batch, max_queue_size, max_queue_token_length] (1 for valid tokens, 0 for padding)
                - queue_slot_mask: [batch, max_queue_size] (1 for occupied slots, 0 for empty)
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

        timing_emb = self.state_feature_norm(timing_emb)
        guild_status_emb = self.state_feature_norm(guild_status_emb)
        deck_status_emb = self.state_feature_norm(deck_status_emb)
        player_resources_emb = self.state_feature_norm(player_resources_emb)
        automa_status_emb = self.state_feature_norm(automa_status_emb)
        
        guild_indices = observations["other_indices"][..., 0].long().clamp(
            min=0,
            max=self.guild_embed.num_embeddings - 1,
        )
        guild_emb = self.guild_embed(guild_indices)  # [batch, main_emb_dim]
        guild_emb = self.state_feature_norm(guild_emb)

        state_emb_input = [timing_emb, guild_status_emb, deck_status_emb, player_resources_emb, automa_status_emb, guild_emb]
        if debug:
            print("Timing embedding:", timing_emb)
            print("Guild status embedding:", guild_status_emb)
            print("Deck status embedding:", deck_status_emb)
            print("Player resources embedding:", player_resources_emb)
            print("Automa status embedding:", automa_status_emb)
            print("Guild embedding:", guild_emb)

        # objectives
        for round_i in range(4):
            objective_ids = observations["other_indices"][..., 1 + round_i].long().clamp(
                min=0,
                max=self.objective_embed.num_embeddings - 1,
            )  # [batch]
            objective_emb = self.objective_embed(objective_ids)  # [batch, main_emb_dim]
            objective_emb = self.state_feature_norm(objective_emb)
            state_emb_input.append(objective_emb)  # combine objective and round info
            if debug:
                print(f"Objective embedding for round {round_i+1}:", objective_emb)

        state_emb_input = torch.stack(state_emb_input, dim=1)  # [batch, num_tokens, main_emb_dim]
        if debug:
            print("State embedding input shape:", state_emb_input.shape)

        # Process cards in hand and on board
        hand_card_ids = observations["hand_card_ids"]  # [batch, max_hand_size]
        slot_types = observations["slot_types"]  # [batch, 12]
        slot_details = observations["slot_details"]  # [batch, 12, 18]
        slot_dragons = observations["dragons_on_slots"]  # [batch, 12]
        display_dragons = observations["card_display_dragons"]  # [batch, 3]
        display_caves = observations["card_display_caves"]  # [batch, 3]

        # Embed hand cards using the same token embedding as actions (since they share the same token space)
        hand_card_embeddings = self.action_sequence_encoder.action_token_embed(hand_card_ids)  # [batch, max_hand_size, main_emb_dim]
        display_dragon_embeddings = self.action_sequence_encoder.action_token_embed(display_dragons)  # [batch, 3, main_emb_dim]
        display_cave_embeddings = self.action_sequence_encoder.action_token_embed(display_caves)  # [batch, 3, main_emb_dim]

        # Embed slot types and details for each of the 12 board spaces
        slot_dragon_embeddings = self.action_sequence_encoder.action_token_embed(slot_dragons)  # [batch, 12, main_emb_dim]
        slot_type_embeddings = self.slot_type_embed(slot_types)  # [batch, 12, main_emb_dim]
        slot_details_embeddings = self.slot_details_encoder(slot_details)  # [batch, 12, main_emb_dim]
        if debug:
            print("Slot dragon embeddings: ", slot_dragon_embeddings)
            print("Slot type embeddings: ", slot_type_embeddings)
            print("Slot details embeddings: ", slot_details_embeddings)

        # Combine slot type and details embeddings (e.g., by summation)
        slot_embeddings = slot_dragon_embeddings + slot_type_embeddings + slot_details_embeddings  # [batch, 12, main_emb_dim]

        # Embed event queue information
        queue_tokens = observations["queue_tokens"]  # [batch, max_queue_size, max_queue_token_length]
        queue_pad_mask = observations["queue_pad_mask"]  # [batch, max_queue_size, max_queue_token_length]

        queue_embeddings = self.action_sequence_encoder(queue_tokens, queue_pad_mask)  # [batch, max_queue_size, main_emb_dim]

        # Combine all embeddings into a single sequence for the state encoder
        encoder_input = torch.cat([
            state_emb_input, 
            slot_embeddings, 
            display_dragon_embeddings, 
            display_cave_embeddings, 
            queue_embeddings,
            hand_card_embeddings, 
        ], dim=1)  # [batch, total_tokens, main_emb_dim]

        # Next, we include the input position embeddings to help the model differentiate between different types of inputs
        # The order constructed is the same, so we can assign fixed position indices to each type of input
        # Note that hand cards have no intrinsic order, so they will all share the same position index
        total_tokens = encoder_input.shape[1]
        input_pos_emb_size = self.input_position_embed.num_embeddings
        position_ids = torch.arange(total_tokens - self.max_hand_size, device=encoder_input.device)
        # add hand card position ids (all the same since hand cards are unordered)
        hand_position_ids = torch.full((self.max_hand_size,), input_pos_emb_size - 1, device=encoder_input.device)  # last index reserved for hand cards
        position_ids = torch.cat([position_ids, hand_position_ids], dim=0)  # [total_tokens]
        position_emb = self.input_position_embed(position_ids)  # [total_tokens, main_emb_dim]
        encoder_input = encoder_input + position_emb.unsqueeze(0)  # [batch, total_tokens, main_emb_dim]

        # next, we construct a mask for the transformer encoder to prevent attention to padded tokens
        # we only have padding in the hand cards and event queue, so we can construct the mask based on those
        hand_card_mask = observations["hand_card_mask"]  # [batch, max_hand_size]
        queue_slot_mask = observations["queue_slot_mask"]  # [batch, max_queue_size]
        other_mask = torch.ones((batch_size, total_tokens - self.max_hand_size - queue_slot_mask.shape[1]), device=encoder_input.device) # [batch, tokens without hand and queue]
        
        encoder_mask = torch.cat([
            other_mask, 
            queue_slot_mask, 
            hand_card_mask
        ], dim=1)  # [batch, total_tokens]
        attention_mask = (encoder_mask == 0)  # True for padded tokens
        if debug:
            print("Encoder input shape:", encoder_input.shape)
            print("Attention mask shape:", attention_mask.shape)

        # Finally, encode the state with the transformer
        output_sequence = self.state_encoder(encoder_input, src_key_padding_mask=attention_mask)  # [batch, total_tokens, main_emb_dim]
        # extract first token as the state embedding (the 'thought' vector)
        state_embedding = output_sequence[:, 0, :]  # [batch, main_emb_dim]

        if debug:
            print("State embedding shape:", state_embedding.shape)
            print("State embedding:", state_embedding)

        # Compute value estimate for the current state
        state_value = self.critic(state_embedding)  # [batch, 1]

        return state_embedding, state_value
    
    def score_actions(self, state_embedding, action_token_ids, action_token_mask, action_mask=None):
        """
        Score legal actions given a state embedding.
        
        This method should be called after forward() to score the available actions.
        
        Args:
            state_embedding: [batch, state_dim] from forward()
            action_token_ids: [batch, num_legal_actions, max_action_tokens]
                Token IDs for each action sequence from the environment
            action_token_mask: [batch, num_legal_actions, max_action_tokens]
                Mask indicating valid tokens (1 for valid, 0 for padding)
            action_mask: [batch, num_legal_actions] optional
                Boolean mask where True indicates legal actions
        
        Returns:
            action_scores: [batch, num_legal_actions]
                Scores for each action (higher = better)
        """
        # Encode action sequences into embeddings using ActionSequenceEncoder
        action_embeddings = self.action_sequence_encoder(action_token_ids, action_token_mask)  # [batch, num_legal_actions, main_emb_dim]
        
        # Score actions based on state using WyrmspanActionScorer
        action_scores = self.actor_scorer(state_embedding, action_embeddings, action_mask)  # [batch, num_legal_actions]
        
        return action_scores

    def policy_value(self, observations: dict):
        """
        Compute action scores and state value from a full observation dict.
        """
        state_embedding, state_value = self.forward(observations)
        action_scores = self.score_actions(
            state_embedding,
            observations["action_token_ids"],
            observations["action_token_mask"],
            observations.get("action_mask"),
        )
        return action_scores, state_value
    

if __name__ == "__main__":
    from game_env import WyrmspanEnv
    import pprint
    import random

    env = WyrmspanEnv()
    agent = WyrmspanAgent(
        main_emb_dim=256,
        fusion_dim=256,
        action_vocab_size=env.action_token_vocab_size,
        action_pad_id=env.pad_token_id,
        max_action_tokens=env.max_action_tokens,
        max_queue_size=env.max_queue_size,
        max_hand_size=env.max_hand_size
    )

    # Test: Run a random action through the environment to verify it steps without errors,
    # gets correct observations, and test the agent's forward pass with those observations
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        # pretty print every 20 steps
        if step_count % 20 == 0:
            print(f"\n=== Step {step_count} ===")
            pprint.pprint(obs)
            
            # run the agent's forward pass with the current observation
            obs_tensor = {k: torch.tensor(v).unsqueeze(0) if not isinstance(v, torch.Tensor) else v.unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                state_embedding, state_value = agent(obs_tensor, debug=False)
                print(f"State value estimate: {state_value.item():.4f}")
                
                # Example: Score actions using the actor_scorer
                if "action_token_ids" in obs_tensor and "action_token_mask" in obs_tensor:
                    action_scores = agent.score_actions(
                        state_embedding,
                        obs_tensor["action_token_ids"],
                        obs_tensor["action_token_mask"],
                        obs_tensor.get("action_mask")
                    )
                    # Print top 5 actions
                    top_k = min(5, action_scores.shape[1])
                    top_scores, top_indices = torch.topk(action_scores[0], top_k)
                    print(f"Top {top_k} action scores: {top_scores.numpy()}")
                
        legal_actions = obs["action_mask"].sum()
        chosen_action = random.randint(0, int(legal_actions) - 1)  # Randomly choose among legal actions
        obs, reward, done, _, info = env.step(chosen_action)

        total_reward += reward
        step_count += 1
    print(f"Episode finished after {step_count} steps with total reward {total_reward:.2f}. Final score: {env.game_state.player.score}, Automa score: {env.game_state.automa.score}")
