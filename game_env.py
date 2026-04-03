from game_states import *

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def get_global_info(game_state: SoloGameState) -> dict:
    """
    Get a dictionary of tensors representing the global information of the game environment.
    Each group of related features will be stored in its own tensor within the dictionary.
    These will be translated into input tokens for the model.
    
    The features include:
    - Game Timing
        - Current Round / Game Phase
        - Number of Coins (Turns left)
        - Automa Deck Size
        - Automa Passed This Round
        - Player Score
        - Automa Score
        - Number of times each cave was explored
    - Guild Board Status
        - Which guild is active (one-hot)
        - Player guild marker position
        - Automa guild marker position
        - Player guild markers remaining
    - Deck Status
        - Dragon deck size
        - Cave deck size
        - Dragon deck tensor
        - Cave deck tensor
    - Player Resources
        - Dragon Cards in hand
        - Cave Cards in hand
        - Coins
        - Each resource type (meat, gold, crystal, milk)
        - Egg totals by location
    - Automa Status
        - Automa score
        - Automa difficulty
        - Automa dragons in hand
        - Automa caves in hand
        - Automa passed this round
        - Automa decision deck cards (one-hot)
    """
    info_dict = {}

    # Game Timing (29 values)
    timing_tensor = torch.zeros(29, dtype=torch.float32)
    timing_tensor[game_state.board["round_tracker"]["round"]] = 1.0  # One-hot encode current round (0-3)
    current_phase_index = PHASE_INDEX.get(game_state.phase, 0)
    timing_tensor[4 + current_phase_index] = 1.0  # One-hot encode current phase (0-7)
    timing_tensor[12] = game_state.player.coins / 10.0  # Normalize coins
    timing_tensor[13] = len(game_state.automa.decision_deck) / 8.0  # Normalize automa deck (0-8 cards)
    timing_tensor[14] = 1.0 if game_state.automa.passed_this_round else 0.0  # Automa passed this round
    timing_tensor[15] = (game_state.player.score) / 100.0  # Player score (normalized estimate)
    timing_tensor[16] = (game_state.automa.score) / 100.0  # Automa score (normalized estimate)
    # cave exploration counts (4 values x 3 caves)
    for i, cave_name in enumerate(CAVE_NAMES):
        num_explored = game_state.player.times_explored[cave_name]
        timing_tensor[17 + i * 4 + num_explored] = 1.0  # One-hot encode number of times explored (0-3)
    info_dict["timing"] = timing_tensor

    # Guild Board Status (51 combined values)
    guild_tensor = torch.zeros(28, dtype=torch.float32)
    guild_index = game_state.board["guild"]["guild_index"]
    # One-hot encode guild (4 possible guilds)
    if guild_index < 4:
        guild_tensor[guild_index] = 1.0
    guild_tensor[4 + game_state.board["guild"]["player_position"]] = 1.0  # Player position on guild track (0-11)
    guild_tensor[16 + game_state.board["guild"]["automa_position"]] = 1.0  # Automa position on guild track
    guild_tensor[27] = game_state.player.guild_markers / 4.0  # Player guild markers remaining
    # Guild ability uses
    guild_ability_tensor = torch.zeros((5, 3), dtype=torch.float32)
    ability_uses = game_state.board["guild"]["ability_uses"]
    for i in range(1, 6):
        uses = ability_uses[i]
        if len(uses) > 0:
            num_uses = min(len(uses), 3)
            guild_ability_tensor[i - 1, num_uses - 1] = 1.0
    # append guild ability uses to the end of guild tensor
    guild_tensor = torch.cat((guild_tensor, guild_ability_tensor.flatten()), dim=0)
    # extra info of end game point ability
    end_game_ability_tensor = torch.zeros((2,4), dtype=torch.float32)
    for player in range(2):
        count = ability_uses[5][2:].count(player)
        if count > 0:
            end_game_ability_tensor[player, count - 1] = 1.0
    guild_tensor = torch.cat((guild_tensor, end_game_ability_tensor.flatten()), dim=0)
    info_dict["guild_status"] = guild_tensor

    # Deck Status (2 + 183 + 75 = 260 values)
    deck_tensor = torch.zeros(2, dtype=torch.float32)
    deck_tensor[0] = len(game_state.dragon_deck) / 183.0  # Dragon deck size
    deck_tensor[1] = len(game_state.cave_deck) / 75.0  # Cave deck size
    # combine dragon and cave deck tensors from game state
    deck_tensor = torch.cat((
        deck_tensor,
        game_state.dragon_deck_tensor,
        game_state.cave_deck_tensor
    ), dim=0)
    info_dict["deck_status"] = deck_tensor

    # Player Resources (13 values: hand sizes, coins, resources, eggs)
    player_tensor = torch.zeros(13, dtype=torch.float32)
    player_tensor[0] = len(game_state.player.dragon_hand) / 10.0  # Dragon cards in hand
    player_tensor[1] = len(game_state.player.cave_hand) / 10.0  # Cave cards in hand
    player_tensor[2] = game_state.player.coins / 10.0  # Coins
    # Resources (meat, gold, crystal, milk)
    for i, resource in enumerate(RESOURCES):
        player_tensor[3 + i] = game_state.player.resources[resource] / 10.0  # Normalize resource counts
    # Egg totals by location
    player_tensor[7] = game_state.player.egg_totals["mat_slots"] / 2.0
    player_tensor[8] = game_state.player.egg_totals["crimson_cavern"] / 12.0
    player_tensor[9] = game_state.player.egg_totals["golden_grotto"] / 12.0
    player_tensor[10] = game_state.player.egg_totals["amethyst_abyss"] / 12.0
    player_tensor[11] = sum(game_state.player.egg_totals.values()) / 40.0  # Total eggs
    player_tensor[12] = sum(game_state.player.num_dragons_played.values()) / 12.0  # Dragons on mat
    info_dict["player_resources"] = player_tensor

    # Automa Status (25 values)
    automa_tensor = torch.zeros(25, dtype=torch.float32)
    automa_tensor[game_state.automa.difficulty] = 1.0  # Automa difficulty level
    automa_tensor[6] = game_state.automa.score / 100.0  # Automa score
    automa_tensor[7] = len(game_state.automa.dragons) / 16.0  # Automa dragons collected
    automa_tensor[8] = len(game_state.automa.caves) / 10.0  # Automa caves collected
    automa_tensor[9] = 1.0 if game_state.automa.passed_this_round else 0.0  # Automa passed this round
    for card_i in game_state.automa.decision_deck:
        automa_tensor[10 + card_i] = 1.0  # One-hot encode automa decision deck cards
    info_dict["automa_status"] = automa_tensor

    return info_dict

def get_player_board_info(game_state: SoloGameState) -> dict:
    """
    Get a dictionary of features representing the player's board information.
    Each group of related features will be stored in its own tensor within the dictionary.
    These will be translated into input tokens for the model.
    
    For each of the 12 mat slots, we will have 3 lists/tensors containing:
    - 1st: Slot Type
        - Is slot empty, excavated, or has a dragon
    - 2nd: Dragon on Slot
        - Which dragon is on the slot, if any
    - 3rd: Slot Details
        - Number of eggs on slot (0-5)
        - Max number of eggs for this slot (0-5)
        - Is this a hatchling (0/1)
        - Is this a grown hatchling (0/1)
        - What resources are cached on this slot (4 values)
        - Number of tucked dragons under this slot (1 value)
    """
    player = game_state.player
    
    # slot types (12 values)
    slot_types = []
    for cave_name in CAVE_NAMES:
        for col in range(4):
            if player.dragons_played[cave_name][col] is not None:
                slot_types.append(2)  # Dragon present
            elif player.caves_played[cave_name][col] is not None:
                slot_types.append(1)  # Excavated but no dragon
            else:
                slot_types.append(0)  # Empty slot
    slot_types_tensor = torch.tensor(slot_types, dtype=torch.int64)

    # dragons on slots (12 values)
    dragons_on_slots = []
    for cave_name in CAVE_NAMES:
        for col in range(4):
            dragon_id = player.dragons_played[cave_name][col]
            dragons_on_slots.append(dragon_id if dragon_id is not None else 0)
    dragons_on_slots_tensor = torch.tensor(dragons_on_slots, dtype=torch.int64)

    # slot details (12 x 17 = 204 total values)
    slot_details = torch.zeros((12, 17), dtype=torch.float32)
    for cave_name in CAVE_NAMES:
        for col in range(4):
            slot_index = CAVE_NAMES.index(cave_name) * 4 + col
            # Number of eggs on slot
            num_eggs, num_nests = player.nested_eggs[cave_name][col]
            ind = min(num_eggs, 2)  # Number of eggs (up to 2, 2+ for 3-5)
            slot_details[slot_index, ind] = 1.0  # One-hot encode number of eggs
            slot_details[slot_index, 3] = num_eggs / 5.0  # Normalized eggs (0-5)
            # Max eggs for slot
            slot_details[slot_index, 4 + num_nests] = 1.0  # One-hot encode max eggs (0-5)
            # Is hatchling / Is grown hatchling (None if no dragon)
            slot_details[slot_index, 10] = 1.0 if (player.hatchling_grown[cave_name][col] == False) else 0.0
            slot_details[slot_index, 11] = 1.0 if (player.hatchling_grown[cave_name][col] == True) else 0.0
            # Cached resources (meat, gold, crystal, milk)
            cached_resources = player.cached_resources[cave_name][col]
            for i, resource in enumerate(RESOURCES):
                slot_details[slot_index, 12 + i] = cached_resources[resource] / 10.0  # Normalize to max 10
            # Tucked dragons count
            slot_details[slot_index, 16] = player.tucked_dragons_count[cave_name][col] / 10.0  # Normalize tucked dragons

    return {
        "slot_types": slot_types_tensor,
        "dragons_on_slots": dragons_on_slots_tensor,
        "slot_details": slot_details
    }


class WyrmspanEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # We assume a max number of legal actions the env will ever return
        self.max_legal_actions = 500
        # The size of your pre-computed Frozen LLM embeddings (e.g., 768 for BERT)
        self.embedding_dim = 768 
        
        self.observation_space = spaces.Dict({
            # 1. Global context (Resources, Guild, Round info)
            "global_stats": spaces.Box(low=0, high=100, shape=(20,), dtype=np.float32),
            
            # 2. Entities (Padded sequences for the Transformer)
            "hand_cards": spaces.Box(low=-5, high=5, shape=(15, self.embedding_dim), dtype=np.float32),
            "board_slots": spaces.Box(low=-5, high=5, shape=(15, self.embedding_dim), dtype=np.float32),
            
            # 3. Action Features (The JSON actions turned into vectors)
            "action_candidates": spaces.Box(low=-5, high=5, shape=(self.max_legal_actions, 128), dtype=np.float32),
            
            # 4. Action Mask (Which indices in action_candidates are real)
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_legal_actions,), dtype=np.int8)
        })

        # The model just outputs the index of the chosen action
        self.action_space = spaces.Discrete(self.max_legal_actions)

    def _get_obs(self):
        # 1. Get JSON actions from your existing engine
        json_actions = self.engine.get_legal_actions()
        
        # 2. Convert JSON actions to vectors (Action Featurizer)
        action_vectors = [self.featurize_json(a) for a in json_actions]
        
        # 3. Pad to max_legal_actions
        padded_actions = np.zeros((self.max_legal_actions, 128))
        mask = np.zeros(self.max_legal_actions)
        
        for i, vec in enumerate(action_vectors):
            padded_actions[i] = vec
            mask[i] = 1
            
        return {
            "global_stats": ...,
            "hand_cards": ...,
            "action_candidates": padded_actions,
            "action_mask": mask
        }