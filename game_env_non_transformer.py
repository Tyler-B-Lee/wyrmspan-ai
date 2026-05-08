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
    timing_tensor[14] = 1.0 if game_state.player.passed_this_round else 0.0  # Player passed this round
    timing_tensor[15] = (game_state.player.score) / 100.0  # Player score (normalized estimate)
    timing_tensor[16] = (game_state.automa.score) / 100.0  # Automa score (normalized estimate)
    # cave exploration counts (4 values x 3 caves)
    for i, cave_name in enumerate(CAVE_NAMES):
        num_explored = game_state.player.times_explored[cave_name]
        timing_tensor[17 + i * 4 + num_explored] = 1.0  # One-hot encode number of times explored (0-3)
    info_dict["timing"] = timing_tensor

    # Guild Board Status (67 combined values)
    guild_tensor = torch.zeros(29, dtype=torch.float32)
    guild_index = game_state.board["guild"]["guild_index"]
    # One-hot encode guild (4 possible guilds)
    if guild_index < 4:
        guild_tensor[guild_index] = 1.0
    guild_tensor[4 + game_state.board["guild"]["player_position"]] = 1.0  # Player position on guild track (0-11)
    guild_tensor[16 + game_state.board["guild"]["automa_position"]] = 1.0  # Automa position on guild track
    guild_tensor[27] = game_state.player.guild_markers / 4.0  # Player guild markers remaining
    guild_tensor[28] = game_state.board["guild"]["automa_markers_ready"] / 4.0  # Automa guild markers ready
    # Guild ability uses per player
    guild_ability_tensor = torch.zeros((5, 6), dtype=torch.float32)
    ability_uses = game_state.board["guild"]["ability_uses"]
    for i in range(1, 6):
        uses = ability_uses[i]
        # get count per player for this ability
        for player_i in range(2):
            count = min(uses.count(player_i), 3)
            if count > 0:
                guild_ability_tensor[i - 1, count - 1 + player_i * 3] = 1.0
    # append guild ability uses to the end of guild tensor
    guild_tensor = torch.cat((guild_tensor, guild_ability_tensor.flatten()), dim=0)
    # extra info of end game point ability
    end_game_ability_tensor = torch.zeros((2,4), dtype=torch.float32)
    for player_i in range(2):
        count = ability_uses[5][2:].count(player_i)
        if count > 0:
            end_game_ability_tensor[player_i, count - 1] = 1.0
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

    # Player Resources (16 values: hand sizes, coins, resources, eggs)
    player_tensor = torch.zeros(16, dtype=torch.float32)
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
    player_tensor[12] = game_state.player.num_dragons_played["crimson_cavern"] / 4.0  # Dragons in crimson cavern
    player_tensor[13] = game_state.player.num_dragons_played["golden_grotto"] / 4.0  # Dragons in golden grotto
    player_tensor[14] = game_state.player.num_dragons_played["amethyst_abyss"] / 4.0  # Dragons in amethyst abyss
    player_tensor[15] = sum(game_state.player.num_dragons_played.values()) / 12.0  # Dragons on mat
    info_dict["player_resources"] = player_tensor

    # Automa Status (29 values)
    automa_tensor = torch.zeros(29, dtype=torch.float32)
    for i in range(4):
        automa_tensor[i] = game_state.board["round_tracker"]["automa_bonus"][i] / 4.0
    automa_tensor[game_state.automa.difficulty + 4] = 1.0  # Automa difficulty level
    automa_tensor[10] = game_state.automa.score / 100.0  # Automa score
    automa_tensor[11] = len(game_state.automa.dragons) / 16.0  # Automa dragons collected
    automa_tensor[12] = len(game_state.automa.caves) / 10.0  # Automa caves collected
    automa_tensor[13] = 1.0 if game_state.automa.passed_this_round else 0.0  # Automa passed this round
    for card_i in game_state.automa.decision_deck:
        automa_tensor[14 + card_i] = 1.0  # One-hot encode automa decision deck cards
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
            slot_details[slot_index, 16] = player.tucked_dragons[cave_name][col] / 10.0  # Normalize tucked dragons

    return {
        "slot_types": slot_types_tensor,
        "dragons_on_slots": dragons_on_slots_tensor,
        "slot_details": slot_details
    }


class WyrmspanEnv(gym.Env):
    ACTION_VEC_SIZE = 128  # Reduced from 192 (uses only 93 dims, padding was 99 dims)
    ACTION_TYPE_OFFSET = 0
    ACTION_TYPE_SIZE = 32
    WRAP_OFFSET = 32
    WRAP_SIZE = 5
    CAVE_OFFSET = 37
    CAVE_SIZE = 5
    COL_OFFSET = 42
    COL_SIZE = 4
    COORD_FLAG_INDEX = 46
    COL_NORM_INDEX = 47
    CAVE_NORM_INDEX = 48
    COST_OFFSET = 49
    COST_SIZE = 9
    SOURCE_OFFSET = 58
    SOURCE_SIZE = 6
    TARGET_OFFSET = 64
    TARGET_SIZE = 4
    DISCOUNT_OFFSET = 68
    DISCOUNT_SIZE = 4
    RESOURCE_TYPE_OFFSET = 72
    RESOURCE_TYPE_SIZE = 4
    PERSONALITY_OFFSET = 76
    PERSONALITY_SIZE = 4
    DRAGON_ID_INDEX = 80
    CAVE_ID_INDEX = 81
    DISPLAY_INDEX_INDEX = 82
    AUX_INDEX = 83
    RAND_FLAG_INDEX = 84
    RAND_SOURCE_OFFSET = 85
    RAND_SOURCE_SIZE = 3
    SEQ_LEN_INDEX = 88
    CHOICE_LEN_INDEX = 89
    SKIP_FLAG_INDEX = 90
    PASS_FLAG_INDEX = 91
    HAS_COST_FLAG_INDEX = 92

    def __init__(self, engine_fn=None):
        super().__init__()
        
        # Function to initialize or access the game engine
        self.engine_fn = engine_fn
        self.engine = None
        
        # We assume a max number of legal actions the env will ever return
        self.max_legal_actions = 180
        self.action_dim = self.ACTION_VEC_SIZE
        # The size of your pre-computed Frozen LLM embeddings (e.g., 768 for BERT)
        self.embedding_dim = 768 

        self.action_type_keys = [
            "play_dragon",
            "play_cave",
            "explore",
            "lay_egg",
            "gain_resource",
            "gain_guild",
            "gain_coin",
            "gain_dragon",
            "gain_cave",
            "cache_from",
            "tuck_from",
            "make_payment",
            "deduct_resources",
            "discard_dragon",
            "discard_cave",
            "swap_dragons",
            "pass",
            "skip",
            "skip_opr",
            "brown_space",
            "4th_space",
            "automa_action",
            "automa_guild_move",
            "top_deck_reveal",
            "draw_decision",
            "other_ability_on_mat",
            "any_resource_decision",
            "end_game",
            "opr_option",
            "refill_dragon_display",
            "refill_cave_display",
        ]
        self.action_type_index = {key: idx for idx, key in enumerate(self.action_type_keys)}
        self.source_keys = ["hand", "display", "deck", "player_supply", "general_supply", "any"]
        self.source_index = {key: idx for idx, key in enumerate(self.source_keys)}
        self.target_keys = ["here", "any", "this_column", "dragon_on_mat"]
        self.target_index = {key: idx for idx, key in enumerate(self.target_keys)}
        self.discount_keys = ["none", "free", "1off", "no_resources"]
        self.discount_index = {key: idx for idx, key in enumerate(self.discount_keys)}
        self.random_source_keys = ["dragon_deck", "cave_deck", "automa_deck"]
        self.random_source_index = {key: idx for idx, key in enumerate(self.random_source_keys)}
        
        # Maximum number of cards that can be referenced by a single action
        self.max_cards_per_action = 5
        
        self.observation_space = spaces.Dict({
            # 1. Global context (Resources, Guild, Round info)
            "global_stats": spaces.Box(low=0, high=100, shape=(20,), dtype=np.float32),
            
            # 2. Card IDs for hand and board (model will handle embeddings)
            "hand_card_ids": spaces.Box(low=0, high=200, shape=(15,), dtype=np.int64),
            "board_slot_card_ids": spaces.Box(low=0, high=200, shape=(12,), dtype=np.int64),
            
            # 3. Action Features (128-dim scalar features only; embedding fusion happens in model)
            "action_candidates": spaces.Box(low=-5, high=5, shape=(self.max_legal_actions, self.action_dim), dtype=np.float32),

            # 4. Action Card References ([card_kind, card_id] pairs per action, padded)
            # card_kind: 0=dragon, 1=cave
            "action_cards": spaces.Box(low=0, high=200, shape=(self.max_legal_actions, self.max_cards_per_action, 2), dtype=np.int64),
            
            # 5. Action Card IDs (legacy, kept for backward compatibility during transition)
            "action_card_ids": spaces.Box(low=0, high=200, shape=(self.max_legal_actions, 2), dtype=np.int64),
            
            # 6. Action Mask (Which indices in action_candidates are real)
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_legal_actions,), dtype=np.int8)
        })

        # The model just outputs the index of the chosen action
        self.action_space = spaces.Discrete(self.max_legal_actions)

    def featurize_json(self, action_json):
        """
        Convert a single JSON action into a fixed-size vector representation.
        """
        vec = np.zeros(self.action_dim, dtype=np.float32)

        def set_one_hot(offset, size, index):
            if index is None:
                return
            if 0 <= index < size:
                vec[offset + index] = 1.0

        def norm(value, denom):
            if denom <= 0:
                return 0.0
            return float(value) / denom

        def merge_costs(costs, cost_dict):
            if not isinstance(cost_dict, dict):
                return
            for key, value in cost_dict.items():
                if key == "egg":
                    if isinstance(value, dict):
                        costs["egg"] = costs.get("egg", 0) + int(value.get("amount", 0))
                    elif isinstance(value, (list, tuple)):
                        costs["egg"] = costs.get("egg", 0) + len(value)
                elif key in ("dragon_card", "cave_card"):
                    if isinstance(value, (list, tuple)):
                        costs[key] = costs.get(key, 0) + len(value)
                    else:
                        costs[key] = costs.get(key, 0) + int(value)
                else:
                    if isinstance(value, dict):
                        continue
                    costs[key] = costs.get(key, 0) + int(value)

        def collect_action_keys(obj, found, depth=3):
            if depth < 0 or not isinstance(obj, dict):
                return
            for key in obj.keys():
                if key in self.action_type_index:
                    found.add(key)
            if "make_payment" in obj:
                inner = obj["make_payment"].get("action", {})
                collect_action_keys(inner, found, depth - 1)
            if "adv_effects" in obj:
                collect_action_keys(obj["adv_effects"], found, depth - 1)
            if "random" in obj:
                collect_action_keys(obj["random"], found, depth - 1)
            if "choice" in obj and isinstance(obj["choice"], list):
                for item in obj["choice"][:3]:
                    collect_action_keys(item, found, depth - 1)
            if "sequence" in obj and isinstance(obj["sequence"], list):
                for item in obj["sequence"][:3]:
                    collect_action_keys(item, found, depth - 1)

        action_keys = set()
        collect_action_keys(action_json, action_keys)
        for key in action_keys:
            set_one_hot(self.ACTION_TYPE_OFFSET, self.ACTION_TYPE_SIZE, self.action_type_index.get(key))

        wrappers = {
            "make_payment": False,
            "adv_effects": False,
            "choice": False,
            "random": False,
            "sequence": False,
        }
        costs = {}
        merge_costs(costs, action_json.get("cost", {}))

        action = action_json
        if "make_payment" in action:
            wrappers["make_payment"] = True
            merge_costs(costs, action["make_payment"].get("cost", {}))
            action = action["make_payment"].get("action", action)
        if "adv_effects" in action:
            wrappers["adv_effects"] = True
            action = action["adv_effects"]

        random_event = None
        if isinstance(action, dict) and "choice" in action:
            wrappers["choice"] = True
            vec[self.CHOICE_LEN_INDEX] = norm(len(action["choice"]), 10.0)
            if action["choice"]:
                action = action["choice"][0]
        if isinstance(action, dict) and "sequence" in action:
            wrappers["sequence"] = True
            vec[self.SEQ_LEN_INDEX] = norm(len(action["sequence"]), 10.0)
            if action["sequence"]:
                action = action["sequence"][0]
        if isinstance(action, dict) and "random" in action:
            wrappers["random"] = True
            random_event = action["random"]
            action = random_event

        for i, key in enumerate(["make_payment", "adv_effects", "choice", "random", "sequence"]):
            if wrappers[key]:
                vec[self.WRAP_OFFSET + i] = 1.0

        for res in RESOURCES:
            if res in costs:
                value = min(costs[res], 10)
                vec[self.COST_OFFSET + RESOURCES.index(res)] = -norm(value, 5.0)
        cost_offset = self.COST_OFFSET + len(RESOURCES)
        coin_cost = min(costs.get("coin", 0), 10)
        vec[cost_offset] = -norm(coin_cost, 5.0)
        vec[cost_offset + 1] = -norm(min(costs.get("egg", 0), 10), 4.0)
        vec[cost_offset + 2] = -norm(min(costs.get("dragon_card", 0), 10), 3.0)
        vec[cost_offset + 3] = -norm(min(costs.get("cave_card", 0), 10), 3.0)
        vec[cost_offset + 4] = -norm(min(costs.get("any_resource", 0), 10), 5.0)
        if costs:
            vec[self.HAS_COST_FLAG_INDEX] = 1.0

        event = action if isinstance(action, dict) else {}
        
        # Extract coordinates from various locations in the action structure
        # Search recursively through make_payment wrappers first
        def find_coords(obj):
            """Recursively search for coords in nested action structures."""
            if not isinstance(obj, dict):
                return None
            # Check top level
            if "coords" in obj and isinstance(obj["coords"], tuple):
                return obj["coords"]
            # Check inside make_payment
            if "make_payment" in obj and isinstance(obj["make_payment"], dict):
                inner_coords = find_coords(obj["make_payment"].get("action", {}))
                if inner_coords:
                    return inner_coords
            # Check inside adv_effects
            if "adv_effects" in obj and isinstance(obj["adv_effects"], dict):
                inner_coords = find_coords(obj["adv_effects"])
                if inner_coords:
                    return inner_coords
            return None
        
        coords = find_coords(action_json)
        cave_name = None
        col = None
        explore_index = None
        if isinstance(coords, tuple) and len(coords) == 2:
            cave_name, col = coords
        elif "cave_location" in event:
            cave_name = event.get("cave_location")
        elif "explore" in event and isinstance(event["explore"], dict):
            cave_name = event["explore"].get("cave_name")
            explore_index = event["explore"].get("index")
        # Check for cave_location nested in action-specific dicts
        if cave_name is None:
            if "play_cave" in event and isinstance(event["play_cave"], dict):
                cave_name = event["play_cave"].get("cave_location")
            elif "draw_decision" in event and isinstance(event["draw_decision"], dict):
                # Try to get cave_location from draw_decision (if present)
                cave_name = event["draw_decision"].get("cave_location")

        if cave_name in CAVE_NAMES:
            set_one_hot(self.CAVE_OFFSET, self.CAVE_SIZE, CAVE_NAMES.index(cave_name))
            vec[self.CAVE_NORM_INDEX] = norm(CAVE_NAMES.index(cave_name), max(1, len(CAVE_NAMES) - 1))
            vec[self.COORD_FLAG_INDEX] = 1.0
        elif cave_name == "mat_slots":
            set_one_hot(self.CAVE_OFFSET, self.CAVE_SIZE, 3)
            vec[self.COORD_FLAG_INDEX] = 1.0
        else:
            set_one_hot(self.CAVE_OFFSET, self.CAVE_SIZE, 4)

        if col is not None:
            col_index = int(col)
            set_one_hot(self.COL_OFFSET, self.COL_SIZE, col_index)
            vec[self.COL_NORM_INDEX] = norm(col_index, 3.0)

        if "play_dragon" in event:
            pd = event["play_dragon"]
            source = pd.get("L1")
            target = pd.get("L2")
            discount = pd.get("discount", "none")
            dragon_id = pd.get("chosen_id", None)
            display_idx = pd.get("chosen_index", None)
            set_one_hot(self.SOURCE_OFFSET, self.SOURCE_SIZE, self.source_index.get(source))
            set_one_hot(self.TARGET_OFFSET, self.TARGET_SIZE, self.target_index.get(target))
            set_one_hot(self.DISCOUNT_OFFSET, self.DISCOUNT_SIZE, self.discount_index.get(discount))
            if dragon_id is not None:
                vec[self.DRAGON_ID_INDEX] = norm(dragon_id, 183.0)
            if display_idx is not None:
                vec[self.DISPLAY_INDEX_INDEX] = norm(display_idx, 2.0)

        if "play_cave" in event:
            pc = event["play_cave"]
            source = pc.get("source")
            cave_id = pc.get("chosen_id", None)
            display_idx = pc.get("chosen_index", None)
            set_one_hot(self.SOURCE_OFFSET, self.SOURCE_SIZE, self.source_index.get(source))
            if cave_id is not None:
                vec[self.CAVE_ID_INDEX] = norm(cave_id, 75.0)
            if display_idx is not None:
                vec[self.DISPLAY_INDEX_INDEX] = norm(display_idx, 2.0)

        if "gain_dragon" in event:
            gd = event["gain_dragon"]
            source = gd.get("source", "any")
            display_idx = gd.get("chosen", None)
            rand_outcome = gd.get("rand_outcome", None)
            set_one_hot(self.SOURCE_OFFSET, self.SOURCE_SIZE, self.source_index.get(source))
            if display_idx is not None:
                vec[self.DISPLAY_INDEX_INDEX] = norm(display_idx, 2.0)
            if rand_outcome is not None and rand_outcome >= 0:
                vec[self.DRAGON_ID_INDEX] = norm(rand_outcome, 183.0)
                vec[self.RAND_FLAG_INDEX] = 1.0

        if "gain_cave" in event:
            gc = event["gain_cave"]
            display_idx = gc.get("chosen", None)
            rand_outcome = gc.get("rand_outcome", None)
            if display_idx is not None:
                vec[self.DISPLAY_INDEX_INDEX] = norm(display_idx, 2.0)
            if rand_outcome is not None and rand_outcome >= 0:
                vec[self.CAVE_ID_INDEX] = norm(rand_outcome, 75.0)
                vec[self.RAND_FLAG_INDEX] = 1.0

        if "gain_resource" in event:
            res_type = event["gain_resource"].get("type")
            if res_type in RESOURCES:
                set_one_hot(self.RESOURCE_TYPE_OFFSET, self.RESOURCE_TYPE_SIZE, RESOURCES.index(res_type))

        if "cache_from" in event:
            cf = event["cache_from"]
            source = cf.get("L1")
            target = cf.get("L2")
            res_type = cf.get("type")
            set_one_hot(self.SOURCE_OFFSET, self.SOURCE_SIZE, self.source_index.get(source))
            set_one_hot(self.TARGET_OFFSET, self.TARGET_SIZE, self.target_index.get(target))
            if res_type in RESOURCES:
                set_one_hot(self.RESOURCE_TYPE_OFFSET, self.RESOURCE_TYPE_SIZE, RESOURCES.index(res_type))

        if "tuck_from" in event:
            tf = event["tuck_from"]
            source = tf.get("L1")
            target = tf.get("L2")
            display_idx = tf.get("chosen_index", None)
            dragon_id = tf.get("chosen_id", None)
            rand_outcome = tf.get("rand_outcome", None)
            include = tf.get("include", None)
            set_one_hot(self.SOURCE_OFFSET, self.SOURCE_SIZE, self.source_index.get(source))
            set_one_hot(self.TARGET_OFFSET, self.TARGET_SIZE, self.target_index.get(target))
            if display_idx is not None:
                vec[self.DISPLAY_INDEX_INDEX] = norm(display_idx, 2.0)
            if dragon_id is not None:
                vec[self.DRAGON_ID_INDEX] = norm(dragon_id, 183.0)
            if rand_outcome is not None and rand_outcome >= 0:
                vec[self.DRAGON_ID_INDEX] = norm(rand_outcome, 183.0)
                vec[self.RAND_FLAG_INDEX] = 1.0
            if include in DRAGON_PERSONALITIES:
                set_one_hot(self.PERSONALITY_OFFSET, self.PERSONALITY_SIZE, DRAGON_PERSONALITIES.index(include))

        if "lay_egg" in event:
            location = event["lay_egg"].get("location")
            set_one_hot(self.TARGET_OFFSET, self.TARGET_SIZE, self.target_index.get(location))

        if "explore" in event:
            exp = event["explore"]
            exp_index = exp.get("index", explore_index)
            if exp_index is not None:
                vec[self.AUX_INDEX] = norm(exp_index, 4.0)

        if isinstance(random_event, dict):
            for key in ("play_cave", "gain_cave"):
                if key in random_event and "possible_outcomes" in random_event[key]:
                    set_one_hot(self.RAND_SOURCE_OFFSET, self.RAND_SOURCE_SIZE, self.random_source_index.get(random_event[key]["possible_outcomes"]))
            for key in ("gain_dragon", "tuck_from"):
                if key in random_event and "possible_outcomes" in random_event[key]:
                    set_one_hot(self.RAND_SOURCE_OFFSET, self.RAND_SOURCE_SIZE, self.random_source_index.get(random_event[key]["possible_outcomes"]))
            if "automa_action" in random_event:
                set_one_hot(self.RAND_SOURCE_OFFSET, self.RAND_SOURCE_SIZE, self.random_source_index.get("automa_deck"))

        if "skip" in event:
            vec[self.SKIP_FLAG_INDEX] = 1.0
        if "pass" in event:
            vec[self.PASS_FLAG_INDEX] = 1.0
        
        if "draw_decision" in event:
            dd = event["draw_decision"]
            chosen_id = dd.get("chosen_id", None)
            if chosen_id is not None:
                vec[self.DRAGON_ID_INDEX] = norm(chosen_id, 183.0)

        return vec

    def _resolve_display_card(self, display_type, index):
        if self.engine is None:
            return None
        game_state = getattr(self.engine, "game_state", None)
        if game_state is None:
            return None
        display = game_state.board.get("card_display", {})
        card_list = display.get(display_type, None)
        if card_list is None:
            return None
        if index is None or index < 0 or index >= len(card_list):
            return None
        return card_list[index]

    def extract_action_card_ids(self, action_json):
        dragon_id = 0
        cave_id = 0

        action = action_json
        if "make_payment" in action:
            action = action["make_payment"].get("action", action)
        if "adv_effects" in action:
            action = action["adv_effects"]
        if isinstance(action, dict) and "choice" in action:
            if action["choice"]:
                action = action["choice"][0]
        if isinstance(action, dict) and "sequence" in action:
            if action["sequence"]:
                action = action["sequence"][0]
        if isinstance(action, dict) and "random" in action:
            action = action["random"]

        event = action if isinstance(action, dict) else {}

        if "play_dragon" in event:
            pd = event["play_dragon"]
            dragon_id = pd.get("chosen_id", 0) or 0
            display_idx = pd.get("chosen_index", None)
            if dragon_id == 0 and display_idx is not None:
                resolved = self._resolve_display_card("dragon_cards", display_idx)
                if resolved is not None:
                    dragon_id = int(resolved)

        if "tuck_from" in event:
            tf = event["tuck_from"]
            dragon_id = tf.get("chosen_id", dragon_id) or dragon_id
            rand_outcome = tf.get("rand_outcome", None)
            if rand_outcome is not None and rand_outcome >= 0:
                dragon_id = int(rand_outcome)
            display_idx = tf.get("chosen_index", None)
            if dragon_id == 0 and display_idx is not None:
                resolved = self._resolve_display_card("dragon_cards", display_idx)
                if resolved is not None:
                    dragon_id = int(resolved)

        if "gain_dragon" in event:
            gd = event["gain_dragon"]
            rand_outcome = gd.get("rand_outcome", None)
            if rand_outcome is not None and rand_outcome >= 0:
                dragon_id = int(rand_outcome)
            display_idx = gd.get("chosen", None)
            if dragon_id == 0 and display_idx is not None:
                resolved = self._resolve_display_card("dragon_cards", display_idx)
                if resolved is not None:
                    dragon_id = int(resolved)

        if "play_cave" in event:
            pc = event["play_cave"]
            cave_id = pc.get("chosen_id", 0) or 0
            display_idx = pc.get("chosen_index", None)
            if cave_id == 0 and display_idx is not None:
                resolved = self._resolve_display_card("cave_cards", display_idx)
                if resolved is not None:
                    cave_id = int(resolved)

        if "gain_cave" in event:
            gc = event["gain_cave"]
            rand_outcome = gc.get("rand_outcome", None)
            if rand_outcome is not None and rand_outcome >= 0:
                cave_id = int(rand_outcome)
            display_idx = gc.get("chosen", None)
            if cave_id == 0 and display_idx is not None:
                resolved = self._resolve_display_card("cave_cards", display_idx)
                if resolved is not None:
                    cave_id = int(resolved)

        return dragon_id, cave_id

    def extract_action_card_refs(self, action_json):
        """
        Extract card references from an action as a list of [card_kind, card_id] pairs.
        card_kind: 0 = dragon, 1 = cave
        Returns a list of [card_kind, card_id] pairs (may be empty for some actions).
        """
        refs = []
        
        action = action_json
        if "make_payment" in action:
            action = action["make_payment"].get("action", action)
        if "adv_effects" in action:
            action = action["adv_effects"]
        if isinstance(action, dict) and "choice" in action:
            if action["choice"]:
                action = action["choice"][0]
        if isinstance(action, dict) and "sequence" in action:
            if action["sequence"]:
                action = action["sequence"][0]
        if isinstance(action, dict) and "random" in action:
            action = action["random"]

        event = action if isinstance(action, dict) else {}

        # Extract dragon references
        if "play_dragon" in event:
            pd = event["play_dragon"]
            dragon_id = pd.get("chosen_id", 0) or 0
            if dragon_id == 0:
                display_idx = pd.get("chosen_index", None)
                if display_idx is not None:
                    resolved = self._resolve_display_card("dragon_cards", display_idx)
                    if resolved is not None:
                        dragon_id = int(resolved)
            if dragon_id > 0:
                refs.append([0, dragon_id])

        if "tuck_from" in event:
            tf = event["tuck_from"]
            dragon_id = tf.get("chosen_id", 0) or 0
            if dragon_id == 0:
                rand_outcome = tf.get("rand_outcome", None)
                if rand_outcome is not None and rand_outcome >= 0:
                    dragon_id = int(rand_outcome)
            if dragon_id == 0:
                display_idx = tf.get("chosen_index", None)
                if display_idx is not None:
                    resolved = self._resolve_display_card("dragon_cards", display_idx)
                    if resolved is not None:
                        dragon_id = int(resolved)
            if dragon_id > 0:
                refs.append([0, dragon_id])

        if "gain_dragon" in event:
            gd = event["gain_dragon"]
            dragon_id = 0
            rand_outcome = gd.get("rand_outcome", None)
            if rand_outcome is not None and rand_outcome >= 0:
                dragon_id = int(rand_outcome)
            if dragon_id == 0:
                display_idx = gd.get("chosen", None)
                if display_idx is not None:
                    resolved = self._resolve_display_card("dragon_cards", display_idx)
                    if resolved is not None:
                        dragon_id = int(resolved)
            if dragon_id > 0:
                refs.append([0, dragon_id])

        # Extract cave references
        if "play_cave" in event:
            pc = event["play_cave"]
            cave_id = pc.get("chosen_id", 0) or 0
            if cave_id == 0:
                display_idx = pc.get("chosen_index", None)
                if display_idx is not None:
                    resolved = self._resolve_display_card("cave_cards", display_idx)
                    if resolved is not None:
                        cave_id = int(resolved)
            if cave_id > 0:
                refs.append([1, cave_id])

        if "gain_cave" in event:
            gc = event["gain_cave"]
            cave_id = 0
            rand_outcome = gc.get("rand_outcome", None)
            if rand_outcome is not None and rand_outcome >= 0:
                cave_id = int(rand_outcome)
            if cave_id == 0:
                display_idx = gc.get("chosen", None)
                if display_idx is not None:
                    resolved = self._resolve_display_card("cave_cards", display_idx)
                    if resolved is not None:
                        cave_id = int(resolved)
            if cave_id > 0:
                refs.append([1, cave_id])

        return refs

    def _get_obs(self):
        """Get observation from game state and legal actions."""
        # 1. Get JSON actions from engine
        if self.engine is None or not hasattr(self.engine, 'get_legal_actions'):
            json_actions = []
        else:
            json_actions = self.engine.get_legal_actions()
        
        # 2. Convert JSON actions to vectors (Action Featurizer)
        action_vectors = [self.featurize_json(a) for a in json_actions]
        action_card_ids = [self.extract_action_card_ids(a) for a in json_actions]  # Legacy: (dragon_id, cave_id)
        action_card_refs = [self.extract_action_card_refs(a) for a in json_actions]  # New: list of [card_kind, card_id]
        
        # 3. Pad to max_legal_actions
        padded_actions = np.zeros((self.max_legal_actions, self.action_dim), dtype=np.float32)
        padded_action_cards = np.zeros((self.max_legal_actions, 2), dtype=np.int64)  # Legacy
        padded_action_refs = np.zeros((self.max_legal_actions, self.max_cards_per_action, 2), dtype=np.int64)  # New
        mask = np.zeros(self.max_legal_actions, dtype=np.int8)
        
        for i, vec in enumerate(action_vectors):
            if i >= self.max_legal_actions:
                break
            padded_actions[i] = vec
            padded_action_cards[i] = action_card_ids[i]  # Legacy
            # Pack variable-length refs into fixed-size tensor
            refs = action_card_refs[i]
            for j, ref in enumerate(refs[:self.max_cards_per_action]):
                padded_action_refs[i, j] = ref
            mask[i] = 1
        
        # 4. Extract card IDs from game state
        hand_card_ids = np.zeros(15, dtype=np.int64)
        board_slot_card_ids = np.zeros(12, dtype=np.int64)
        
        game_state = getattr(self.engine, "game_state", None) if self.engine else None
        if game_state is not None:
            player = game_state.player
            # Hand: player.dragon_hand is a list of dragon IDs
            for i, dragon_id in enumerate(player.dragon_hand[:15]):
                hand_card_ids[i] = dragon_id
            # Board: player.dragons_played[cave_name][col] has dragon IDs in 4 slots per cave (3 caves = 12 total)
            board_idx = 0
            for cave_name in ["crimson_cavern", "golden_grotto", "amethyst_abyss"]:
                if cave_name in player.dragons_played:
                    for dragon_id in player.dragons_played[cave_name]:
                        if board_idx < 12:
                            if dragon_id is not None:
                                board_slot_card_ids[board_idx] = dragon_id
                            board_idx += 1
            
        return {
            "global_stats": np.zeros(20, dtype=np.float32),
            "hand_card_ids": hand_card_ids,
            "board_slot_card_ids": board_slot_card_ids,
            "action_candidates": padded_actions,
            "action_cards": padded_action_refs,
            "action_card_ids": padded_action_cards,  # Legacy, kept for backward compatibility
            "action_mask": mask
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.engine_fn:
            self.engine = self.engine_fn()
        return self._get_obs(), {}
        
    def step(self, action_idx):
        json_list = self.engine.get_legal_actions()
        chosen_json = json_list[action_idx]
        
        # Example engine hook - adapt this to your real game logic/engine execute setup
        new_state, reward, done = self.engine.execute(chosen_json)
        
        obs = self._get_obs()
        return obs, reward, done, False, {}