from game_states import *

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any

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

    def __init__(self, engine_fn=None):
        super().__init__()
        
        # Function to initialize or access the game engine
        self.engine_fn = engine_fn
        self.engine = None
        
        # We assume a max number of legal actions the env will ever return
        self.max_legal_actions = 180
        self.action_dim = self.ACTION_VEC_SIZE

        self.token_strings = [
            # numbers (for amounts of resources, costs, points, etc.)
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",

            # basic actions
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
            "skip_choice", # for draw/any resource decisions
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

            # action modifiers and contexts
            "L1",
            "L2",
            "discount",
            "hand",
            "display",
            "deck",
            "player_supply",
            "general_supply",
            "any",
            "here",
            "this_column",
            "this_cave",
            "each_this_column",
            "each_this_cave",
            "dragon_on_mat",
            "ortho",
            "none",
            "free",
            "1off",
            "no_resources",
            "dragon_deck",
            "cave_deck",
            "automa_deck",
            "include",
            "exclude",
            "keep_card",
            "gain_meat",
            "gain_milk",

            # resources
            "meat",
            "gold",
            "crystal",
            "milk",
            "any_resource",
            "egg",
            "dragon_card",
            "cave_card",
            "coin",

            # cave_names
            "crimson_cavern",
            "golden_grotto",
            "amethyst_abyss",
            "mat_slots",

            # columns
            "col_0",
            "col_1",
            "col_2",
            "col_3",

            # Display Positions
            "display_0",
            "display_1",
            "display_2",

            # Explore Indices
            "explore_0",
            "explore_1",
            "explore_2",
            "explore_3",
            "explore_4",
            "explore_5",
            "explore_6",
            "explore_7",
            "explore_8",

            # Brown Space Abilities
            "brown_space_1",
            "brown_space_2",
            "brown_space_3",
            "brown_space_4",
            "brown_space_5",

            # Personalities
            "Helpful",
            "Shy",
            "Playful",
            "Aggressive",

            # Sizes
            "Hatchling",
            "Small",
            "Medium",
            "Large",
            
            # Ability Types
            "once_per_round",
            "if_activated",
            "end_game",
            "when_played",

            # advanced effects
            "adv_effects_start",
            "adv_effects_end",
            "sequence_start",
            "sequence_end",
            "sequence_item_start",
            "sequence_item_end",
            "choice_start",
            "choice_end",
            "choice_item_start",
            "choice_item_end",
            "random_start",
            "random_end",
            "cost_start",
            "cost_end",
            "condition_start",
            "condition_end",
            "or_start",
            "or_end",
            "or_option_start",
            "or_option_end",
            "and_start",
            "and_end",
            "and_option_start",
            "and_option_end",
            "coords",
            "max_uses",
            "chosen_payment_start",
            "chosen_payment_end",

            # other modifiers
            "rand_outcome",
            "gain_from_cost",
            "loc_info",
            "draw_decision_keep",
            "draw_decision_discard",
            "draw_decision_tuck_here",
            "draw_decision_tuck_any",
            "draw_decision_chosen_id",
            "draw_decision_remaining_dragons",
            "any_resource_decision_cache_any",
            "any_resource_decision_cache_here",
            "any_resource_decision_keep",
            "any_resource_decision_chosen_type",
            "any_resource_decision_remaining_resources",

            # placeholders for very long or complex actions we don't have room to fully encode
            "placeholder_tuck_from"
        ]

        # add dragons and caves
        for dragon_id in range(1, 184):
            self.token_strings.append(f"dragon_{dragon_id}")
        for cave_id in range(1, 76):
            self.token_strings.append(f"cave_{cave_id}")

        self.token_index = {key: idx for idx, key in enumerate(self.token_strings)}
        
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

    def tokenize_json(self, action_json, as_ids=False):
        """
        Convert a single JSON action into a sequence of semantic tokens.

        Returns a list of token strings by default.
        Pass `as_ids=True` to map known tokens to their integer vocabulary ids.
        """
        def encode_number(value: Any) -> str:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return str(value)
            if num.is_integer():
                return str(int(num))
            return str(num)

        def to_card_token(card_kind: str, card_id: Any) -> str:
            if card_kind == "dragon":
                return f"dragon_{int(card_id)}"
            if card_kind == "cave":
                return f"cave_{int(card_id)}"
            return encode_number(card_id)

        def normalize_cost_value(cost_key: str, raw_value: Any):
            if cost_key in ("dragon_card", "cave_card") and isinstance(raw_value, (list, tuple)):
                return len(raw_value)
            if cost_key == "egg":
                if isinstance(raw_value, dict):
                    return raw_value
                if isinstance(raw_value, (list, tuple)):
                    return raw_value
            if isinstance(raw_value, dict):
                return None
            return raw_value

        def emit_egg_locations(tokens, raw_value):
            if isinstance(raw_value, dict):
                amount = raw_value.get("amount", 0)
                location = raw_value.get("location")
                tokens.append("egg")
                tokens.append(encode_number(amount))
                if location is not None:
                    emit_scalar(tokens, location)
                return

            if isinstance(raw_value, (list, tuple)):
                tokens.append("egg")
                for item in raw_value:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        cave_name, col = item
                        if cave_name is not None:
                            tokens.append(str(cave_name))
                        if col is not None:
                            try:
                                tokens.append(f"col_{int(col)}")
                            except (TypeError, ValueError):
                                tokens.append(str(col))
                    else:
                        emit_scalar(tokens, item)
                return

            tokens.append("egg")
            emit_scalar(tokens, raw_value)

        def emit_coords(tokens, coords):
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                return
            cave_name, col = coords
            tokens.append("coords")
            if cave_name is not None:
                tokens.append(str(cave_name))
            if col is not None:
                try:
                    tokens.append(f"col_{int(col)}")
                except (TypeError, ValueError):
                    tokens.append(str(col))

        def emit_cost(tokens, cost_dict, include_card_choices: bool = False):
            if not isinstance(cost_dict, dict) or not cost_dict:
                return
            tokens.append("cost_start")
            for key, raw_value in cost_dict.items():
                if include_card_choices and key in {"dragon_card", "cave_card"} and isinstance(raw_value, (list, tuple)):
                    tokens.append(str(key))
                    tokens.append(encode_number(len(raw_value)))
                    for card_id in raw_value:
                        if key == "dragon_card":
                            tokens.append(to_card_token("dragon", card_id))
                        else:
                            tokens.append(to_card_token("cave", card_id))
                    continue

                normalized = normalize_cost_value(key, raw_value)
                if normalized is None:
                    continue
                if key == "egg":
                    emit_egg_locations(tokens, normalized)
                    continue
                tokens.append(str(key))
                tokens.append(encode_number(normalized))
            tokens.append("cost_end")

        def emit_scalar(tokens, value):
            if isinstance(value, bool):
                tokens.append("1" if value else "0")
            elif isinstance(value, (int, float)):
                tokens.append(encode_number(value))
            elif value is not None:
                tokens.append(str(value))

        def emit_key_value(tokens, action_key: str, field_key: str, value):
            if field_key == "chosen_id":
                if action_key in {"play_dragon", "gain_dragon", "tuck_from", "draw_decision", "discard_dragon"}:
                    tokens.append(to_card_token("dragon", value))
                    return
                if action_key in {"play_cave", "gain_cave", "discard_cave"}:
                    tokens.append(to_card_token("cave", value))
                    return
            
            if field_key == "dragon_id":
                tokens.append(to_card_token("dragon", value))
                return

            if field_key in {"chosen", "chosen_index"}:
                try:
                    tokens.append(f"display_{int(value)}")
                except (TypeError, ValueError):
                    emit_scalar(tokens, value)
                return

            if field_key in {"coords", "loc_info"}:
                emit_coords(tokens, value)
                return

            if field_key in {"cave_location", "location", "cave_name"}:
                emit_scalar(tokens, value)
                return

            if field_key == "index":
                try:
                    tokens.append(f"explore_{int(value)}")
                except (TypeError, ValueError):
                    emit_scalar(tokens, value)
                return

            if field_key == "rand_outcome":
                tokens.append("rand_outcome")
                if action_key in {"gain_dragon", "tuck_from", "play_dragon", "draw_decision"}:
                    tokens.append(to_card_token("dragon", value))
                elif action_key in {"gain_cave", "play_cave"}:
                    tokens.append(to_card_token("cave", value))
                else:
                    emit_scalar(tokens, value)
                return

            # Special handling for draw_decision modifiers
            if action_key == "draw_decision":
                if field_key == "limits" and isinstance(value, dict):
                    # emit tokens like draw_decision_keep 1, draw_decision_tuck_here 1
                    for lim_key, lim_val in value.items():
                        tokens.append(f"draw_decision_{lim_key}")
                        emit_scalar(tokens, lim_val)
                    return
                if field_key == "remaining_dragons":
                    tokens.append("draw_decision_remaining_dragons")
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            tokens.append(to_card_token("dragon", v))
                    else:
                        tokens.append(to_card_token("dragon", value))
                    return

            # Swap dragons
            if action_key == "swap_dragons":
                if field_key.startswith("dragon_id"):
                    tokens.append(str(field_key))
                    tokens.append(to_card_token("dragon", value))
                    return
                if field_key.startswith("coords"):
                    tokens.append(str(field_key))
                    cave_name, col = value
                    if cave_name is not None:
                        tokens.append(str(cave_name))
                    if col is not None:
                        try:
                            tokens.append(f"col_{int(col)}")
                        except (TypeError, ValueError):
                            tokens.append(str(col))
                    return

            # Special handling for any_resource_decision modifiers
            if action_key == "any_resource_decision":
                if field_key == "limits" and isinstance(value, dict):
                    for lim_key, lim_val in value.items():
                        tokens.append(f"any_resource_decision_{lim_key}")
                        emit_scalar(tokens, lim_val)
                    return
                if field_key == "remaining_resources":
                    tokens.append("any_resource_decision_remaining_resources")
                    emit_scalar(tokens, value)
                    return

            if field_key == "possible_outcomes":
                emit_scalar(tokens, value)
                return
            
            if isinstance(value, bool):
                if value:
                    tokens.append(str(field_key))
                return

            if isinstance(value, dict):
                emit_action(tokens, value)
                return

            if isinstance(value, (list, tuple)):
                for item in value:
                    emit_scalar(tokens, item)
                return

            emit_scalar(tokens, value)

        def emit_action(tokens, obj):
            if not isinstance(obj, dict):
                emit_scalar(tokens, obj)
                return

            if "make_payment" in obj and isinstance(obj["make_payment"], dict):
                mp = obj["make_payment"]
                tokens.append("make_payment")
                # Chosen payment may contain explicit card ids (e.g. dragon_card: (85, 36, 104)).
                emit_cost(tokens, mp.get("cost", {}), include_card_choices=True)
                emit_action(tokens, mp.get("action", {}))
                if "coords" in obj:
                    emit_coords(tokens, obj.get("coords"))
                return

            if "adv_effects" in obj and isinstance(obj["adv_effects"], dict):
                tokens.append("adv_effects_start")
                emit_action(tokens, obj["adv_effects"])
                tokens.append("adv_effects_end")

            if "sequence" in obj and isinstance(obj["sequence"], list):
                tokens.append("sequence_start")
                for item in obj["sequence"]:
                    tokens.append("sequence_item_start")
                    emit_action(tokens, item)
                    tokens.append("sequence_item_end")
                tokens.append("sequence_end")

            if "choice" in obj and isinstance(obj["choice"], list):
                tokens.append("choice_start")
                for item in obj["choice"]:
                    tokens.append("choice_item_start")
                    emit_action(tokens, item)
                    tokens.append("choice_item_end")
                tokens.append("choice_end")

            if "random" in obj:
                tokens.append("random_start")
                emit_action(tokens, obj["random"])
                tokens.append("random_end")

            if "cost" in obj:
                emit_cost(tokens, obj["cost"])

            normal_keys = [
                key for key in obj.keys()
                if key not in {"make_payment", "adv_effects", "sequence", "choice", "random", "cost", "coords", "opponent_effect"}
            ]

            for key in normal_keys:
                value = obj[key]

                if isinstance(value, bool):
                    if value:
                        tokens.append(str(key))
                    continue
                elif key == "brown_space":
                    tokens.append(f"brown_space_{int(value)}")
                    continue
                else:
                    tokens.append(str(key))

                if isinstance(value, dict):
                    for inner_key, inner_val in value.items():
                        emit_key_value(tokens, str(key), inner_key, inner_val)
                elif isinstance(value, list):
                    for item in value:
                        emit_action(tokens, item)
                else:
                    emit_scalar(tokens, value)

            if "coords" in obj:
                emit_coords(tokens, obj["coords"])

        tokens = []
        emit_action(tokens, action_json)

        # Remove accidental empty tokens while preserving order.
        tokens = [tok for tok in tokens if tok != ""]

        if as_ids:
            return [self.token_index.get(tok, tok) for tok in tokens]
        return tokens

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
    

if __name__ == "__main__":
    env = WyrmspanEnv()

    test_file_name = input("Enter path to test game to load actions from: ")
    errors = []
    lengths = []
    with open(f"logs/{test_file_name}", "r") as f:
        # read line by line, look for actions
        for line in f:
            if "Best move" in line:
                json_str = line.split("Best move: ")[1].strip()
                # this is actually a printed Python dict
                print(f"Testing tokenization for action JSON: {json_str}")
                # for safety
                assert json_str.startswith("{") and json_str.endswith("}"), "Expected a JSON-like dict string"
                action_json = eval(json_str)  # Caution: using eval on untrusted input can be dangerous
                tokens = env.tokenize_json(action_json)
                print(f"Tokens for action: {tokens}\n")
                token_ids = env.tokenize_json(action_json, as_ids=True)
                lengths.append(len(tokens))
                print(f"Token IDs for action: {token_ids}\n")
                if any(isinstance(tok, str) for tok in token_ids):
                    print(f">>>>> Warning: Some tokens were not recognized and kept as strings: {token_ids} <<<<<\n\n\n")
                    errors.append((json_str, token_ids))
    if errors:
        print("Some actions had unrecognized tokens:")
        for json_str, token_ids in errors:
            print(f"Action JSON: {json_str}")
            print(f"Token IDs: {token_ids}\n")
    else:
        print("All actions were successfully tokenized into known token IDs.")

    # length statistics
    if lengths:
        print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
        print(f"Median length: {sorted(lengths)[len(lengths)//2]}")