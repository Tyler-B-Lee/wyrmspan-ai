from game_states import *
from game_logic import get_next_state, get_random_outcome

import time
import random
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

    # Game Timing (30 values)
    timing_tensor = np.zeros(30, dtype=np.float32)
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
    timing_tensor[29] = 1.0 if game_state.board["round_tracker"]["egg_given"] else 0.0  # Player passed this round
    info_dict["timing"] = timing_tensor

    # Guild Board Status (67 combined values)
    guild_tensor = np.zeros(29, dtype=np.float32)
    guild_index = game_state.board["guild"]["guild_index"]
    # One-hot encode guild (4 possible guilds)
    if guild_index < 4:
        guild_tensor[guild_index] = 1.0
    guild_tensor[4 + game_state.board["guild"]["player_position"]] = 1.0  # Player position on guild track (0-11)
    guild_tensor[16 + game_state.board["guild"]["automa_position"]] = 1.0  # Automa position on guild track
    guild_tensor[27] = game_state.player.guild_markers / 4.0  # Player guild markers remaining
    guild_tensor[28] = game_state.board["guild"]["automa_markers_ready"] / 4.0  # Automa guild markers ready
    # Guild ability uses per player
    guild_ability_tensor = np.zeros((5, 6), dtype=np.float32)
    ability_uses = game_state.board["guild"]["ability_uses"]
    for i in range(1, 6):
        uses = ability_uses[i]
        # get count per player for this ability
        for player_i in range(2):
            count = min(uses.count(player_i), 3)
            if count > 0:
                guild_ability_tensor[i - 1, count - 1 + player_i * 3] = 1.0
    # append guild ability uses to the end of guild tensor
    guild_tensor = np.concatenate((guild_tensor, guild_ability_tensor.flatten()), axis=0)
    # extra info of end game point ability
    end_game_ability_tensor = np.zeros((2,4), dtype=np.float32)
    for player_i in range(2):
        count = ability_uses[5][2:].count(player_i)
        if count > 0:
            end_game_ability_tensor[player_i, count - 1] = 1.0
    guild_tensor = np.concatenate((guild_tensor, end_game_ability_tensor.flatten()), axis=0)
    info_dict["guild_status"] = guild_tensor

    # Deck Status (2 + 183 + 75 = 260 values)
    deck_tensor = np.zeros(2, dtype=np.float32)
    deck_tensor[0] = len(game_state.dragon_deck) / 183.0  # Dragon deck size
    deck_tensor[1] = len(game_state.cave_deck) / 75.0  # Cave deck size
    # combine dragon and cave deck tensors from game state
    deck_tensor = np.concatenate((
        deck_tensor,
        game_state.dragon_deck_array,
        game_state.cave_deck_array
    ), axis=0)
    info_dict["deck_status"] = deck_tensor

    # Player Resources (16 values: hand sizes, coins, resources, eggs)
    player_tensor = np.zeros(16, dtype=np.float32)
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
    automa_tensor = np.zeros(29, dtype=np.float32)
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
        - Once-Per-Round ability used (0/1)
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
    slot_types_tensor = np.array(slot_types, dtype=np.int64)

    # dragons on slots (12 values)
    dragons_on_slots = []
    for cave_name in CAVE_NAMES:
        for col in range(4):
            dragon_id = player.dragons_played[cave_name][col]
            dragons_on_slots.append(dragon_id if dragon_id is not None else 0)
    dragons_on_slots_tensor = np.array(dragons_on_slots, dtype=np.int64)

    # slot details (12 x 18 = 216 total values)
    slot_details = np.zeros((12, 18), dtype=np.float32)
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
            slot_details[slot_index, 16] = len(player.tucked_dragons[cave_name][col]) / 10.0  # Normalize tucked dragons
            # Once-Per-Round ability used
            if game_state.board["round_tracker"]["opr_remaining"] is None:
                slot_details[slot_index, 17] = 0.0  # No usage at the moment
            else:
                assert isinstance(game_state.board["round_tracker"]["opr_remaining"], list)
                this_dragon_id = player.dragons_played[cave_name][col]
                if this_dragon_id is not None and this_dragon_id in game_state.board["round_tracker"]["opr_remaining"]:
                    slot_details[slot_index, 17] = 0.0  # OPR used for this dragon
                else:
                    slot_details[slot_index, 17] = 1.0  # OPR not used for this dragon
    return {
        "slot_types": slot_types_tensor,
        "dragons_on_slots": dragons_on_slots_tensor,
        "slot_details": slot_details
    }


class WyrmspanEnv(gym.Env):
    # Multi-component reward configuration for training
    # Tuned to guide agent toward competitive victory (beating automa) while exploring diverse strategies
    REWARD_CONFIG = {
        'margin_scaling': 100.0,      # Scaling factor for margin-based reward component
        'point_weight': 0.3,          # Reduced weight on point accumulation (was 1.0, now 0.3)
        'round_bonus_threshold': -5,  # Bonus if within X points of automa at round end
        'round_bonus_amount': 0.15,   # Small competitive bonus awarded at round end
        'target_score': 75,           # End-game target score for difficulty 0 automa
        'win_bonus_base': 2.0,        # Base bonus for winning (beating automa)
        'win_bonus_per_margin': 0.05, # Bonus per point of margin (incentivizes larger victories)
        'target_bonus_max': 1.0,      # Max bonus for exceeding target score
        'loss_penalty': -0.5,         # Small penalty for losing (not too harsh, encourages exploration)
    }

    def __init__(self):
        super().__init__()
        
        # SoloGameState contains all the game logic and state management; we will interact with it to step through the game
        self.game_state = SoloGameState(automa_difficulty=0)
        
        # Track automa score to compute margin-based rewards
        self.prev_automa_score = 0
        
        # We assume a max number of legal actions the env will ever return
        self.max_legal_actions = 180
        self.max_action_tokens = 100
        self.max_hand_size = 15
        self.max_queue_size = 5

        self.token_strings = [
            "<pad>",
            "<unk>",

            # numbers (for amounts of resources, costs, points, etc.)
            "-2",
            "-1",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",

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
            "this_position",
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
            "condition",
            "if_true",
            "min_spaces_excavated",
            "cave",
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
            "amount",
            "point_amount",

            # end game keywords
            "double_value",
            "tuck",
            "egg_pairs",
            "min_dragons_in_cave",
            "max_dragons_in_cave",
            "min_dragons_this_column",
            "for_each",
            "payments",
            "lowest_objectives",
            "type",
            "any_dragon",
            "exclude_self",
            "guild_markers",
            "set_of_traits",
            "set_of_sizes",
            "max",

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
        ]

        self.normal_token_counter = len(self.token_strings)
        self.dragon_token_start_index = self.normal_token_counter
        self.cave_token_start_index = self.normal_token_counter + 183

        # add dragons and caves
        for dragon_id in range(1, 184):
            self.token_strings.append(f"dragon_{dragon_id}")
        for cave_id in range(1, 76):
            self.token_strings.append(f"cave_{cave_id}")

        self.token_index = {key: idx for idx, key in enumerate(self.token_strings)}
        self.pad_token_id = self.token_index["<pad>"]
        self.unk_token_id = self.token_index["<unk>"]
        self.action_token_vocab_size = len(self.token_strings)
        
        self.observation_space = spaces.Dict({
            # 1. Global context (Resources, Guild, Round info)
            "timing": spaces.Box(low=0, high=3, shape=(30,), dtype=np.float32),
            "guild_status": spaces.Box(low=0, high=3, shape=(67,), dtype=np.float32),
            "deck_status": spaces.Box(low=0, high=3, shape=(260,), dtype=np.float32),
            "player_resources": spaces.Box(low=0, high=3, shape=(16,), dtype=np.float32),
            "automa_status": spaces.Box(low=0, high=3, shape=(29,), dtype=np.float32),

            # 2. Card display (dragon and cave cards on display)
            "card_display_dragons": spaces.Box(low=0, high=500, shape=(3,), dtype=np.int64),
            "card_display_caves": spaces.Box(low=0, high=600, shape=(3,), dtype=np.int64),

            # 3. Card IDs for hand and board (model will handle embeddings)
            "hand_card_ids": spaces.Box(low=0, high=500, shape=(self.max_hand_size,), dtype=np.int64),
            "hand_card_mask": spaces.Box(low=0, high=1, shape=(self.max_hand_size,), dtype=np.int8),  # mask to indicate real cards in hand

            "slot_types": spaces.Box(low=0, high=3, shape=(12,), dtype=np.int64),
            "dragons_on_slots": spaces.Box(low=0, high=200, shape=(12,), dtype=np.int64),
            "slot_details": spaces.Box(low=0, high=3, shape=(12, 18), dtype=np.float32),
            
            # 4. Other items to be tokenized - Guild, Objectives
            "other_indices": spaces.Box(low=0, high=20, shape=(5,), dtype=np.int64),

            # 5. event queue (current stack of events that will resolve after the current action)
            "queue_tokens": spaces.Box(
                low=0, 
                high=self.action_token_vocab_size - 1, 
                shape=(self.max_queue_size, self.max_action_tokens), 
                dtype=np.int64
            ),

            # 6. queue pad mask (which positions in queue_tokens are real)
            "queue_pad_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self.max_queue_size, self.max_action_tokens),
                dtype=np.int8,
            ),
            
            # 7. valid queue slot mask (Which indices in queue_tokens are real)
            "queue_slot_mask": spaces.Box(low=0, high=1, shape=(self.max_queue_size,), dtype=np.int8),

            # 8. Action tokens (tokenized JSON actions)
            "action_token_ids": spaces.Box(
                low=0,
                high=self.action_token_vocab_size - 1,
                shape=(self.max_legal_actions, self.max_action_tokens),
                dtype=np.int64,
            ),

            # 9. Action token mask (which positions in action_token_ids are real)
            "action_token_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self.max_legal_actions, self.max_action_tokens),
                dtype=np.int8,
            ),
            
            # 10. Action Mask (Which indices in action_token_ids are real)
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

        def emit_condition(tokens, condition_dict):
            tokens.append("condition_start")
            if not isinstance(condition_dict, dict):
                emit_scalar(tokens, condition_dict)
                tokens.append("condition_end")
                return
            for key, value in condition_dict.items():
                if key == "or" and isinstance(value, list):
                    tokens.append("or_start")
                    for option in value:
                        tokens.append("or_option_start")
                        emit_condition(tokens, option)
                        tokens.append("or_option_end")
                    tokens.append("or_end")
                    continue
                elif key == "and" and isinstance(value, list):
                    tokens.append("and_start")
                    for option in value:
                        tokens.append("and_option_start")
                        emit_condition(tokens, option)
                        tokens.append("and_option_end")
                    tokens.append("and_end")
                    continue
                
                emit_action(tokens, value)
            tokens.append("condition_end")

        def emit_end_game(tokens, end_game_dict):
            tokens.append("end_game")
            if not isinstance(end_game_dict, dict):
                emit_scalar(tokens, end_game_dict)
                return
            for key, value in end_game_dict.items():
                if key == "if_true" and isinstance(value, dict):
                    tokens.append("if_true")
                    tokens.append("point_amount")
                    tokens.append(encode_number(value.get("amount", 0)))
                    emit_condition(tokens, value.get("condition", {}))
                    continue
                
                elif key == "for_each" and isinstance(value, dict):
                    tokens.append("for_each")
                    for field_key, field_value in value.items():
                        if field_key == "amount":
                            tokens.append("point_amount")
                            tokens.append(encode_number(field_value))
                        else:
                            emit_key_value(tokens, "end_game", field_key, field_value)
                    continue

                elif key == "payments" and isinstance(value, list):
                    tokens.append("payments")
                    for payment in value:
                        tokens.append("point_amount")
                        tokens.append(encode_number(payment.get("amount", 0)))
                        emit_cost(tokens, payment.get("cost", {}), include_card_choices=True)
                    continue
                
                tokens.append(str(key))
                emit_scalar(tokens, value)

        def emit_key_value(tokens, action_key: str, field_key: str, value):
            if field_key == "chosen_id":
                if action_key in {"play_dragon", "gain_dragon", "tuck_from", "draw_decision", "discard_dragon"}:
                    tokens.append(to_card_token("dragon", value))
                    return
                if action_key in {"play_cave", "gain_cave", "discard_cave"}:
                    tokens.append(to_card_token("cave", value))
                    return

            if field_key in {"dragon_id1", "dragon_id2"}:
                tokens.append(to_card_token("dragon", value))
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

            if field_key in {"coords1", "coords2"}:
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
            
            if "end_game" in obj:
                emit_end_game(tokens, obj["end_game"])

            normal_keys = [
                key for key in obj.keys()
                if key not in {"make_payment", "adv_effects", "sequence", "choice", "random", "cost", "coords", "opponent_effect", "end_game"}
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
            return [self.token_index.get(tok, self.unk_token_id) for tok in tokens]
        return tokens

    def save_state_action_info(self, action: dict):
        """Utility function to save the current game state and a chosen action to a log file for testing."""
        timestamp = int(time.time())
        filename = f"logs/state_action_{timestamp}.json"
        with open(filename, "w") as f:
            # save string representation of game state for debugging
            f.write("Game State:\n")
            f.write(str(self.game_state))
            f.write("\n\nChosen Action:\n")
            json.dump(action, f, indent=2)
        print(f"Saved state and action info to {filename}")

    def _get_obs(self):
        """Get observation from game state and legal actions."""
        # 1. Global context tensors
        final_obs = get_global_info(self.game_state)

        # 2. Card display (dragon and cave cards on display)
        card_display = self.game_state.board["card_display"]
        display_dragons = card_display["dragon_cards"]
        display_caves = card_display["cave_cards"]
        # Replace None with unk token
        display_dragons = [f"dragon_{card_id}" if card_id is not None else "<unk>" for card_id in display_dragons]
        display_caves = [f"cave_{card_id}" if card_id is not None else "<unk>" for card_id in display_caves]
        display_dragon_ids = [self.token_index.get(card_str, self.unk_token_id) for card_str in display_dragons]
        display_cave_ids = [self.token_index.get(card_str, self.unk_token_id) for card_str in display_caves]
        
        final_obs["card_display_dragons"] = np.array(display_dragon_ids, dtype=np.int64)
        final_obs["card_display_caves"] = np.array(display_cave_ids, dtype=np.int64)

        # 3. Card IDs for hand and board
        hand_card_strings = [f"dragon_{card_id}" for card_id in self.game_state.player.dragon_hand] + [f"cave_{card_id}" for card_id in self.game_state.player.cave_hand]
        hand_card_ids = [self.token_index.get(card_str, self.unk_token_id) for card_str in hand_card_strings]
        # Pad hand card ids to fixed length
        hand_card_ids += [self.unk_token_id] * (self.max_hand_size - len(hand_card_ids))
        final_obs["hand_card_ids"] = np.array(hand_card_ids, dtype=np.int64)
        hand_card_mask = [1 if card_id != self.unk_token_id else 0 for card_id in hand_card_ids]
        final_obs["hand_card_mask"] = np.array(hand_card_mask, dtype=np.int8)

        player_board_info = get_player_board_info(self.game_state)
        final_obs.update(player_board_info)

        # 4. Other items to be tokenized - Guild, Objectives
        other_indices = np.zeros(5, dtype=np.int64)
        other_indices[0] = self.game_state.board["guild"]["guild_index"]  # Guild index (0-3)
        # Objectives - we will encode the presence of each objective type as a binary feature
        for i, (tile_index, side) in enumerate(self.game_state.board["round_tracker"]["objectives"]):
            other_indices[i + 1] = 2 * tile_index + (1 if side == "side_b" else 0)  # Encode objective as a single integer
        final_obs["other_indices"] = other_indices

        # 5-7. Event queue tokens, pad mask, and slot mask
        queue_jsons = self.game_state.event_queue
        full_queue_tokens = np.full((self.max_queue_size, self.max_action_tokens), self.pad_token_id, dtype=np.int64)
        queue_pad_mask = np.zeros((self.max_queue_size, self.max_action_tokens), dtype=np.int8)
        queue_slot_mask = np.zeros((self.max_queue_size,), dtype=np.int8)
        
        error = False
        for i in range(1, len(queue_jsons) + 1):
            # read queue in reverse order so that the next event to resolve is in slot 0
            event = queue_jsons[-i]
            if i > self.max_queue_size:
                error = True
                print("Warning: Event queue exceeds max_queue_size. Some events will be truncated.")
                break
            event_tokens = self.tokenize_json(event, as_ids=True)
            if self.unk_token_id in event_tokens:
                error = True
                print(f"Warning: Event {event} contains tokens that were not recognized and mapped to <unk>: {event_tokens}")
            if not event_tokens:
                continue
            # clip tokens to max_action_tokens
            if len(event_tokens) > self.max_action_tokens:
                error = True
                print(f"Warning: Tokenized event exceeds max_action_tokens and will be truncated: {event_tokens}")
            event_tokens = event_tokens[: self.max_action_tokens]
            L = len(event_tokens)
            full_queue_tokens[i - 1, :L] = np.array(event_tokens, dtype=np.int64)
            queue_pad_mask[i - 1, :L] = 1
            queue_slot_mask[i - 1] = 1

            if error:
                self.save_state_action_info(event)
                error = False

        final_obs["queue_tokens"] = full_queue_tokens
        final_obs["queue_pad_mask"] = queue_pad_mask
        final_obs["queue_slot_mask"] = queue_slot_mask

        # 8-10. Action tokens, token mask, and action mask
        if self.game_state.current_choice is not None:
            action_jsons = self.game_state.current_choice
        else:
            action_jsons = []

        # Initialize arrays: token ids filled with pad_token_id, token mask zeros
        full_token_ids = np.full(
            (self.max_legal_actions, self.max_action_tokens),
            self.pad_token_id,
            dtype=np.int64,
        )
        full_token_mask = np.zeros((self.max_legal_actions, self.max_action_tokens), dtype=np.int8)
        action_mask = np.zeros((self.max_legal_actions,), dtype=np.int8)

        error = False
        for i, action in enumerate(action_jsons):
            if i >= self.max_legal_actions:
                error = True
                print("Warning: Number of legal actions exceeds max_legal_actions. Some actions will be truncated.")
                break
            token_ids = self.tokenize_json(action, as_ids=True)
            if self.unk_token_id in token_ids:
                error = True
                print(f"Warning: Action {action} contains tokens that were not recognized and mapped to <unk>: {token_ids}")
            if not token_ids:
                # empty action -> leave as pad
                action_mask[i] = 1
                continue
            # clip tokens to max_action_tokens
            if len(token_ids) > self.max_action_tokens:
                error = True
                print(f"Warning: Tokenized action exceeds max_action_tokens and will be truncated: {token_ids}")
            token_ids = token_ids[: self.max_action_tokens]
            L = len(token_ids)
            full_token_ids[i, :L] = np.array(token_ids, dtype=np.int64)
            full_token_mask[i, :L] = 1
            action_mask[i] = 1
            
            if error:
                self.save_state_action_info(action)
                error = False

        final_obs["action_token_ids"] = full_token_ids
        final_obs["action_token_mask"] = full_token_mask
        final_obs["action_mask"] = action_mask

        return final_obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # set rng seed for the game engine
        random.seed(seed)
        # reset the game engine to start a new game
        self.game_state = SoloGameState(automa_difficulty=0)
        self.game_state.create_game()
        
        # Initialize automa score tracking for margin-based rewards
        self.prev_automa_score = self.game_state.automa.score

        # progress to the first choice point
        while not self.game_state.is_halted():
            self.game_state = get_next_state(self.game_state)

        return self._get_obs(), {}
        
    def step(self, action_idx):
        prev_player_score = self.game_state.player.score
        prev_round = self.game_state.board["round_tracker"]["round"]
        terminated = False
        next_state = get_next_state(self.game_state, action_idx)
        # continue progressing through the game until we reach the next choice point
        while next_state.phase != PHASE_END_GAME:
            if next_state.current_choice is not None:
                break
            if next_state.current_random_event is not None:
                # resolve this random event immediately and continue
                outcome = get_random_outcome(next_state, next_state.current_random_event, next_state.player)
                next_state = get_next_state(next_state, outcome)
            else:
                next_state = get_next_state(next_state)

        # we either have a choice or we've reached the end of the game
        self.game_state = next_state
        
        # --- Multi-component reward calculation ---
        
        # Component 1: Score margin shaping
        # Reward for improving (or maintaining) relative position to automa
        current_margin = self.game_state.player.score - self.game_state.automa.score
        prev_margin = prev_player_score - self.prev_automa_score
        margin_delta = current_margin - prev_margin
        reward_margin = margin_delta / self.REWARD_CONFIG['margin_scaling']
        
        # Component 2: Point accumulation (reduced weight to emphasize winning over point hoarding)
        point_delta = self.game_state.player.score - prev_player_score
        reward_points = (point_delta / 100.0) * self.REWARD_CONFIG['point_weight']
        
        # Component 3: Round-end competitive bonus
        # Small incentive for being competitive at round boundaries
        reward_round_bonus = 0.0
        current_round = self.game_state.board["round_tracker"]["round"]
        if (current_round > prev_round) and current_margin >= self.REWARD_CONFIG['round_bonus_threshold']:
            reward_round_bonus = self.REWARD_CONFIG['round_bonus_amount']
        
        # Component 4: End-game scoring
        reward_end_game = 0.0
        if self.game_state.phase == PHASE_END_GAME:
            terminated = True
            if self.game_state.player.score >= self.game_state.automa.score:
                # Win bonus: base + scaled by victory margin
                margin_at_end = self.game_state.player.score - self.game_state.automa.score
                margin_bonus = min(margin_at_end * self.REWARD_CONFIG['win_bonus_per_margin'], 
                                  self.REWARD_CONFIG['target_bonus_max'])
                reward_end_game = self.REWARD_CONFIG['win_bonus_base'] + margin_bonus
                
                # Additional bonus for reaching target score (beating typical difficulty 0 score)
                if self.game_state.player.score >= self.REWARD_CONFIG['target_score']:
                    target_bonus = min(
                        (self.game_state.player.score - self.REWARD_CONFIG['target_score']) / 50.0,
                        self.REWARD_CONFIG['target_bonus_max']
                    )
                    reward_end_game += target_bonus
            else:
                # Loss: small penalty to discourage defeat but not too harsh
                reward_end_game = self.REWARD_CONFIG['loss_penalty']
        
        # Combine all reward components
        reward = reward_margin + reward_points + reward_round_bonus + reward_end_game
        
        # Update tracked automa score for next step's margin calculation
        self.prev_automa_score = self.game_state.automa.score
        
        obs = self._get_obs()

        return obs, reward, terminated, False, {}
    

if __name__ == "__main__":
    env = WyrmspanEnv()
    print(f"Number of tokens in action token vocabulary: {env.action_token_vocab_size}")

    # Test: Load a test game log and tokenize the actions to verify the tokenization logic.
    # test_file_name = input("Enter path to test game to load actions from: ")
    # errors = []
    # lengths = []
    # with open(f"logs/{test_file_name}", "r") as f:
    #     # read line by line, look for actions
    #     for line in f:
    #         if "Best move" in line:
    #             json_str = line.split("Best move: ")[1].strip()
    #             # this is actually a printed Python dict
    #             print(f"Testing tokenization for action JSON: {json_str}")
    #             # for safety
    #             assert json_str.startswith("{") and json_str.endswith("}"), "Expected a JSON-like dict string"
    #             action_json = eval(json_str)  # Caution: using eval on untrusted input can be dangerous
    #             tokens = env.tokenize_json(action_json)
    #             print(f"Tokens for action: {tokens}\n")
    #             token_ids = env.tokenize_json(action_json, as_ids=True)
    #             lengths.append(len(tokens))
    #             print(f"Token IDs for action: {token_ids}\n")
    #             if any(isinstance(tok, str) for tok in token_ids):
    #                 print(f">>>>> Warning: Some tokens were not recognized and kept as strings: {token_ids} <<<<<\n\n\n")
    #                 errors.append((json_str, token_ids))
    # if errors:
    #     print("Some actions had unrecognized tokens:")
    #     for json_str, token_ids in errors:
    #         print(f"Action JSON: {json_str}")
    #         print(f"Token IDs: {token_ids}\n")
    # else:
    #     print("All actions were successfully tokenized into known token IDs.")

    # # length statistics
    # if lengths:
    #     print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
    #     print(f"Median length: {sorted(lengths)[len(lengths)//2]}")

    # Test: Run a random action through the environment to verify it steps without errors and returns an observation of the correct format.
    import pprint
    from playout_compare import get_sim_algo

    sim_algo = get_sim_algo("greedy_action_priority", {'dragon_weight': 2.845, 'cave_weight': 2.056, 'explore_weight': 1.431})
    
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        # pretty print every 20 steps
        # if step_count % 20 == 0:
        #     print(f"Step {step_count} observation:")
        #     pprint.pprint(obs)
            
        legal_actions = obs["action_mask"].sum()
        chosen_action = sim_algo(env.game_state, None)
        # chosen_action = random.randint(0, legal_actions - 1)  # Randomly choose among legal actions
        
        # check reward per step
        print(f"Step {step_count}: Chosen action index {chosen_action} out of {legal_actions} legal actions.")
        print(f"Round {env.game_state.board['round_tracker']['round']}, Player score: {env.game_state.player.score}, Automa score: {env.game_state.automa.score}")
        obs, reward, done, _, info = env.step(chosen_action)
        print(f"After step - Round {env.game_state.board['round_tracker']['round']}, Player score: {env.game_state.player.score}, Automa score: {env.game_state.automa.score}, Reward: {reward:.2f}\n")

        total_reward += reward
        step_count += 1
    print(f"Episode finished after {step_count} steps with total reward {total_reward:.2f}. Final score: {env.game_state.player.score}, Automa score: {env.game_state.automa.score}")
