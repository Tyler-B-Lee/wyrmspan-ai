import logging
import random
import time
from game_states import (
    GameState,
    SoloGameState,
    PlayerState,
    DRAGON_CARDS,
    CAVE_CARDS,
    OBJECTIVE_TILES,
    GUILD_TILES,
    CAVE_NAMES,
    DRAGON_PERSONALITIES,
    DRAGON_SIZES,
    EXPLORE_CAVE_EFFECTS,
)
import game_logic as logic

# File for comparing different algorithms for simulating game playouts in Wyrmspan.

DEFAULT_BATCH_SIZE = 32


def _action_effect(action: dict) -> dict:
    """Unwrap the immediate effect for a choice action."""
    if "adv_effects" in action and isinstance(action["adv_effects"], dict):
        return action["adv_effects"]
    return action


def _action_cost(action: dict) -> dict:
    """Get the explicit cost component for a choice action."""
    cost = action.get("cost")
    return cost if isinstance(cost, dict) else {}


def _sum_printed_cost(card_info: dict) -> int:
    total = 0
    for key, value in card_info.items():
        if key.endswith("_cost") and isinstance(value, int):
            total += value
    return total


def _objective_player_count(player, objective_info: dict) -> int:
    """Compute the player's count for an objective using the same categories as scoring."""
    obj_item = objective_info["for_each"]
    obj_type = obj_item["type"]
    count = 0
    if obj_type == "eggs":
        return sum(player.egg_totals.values())
    if obj_type == "total_cards_in_cave":
        cave_name = obj_item["location"]
        for col in range(4):
            if player.dragons_played[cave_name][col] is not None:
                count += 1
            cave_card = player.caves_played[cave_name][col]
            if cave_card is not None and cave_card != -1:
                count += 1
        return count

    for cave_name in CAVE_NAMES:
        for col in range(4):
            dragon_id = player.dragons_played[cave_name][col]
            cave_card = player.caves_played[cave_name][col]
            if obj_type in DRAGON_PERSONALITIES:
                if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == obj_type:
                    count += 1
            elif obj_type in DRAGON_SIZES:
                if dragon_id is not None and DRAGON_CARDS[dragon_id]["size"] == obj_type:
                    count += 1
            elif obj_type == "dragon_abilities":
                if dragon_id is not None:
                    dragon_info = DRAGON_CARDS[dragon_id]
                    if any(ability in dragon_info for ability in obj_item["ability_types"]):
                        count += 1
            elif obj_type == "dragon_cost":
                if dragon_id is not None:
                    dragon_info = DRAGON_CARDS[dragon_id]
                    item_cost = _sum_printed_cost(dragon_info)
                    if obj_item["min_cost"] <= item_cost <= obj_item["max_cost"]:
                        count += 1
            elif obj_type == "cave_cards_played":
                if cave_card is not None and cave_card != -1:
                    count += 1
            elif obj_type == "egg_capacity":
                count += player.nested_eggs[cave_name][col][1]
            elif obj_type == "cached_resources":
                count += sum(player.cached_resources[cave_name][col].values())
            elif obj_type == "guild_position":
                count += player.guild_markers
    return count


class RolloutCache:
    """Rollout-local memoization for a single playout."""

    def __init__(self):
        self._signature = None
        self._summary = None
        self._dragon_choice_cache = {}
        self._cave_choice_cache = {}
        self._dragon_feature_cache = {}
        self._cave_feature_cache = {}
        self._objective_synergy_cache = None

    def _state_signature(self, game_state: GameState):
        player = game_state.player
        automa_score = getattr(getattr(game_state, "automa", None), "score", 0)
        round_num = game_state.board.get("round_tracker", {}).get("round", 0)
        return (
            game_state.phase,
            game_state.turn,
            round_num,
            getattr(game_state, "current_player", 0),
            player.score,
            player.coins,
            automa_score,
            tuple(player.dragon_hand),
            tuple(player.cave_hand),
            tuple(player.resources.items()),
            tuple(player.egg_totals.items()),
            tuple(player.times_explored[cave] for cave in CAVE_NAMES),
            tuple(player.num_dragons_played[cave] for cave in CAVE_NAMES),
            tuple(tuple(player.caves_played[cave]) for cave in CAVE_NAMES),
            tuple(tuple(player.dragons_played[cave]) for cave in CAVE_NAMES),
            len(game_state.dragon_deck),
            len(game_state.cave_deck),
            len(game_state.event_queue),
            len(game_state.current_choice) if game_state.current_choice is not None else -1,
            repr(game_state.current_random_event),
        )

    def refresh(self, game_state: GameState):
        signature = self._state_signature(game_state)
        if signature != self._signature:
            self._signature = signature
            self._summary = self._build_summary(game_state)
            self._dragon_choice_cache.clear()
            self._cave_choice_cache.clear()
        return self._summary

    def _build_summary(self, game_state: GameState):
        player = game_state.player
        round_num = game_state.board["round_tracker"]["round"]
        objective_info = [OBJECTIVE_TILES[idx][side] for idx, side in game_state.board["round_tracker"]["objectives"]]
        objective_counts = []
        for idx, info in enumerate(objective_info):
            player_count = _objective_player_count(player, info)
            automa_count = info["automa_values"][idx]
            automa_bonus = game_state.board["round_tracker"].get("automa_bonus")
            if automa_bonus is not None:
                automa_count += automa_bonus[idx]
            objective_counts.append(
                {
                    "objective": info,
                    "player_count": player_count,
                    "automa_count": automa_count,
                    "gap": player_count - automa_count,
                }
            )

        dragon_trait_counts = {
            "personality": {name: 0 for name in DRAGON_PERSONALITIES},
            "size": {name: 0 for name in DRAGON_SIZES},
            "abilities": {
                "if_activated": 0,
                "when_played": 0,
                "on_feed": 0,
                "on_grow_up": 0,
            },
            "printed_cost_total": 0,
        }
        dragon_board_value = 0
        cave_board_value = 0
        for cave_name in CAVE_NAMES:
            for col in range(4):
                dragon_id = player.dragons_played[cave_name][col]
                cave_card = player.caves_played[cave_name][col]
                if dragon_id is not None:
                    dragon_info = DRAGON_CARDS[dragon_id]
                    dragon_trait_counts["personality"][dragon_info["personality"]] += 1
                    dragon_trait_counts["size"][dragon_info["size"]] += 1
                    dragon_trait_counts["printed_cost_total"] += _sum_printed_cost(dragon_info)
                    for ability_key in dragon_trait_counts["abilities"]:
                        if ability_key in dragon_info:
                            dragon_trait_counts["abilities"][ability_key] += 1
                    dragon_board_value += dragon_info.get("VP", 0)
                if cave_card is not None and cave_card != -1:
                    cave_board_value += 1

        return {
            "round": round_num,
            "objective_info": objective_info,
            "objective_counts": objective_counts,
            "dragon_trait_counts": dragon_trait_counts,
            "deck_counts": {
                "dragon": len(game_state.dragon_deck),
                "cave": len(game_state.cave_deck),
            },
            "resource_total": sum(player.resources.values()),
            "egg_total": sum(player.egg_totals.values()),
            "guild_index": game_state.board["guild"]["guild_index"],
            "guild_markers": player.guild_markers,
            "dragon_board_value": dragon_board_value,
            "cave_board_value": cave_board_value,
        }

    def static_dragon_feature_score(self, dragon_id: int) -> float:
        if dragon_id in self._dragon_feature_cache:
            return self._dragon_feature_cache[dragon_id]
        dragon_info = DRAGON_CARDS[dragon_id]
        score = 0.15 * dragon_info.get("VP", 0)
        score += 0.35 * dragon_info.get("capacity", 0)
        score += 0.45 * dragon_info.get("egg_cost", 0)
        score += 0.25 * dragon_info.get("coin_cost", 0)
        score += 0.25 * sum(dragon_info.get(f"{resource}_cost", 0) for resource in ("meat", "gold", "crystal", "milk"))
        if "if_activated" in dragon_info:
            score += 0.8
        if "when_played" in dragon_info:
            score += 0.6
        if "on_feed" in dragon_info:
            score += 0.7
        if "on_grow_up" in dragon_info:
            score += 0.5
        if dragon_info.get("personality") == "Shy":
            score += 0.15
        self._dragon_feature_cache[dragon_id] = score
        return score

    def static_cave_feature_score(self, cave_id: int) -> float:
        if cave_id in self._cave_feature_cache:
            return self._cave_feature_cache[cave_id]
        cave_info = CAVE_CARDS[cave_id]
        score = 0.2
        when_played = cave_info.get("when_played", {})
        if "other_ability_on_mat" in when_played:
            score += 1.0
        if "adv_effects" in when_played:
            adv = when_played["adv_effects"]
            if "sequence" in adv:
                score += 0.8 + 0.15 * len(adv["sequence"])
            elif "choice" in adv:
                score += 0.7
            else:
                score += 0.5
        text = cave_info.get("text", "")
        if "Cache" in text:
            score += 0.6
        if "Gain [CaveCard]" in text:
            score += 0.7
        if "DragonGuild" in text:
            score += 0.5
        if "Lay [Egg]" in text:
            score += 0.4
        self._cave_feature_cache[cave_id] = score
        return score

    def objective_synergy_cache(self):
        if self._objective_synergy_cache is not None:
            return self._objective_synergy_cache
        self._objective_synergy_cache = {
            "eggs": {"dragon": 0.15, "cave": 0.25, "explore": 0.55},
            "cached_resources": {"dragon": 0.55, "cave": 0.35, "explore": 0.4},
            "cave_cards_played": {"dragon": 0.1, "cave": 0.8, "explore": 0.35},
            "total_cards_in_cave": {"dragon": 0.7, "cave": 0.5, "explore": 0.2},
            "egg_capacity": {"dragon": 0.85, "cave": 0.15, "explore": 0.1},
            "guild_position": {"dragon": 0.25, "cave": 0.55, "explore": 0.7},
            "dragon_cost": {"dragon": 0.9, "cave": 0.1, "explore": 0.1},
        }
        return self._objective_synergy_cache

    def best_dragon_option_score(self, game_state: GameState) -> float:
        summary = self.refresh(game_state)
        player = game_state.player
        best_score = float("-inf")
        objective_synergy = self.objective_synergy_cache()
        for dragon_id in player.dragon_hand:
            cache_key = (self._signature, dragon_id)
            if cache_key not in self._dragon_choice_cache:
                dragon_info = DRAGON_CARDS[dragon_id]
                self._dragon_choice_cache[cache_key] = logic.get_dragon_enticement_options(player, dragon_info)
            costs = self._dragon_choice_cache[cache_key]
            if not costs:
                continue

            dragon_info = DRAGON_CARDS[dragon_id]
            score = self.static_dragon_feature_score(dragon_id)
            score += 0.4 * len(costs)
            if dragon_info.get("personality") == "Shy":
                score += 0.15 * summary["dragon_trait_counts"]["personality"]["Shy"]

            for obj in summary["objective_counts"]:
                obj_type = obj["objective"]["for_each"]["type"]
                synergy = objective_synergy.get(obj_type)
                if synergy is None:
                    continue
                score += synergy["dragon"]
                if obj_type in DRAGON_PERSONALITIES and dragon_info["personality"] == obj_type:
                    score += 1.1
                elif obj_type in DRAGON_SIZES and dragon_info["size"] == obj_type:
                    score += 0.9
                elif obj_type == "dragon_abilities":
                    abilities = obj["objective"]["for_each"].get("ability_types", [])
                    if any(ability in dragon_info for ability in abilities):
                        score += 1.2
                elif obj_type == "dragon_cost":
                    total_cost = _sum_printed_cost(dragon_info)
                    bounds = obj["objective"]["for_each"]
                    if bounds["min_cost"] <= total_cost <= bounds["max_cost"]:
                        score += 1.0
                elif obj_type == "cached_resources" and any(key in dragon_info for key in ("if_activated", "on_feed", "when_played")):
                    score += 0.8

            best_score = max(best_score, score)

        return best_score

    def best_cave_option_score(self, game_state: GameState) -> float:
        summary = self.refresh(game_state)
        player = game_state.player
        best_score = float("-inf")
        objective_synergy = self.objective_synergy_cache()

        for cave_id in player.cave_hand:
            cave_static = self.static_cave_feature_score(cave_id)
            cave_info = CAVE_CARDS[cave_id]
            for cave_name in CAVE_NAMES:
                cache_key = (self._signature, cave_id, cave_name)
                if cache_key not in self._cave_choice_cache:
                    self._cave_choice_cache[cache_key] = logic.can_excavate_cave(player, cave_name)
                can_excavate, slot_index = self._cave_choice_cache[cache_key]
                if not can_excavate:
                    continue

                score = cave_static
                score += max(0, 0.35 - 0.1 * slot_index)
                if slot_index == 0:
                    score += 0.45
                elif slot_index == 1:
                    score += 0.25
                elif slot_index == 2:
                    score += 0.1

                when_played = cave_info.get("when_played", {})
                if "other_ability_on_mat" in when_played:
                    required_type = when_played["other_ability_on_mat"].get("type")
                    if required_type == "once_per_round":
                        score += 0.45 * summary["dragon_trait_counts"]["abilities"]["if_activated"]
                    elif required_type == "when_played":
                        score += 0.35 * summary["dragon_trait_counts"]["abilities"]["when_played"]
                if "adv_effects" in when_played:
                    adv = when_played["adv_effects"]
                    if "sequence" in adv:
                        for seq_action in adv["sequence"]:
                            if "gain_cave" in seq_action:
                                score += 0.45
                            elif "gain_guild" in seq_action:
                                score += 0.4
                            elif "gain_resource" in seq_action:
                                score += 0.25
                            elif "lay_egg" in seq_action:
                                score += 0.35
                            elif "cache_from" in seq_action:
                                score += 0.6
                    elif "choice" in adv:
                        score += 0.5

                for obj in summary["objective_counts"]:
                    obj_type = obj["objective"]["for_each"]["type"]
                    synergy = objective_synergy.get(obj_type)
                    if synergy is None:
                        continue
                    score += synergy["cave"]
                    if obj_type == "cave_cards_played":
                        score += 0.95
                    elif obj_type == "total_cards_in_cave" and obj["objective"]["for_each"].get("location") == cave_name:
                        score += 0.85
                    elif obj_type == "guild_position":
                        score += 0.65

                if cave_info.get("text", "").startswith("Gain [CaveCard]"):
                    score += 0.25 * summary["deck_counts"]["cave"] / 75.0

                best_score = max(best_score, score)

        return best_score

    def explore_value(self, game_state: GameState, cave_name: str) -> float:
        """
        Evaluate the value of exploring a specific cave.
        
        WYRMSPAN EXPLORE MECHANICS (as clarified):
        ==========================================
        1. ALTERNATING SEQUENCE: When exploring, effects activate left-to-right (slots 0-3):
           - Cave effect at slot 0
           - Dragon effect at slot 0 (if dragon present AND has "if_activated" ability type)
           - Cave effect at slot 1
           - Dragon effect at slot 1 (if dragon present AND has "if_activated" ability type)
           - ... and so on
        
        2. DRAGON ABILITY TRIGGER: Dragon abilities ONLY activate if the dragon card has 
           "if_activated" in its dictionary. Otherwise, the explore passes over that dragon
           and continues to the next slot.
        
        3. EARLY STOPPING: If a dragon slot has NO dragon placed, exploring stops immediately.
           This is critical: you may not get effects from later cave slots if you hit an
           empty dragon slot early!
        
        4. CAVE DECK DRAW RATE: Solo games draw ~20-30 caves out of 75, so deck depletion
           is not a primary concern. However, deck scarcity still affects card value.
        """
        summary = self.refresh(game_state)
        player = game_state.player
        round_num = summary["round"]
        times_explored = player.times_explored[cave_name]
        cave_effects = EXPLORE_CAVE_EFFECTS[cave_name]
        
        # Base score for taking an explore action
        score = 0.95
        
        # Bonus for first explore (most effects are new), penalty for repeated explores
        if times_explored == 0:
            score += 0.35
        elif times_explored == 1:
            score += 0.2
        else:
            score += 0.1
        
        # Simulate the explore sequence slot-by-slot to account for early stopping
        # and dragon ability conditions.
        for slot_idx in range(4):
            # Get the cave effect for this slot
            effect = cave_effects[slot_idx]
            
            # Score the cave effect (same as before)
            if "gain_resource" in effect:
                resource_type = effect["gain_resource"]["type"]
                score += 0.5
                if resource_type == "any":
                    score += 0.2
                if summary["objective_counts"][round_num]["objective"]["for_each"]["type"] in ("cached_resources", "eggs"):
                    score += 0.35
            elif "gain_dragon" in effect:
                score += 0.85
                if summary["objective_counts"][round_num]["objective"]["for_each"]["type"] in DRAGON_PERSONALITIES:
                    score += 0.35
            elif "gain_cave" in effect:
                score += 0.8
                if summary["objective_counts"][round_num]["objective"]["for_each"]["type"] == "cave_cards_played":
                    score += 0.4
            elif "gain_guild" in effect:
                score += 0.65
                if summary["objective_counts"][round_num]["objective"]["for_each"]["type"] == "guild_position":
                    score += 0.5
            elif "lay_egg" in effect:
                score += 0.6
                if summary["objective_counts"][round_num]["objective"]["for_each"]["type"] == "eggs":
                    score += 0.55
            elif "adv_effects" in effect:
                adv = effect["adv_effects"]
                if "choice" in adv:
                    score += 0.7
                if "sequence" in adv:
                    score += 0.45 + 0.15 * len(adv["sequence"])
                if "cache_from" in adv:
                    score += 0.8
            
            # Check for dragon at this slot
            dragon_id = player.dragons_played[cave_name][slot_idx]
            
            if dragon_id is None:
                # EARLY STOPPING: No dragon in this slot, explore terminates here.
                # We don't get to process any further slots.
                break
            
            # Dragon is present. Check if it has an "if_activated" ability type.
            dragon_info = DRAGON_CARDS[dragon_id]
            if "if_activated" in dragon_info:
                # Dragon ability activates during explore.
                # Evaluate the ability's value using existing dragon scoring logic.
                dragon_feature_score = self.static_dragon_feature_score(dragon_id)
                # Apply a reduced weight since this is a side effect during explore,
                # not the primary action of playing the dragon.
                score += 0.35 * dragon_feature_score
        
        # Deck scarcity adjustments (note: we don't heavily penalize for depletion risk,
        # as solo games rarely exhaust the deck; instead, scarcity makes cards slightly more valuable)
        if summary["deck_counts"]["dragon"] <= 25:
            score += 0.15
        if summary["deck_counts"]["cave"] <= 15:
            score += 0.1
        
        return score

    def score_action(self, game_state: GameState, action: dict,
                     dragon_weight=2.845,
                     cave_weight=2.056,
                     explore_weight=1.431,
                     pass_penalty=1.5) -> float:
        summary = self.refresh(game_state)
        player = game_state.player
        effect = _action_effect(action)
        cost = _action_cost(action)
        round_num = summary["round"]
        objective_info = summary["objective_counts"][round_num if round_num < len(summary["objective_counts"]) else -1]["objective"] if summary["objective_counts"] else None

        score = 0.0
        if "play_dragon" in effect:
            score = dragon_weight + 0.35
            best_dragon = self.best_dragon_option_score(game_state)
            if best_dragon > float("-inf"):
                score += 0.4 * best_dragon
            if player.coins <= 1:
                score -= 0.15
            if summary["objective_counts"]:
                score += 0.35 * max(obj["gap"] for obj in summary["objective_counts"])
        elif "play_cave" in effect:
            score = cave_weight + 0.25
            best_cave = self.best_cave_option_score(game_state)
            if best_cave > float("-inf"):
                score += 0.35 * best_cave
            if player.cave_hand:
                score += 0.15 * len(player.cave_hand)
        elif "explore" in effect:
            cave_name = effect["explore"]["cave_name"]
            score = explore_weight + self.explore_value(game_state, cave_name)
            if objective_info is not None:
                obj_type = objective_info["for_each"]["type"]
                if obj_type in ("eggs", "cached_resources", "cave_cards_played"):
                    score += 0.35
                if obj_type == "guild_position":
                    score += 0.2
        elif "gain_resource" in effect:
            score = 1.3
            if objective_info is not None and objective_info["for_each"]["type"] == "cached_resources":
                score += 0.5
        elif "lay_egg" in effect:
            score = 1.2
            if objective_info is not None and objective_info["for_each"]["type"] == "eggs":
                score += 0.6
        elif "gain_dragon" in effect:
            score = 1.0
        elif "gain_cave" in effect:
            score = 0.95
        elif "gain_guild" in effect:
            score = 0.9
        elif "pass" in effect:
            score = -pass_penalty
            if player.coins > 0:
                score -= 0.6
            if summary["objective_counts"]:
                score -= 0.1 * max(0, max(obj["gap"] for obj in summary["objective_counts"]))
        elif "skip" in effect or "skip_opr" in effect:
            score = -(pass_penalty * 0.7)
        else:
            score = 0.5

        if cost:
            if cost.get("coin", 0) > player.coins:
                score -= 5.0
            egg_cost = cost.get("egg")
            if isinstance(egg_cost, dict) and egg_cost.get("amount", 0) > summary["egg_total"]:
                score -= 4.0
            if cost.get("any_resource", 0) > summary["resource_total"]:
                score -= 1.5
            if cost.get("cave_card", 0) > len(player.cave_hand):
                score -= 2.0
            if cost.get("dragon_card", 0) > len(player.dragon_hand):
                score -= 2.0

        if summary["deck_counts"]["dragon"] <= 12 and "play_dragon" in effect:
            score += 0.2
        if summary["deck_counts"]["cave"] <= 12 and "play_cave" in effect:
            score += 0.2
        return score


def _configure_worker_logging(level=logging.WARNING):
    """Keep worker processes quiet so logging does not dominate simulation time."""
    logging.getLogger().setLevel(level)


def _chunk_count(total_items: int, batch_size: int):
    batch_size = max(1, batch_size)
    for start in range(0, total_items, batch_size):
        yield start, min(start + batch_size, total_items)


def _run_seeded_game_batch(game_state: GameState, algo_name, algo_kwargs, display_name, seeds, worker_log_level=logging.WARNING):
    """Run several simulations from the same starting state inside one worker."""
    _configure_worker_logging(worker_log_level)
    total_score = 0
    total_time = 0.0
    for seed in seeds:
        score, _, elapsed = simulate_game(game_state.make_copy(), algo_name, algo_kwargs, display_name, seed)
        total_score += score
        total_time += elapsed
    return total_score, display_name, total_time, len(seeds)


def _run_simulation_task_batch(task_batch, worker_log_level=logging.WARNING):
    """Run a batch of independent simulation tasks inside one worker."""
    _configure_worker_logging(worker_log_level)
    results = []
    for game_state, algo_name, algo_kwargs, display_name, seed in task_batch:
        results.append(simulate_game(game_state, algo_name, algo_kwargs, display_name, seed))
    return results

class RNGOrder:
    """
    A class to manage the order of random events in a game state.
    This is used to ensure that the same random events are chosen in the same order
    during actual games, which is important for reproducibility.

    The order is maintained as lists of outcomes.
    We can create this object at the start of a game along with the game state,
    then use it whenever a random event needs to be resolved.
    """
    def __init__(self, game_state: GameState):
        # assume the game_state has just been initialized,
        # create_game has been run, and nothing has happened yet
        self.order = []
        self.game_state = game_state
        # Initialize the RNG order with the current game state
        # setup the dragon and cave decks
        dragon_deck = game_state.dragon_deck.copy()
        random.shuffle(dragon_deck)
        cave_deck = game_state.cave_deck.copy()
        random.shuffle(cave_deck)
        # setup the automa decks for each round
        automa_decks = []
        for r in range(4):
            rnd_automa_deck = game_state.automa.decision_deck.copy()
            random.shuffle(rnd_automa_deck)
            rnd_automa_deck.pop()  # remove the last card, as only 7 are used in each round
            automa_decks.extend(rnd_automa_deck)

        # Keep deck orders immutable and only mutate tiny cursor indices.
        self.dragon_deck = tuple(dragon_deck)
        self.cave_deck = tuple(cave_deck)
        self.automa_decks = tuple(automa_decks)
        self._dragon_idx = len(self.dragon_deck) - 1
        self._cave_idx = len(self.cave_deck) - 1
        self._automa_idx = len(self.automa_decks) - 1

    @classmethod
    def _from_existing(cls, source: "RNGOrder") -> "RNGOrder":
        """Fast copy constructor that shares immutable deck order data."""
        new_obj = cls.__new__(cls)
        new_obj.order = source.order
        new_obj.game_state = source.game_state
        new_obj.dragon_deck = source.dragon_deck
        new_obj.cave_deck = source.cave_deck
        new_obj.automa_decks = source.automa_decks
        new_obj._dragon_idx = source._dragon_idx
        new_obj._cave_idx = source._cave_idx
        new_obj._automa_idx = source._automa_idx
        return new_obj

    def _draw_from_deck(self, deck_name: str):
        if deck_name == "dragon":
            if self._dragon_idx < 0:
                return None
            ret = self.dragon_deck[self._dragon_idx]
            self._dragon_idx -= 1
            return ret
        if deck_name == "cave":
            if self._cave_idx < 0:
                return None
            ret = self.cave_deck[self._cave_idx]
            self._cave_idx -= 1
            return ret
        if self._automa_idx < 0:
            return None
        ret = self.automa_decks[self._automa_idx]
        self._automa_idx -= 1
        return ret

    def get_copy(self):
        """
        Get a copy of the RNGOrder object.
        This is useful for simulating multiple games from the same initial state.
        """
        return RNGOrder._from_existing(self)

    def get_random_outcome(self, game_state: GameState, event:dict, player:PlayerState):
        """
        Get the next random outcome for a given event.
        This will return the next outcome in the order and remove the outcome from the list.
        """
        # check the event type
        if "automa_action" in event:
            # we return one card sampled from the automa's decision deck
            return self._draw_from_deck("automa")
        elif ("top_deck_reveal" in event) or ("refill_dragon_display" in event) or ("gain_dragon" in event) or ("tuck_from" in event):
            # we return one card sampled from the dragon deck
            return self._draw_from_deck("dragon")
        elif ("refill_cave_display" in event) or ("play_cave" in event) or ("gain_cave" in event):
            # we return one card sampled from the cave deck
            return self._draw_from_deck("cave")
        elif "draw_decision" in event:
            # we return a number of cards sampled from the dragon deck
            num_cards = event["draw_decision"]["amount"]
            if num_cards == "shy_this_cave":
                # we find the amount to draw
                cave_name, col = event["coords"]
                num_cards = 0
                for col in range(4):
                    dragon_id = player.dragons_played[cave_name][col]
                    if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == "Shy":
                        num_cards += 1
            ret = []
            for _ in range(num_cards):
                ret.append(self._draw_from_deck("dragon"))
            return tuple(ret)
        raise ValueError(f"Invalid random event: {event}")

def alg_uniform_random(game_state: GameState, rollout_cache=None) -> int:
    """
    Choose an action to play during the simulation, assuming the game state
    has a choice (game_state.current_choice is a list of actions/events).
    This is an evenly random choice, which is not optimal for Wyrmspan.

    Returns the index of the action to take from the current choice list.
    """
    return random.randint(0, len(game_state.current_choice) - 1)

def alg_non_pass(game_state: GameState, rollout_cache=None) -> int:
    """
    Choose an action to play during the simulation, assuming the game state
    has a choice (game_state.current_choice is a list of actions/events).
    By default, you can choose a random action, but this is not optimal for Wyrmspan.

    Returns the index of the action to take from the current choice list.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        # we only have one action to choose from
        return 0
    # assume we have more than 1 action now
    pass_index = None
    skip_index = None
    # iterate from the end because pass is often near the end of the list
    for i in range(len(current_choice) - 1, -1, -1):
        action = current_choice[i]
        if "pass" in action:
            pass_index = i
            break
        elif "skip" in action or "skip_opr" in action:
            skip_index = i
            break
    if pass_index is not None:
        # we have a pass action
        if game_state.player.coins <= 4 and random.random() < (1 / (game_state.player.coins * 100)):
            # we can very rarely pass if we have fewer coins
            return pass_index
        # otherwise choose a non-pass action
        return random.choice([i for i in range(len(current_choice)) if i != pass_index])
    elif skip_index is not None:
        # we have a skip action
        # skip less often than uniform random
        if random.random() < (1 / (len(current_choice) * 2)):
            return skip_index
        return random.choice([i for i in range(len(current_choice)) if i != skip_index])
    else:
        return random.randint(0, len(current_choice) - 1)

def alg_play_dragon_cave(game_state: GameState, entice_prob=0.7, excavate_prob=0.7, rollout_cache=None) -> int:
    """
    This algorithm specifically tries to play dragons and caves more often
    when they are available in the current choice. Otherwise, it defers to another algorithm.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        # we only have one action to choose from
        return 0
    # assume we have more than 1 action now
    entice_index = None
    excavate_index = None
    for i, action in enumerate(current_choice):
        if "play_cave" in action:
            excavate_index = i
        elif "play_dragon" in action:
            entice_index = i
        if entice_index is not None and excavate_index is not None:
            break
    if entice_index is not None and random.random() < entice_prob:
        # play dragon if available and with some probability
        return entice_index
    elif excavate_index is not None and random.random() < excavate_prob:
        # play cave if available and with some probability
        return excavate_index
    else:
        # defer to another algorithm
        return alg_uniform_random(game_state)
        # return alg_non_pass(game_state)

# original weights were 3.2, 2.8, 2.1 with pass_penalty 1.5 and tie_threshold 0.35
def alg_greedy_action_priority(game_state: GameState,
                               dragon_weight=2.845,
                               cave_weight=2.056,
                               explore_weight=1.431,
                               pass_penalty=1.5,
                               tie_threshold=0.35,
                               rollout_cache=None) -> int:
    """
    Fast heuristic playout policy that uses action-type priorities to guide rollouts.

    GAME MECHANICS APPLIED:
    ======================
    1. DRAGON PLAYS: Value depends on:
       - Available enticing options (dragons can have 1+ ways to be played at different costs)
       - Dragon hand size (empty hand makes playing impossible)
       - Coin availability (most dragon plays require coins)
       - Objective synergies (some objectives reward specific dragon types)
    
    2. CAVE PLAYS: Value depends on:
       - Slot position (earlier slots generally better for accessing dragon effects)
       - Cave hand size (empty hand makes playing impossible)
       - Free vs. paid plays (free caves get bonus)
       - Objective alignment (cave_cards_played objective values cave placement)
    
    3. EXPLORE ACTIONS: Value depends on:
       - Cave state (which dragons are placed, early stopping on empty slots)
       - Dragon abilities with "if_activated" type (only these trigger during explore)
       - Times previously explored (diminishing returns)
       - Current objectives (eggs, cached_resources, cave_cards_played, guild_position)
       
    4. RESOURCE EFFICIENCY: Deck draw rates (~20-30 caves, ~40-60 dragons out of 75/183)
       mean decks rarely run out, so high-value cards should be prioritized when available.
    
    5. PASS PENALTY: Passing ends a turn, so it's heavily penalized unless coin generation
       is the only productive action available.
    
    The policy uses tie-breaking randomness to add rollout diversity while maintaining
    greedy prioritization of high-value actions.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        return 0

    player = game_state.player
    coins = player.coins
    total_eggs = sum(player.egg_totals.values())
    resources = player.resources
    total_resources = sum(resources.values())
    round_num = game_state.board["round_tracker"]["round"]

    def get_effect(action: dict) -> dict:
        """Extract the immediate action payload if wrapped in adv_effects."""
        if "adv_effects" in action and isinstance(action["adv_effects"], dict):
            return action["adv_effects"]
        return action

    def get_cost(action: dict) -> dict:
        if "cost" in action and isinstance(action["cost"], dict):
            return action["cost"]
        return {}

    best_idx = 0
    best_score = float("-inf")
    second_idx = None
    second_score = float("-inf")

    for i, action in enumerate(current_choice):
        effect = get_effect(action)
        cost = get_cost(action)

        if "play_dragon" in effect:
            score = dragon_weight + 0.35
            if len(player.dragon_hand) == 0:
                score -= 3.0
            if coins <= 1:
                score -= 0.2
            if round_num <= 2:
                score += 0.25
        elif "play_cave" in effect:
            score = cave_weight
            if len(player.cave_hand) == 0:
                score -= 3.0
            cave_payload = effect.get("play_cave", {})
            if isinstance(cave_payload, dict) and cave_payload.get("free", False):
                score += 0.45
            if round_num <= 2:
                score += 0.2
        elif "explore" in effect:
            score = explore_weight
            if round_num <= 2:
                score += 0.45
        elif "gain_resource" in effect:
            score = 1.7
        elif "lay_egg" in effect:
            score = 1.4
        elif "draw_decision" in effect or "gain_dragon" in effect or "gain_cave" in effect:
            score = 1.2
        elif "pass" in effect:
            score = -pass_penalty
            if coins > 0:
                score -= 0.6
        elif "skip" in effect or "skip_opr" in effect:
            score = -(pass_penalty * 0.7)
        else:
            score = 0.8

        if score > best_score:
            second_score, second_idx = best_score, best_idx
            best_score, best_idx = score, i
        elif score > second_score:
            second_score, second_idx = score, i

    if second_idx is not None and abs(best_score - second_score) <= tie_threshold:
        return random.choice([best_idx, second_idx])
    return best_idx


def alg_strategic_objective_aware(
    game_state: GameState,
    dragon_weight=3.677,
    cave_weight=2.569,
    explore_weight=1.545,
    pass_penalty=1.608,
    tie_threshold=0.423,
    rollout_cache=None,
) -> int:
    """
    A heavier, objective-aware rollout policy using rollout-local memoization.
    
    DIFFERENCES FROM GREEDY_ACTION_PRIORITY:
    ========================================
    This policy is more expensive but more strategic. It:
    
    1. CACHES STATE FEATURES: Uses RolloutCache to evaluate:
       - Dragon trait distributions (personalities, sizes, abilities)
       - Board value (dragons and caves already played)
       - Objective gaps (how far behind the automa on each objective)
       - Deck counts (dragons/caves remaining, affecting scarcity bonuses)
    
    2. EVALUATES ACTION CONSEQUENCES using:
       - Best available dragon option scores (considering entice costs, traits, objectives)
       - Best available cave option scores (considering slot positions, abilities, objectives)
       - Objective synergy tables (mapping objective types to action value weights)
       - Cost feasibility (rejecting actions the player can't afford)
    
    3. APPLIES WYRMSPAN MECHANICS:
       - Early exploration stopping (empty dragon slots terminate explore sequences)
       - Dragon ability types (only if_activated triggers during explore)
       - Deck draw rates (20-30 caves, 40-60 dragons; scarcity is mild)
       - Optional coin exchanges (4th cave card in a column offers 3-for-1 resource trade)
       - Objective scoring alignment (matching action values to scoring categories)
    
    4. USES OBJECTIVE LOOKAHEAD:
       - Scores actions based on how they help meet current round objectives
       - Adjusts weights dynamically based on objective type
       - Prioritizes "close objective gaps" to maximize end-of-round scoring
    
    The policy is 2-3x more expensive per evaluation than greedy_action_priority,
    but substantially better at planning multi-step sequences and objective alignment.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        return 0

    cache = rollout_cache if rollout_cache is not None else RolloutCache()
    cache.refresh(game_state)

    best_idx = 0
    best_score = float("-inf")
    second_idx = None
    second_score = float("-inf")

    for i, action in enumerate(current_choice):
        effect = _action_effect(action)
        score = cache.score_action(
            game_state,
            action,
            dragon_weight=dragon_weight,
            cave_weight=cave_weight,
            explore_weight=explore_weight,
            pass_penalty=pass_penalty,
        )

        # Encourage branches that improve the most urgent objective gap.
        if cache._summary["objective_counts"]:
            round_num = cache._summary["round"]
            current_gap = cache._summary["objective_counts"][round_num]["gap"]
            if "play_dragon" in effect or "play_cave" in effect:
                score += 0.15 * max(0, current_gap)
            elif "explore" in effect and current_gap < 0:
                score += 0.2

        if score > best_score:
            second_score, second_idx = best_score, best_idx
            best_score, best_idx = score, i
        elif score > second_score:
            second_score, second_idx = score, i

    if second_idx is not None and abs(best_score - second_score) <= tie_threshold:
        return random.choice([best_idx, second_idx])
    return best_idx

def get_sim_algo(algo_name, algo_kwargs):
    """
    Dispatcher for simulation algorithms.
    """
    if algo_name == "uniform_random":
        return lambda gs, rollout_cache=None: alg_uniform_random(gs, rollout_cache=rollout_cache)
    elif algo_name == "non_pass":
        return lambda gs, rollout_cache=None: alg_non_pass(gs, rollout_cache=rollout_cache)
    elif algo_name == "play_dragon_cave":
        def algo(gs, rollout_cache=None):
            return alg_play_dragon_cave(gs, rollout_cache=rollout_cache, **algo_kwargs)
        return algo
    elif algo_name == "greedy_action_priority":
        def algo(gs, rollout_cache=None):
            return alg_greedy_action_priority(gs, rollout_cache=rollout_cache, **algo_kwargs)
        return algo
    elif algo_name == "strategic_objective_aware":
        def algo(gs, rollout_cache=None):
            return alg_strategic_objective_aware(gs, rollout_cache=rollout_cache, **algo_kwargs)
        return algo
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def simulate_game(game_state: GameState, algo_name, algo_kwargs, display_name, seed=None) -> tuple:
    """
    Simulate a random game from the given game state until a terminal state is reached.
    Returns a score for the simulation, the name of the algorithm used, and the time taken to simulate.
    """
    if seed is not None:
        random.seed(seed)
    sim_algo = get_sim_algo(algo_name, algo_kwargs)
    rollout_cache = RolloutCache() if algo_name == "strategic_objective_aware" else None
    start_time = time.time()
    while game_state.phase != logic.PHASE_END_GAME:
        # check if we have a choice or random event
        if game_state.current_choice is not None:
            # we have a choice to make
            chosen_input = sim_algo(game_state, rollout_cache)
            game_state = logic.get_next_state(game_state, chosen_input)
        elif game_state.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = logic.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            game_state = logic.get_next_state(game_state, chosen_input)
        else:
            # progress the game
            game_state = logic.get_next_state(game_state, chosen_input=None)
    end_time = time.time()
    # return 1 if game_state.player.score > 30 else 0
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    # if game_state.player.score >= game_state.automa.score:
    #     return (5000 + (game_state.player.score - game_state.automa.score) ** 2) / 10000, display_name, end_time - start_time
    # else:
    #     return (5000 - (game_state.automa.score - game_state.player.score) ** 2) / 10000, display_name, end_time - start_time
    
    score_based_reward = (game_state.player.score ** 2) / 40_000
    if game_state.player.score >= game_state.automa.score:
        return score_based_reward + 0.75, display_name, end_time - start_time
    else:
        return score_based_reward, display_name, end_time - start_time

    # Other Objectives to Maximize
    #get as many cached resources as possible
    # score = 0
    # for cache_list in game_state.player.cached_resources.values():
    #     score += sum(num_caches for cache_dict in cache_list for num_caches in cache_dict.values())
    # return score / 50, display_name, end_time - start_time


def simulate_multiple_games(game_state: GameState, algo_name, algo_kwargs, display_name, num_simulations, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple:
    """
    Simulate multiple games from the given game state using the specified algorithm.
    Returns the average score for the simulations, the name of the algorithm used, and the time taken to simulate.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    total_score = 0
    start_time = time.time()
    base_seed = int(time.time() * 1000)  # Use current time in milliseconds as a base seed
    batch_size = max(1, batch_size)
    with ProcessPoolExecutor() as executor:
        futures = []
        for start, end in _chunk_count(num_simulations, batch_size):
            seeds = [base_seed + s for s in range(start, end)]
            futures.append(
                executor.submit(
                    _run_seeded_game_batch,
                    game_state,
                    algo_name,
                    algo_kwargs,
                    display_name,
                    seeds,
                )
            )
        # wait for all futures to complete and sum the scores
        for future in as_completed(futures):
            score, _, _, _ = future.result()
            total_score += score
    end_time = time.time()
    return total_score, display_name, end_time - start_time

def simulate_game_given_rng(game_state: GameState, algo_name, algo_kwargs, display_name, rng: RNGOrder=None) -> tuple:
    """
    Simulate a game following the given RNG order.
    Returns a score for the simulation, the name of the algorithm used, and the time taken to simulate.
    """
    sim_algo = get_sim_algo(algo_name, algo_kwargs)
    rollout_cache = RolloutCache() if algo_name == "strategic_objective_aware" else None
    start_time = time.time()
    this_rng = rng.get_copy()
    while game_state.phase != logic.PHASE_END_GAME:
        # check if we have a choice or random event
        if game_state.current_choice is not None:
            # we have a choice to make
            chosen_input = sim_algo(game_state, rollout_cache)
            game_state = logic.get_next_state(game_state, chosen_input)
        elif game_state.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = this_rng.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            game_state = logic.get_next_state(game_state, chosen_input)
        else:
            # progress the game
            game_state = logic.get_next_state(game_state, chosen_input=None)
    end_time = time.time()
    # return 1 if game_state.player.score > 30 else 0
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    # if game_state.player.score >= game_state.automa.score:
    #     return (5000 + (game_state.player.score - game_state.automa.score) ** 2) / 10000, display_name, end_time - start_time
    # else:
    #     return (5000 - (game_state.automa.score - game_state.player.score) ** 2) / 10000, display_name, end_time - start_time
    score_based_reward = (game_state.player.score ** 2) / 40000
    if game_state.player.score >= game_state.automa.score:
        return score_based_reward + 0.75, display_name, end_time - start_time
    else:
        return score_based_reward, display_name, end_time - start_time

def compare_algorithms(game_state: GameState = None, num_simulations: int = 500, algos=None, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Compare the performance of different simulation algorithms by running
    multiple simulations and returning the average score for each algorithm.
    """
    run_results = {}
    if algos is None:
        algos = []
        for entice_prob in [0.6, 0.75, 0.9]:
            for excavate_prob in [0.6, 0.75, 0.9]:
                algo_kwargs = {'entice_prob': entice_prob, 'excavate_prob': excavate_prob}
                algos.append(("play_dragon_cave", algo_kwargs, f"play_dragon_cave_{entice_prob:.4f}_{excavate_prob:.4f}"))
        # define some random play_dragon_cave probabilities
        # for i in range(6):
        #     entice_prob = random.random()
        #     excavate_prob = random.random()
        #     algo_kwargs = {'entice_prob': entice_prob, 'excavate_prob': excavate_prob}
        #     algos.append(("play_dragon_cave", algo_kwargs, f"play_dragon_cave_{entice_prob:.4f}_{excavate_prob:.4f}"))
        # constant algorithms
        # algos += [("uniform_random", {}, "uniform_random"), ("non_pass", {}, "non_pass")]
    for _, _, name in algos:
        run_results[name] = {'total_score': 0, 'num_simulations': 0, 'total_time': 0.0}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    batch_size = max(1, batch_size)
    # run each algorithm in parallel num_simulations times
    with ProcessPoolExecutor() as executor:
        futures = []
        task_batch = []
        task_seed = int(time.time() * 1000)
        for _ in range(num_simulations):
            if game_state is None:
                this_game_state = SoloGameState(automa_difficulty=1)
                this_game_state.create_game()
            else:
                this_game_state = game_state
            for algo_name, algo_kwargs, display_name in algos:
                # batch the simulation tasks to reduce executor overhead
                task_batch.append((this_game_state.make_copy(), algo_name, algo_kwargs, display_name, task_seed))
                task_seed += 1
                if len(task_batch) >= batch_size:
                    futures.append(executor.submit(_run_simulation_task_batch, task_batch))
                    task_batch = []

        if task_batch:
            futures.append(executor.submit(_run_simulation_task_batch, task_batch))

        for i, future in enumerate(as_completed(futures)):
            for score, algo_name, time_taken in future.result():
                run_results[algo_name]['total_score'] += score
                run_results[algo_name]['num_simulations'] += 1
                run_results[algo_name]['total_time'] += time_taken
            if i % 500 == 0:
                print(f"\nCompleted {i+1} simulations")
                for display_name, results in run_results.items():
                    if results['num_simulations'] > 0:
                        print(f"\t{display_name}: {results['total_score'] / results['num_simulations']:.4f} (Total: {results['total_score']}, Simulations: {results['num_simulations']}, Time: {results['total_time']:.2f} seconds)")
    # calculate the average scores
    print(f"\nSimulations completed per algorithm: {num_simulations}")
    for display_name, results in run_results.items():
        if results['num_simulations'] > 0:
            average_score = results['total_score'] / results['num_simulations']
            print(f"{display_name}: {average_score:.4f} (Total: {results['total_score']}, Simulations: {results['num_simulations']}, Time: {results['total_time']:.2f} seconds)")
        else:
            print(f"{display_name}: No simulations run.")
    
    return run_results

def evolutionary_compare_algorithms(game_state: GameState = None, num_simulations: int = 200):
    """
    Find the best hyperparameters for the play_dragon_cave algorithm using an evolutionary algorithm.
    Starting with a set of random hyperparameters, evolve them over multiple generations
    by selecting the best performing ones and mutating them slightly.
    """
    import numpy as np

    # Define the initial population size and number of generations
    population_size = 10
    num_generations = 30

    # Initialize a random population of hyperparameters
    population = [(random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)) for _ in range(population_size)]
    # population.append((0.8718, 0.7396)) # add previous best hyperparameters if available
    population.append((0.97, 0.75)) # add previous best hyperparameters if available

    for generation in range(num_generations):
        print(f"\n\n*> Generation {generation + 1}/{num_generations}")
        print(f"Current population of {len(population)}: {population}")
        # generate possible offspring from the current population
        offspring = []
        remaining_parents = population.copy()
        while len(remaining_parents) > 1:
            # randomly pair two individuals from the population
            parent1_index = random.choice(range(len(remaining_parents)))
            parent1 = remaining_parents[parent1_index]
            remaining_parents.pop(parent1_index)
            parent2_index = random.choice(range(len(remaining_parents)))
            parent2 = remaining_parents[parent2_index]
            remaining_parents.pop(parent2_index)
            print(f"Pairing parents: {parent1} and {parent2}")
            # create a new individual by averaging the parents' hyperparameters
            new_individual = (
                (parent1[0] + parent2[0]) / 2 + np.random.normal(0, 0.05),  # entice_prob
                (parent1[1] + parent2[1]) / 2 + np.random.normal(0, 0.05)   # excavate_prob
            )
            # ensure the new individual is within bounds
            new_individual = (max(0.1, min(1.0, new_individual[0])), max(0.1, min(1.0, new_individual[1])))
            # add the new individual to the offspring
            offspring.append(new_individual)
        # Use the full population for the next generation
        population = offspring + population
        print(f"\nOffspring generated: {offspring}")
        print("\nEvaluating offspring...")
        algos = [("play_dragon_cave", {'entice_prob': ep, 'excavate_prob': ex}, f"play_dragon_cave_{ep:.4f}_{ex:.4f}") for ep, ex in population]
        results = compare_algorithms(game_state=game_state, num_simulations=num_simulations, algos=algos)

        # Extract scores for the current population
        scores = []
        for _, _, display_name in algos:
            if display_name in results:
                scores.append(results[display_name]['total_score'] / results[display_name]['num_simulations'])
            else:
                scores.append(0)
        # Sort the population based on scores
        sorted_population = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)

        # Keep the top 'population_size' individuals for the next generation
        population = [ind for ind, score in sorted_population[:population_size]]

    # Return the best hyperparameters found
    best_entice_prob, best_excavate_prob = sorted_population[0][0]
    print(f"\n\n>> Best hyperparameters: entice_prob={best_entice_prob:.4f}, excavate_prob={best_excavate_prob:.4f}")

if __name__ == "__main__":
    algos = [
        ("uniform_random", {}, "uniform_random"),
        #("non_pass", {}, "non_pass"),
        ("play_dragon_cave", {'entice_prob': 0.8718, 'excavate_prob': 0.7396}, "play_dragon_cave_0.8718_0.7396"),
        #("greedy_action_priority", {'dragon_weight': 3.2, 'cave_weight': 2.8, 'explore_weight': 2.1}, "greedy_action_priority_original"),
        ("greedy_action_priority", {'dragon_weight': 2.845, 'cave_weight': 2.056, 'explore_weight': 1.431}, "greedy_action_priority_tuned?"),
        ("strategic_objective_aware", {'dragon_weight': 3.677, 'cave_weight': 2.569, 'explore_weight': 1.545, 'pass_penalty': 1.608, 'tie_threshold': 0.423}, "strategic_objective_aware_1"),
        ("strategic_objective_aware", {'dragon_weight': 11.649, 'cave_weight': 6.571, 'explore_weight': 4.927, 'pass_penalty': 1.57, 'tie_threshold': 1.496}, "strategic_objective_aware_2"),
        ("strategic_objective_aware", {'dragon_weight': 12.441, 'cave_weight': 7.055, 'explore_weight': 4.927, 'pass_penalty': 1.541, 'tie_threshold': 1.535}, "strategic_objective_aware_3"),
        ("strategic_objective_aware", {'dragon_weight': 17.596, 'cave_weight': 6.939, 'explore_weight': 5.447, 'pass_penalty': 4.461, 'tie_threshold': 1.219}, "strategic_objective_aware_4"),
        ("strategic_objective_aware", {'dragon_weight': 7.65, 'cave_weight': 3.409, 'explore_weight': 1.864, 'pass_penalty': 1.9, 'tie_threshold': 0.25}, "strategic_objective_aware_5"),
    ]
    compare_algorithms(num_simulations=5000, algos=algos)
    
    # evolutionary_compare_algorithms(num_simulations=300)

    # import logging
    # logging.basicConfig(
    #     filename='playout_compare.log',
    #     level=logging.DEBUG,
    #     # level=logging.INFO,
    #     # level=logging.WARNING,
    #     format='%(asctime)s:%(levelname)s:%(message)s',
    #     filemode='w'
    # )
    # logger = logging.getLogger(__name__)
    # game = SoloGameState()
    # game.create_game()
    # # simulate_game(game, "play_dragon_cave", {'entice_prob': 0.8718, 'excavate_prob': 0.7396}, "play_dragon_cave_0.8718_0.7396")

    # rng = RNGOrder(game)
    # print(f"Opening hand: {game.player.dragon_hand}")
    # print(any(d in rng.dragon_deck for d in game.player.dragon_hand))
    # print(f"Dragon Deck order: {rng.dragon_deck}")
    # print(f"Cave deck order: {rng.cave_deck}")
    # print(f"Automa decks: {rng.automa_decks}")