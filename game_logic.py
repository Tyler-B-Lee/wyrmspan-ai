import logging
logging.basicConfig(
    filename='file.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

from game_states import *
import itertools
from typing import Union
import math

CaveName = str
DragonNumber = int
CaveNumber = int


# state conversion functions
def player_state_to_dict(player_state: PlayerState) -> dict:
    return {
        "dragon_hand": player_state.dragon_hand,
        "cave_hand": player_state.cave_hand,
        "resources": player_state.resources,
        "egg_totals": player_state.egg_totals,
        "num_dragons_played": player_state.num_dragons_played,
        "score": player_state.score,
        "guild_markers": player_state.guild_markers,
        "coins": player_state.coins,
        "caves_played": player_state.caves_played,
        "dragons_played": player_state.dragons_played,
        "cached_resources": player_state.cached_resources,
        "tucked_dragons": player_state.tucked_dragons,
        "nested_eggs": player_state.nested_eggs,
        "times_explored": player_state.times_explored,
        "adventurer_position": player_state.adventurer_position
    }

def dict_to_player_state(data: dict) -> PlayerState:
    player_state = PlayerState()
    player_state.dragon_hand = data["dragon_hand"]
    player_state.cave_hand = data["cave_hand"]
    player_state.resources = data["resources"]
    player_state.egg_totals = data["egg_totals"]
    player_state.num_dragons_played = data["num_dragons_played"]
    player_state.score = data["score"]
    player_state.guild_markers = data["guild_markers"]
    player_state.coins = data["coins"]
    player_state.caves_played = data["caves_played"]
    player_state.dragons_played = data["dragons_played"]
    player_state.cached_resources = data["cached_resources"]
    player_state.tucked_dragons = data["tucked_dragons"]
    player_state.nested_eggs = data["nested_eggs"]
    player_state.times_explored = data["times_explored"]
    player_state.adventurer_position = data["adventurer_position"]
    return player_state

def automa_state_to_dict(automa_state: AutomaState) -> dict:
    return {
        "dragons": automa_state.dragons,
        "caves": automa_state.caves,
        "score": automa_state.score,
        "difficulty": automa_state.difficulty,
    }

def dict_to_automa_state(data: dict) -> AutomaState:
    automa_state = AutomaState()
    automa_state.dragons = data["dragons"]
    automa_state.caves = data["caves"]
    automa_state.score = data["score"]
    automa_state.difficulty = data["difficulty"]
    return automa_state

def game_state_to_dict(game_state: GameState) -> dict:
    return {
        "turn": game_state.turn,
        "phase": game_state.phase,
        "board": game_state.board,
        "dragon_deck": game_state.dragon_deck,
        "cave_deck": game_state.cave_deck,
        "dragon_discard": game_state.dragon_discard,
        "cave_discard": game_state.cave_discard,
        "event_queue": game_state.event_queue,
        "players": [player_state_to_dict(player) for player in game_state.players],
        "round_start_player": game_state.round_start_player,
        "current_player": game_state.current_player,
    }

def dict_to_game_state(data: dict) -> GameState:
    game_state = GameState()
    game_state.turn = data["turn"]
    game_state.phase = data["phase"]
    game_state.board = data["board"]
    game_state.dragon_deck = data["dragon_deck"]
    game_state.cave_deck = data["cave_deck"]
    game_state.dragon_discard = data["dragon_discard"]
    game_state.cave_discard = data["cave_discard"]
    game_state.event_queue = data["event_queue"]
    game_state.players = [dict_to_player_state(player) for player in data["players"]]
    game_state.round_start_player = data["round_start_player"]
    game_state.current_player = data["current_player"]
    return game_state

def solo_game_state_to_dict(solo_game_state: SoloGameState) -> dict:
    return {
        "turn": solo_game_state.turn,
        "phase": solo_game_state.phase,
        "board": solo_game_state.board,
        "dragon_deck": solo_game_state.dragon_deck,
        "cave_deck": solo_game_state.cave_deck,
        "dragon_discard": solo_game_state.dragon_discard,
        "cave_discard": solo_game_state.cave_discard,
        # turn deque into list
        "event_queue": solo_game_state.event_queue,
        "automa_difficulty": solo_game_state.automa_difficulty,
        "automa": automa_state_to_dict(solo_game_state.automa),
        "player": player_state_to_dict(solo_game_state.player),
        "round_start_player": solo_game_state.round_start_player,
        "current_player": solo_game_state.current_player,
    }

def dict_to_solo_game_state(data: dict) -> SoloGameState:
    solo_game_state = SoloGameState()
    solo_game_state.turn = data["turn"]
    solo_game_state.phase = data["phase"]
    solo_game_state.board = data["board"]
    solo_game_state.dragon_deck = data["dragon_deck"]
    solo_game_state.cave_deck = data["cave_deck"]
    solo_game_state.dragon_discard = data["dragon_discard"]
    solo_game_state.cave_discard = data["cave_discard"]
    solo_game_state.event_queue = data["event_queue"]
    solo_game_state.automa_difficulty = data["automa_difficulty"]
    solo_game_state.automa = dict_to_automa_state(data["automa"])
    solo_game_state.player = dict_to_player_state(data["player"])
    solo_game_state.round_start_player = data["round_start_player"]
    solo_game_state.current_player = data["current_player"]
    return solo_game_state


# basic / logical functions
def can_lay_egg_at(player_state: PlayerState, coords:tuple) -> bool:
    """
    Check if the player can lay an egg at the specified location.
    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    # check the nested_eggs of the player
    ne = player_state.nested_eggs[cave_name][col]
    return ne[0] < ne[1]

def get_total_eggs(player_state: PlayerState) -> int:
    """
    Get the total number of eggs laid by the player.
    """
    return sum(player_state.egg_totals.values())

def get_dragon_list(player_state: PlayerState, location: str) -> list[tuple]:
    """
    Get the list of dragons played by the player on their mat in the specified location.
    The location can be "any", "col0/1/2/3", or the name of a cave.

    Returns a list of tuples (dragon, location) where dragon is the dragon number
    and location is a tuple (cave_name, col_index).
    """
    dragons = []
    col = int(location[-1]) if location.startswith("col") else None
    for cave_name, dragon_list in player_state.dragons_played.items():
        # check each cave (horizontal row)
        if (location not in CAVE_NAMES) or \
                (location in CAVE_NAMES and cave_name == location):
            for col_index, dragon in enumerate(dragon_list):
                # check each slot in cave (vertical column)
                if col is None or col_index == col:
                    if dragon:
                        # add the dragon and its location to the list
                        dragons.append( (dragon, (cave_name, col_index) ) )
                    else:
                        break # no more dragons in this cave
    return dragons

def can_pay_resources(player_state: PlayerState, cost_dict:dict) -> bool:
    """
    Check if the player can pay the specified resources.
    The cost is a dictionary with the resource names as keys and the amounts as values.
    The costs are for meat, gold, crystal, milk, and 'any' for any resource.
    
    At any point, the player can exchange 2 resources of any combination
    for 1 resource of any type.
    """
    player_amounts = player_state.resources.copy()
    # try to pay the cost with the player's resources
    remaining_cost = {}
    # check if the player has enough resources for the cost
    for resource, cost_amount in cost_dict.items():
        if resource == "any":
            # skip for now
            continue
        if player_amounts[resource] >= cost_amount:
            player_amounts[resource] -= cost_amount
        else:
            remaining_cost[resource] = cost_amount - player_amounts[resource]
            player_amounts[resource] = 0
    # check if the player has enough resources for the remaining cost
    free_resources = sum(player_amounts.values())
    # check if the player can pay the remaining cost by exchanging resources
    return (free_resources - 2 * sum(remaining_cost.values()) - cost_dict.get("any",0)) >= 0

def can_excavate_cave(player_state: PlayerState, cave_name: CaveName, free: bool=False) -> tuple[bool, int]:
    """
    Check if the player can excavate a cave (a row on their mat).
    Does not check for where the cave card is being used from.
    If free is False (default), check if the player has enough resources for later caves.

    Returns a tuple (can_excavate, cave_id) where can_excavate is a boolean
    """
    cave_list = player_state.caves_played[cave_name]
    if cave_list[-1]: # check if last cave is already excavated
        return False, -1
    if free: # check if the cave is free to excavate
        return True, cave_list.index(None)
    # not free - check if the player has enough resources for later caves
    if not cave_list[1]: # slot 2 has no extra cost
        return True, cave_list.index(None)
    # check cave slots 2 and 3
    num_eggs = get_total_eggs(player_state)
    if not cave_list[2]:
        return num_eggs >= 1, cave_list.index(None)
    # only slot 3 is available, since we saw this last slot is not excavated
    return num_eggs >= 2, cave_list.index(None)

def get_dragon_enticement_options(player_state: PlayerState, dragon_info: dict, discount:str="none") -> list[dict]:
    """
    Check if the player can entice a specific dragon in a specific cave. Does not check
    where this dragon is coming from. The discount is described by a string, which can be:
    "none", "free", "1off", "no_resources".

    Returns a list of cost dictionaries, each with a valid cost for the dragon.
    An empty dictionary means the player can entice the dragon for free.
    An empty list means the player cannot entice the dragon at all.
    """
    logger.debug("\t*- Getting dragon entice options for %s", dragon_info["name"])
    costs = []
    resource_cost_dict = {}
    for resource_name in RESOURCES:
        cost_name = f"{resource_name}_cost"
        if dragon_info[cost_name] > 0:
            resource_cost_dict[resource_name] = dragon_info[cost_name]
    logger.debug("\t- Resource cost dict: %s", resource_cost_dict)
    # check if the player has enough resources for the dragon
    if discount == "1off":
        # player can pay 1 less of any resource, so we check each discount possible
        for resource in resource_cost_dict:
            new_cost = resource_cost_dict.copy()
            new_cost[resource] -= 1
            # check if the player has enough resources for the cost
            if can_pay_resources(player_state, new_cost):
                costs.append(new_cost)
    elif discount == "none":
        # player must pay the full resource cost
        if can_pay_resources(player_state, resource_cost_dict):
            costs.append(resource_cost_dict)
    else:
        # player does not pay any resources
        costs.append({})
    if len(costs) == 0:
        logger.debug("\t\tPlayer cannot afford the resource cost")
        return costs
    
    # now we check other possible costs and ajust the current costs
    if discount != "free":
        # check coin cost
        effective_coin_cost = dragon_info["coin_cost"]
        if discount == "none":
            effective_coin_cost += 1 # player must pay extra entice cost
        if effective_coin_cost > player_state.coins:
            logger.debug("\t\tPlayer cannot afford the coin cost")
            return []
        if effective_coin_cost > 0:
            logger.debug("\t- Coin cost: %s", effective_coin_cost)
            for cost in costs:
                cost["coin"] = effective_coin_cost
        # check egg cost
        if dragon_info["egg_cost"] > 0:
            if dragon_info["egg_cost"] > get_total_eggs(player_state):
                logger.debug("\t\tPlayer cannot afford the egg cost")
                return []
            for cost in costs:
                cost["egg"] = {
                    "amount": dragon_info["egg_cost"],
                    "location": "any"
                }
    logger.debug("\t- Final costs: %s", costs)
    return costs

def can_pay_cost(player_state: PlayerState, cost_dict:dict) -> bool:
    """
    Check if the player can pay the specified cost.
    This is a generic checking function for cost parameters used for effects.

    The cost is a dictionary with the resource names as keys and the amounts as values.
    """
    # check each cost in the cost dictionary
    for name, value in cost_dict.items():
        if name == "any_resource":
            if not can_pay_resources(player_state, { "any": value }):
                return False
        elif name == "cave_card":
            # check if the player has enough cave cards in hand
            if len(player_state.cave_hand) < value:
                return False
        elif name == "dragon_card":
            # check if the player has enough dragon cards in hand
            if len(player_state.dragon_hand) < value:
                return False
        elif name == "egg":
            # trickiest case, we must check for the location specified
            amount = value["amount"]
            location = value["location"]
            if location == "any":
                # check if the player has enough eggs in any location
                available_eggs = get_total_eggs(player_state)
            elif location == "ortho":
                # check orthogonally adjacent dragons
                main_cave, main_col = cost_dict["coords"] # assume coords are given
                available_eggs = 0
                mci = CAVE_NAMES.index(main_cave)
                # check the 4 orthogonal directions
                # up
                if mci > 0:
                    available_eggs += player_state.nested_eggs[CAVE_NAMES[mci-1]][main_col][0]
                # down
                if mci < 2:
                    available_eggs += player_state.nested_eggs[CAVE_NAMES[mci+1]][main_col][0]
                # left
                if main_col > 0:
                    available_eggs += player_state.nested_eggs[main_cave][main_col-1][0]
                # right
                if main_col < 3:
                    available_eggs += player_state.nested_eggs[main_cave][main_col+1][0]
            elif location == "this_cave":
                # check the eggs in the specified cave
                main_cave, main_col = cost_dict["coords"] # assume coords are given
                cave_eggs = player_state.nested_eggs[main_cave]
                available_eggs = sum(eggs for eggs,nests in cave_eggs)
            # check if the player has enough eggs for the cost
            if available_eggs < amount:
                return False
        elif name == "coin":
            # check if the player has enough coins for the cost
            if player_state.coins < value:
                return False
        elif name == "choice":
            # check possible costs to pay
            # value is a list of cost dictionaries
            return any(can_pay_cost(player_state, cost) for cost in value)
    
    return True

# manipulation functions
def lay_egg(player_state: PlayerState, coords:tuple) -> None:
    """
    Lay an egg at the specified location.
    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    if cave_name != "mat_slots":
        player_state.nested_eggs[cave_name][col][0] += 1 # increment the number of eggs laid
    player_state.egg_totals[cave_name] += 1 # increment the total number of eggs laid
    player_state.score += 1 # increment the score for the player
    logger.info("> Player laid an egg at %s", coords)

def pay_egg(player_state: PlayerState, coords:tuple) -> None:
    """
    Pay an egg at the specified location.
    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    if cave_name != "mat_slots":
        player_state.nested_eggs[cave_name][col][0] -= 1 # decrement the number of eggs laid
    player_state.egg_totals[cave_name] -= 1 # decrement the total number of eggs laid
    player_state.score -= 1 # decrement the score for the player
    logger.info("> Player paid an egg from %s", coords)

def discard_dragon(player_state: PlayerState, game_state: GameState, dragon: DragonNumber) -> None:
    """
    Discard a dragon from the player's hand and add it to the discard pile.
    """
    player_state.dragon_hand.remove(dragon) # remove the dragon from the hand
    game_state.dragon_discard.append(dragon) # add the dragon to the discard pile
    logger.info(f"> Player discarded dragon {dragon} ({DRAGON_CARDS[dragon]['name']})")

def discard_cave(player_state: PlayerState, game_state: GameState, cave: CaveNumber) -> None:
    """
    Discard a cave from the player's hand and add it to the discard pile.
    """
    player_state.cave_hand.remove(cave) # remove the cave from the hand
    game_state.cave_discard.append(cave) # add the cave to the discard pile
    logger.info(f"> Player discarded cave {cave} ({CAVE_CARDS[cave]['text']})")

def deduct_resources(player_state: PlayerState, cost_dict:dict) -> None:
    """
    Deduct the specified resources from the player's resources.
    It is assumed that the player has enough resources for the cost.

    The cost is a dictionary with the resource names as keys and the amounts as values.
    The costs are for meat, gold, crystal, and milk.
    """
    for resource, cost_amount in cost_dict.items():
        player_state.resources[resource] -= cost_amount
        assert player_state.resources[resource] >= 0, f"Player has negative resources: {player_state.resources}"
        logger.info(f"> Deducted {cost_amount} {resource} from player resources")

def cache_resource(player_state: PlayerState, game_state:GameState, resource:str, coords:tuple) -> None:
    """
    Cache a resource at the specified location.
    Does not affect the player's resources.
    Triggers any hatchling effects if applicable.

    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    player_state.cached_resources[cave_name][col][resource] += 1 # increment the cached resource
    player_state.score += 1 # increment the score for the player
    logger.info(f"> Cached {resource} at {coords}")
    # check for hatchling
    dragon_info = DRAGON_CARDS[player_state.dragons_played[cave_name][col]]
    if "on_feed" in dragon_info:
        # check for matching cache type
        if dragon_info["on_feed"]["type"] == resource:
            # we activate the effect
            logger.info(f"\tActivated hatchling effect for the {dragon_info['name']}")
            if (
                player_state.hatchling_grown[cave_name][col] == False and
                player_state.cached_resources[cave_name][col][resource] >= 3
            ):
                # we can grow the hatchling
                logger.info(f"\t\tHatchling also grew up!")
                player_state.hatchling_grown[cave_name][col] = True
                game_state.event_queue.append({
                    "adv_effects": {
                        "sequence": [
                            dragon_info["on_feed"]["effect"],
                            dragon_info["on_grow_up"]
                        ]
                    }
                })
            else:
                # we can only activate the effect
                game_state.event_queue.append(dragon_info["on_feed"]["effect"])
    
def tuck_dragon(player_state: PlayerState, game_state:GameState, dragon: DragonNumber, coords:tuple) -> None:
    """
    Tuck a dragon at the specified location.
    Does not affect the player's hand.

    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    player_state.tucked_dragons[cave_name][col].append(dragon) # add the dragon to the tucked dragons
    player_state.score += 1 # increment the score for the player
    logger.info(f"> Tucked dragon {dragon} at {coords}")
    # check for hatchling
    dragon_info = DRAGON_CARDS[player_state.dragons_played[cave_name][col]]
    if "on_feed" in dragon_info:
        # check for matching cache type
        if dragon_info["on_feed"]["type"] == "tuck":
            # we activate the effect
            logger.info(f"\tActivated hatchling effect for the {dragon_info['name']}")
            if (
                player_state.hatchling_grown[cave_name][col] == False and
                len(player_state.tucked_dragons[cave_name][col]) >= 3
            ):
                # we can grow the hatchling
                logger.info(f"\t\tHatchling also grew up!")
                player_state.hatchling_grown[cave_name][col] = True
                game_state.event_queue.append({
                    "adv_effects": {
                        "sequence": [
                            dragon_info["on_feed"]["effect"],
                            dragon_info["on_grow_up"]
                        ]
                    }
                })
            else:
                # we can only activate the effect
                game_state.event_queue.append(dragon_info["on_feed"]["effect"])

def excavate_cave(player_state: PlayerState, game_state:GameState, event:dict, cave_id:int) -> None:
    """
    Excavate a cave for the player specified by the event.
    """
    cave_location_name = event["cave_location"]
    index_to_excavate = player_state.caves_played[cave_location_name].index(None)
    # excavate the cave
    player_state.caves_played[cave_location_name][index_to_excavate] = cave_id
    logger.info(f"> Excavated cave {cave_id} at {cave_location_name}")
    # add effects from cave to the event queue
    cave_effect = CAVE_CARDS[cave_id]["when_played"]
    if index_to_excavate == 3:
        # we must check if the player can do the 4th space exchange
        if len(player.dragon_hand) + len(player.cave_hand) + sum(player.resources.values()) >= 3:
            # add the cave effect to the event queue
            logger.info("\tPlayer can perform the 4th space exchange")
            new_event = {
                "adv_effects": 
                    {"choice": [
                        {"adv_effects": {"sequence":[{"4th_space": False}, cave_effect]}},
                        {"adv_effects": {"sequence":[cave_effect, {"4th_space": False}]}}
                    ]}
            }
            game_state.event_queue.append(new_event)
            return
    # add the cave effect to the event queue
    game_state.event_queue.append(cave_effect)

def place_dragon(player_state: PlayerState, game_state:GameState, coords:tuple, dragon_id:int) -> None:
    """
    Place a dragon for the player specified by the event.
    """
    cave_loc_name, index_to_place = coords
    # place the dragon
    # TODO - Guild of Highlands effect: play dragon on top of another dragon
    # update player state
    dragon_info = DRAGON_CARDS[dragon_id]
    player_state.dragons_played[cave_loc_name][index_to_place] = dragon_id
    player_state.num_dragons_played[cave_loc_name] += 1
    player_state.score += dragon_info["VP"] # increment the score for the player
    player_state.nested_eggs[cave_loc_name][index_to_place][1] = dragon_info["capacity"]
    if dragon_info["size"] == "Hatchling":
        player_state.hatchling_grown[cave_loc_name][index_to_place] = False
    logger.info(f"> Placed dragon {dragon_id} ({dragon_info['name']}) at {coords}")

    # add effects from dragon to the event queue
    dragon_effect = dragon_info.get("when_played", None)
    if dragon_effect:
        logger.info(f"\tDragon has a when played effect: {dragon_effect}")
        dragon_effect = copy.deepcopy(dragon_effect)
        dragon_effect["coords"] = (cave_loc_name, index_to_place)
        game_state.event_queue.append(dragon_effect)

def refresh_cave_deck(game_state: GameState) -> None:
    """
    Refresh the cave deck by copying the discard pile back to the deck
    and clearing the discard pile.
    """
    game_state.cave_deck = game_state.cave_discard.copy()
    game_state.cave_discard.clear() # clear the discard pile
    logger.info("> Refreshed cave deck")

def refresh_dragon_deck(game_state: GameState) -> None:
    """
    Refresh the dragon deck by copying the discard pile back to the deck
    and clearing the discard pile.
    """
    game_state.dragon_deck = game_state.dragon_discard.copy()
    game_state.dragon_discard.clear() # clear the discard pile
    logger.info("> Refreshed dragon deck")

# main game functions
def get_current_player(game_state:GameState) -> Union[PlayerState, AutomaState]:
    """
    Get the current player from the game state.
    The current player is the player whose turn it is.
    """
    if isinstance(game_state, SoloGameState):
        return game_state.player if game_state.current_player == 0 else game_state.automa
    # for multiplayer game, return the current player
    return game_state.players[game_state.current_player]

# theory dump / thinking
# so every game state can be divided into several categories:
# - Halted: the game cannot progress automatically, since there
#   are multiple possible future states to follow this one. Either
#   the player must make a choice or a random event must be resolved.
#   When this state is reached a new node is created in the game tree.
#   This happens when the next event in the queue is a choice or random event.
#
# - Active: the game can progress automatically. Either we are in the
#   middle of resolving an event or the event queue has non-choice/random
#   events left in the front to resolve. We must resolve these first before
#   another node can be created.
#
# - Terminal: the game is over. The game state is not modified anymore.
#   We can calculate the final score for the player.
#
# So if the queue is ever empty, we are technically in an active state;
# we can always try to find the next choice or random event to resolve, or
# the game ends. We need functions that can progress the game, even if the queue is empty.
def progress_game(game_state:GameState) -> GameState:
    """
    Progress the game by one step.

    This function will check the game state and try to find the next
    choice or random event to resolve. If the game is over, it will
    return the final game state.
    """
    # loop until we find a choice or random event to resolve
    logger.info("** Progressing game. Current phase: %s", game_state.phase)
    while True:
        current_phase = game_state.phase
        current_player_obj = get_current_player(game_state)
        logger.info("Current player: %s", current_player_obj)
        game_state.log_card_display()

        if current_phase == PHASE_BEFORE_ACTION:
            # start of the turn
            logger.info("Phase: PHASE_BEFORE_ACTION")
            if game_state.all_players_passed():
                # all players have passed, so we move to the end of the round
                logger.info("All players have passed. Moving to PHASE_END_ROUND.")
                game_state.phase = PHASE_END_ROUND
                game_state.current_player = game_state.round_start_player
            elif current_player_obj.passed_this_round:
                # player has passed, so we move to the next player
                logger.info("Player %s has passed. Moving to the next player.", game_state.current_player)
                if isinstance(game_state, SoloGameState):
                    game_state.current_player = 1 - game_state.current_player
                else:
                    game_state.current_player = (game_state.current_player + 1) % len(game_state.players)
            else:
                # player has not passed, so we check if they can take an action
                logger.info("Player %s has not passed.", game_state.current_player)
                if isinstance(current_player_obj, AutomaState):
                    logger.info("\tResolving automa action.")
                    # TODO - implement automa action resolution
                    current_player_obj.passed_this_round = True
                else:
                    new_event = get_main_action_choice(current_player_obj)
                    if len(new_event["choice"]) > 0:
                        logger.info("\tAdding main action choice to event queue.")
                        new_event["choice"].append({"pass": True})
                        game_state.event_queue.append({"adv_effects": new_event})
                        game_state.phase = PHASE_MAIN_ACTION
                        break
                    else:
                        logger.info("\tNo valid actions for player %s. Forcing pass.", game_state.current_player)
                        current_player_obj.passed_this_round = True
                        game_state.phase = PHASE_END_TURN

        elif current_phase == PHASE_MAIN_ACTION:
            # we should have resolved the main action event
            # so we assume the event queue is empty
            # and the player's turn is over
            logger.info("Phase: PHASE_MAIN_ACTION")
            assert len(game_state.event_queue) == 0, "Main action event queue is not empty"
            logger.info("Main action is complete. Moving to PHASE_END_TURN.")
            game_state.phase = PHASE_END_TURN

        elif current_phase == PHASE_END_TURN:
            logger.info("Phase: PHASE_END_TURN")
            if current_player_obj.coins > 9:
                logger.info(">> Player has more than 9 coins, so they must discard down to 9.")
            current_player_obj.coins = min(current_player_obj.coins, 9)
            if len(current_player_obj.dragon_hand) + len(current_player_obj.cave_hand) > 9:
                logger.info(">> Player has more than 9 cards. Adding discard event.")
                new_event = {"choice": []}
                for dragon in current_player_obj.dragon_hand:
                    new_event["choice"].append({"discard_dragon": {"dragon": dragon}})
                for cave in current_player_obj.cave_hand:
                    new_event["choice"].append({"discard_cave": {"cave": cave}})
                game_state.event_queue.append({"adv_effects": new_event})
                break
            if sum(current_player_obj.resources.values()) > 9:
                logger.info(">> Player has more than 9 resources. Adding resource discard event.")
                new_event = {"choice": []}
                for resource in current_player_obj.resources.keys():
                    if current_player_obj.resources[resource] > 0:
                        new_event["choice"].append({"deduct_resources": {"cost": {resource: 1}}})
                game_state.event_queue.append({"adv_effects": new_event})
                break
            logger.info("> Player has at most 9 of all resources.")
            # refill card display from decks
            # this would create a random event
            full_display = True
            for i in range(3):
                if game_state.board["card_display"]["dragon_cards"][i] is None:
                    # random event to draw dragon for this slot
                    logger.info(">> Refilling dragon display slot %s", i)
                    new_event = {
                        "random": {
                            "refill_dragon_display": {
                            "slot": i,
                            "possible_outcomes": "dragon_deck",
                        }}
                    }
                    game_state.event_queue.append(new_event)
                    full_display = False
                if game_state.board["card_display"]["cave_cards"][i] is None:
                    # random event to draw cave for this slot
                    logger.info(">> Refilling cave display slot %s", i)
                    new_event = {
                        "random": {
                            "refill_cave_display": {
                            "slot": i,
                            "possible_outcomes": "cave_deck",
                        }}
                    }
                    game_state.event_queue.append(new_event)
                    full_display = False
            if not full_display:
                # we are not done with the refill, so we break
                break
            # display is full, move to the next player
            logger.info("> Display is full. Moving to the next player.")
            if isinstance(game_state, SoloGameState):
                game_state.current_player = 1 - game_state.current_player
            else:
                game_state.current_player = (game_state.current_player + 1) % len(game_state.players)
            game_state.phase = PHASE_BEFORE_ACTION

        elif current_phase == PHASE_END_ROUND:
            logger.info("Phase: PHASE_END_ROUND")
            # first trigger once per round abilities
            # in order starting from the round start player
            # NOTE - Assume solo game for now
            if not game_state.board["round_tracker"]["finished_opr"]:
                # check if we need to find once per round abilities
                logger.info("* We are not finished handling once per round abilities")
                if game_state.board["round_tracker"]["opr_remaining"] is None:
                    logger.info("\t* Finding once per round abilities...")
                    player_dragons = get_dragon_list(current_player_obj, "any")
                    opr_list = []
                    for dragon_id,coords in player_dragons:
                        if "once_per_round" in DRAGON_CARDS[dragon_id]:
                            # add the dragon to the list of once per round abilities
                            opr_list.append((dragon_id, coords))
                    game_state.board["round_tracker"]["opr_remaining"] = opr_list
                else:
                    opr_list = game_state.board["round_tracker"]["opr_remaining"]
                # check if we have any once per round abilities to trigger
                if len(opr_list) > 0:
                    # add the event to the event queue
                    logger.info("* Adding once per round abilities to event queue")
                    new_event = {"choice": [{"skip_opr": True}]}
                    for dragon_id, coords in opr_list:
                        new_event["choice"].append(
                            {"opr_option": {
                                "dragon_id": dragon_id,
                                "coords": coords,
                            }}
                        )
                    game_state.event_queue.append({"adv_effects": new_event})
                    break
                else:
                    # we are done with the once per round abilities
                    logger.info("* No once per round abilities left to trigger")
                    game_state.board["round_tracker"]["finished_opr"] = True
            # we are done with the once per round abilities
            logger.info("* We are done with the once per round abilities")
            # now score the round objective
            curr_round = game_state.board["round_tracker"]["round"]
            score_objective(game_state, curr_round)
            # now we try moving to the next round
            if (curr_round + 1) == game_state.ending_round:
                # we are done with the game
                game_state.phase = PHASE_END_GAME
            else:
                # move to the next round
                game_state.phase = PHASE_ROUND_START
                game_state.board["round_tracker"]["round"] += 1
                game_state.board["round_tracker"]["finished_opr"] = False
                game_state.board["round_tracker"]["opr_remaining"] = None
                game_state.current_player = game_state.round_start_player
                logger.info(f">>> Beginning round {game_state.board['round_tracker']['round'] + 1}")
                # refresh display
                for i in range(3):
                    dragon_id = game_state.board["card_display"]["dragon_cards"][i]
                    if dragon_id is not None:
                        game_state.board["card_display"]["dragon_cards"][i] = None
                        game_state.dragon_discard.append(dragon_id)
                    cave_id = game_state.board["card_display"]["cave_cards"][i]
                    if cave_id is not None:
                        game_state.board["card_display"]["cave_cards"][i] = None
                        game_state.cave_discard.append(cave_id)
                # TODO other resetting effects / Automa effects

        elif current_phase == PHASE_ROUND_START:
            logger.info("Phase: PHASE_ROUND_START")
            # we are done with the round start phase
            # so we need to check if we need to refill the display
            # refill card display from decks
            # this would create a random event
            full_display = True
            for i in range(3):
                if game_state.board["card_display"]["dragon_cards"][i] is None:
                    # random event to draw dragon for this slot
                    logger.info(">> Refilling dragon display slot %s", i)
                    new_event = {
                        "random": {
                            "refill_dragon_display": {
                            "slot": i,
                            "possible_outcomes": "dragon_deck",
                        }}
                    }
                    game_state.event_queue.append(new_event)
                    full_display = False
                if game_state.board["card_display"]["cave_cards"][i] is None:
                    # random event to draw cave for this slot
                    logger.info(">> Refilling cave display slot %s", i)
                    new_event = {
                        "random": {
                            "refill_cave_display": {
                            "slot": i,
                            "possible_outcomes": "cave_deck",
                        }}
                    }
                    game_state.event_queue.append(new_event)
                    full_display = False
            if not full_display:
                # we are not done with the refill, so we break
                break
            # display is full, check if the bonus has been given yet
            if not game_state.board["round_tracker"]["egg_given"]:
                logger.info(">> Giving 6 Coins and 1 Egg to the player.")
                # give the player 6 coins and 1 egg
                current_player_obj.coins += 6
                game_state.event_queue.append({"lay_egg": {"location": "any"}})
                game_state.board["round_tracker"]["egg_given"] = True
                break
            # we are done with the round start phase
            logger.info(">> Round start phase complete.")
            game_state.board["round_tracker"]["egg_given"] = False
            # move to the starting player
            game_state.current_player = game_state.round_start_player
            game_state.phase = PHASE_BEFORE_ACTION
            # reset player states
            game_state.player.passed_this_round = False
            game_state.automa.passed_this_round = False
            for cave_name in CAVE_NAMES:
                game_state.player.times_explored[cave_name] = 0

        elif current_phase == PHASE_SETUP:
            logger.info("Phase: PHASE_SETUP")
            new_event = {"choice": []}
            # setup phase - player must discard down to 4 total cards
            # and then choose any 3 resources
            if len(current_player_obj.dragon_hand) + len(current_player_obj.cave_hand) > 4:
                # discard dragons and caves
                logger.info(">> Player must discard down to 4 cards.")
                for dragon in current_player_obj.dragon_hand:
                    new_event["choice"].append({"discard_dragon": {"dragon": dragon}})
                for cave in current_player_obj.cave_hand:
                    new_event["choice"].append({"discard_cave": {"cave": cave}})
                game_state.event_queue.append({"adv_effects": new_event})
                break
            elif sum(current_player_obj.resources.values()) < 3:
                # choose resources
                logger.info(">> Player must choose 3 resources.")
                for resource in current_player_obj.resources.keys():
                    new_event["choice"].append({"gain_resource": {"type": resource}})
                game_state.event_queue.append({"adv_effects": new_event})
                break
            else:
                # we are done with the setup phase
                if isinstance(game_state, SoloGameState):
                    game_state.phase = PHASE_BEFORE_ACTION
                else:
                    game_state.current_player += 1 # move to the next player
                    if game_state.current_player >= len(game_state.players):
                        # we are done with the setup phase
                        game_state.current_player = game_state.round_start_player
                        game_state.phase = PHASE_BEFORE_ACTION
        
        elif current_phase == PHASE_END_GAME:
            logger.info("Phase: PHASE_END_GAME")
            # we are done with the game, so we need to score the game
            # and return the final game state
            # NOTE - assume we have a SoloGameState for now
            # score the game for the player and automa
            score_game(game_state)
            break # we are done with the game
    
    logger.debug("Game progression complete. Returning game state.")
    return game_state

def score_objective(game_state:GameState, round_number:int) -> None:
    """
    Scores the round objective for the given round in the game.
    Args:
        game_state (GameState): The current state of the game, including player and board information.
        round_number (int): The index of the current round being scored (0-based).
    Returns:
        None
    Functionality:
        - Logs the round objective and its details.
        - Calculates the player's count for the objective based on various criteria such as eggs, 
            total cards in a cave, dragon personalities, dragon sizes, dragon abilities, dragon cost, 
            cave cards played, egg capacity, and cached resources.
        - Compares the player's count with the automa's count for the objective.
        - Determines the player's and automa's positions (e.g., "1st", "2nd", "Other") based on their counts.
        - Updates the scoring for the round in the game state.
        - Adjusts the player's and automa's scores based on their positions.
    Notes:
        - Assumes a SoloGameState for now.
        - Automa scoring is partially implemented and may require further development.
    """
    logger.info(">>> Scoring round objective for round %s", (round_number + 1))
    # NOTE - assume we have a SoloGameState for now
    player = game_state.player
    obj_index, side = game_state.board["round_tracker"]["objectives"][round_number]
    objective_info = OBJECTIVE_TILES[obj_index][side]
    logger.info("\tObjective: %s", objective_info['text'])
    automa_count = objective_info['automa_values'][round_number] # TODO - implement automa scoring
    obj_item = objective_info["for_each"]

    # finding the player's count for the objective
    player_count = 0
    if obj_item["type"] == "eggs":
        player_count = sum(player.egg_totals.values())
    elif obj_item["type"] == "total_cards_in_cave":
        # we are looking for the total number of cards in one cave
        cave_name = obj_item["location"]
        for col in range(4):
            if player.dragons_played[cave_name][col] is not None:
                player_count += 1
            if (player.caves_played[cave_name][col] is not None) and (player.caves_played[cave_name][col] != -1):
                player_count += 1
    else:
        # we loop through all spaces on the mat
        for cave_name in CAVE_NAMES:
            for col in range(4):
                if obj_item["type"] in DRAGON_PERSONALITIES:
                    dragon_id = player.dragons_played[cave_name][col]
                    if (dragon_id is not None):
                        dragon_info = DRAGON_CARDS[dragon_id]
                        if dragon_info["personality"] == obj_item["type"]:
                            player_count += 1
                elif obj_item["type"] in DRAGON_SIZES:
                    dragon_id = player.dragons_played[cave_name][col]
                    if (dragon_id is not None):
                        dragon_info = DRAGON_CARDS[dragon_id]
                        if dragon_info["size"] == obj_item["type"]:
                            player_count += 1
                elif obj_item["type"] == "dragon_abilities":
                    dragon_id = player.dragons_played[cave_name][col]
                    if (dragon_id is not None):
                        dragon_info = DRAGON_CARDS[dragon_id]
                        if any(ability in dragon_info for ability in obj_item["ability_types"]):
                            player_count += 1
                elif obj_item["type"] == "dragon_cost":
                    # we are looking for the number of dragons with a cost
                    dragon_id = player.dragons_played[cave_name][col]
                    if (dragon_id is not None):
                        dragon_info = DRAGON_CARDS[dragon_id]
                        item_cost = sum(val for key, val in dragon_info if ("cost" in key))
                        if item_cost >= obj_item["min_cost"] and item_cost <= obj_item["max_cost"]:
                            player_count += 1
                elif obj_item["type"] == "cave_cards_played":
                    # we are looking for the number of cave cards played
                    if (player.caves_played[cave_name][col] is not None) and (player.caves_played[cave_name][col] != -1):
                        player_count += 1
                elif obj_item["type"] == "egg_capacity":
                    player_count += player.nested_eggs[cave_name][col][1]
                elif obj_item["type"] == "cached_resources":
                    player_count += sum(player.cached_resources[cave_name][col].values())
    # now we have the player's count for the objective
    logger.info(f">> Player count: {player_count}")
    logger.info(f">> Automa count: {automa_count}")
    # now we need to score the objective
    if player_count == 0:
        player_pos = "Other"
    elif player_count >= automa_count:
        player_pos = "1st"
    else:
        player_pos = "2nd"
    if automa_count == 0:
        automa_pos = "Other"
    elif automa_count >= player_count:
        automa_pos = "1st"
    else:
        automa_pos = "2nd"
    # update the scoring
    game_state.board["round_tracker"]["scoring"][round_number][player_pos].append(("Player", player_count))
    game_state.board["round_tracker"]["scoring"][round_number][automa_pos].append(("Automa", automa_count))
    player.score += OBJECTIVE_POSITION_SCORES[round_number][player_pos]
    game_state.automa.score += OBJECTIVE_POSITION_SCORES[round_number][automa_pos]

def score_game(game_state:GameState) -> GameState:
    """
    Score the game for the player and automa.
    """
    # TODO - implement scoring for the game
    pass

def get_random_outcome(game_state:GameState, event:dict, player:PlayerState) -> Union[int, list[int]]:
    """
    Get the random outcome for the given event.
    The event is a dictionary with the base random event.
    Depending on the event, it may contain information about where
    certain outcomes are coming from.

    Returns the outcome of the random event, which could be a
    single integer or a list of integers.
    """
    # check the event type
    if ("top_deck_reveal" in event) or ("refill_dragon_display" in event) or ("gain_dragon" in event) or ("tuck_from" in event):
        # we return one card sampled from the dragon deck
        return random.choice(game_state.dragon_deck)
    elif ("refill_cave_display" in event) or ("play_cave" in event) or ("gain_cave" in event):
        # we return one card sampled from the cave deck
        return random.choice(game_state.cave_deck)
    elif "draw_decision" in event:
        # we return a number of cards sampled from the dragon deck
        num_cards = event["draw_decision"]["amount"]
        if num_cards == "shy_this_cave":
            # we find the amount to draw
            cave_name, col = event["draw_decision"]["coords"]
            num_cards = 0
            for col in range(4):
                dragon_id = player.dragons_played[cave_name][col]
                if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == "Shy":
                    num_cards += 1
        return random.sample(game_state.dragon_deck, num_cards)
    raise ValueError(f"Invalid random event: {event}")

def get_num_random_outcomes(game_state:GameState, event:dict, player:PlayerState) -> int:
    """
    Get the number of random outcomes for the given event.
    """
    # check the event type
    if ("top_deck_reveal" in event) or ("refill_dragon_display" in event) or ("gain_dragon" in event) or ("tuck_from" in event):
        # we return one card sampled from the dragon deck
        return len(game_state.dragon_deck)
    elif ("refill_cave_display" in event) or ("play_cave" in event) or ("gain_cave" in event):
        # we return one card sampled from the cave deck
        return len(game_state.cave_deck)
    elif "draw_decision" in event:
        # we return a number of cards sampled from the dragon deck
        num_cards = event["draw_decision"]["amount"]
        if num_cards == "shy_this_cave":
            # we find the amount to draw
            cave_name, col = event["draw_decision"]["coords"]
            num_cards = 0
            for col in range(4):
                dragon_id = player.dragons_played[cave_name][col]
                if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == "Shy":
                    num_cards += 1
        return math.comb(len(game_state.dragon_deck), num_cards)
    raise ValueError(f"Invalid random event: {event}")

def apply_action(game_state:GameState, action:dict) -> GameState:
    """
    Apply the given action to the game_state in place and return the game state.
    """
    player = get_current_player(game_state) # get the current player

    # check the next action
    # we need to process advanced effects if they exist
    # otherwise we extract the action directly
    logger.debug("** Applying action: %s", action)
    if "adv_effects" in action:
        # check for other parameters and variables for this action
        if "max_uses" in action:
            adjusted_action = copy.deepcopy(action)
            max_uses = adjusted_action.pop("max_uses")
            cloned_single_action = copy.deepcopy(adjusted_action)
            if max_uses == "guild_markers":
                # special case where the amount is equal to
                # the number of guild markers placed by the player
                guild_ability_uses = game_state.board["guild"]["ability_uses"]
                max_uses = 0
                for lst in guild_ability_uses.values():
                    max_uses += lst.count(game_state.current_player)
            # now we adjust the queue with a reduced max_uses count
            # copy of this action and a single of this action after it
            if max_uses > 1:
                # add the adjusted action to the queue
                adjusted_action["max_uses"] = max_uses - 1
                game_state.event_queue.append(adjusted_action)
            if max_uses > 0:
                # add the single action to the queue
                game_state.event_queue.append(cloned_single_action)
            return game_state
        elif "condition" in action:
            # the condition must be true for the action to be applied
            if condition_is_met(game_state, action["condition"], player, action.get("coords", None)):
                # the condition is met, so we can apply the action
                adjusted_action = copy.deepcopy(action)
                adjusted_action.pop("condition")
                game_state.event_queue.append(adjusted_action)
            return game_state
        elif "cost" in action:
            # the adv_effect action has a cost, so we need to check if the player can pay it
            if len(action["cost"]) == 0:
                # no cost, so we can apply the action directly
                logger.debug("\t** No cost left for action, applying directly.")
                action_effect = action["adv_effects"]
            else:
                # check if the player can pay the cost
                logger.debug("\t** Checking if player can pay cost: %s", action["cost"])
                altered_cost = action["cost"].copy()
                altered_cost["coords"] = action.get("coords", None)
                if can_pay_cost(player, altered_cost):
                    # the player can pay the cost, so we must
                    # now find all valid payments for the cost
                    logger.debug("\t\t** Player can pay cost, checking for payments.")
                    payments_possible = get_all_payments(player, altered_cost)
                    if len(payments_possible) == 1:
                        # only one payment possible, so we pay it automatically
                        logger.debug("\t\t** Only one payment possible, paying it automatically.")
                        new_event = {
                            "make_payment": {
                                "cost": payments_possible[0],
                                "action": action,
                            }
                        }
                        game_state.event_queue.append(new_event)
                    else:
                        # add the payments to the event queue
                        logger.debug("\t\t** Multiple payments possible, adding to event queue.")
                        new_event = {"choice": []}
                        for payment in payments_possible:
                            new_event["choice"].append(
                                {
                                    "make_payment": {
                                        "cost": payment,
                                        "action": action,
                                    }
                                }
                            )
                        # add the event to the event queue
                        game_state.event_queue.append({"adv_effects": new_event})
                return game_state
        else:
            # no other effects, so extract and add the action
            action_effect = action["adv_effects"]
    else:
        action_effect = action

    # now process the raw action effect
    if "sequence" in action_effect:
        # we have a sequence of actions to apply
        # check for some special cases
        invalid = False
        for seq_action in action_effect["sequence"]:
            if "tuck_from" in seq_action and seq_action["tuck_from"]["L1"] == "hand":
                # the player must have at least 1 dragon in hand
                if len(player.dragon_hand) == 0:
                    invalid = True
                    break
            elif "cache_from" in seq_action and seq_action["cache_from"]["L1"] == "player_supply":
                # the player must have at least 1 resource in supply
                if sum(player.resources.values()) == 0:
                    invalid = True
                    break
        if not invalid:
            # add the sequence of actions to the event queue
            # in reverse order since it is like a stack
            for seq_action in reversed(action_effect["sequence"]):
                game_state.event_queue.append(seq_action)
    elif "choice" in action_effect:
        # extract the choice from the action
        choice = action_effect["choice"]
        game_state.current_choice = choice
    elif "random" in action_effect:
        # extract the random event from the action
        random_event = action_effect["random"]
        game_state.current_random_event = random_event
    else:
        # no other effects, so we can handle the action directly
        handle_simple_event(game_state, action_effect, player)

    return game_state

def get_main_action_choice(player:PlayerState) -> dict:
    """
    Get the main action choice for the player.
    The main action choice is a dictionary with the possible actions
    the player can take on their turn.

    The possible actions are:
    - Excavate: Play a cave card into 1 of the player's caves.
    - Entice: Add a dragon to the player's mat.
    - Explore: Walk the adventurer piece through one of the three caves.
    - Pass: The player passes, unable to take any more actions this round.
    """
    # get the possible actions for the player
    possible_actions = {"choice": []}
    if (player.coins == 0):
        # player must pass, since they have no coins
        return possible_actions
    # player has at least 1 coin, so they can take an action
    # check if the player can excavate somewhere
    if len(player.cave_hand) > 0:
        for cave_name in CAVE_NAMES:
            can_ex, cave_id = can_excavate_cave(player, cave_name)
            if can_ex:
                # add the action to the list of possible actions
                # we will calculate all possible plays later
                possible_actions["choice"].append({"play_cave": {"source": "hand", "free": False}})
                break
    # check if the player can entice any one dragon
    if len(player.dragon_hand) > 0:
        # check caves for enticing dragons
        break_out = False
        for cave_name in CAVE_NAMES:
            for col in range(4):
                if player.caves_played[cave_name][col] is not None:
                    # this cave is excavated, check for a dragon here too
                    if player.dragons_played[cave_name][col] is None:
                        # this cave is empty, so we can entice a dragon here
                        for dragon in player.dragon_hand:
                            # check if the player can entice this dragon
                            dragon_info = DRAGON_CARDS[dragon]
                            if not dragon_info[cave_name]:
                                # this dragon cannot be enticed here
                                continue
                            # check the cost
                            costs = get_dragon_enticement_options(player, dragon_info)
                            if len(costs) > 0:
                                # the player can entice this dragon
                                possible_actions["choice"].append({"play_dragon": {"L1": "hand", "L2": "any"}})
                                break_out = True
                                break
                        if break_out:
                            break
            if break_out:
                break
    # check if the player can explore each cave
    for cave_name in CAVE_NAMES:
        times_explored = player.times_explored[cave_name]
        if times_explored < 3:
            # the player can explore this cave this round still
            # check if the player has enough resources to explore
            total_eggs = get_total_eggs(player)
            exp_event = {"explore": {"cave_name": cave_name, "index": 0}}
            if times_explored == 0:
                # first time exploring, so we need to pay 1 coin
                if player.coins > 0:
                    possible_actions["choice"].append(
                        {"adv_effects": exp_event, "cost": {"coin": 1}}
                    )
            elif times_explored == 1:
                # second time exploring, so we need to pay 1 coin and 1 egg
                if player.coins > 0 and total_eggs > 0:
                    possible_actions["choice"].append(
                        {"adv_effects": exp_event, "cost": {"coin": 1, "egg": {"amount": 1, "location": "any"}}}
                    )
            else:
                # third time exploring, so we need to pay 1 coin and 2 eggs
                if player.coins > 0 and total_eggs > 1:
                    possible_actions["choice"].append(
                        {"adv_effects": exp_event, "cost": {"coin": 1, "egg": {"amount": 2, "location": "any"}}}
                    )

    return possible_actions

def get_all_payments(player:PlayerState, cost_dict:dict) -> list[dict]:
    """
    Get all possible payments for the given cost dictionary.
    The cost is a dictionary with the resource names as keys and the amounts as values.

    Only one cost is handled at a time, and we assume that
    the player can pay the full cost in cost_dict.

    The cost is a dictionary with the resource names as keys and the amounts as values.
    The function returns a list of dictionaries, each with a valid payment.
    """
    logger.debug("- Getting all payments for cost: %s", cost_dict)
    # check each cost in the cost dictionary
    payments = []
    for name, value in cost_dict.items():
        if name == "choice":
            # check possible costs to pay
            # value is a list of cost dictionaries
            # assume this is the first item in cost_dict
            for payment in value:
                payments.extend(get_all_payments(player, payment))
        elif name == "coin":
            # add the one coin payment
            payments.append({ "coin": value })
        elif name == "dragon_card":
            # the player must pay with some number of dragon cards
            # value is the number of cards to pay
            for card_comb in itertools.combinations(player.dragon_hand, value):
                payments.append({ "dragon_card": card_comb })
        elif name == "cave_card":
            # the player must pay with some number of cave cards
            # value is the number of cards to pay
            for card_comb in itertools.combinations(player.cave_hand, value):
                payments.append({ "cave_card": card_comb })
        elif name == "any_resource":
            # get all resource payments for the cost
            combinations = []
            get_resource_combinations_helper(
                value,
                [player.resources[res] for res in RESOURCES],
                0,
                [0,0,0,0],
                combinations
            )
            # add the combinations to the payments
            for comb in combinations:
                # convert the list of resources to a dictionary
                payment = {}
                for i, res in enumerate(RESOURCES):
                    if comb[i] > 0:
                        payment[res] = comb[i]
                payments.append(payment)
        elif name in RESOURCES:
            # check if the player has enough resources for the cost
            # a player can exchange 2 of any resource for 1 of any type
            # we assume the player can afford the cost and any exchanges
            logger.debug("Checking resource cost: %s", name)
            remaining_resource_cost = {}
            for res in RESOURCES:
                if cost_dict.get(res, 0) > 0:
                    remaining_resource_cost[res] = cost_dict[res]
            logger.debug(f"\tRemaining resource cost: {remaining_resource_cost}")
            # try paying the exact cost first
            const_payment = {res: 0 for res in remaining_resource_cost}
            player_resources = player.resources.copy()
            for res in remaining_resource_cost:
                if player_resources[res] >= remaining_resource_cost[res]:
                    const_payment[res] = remaining_resource_cost[res]
                    player_resources[res] -= remaining_resource_cost[res]
                    remaining_resource_cost[res] = 0
                else:
                    const_payment[res] = player_resources[res]
                    player_resources[res] = 0
                    remaining_resource_cost[res] -= const_payment[res]
            logger.debug(f"\tConst payment: {const_payment}")
            logger.debug(f"\tRemaining resource cost after payment: {remaining_resource_cost}")
            # check if we have paid the cost
            if all(v == 0 for v in remaining_resource_cost.values()):
                # we have paid the cost, no exchanges needed
                payments.append(const_payment)
            else:
                # now check for valid exchanges
                # get all resource payments for the cost
                exchange_cost = sum(remaining_resource_cost.values()) * 2
                res_counts = [player_resources[res] for res in RESOURCES]
                logger.debug(f"\t\tExchange cost: {exchange_cost}")
                logger.debug(f"\t\tResource counts: {res_counts}")
                combinations = []
                get_resource_combinations_helper(
                    exchange_cost,
                    res_counts,
                    0,
                    [0,0,0,0],
                    combinations
                )
                logger.debug(f"\t\tCombinations: {combinations}")
                # add the combinations to the payments
                for comb in combinations:
                    # convert the list of resources to a dictionary
                    payment = {}
                    for i, res in enumerate(RESOURCES):
                        total = comb[i] + const_payment.get(res, 0)
                        if total > 0:
                            payment[res] = total
                    payments.append(payment)
        elif name == "egg":
            amount = value["amount"]
            target_location = value["location"]
            coords = cost_dict["coords"] # assume there is a coords key

            egg_locations = []
            egg_counts = []
            if target_location == "any":
                # check eggs on mat
                if player.egg_totals["mat_slots"] > 0:
                    egg_locations.append(("mat_slots", 0))
                    egg_counts.append(player.egg_totals["mat_slots"])
                # now check the caves
                for cave_name in CAVE_NAMES:
                    # check the eggs in the specified cave
                    cave_eggs = player.nested_eggs[cave_name]
                    for col in range(4):
                        if cave_eggs[col][0] > 0:
                            egg_locations.append((cave_name, col))
                            egg_counts.append(cave_eggs[col][0])
            elif target_location == "this_cave":
                # check the eggs in the specified cave
                cave_name, col = coords
                cave_eggs = player.nested_eggs[cave_name]
                for col in range(4):
                    if cave_eggs[col][0] > 0:
                        egg_locations.append((cave_name, col))
                        egg_counts.append(cave_eggs[col][0])
            elif target_location == "ortho":
                # check orthogonally adjacent dragons
                main_cave, main_col = coords
                mci = CAVE_NAMES.index(main_cave)
                # check the 4 orthogonal directions
                # up
                if mci > 0 and (player.nested_eggs[CAVE_NAMES[mci-1]][main_col][0] > 0):
                    egg_locations.append((CAVE_NAMES[mci-1], main_col))
                    egg_counts.append(player.nested_eggs[CAVE_NAMES[mci-1]][main_col][0])
                # down
                if mci < 2 and (player.nested_eggs[CAVE_NAMES[mci+1]][main_col][0] > 0):
                    egg_locations.append((CAVE_NAMES[mci+1], main_col))
                    egg_counts.append(player.nested_eggs[CAVE_NAMES[mci+1]][main_col][0])
                # left
                if main_col > 0 and (player.nested_eggs[main_cave][main_col-1][0] > 0):
                    egg_locations.append((main_cave, main_col-1))
                    egg_counts.append(player.nested_eggs[main_cave][main_col-1][0])
                # right
                if main_col < 3 and (player.nested_eggs[main_cave][main_col+1][0] > 0):
                    egg_locations.append((main_cave, main_col+1))
                    egg_counts.append(player.nested_eggs[main_cave][main_col+1][0])
            # find all valid payments for the egg cost
            combinations = []
            get_egg_payments_helper(
                amount,
                egg_locations,
                egg_counts,
                0,
                [],
                combinations
            )
            # add the combinations to the payments
            for comb in combinations:
                # store each tuple of locations in a dictionary
                payments.append({"egg": comb})
        # we will only handle one payment here
        break
    
    return payments

def get_egg_payments_helper(
        egg_cost_left:int, 
        egg_locations:list[tuple],
        egg_counts:list[int],
        location_index:int,
        current_locations:list[tuple],
        master_list:list[tuple]
        ) -> None:
    """
    Recursive helper function to get all possible egg payments for the given cost.
    
    player is the player who is paying the cost.
    egg_cost_left is the number of eggs left to pay.
    egg_locations is a list of tuples (cave:str, col:int) that are valid locations to pay eggs.
    egg_counts is a list of integers that are the number of eggs in each location.
    location_index is the index of the location to check.
    current_locations is a list of tuples (cave:str, col:int) that are already added.
    master_list is a list of tuples

    This adds valid payments to the master_list in place and returns None.
    """
    if egg_cost_left == 0:
        # we have a valid payment, so add it to the master list
        master_list.append(tuple(current_locations))
        return
    if location_index >= len(egg_locations):
        # we have checked all locations, so return
        return
    # check the current location
    cave_name, col = egg_locations[location_index]
    # check if the player has enough eggs in this location
    if egg_counts[location_index] > 0:
        # we can pay an egg from this location
        new_locations = current_locations.copy()
        new_locations.append((cave_name, col))
        new_egg_counts = egg_counts.copy()
        new_egg_counts[location_index] -= 1
        get_egg_payments_helper(
            egg_cost_left - 1,
            egg_locations,
            new_egg_counts,
            location_index,
            new_locations,
            master_list
        )
    # skip the current location
    get_egg_payments_helper(
        egg_cost_left,
        egg_locations,
        egg_counts,
        location_index + 1,
        current_locations,
        master_list
    )

def get_resource_combinations_helper(
        res_cost_left:int, 
        res_counts:list[int],
        res_index:int,
        current_resources:list,
        master_list:list[tuple]
        ) -> None:
    """
    When a player pays a cost with resources, they can pay 2 of any resource
    for 1 of any resource. This function finds all possible exchanges a player
    can perform and adds them to the master list.
    We assume that the player has already paid any resources that they
    can afford exactly, so we only need to check for exchanges (twice the remaining cost).
    
    res_cost_left is the number of resources left to exchange.
    res_counts is a list of integers that are the number of resources of each type.
    res_index is the index of the resource to check.
    current_resources is a list of the amounts of each resource we have counted.
        (assume current_resources starts as list of 0s of the same length as res_counts)
    master_list is a list of integers that are the resources that have been used.
    """
    if res_cost_left == 0:
        # we have a valid payment, so add it to the master list
        master_list.append(tuple(current_resources))
        return
    if res_index >= len(res_counts):
        # we have checked all resources, so return
        return
    # check the current resource
    if res_counts[res_index] > 0:
        # we can pay with this resource
        new_resources = current_resources.copy()
        new_resources[res_index] += 1
        new_res_counts = res_counts.copy()
        new_res_counts[res_index] -= 1
        get_resource_combinations_helper(
            res_cost_left - 1,
            new_res_counts,
            res_index,
            new_resources,
            master_list
        )
    # skip the current resource
    get_resource_combinations_helper(
        res_cost_left,
        res_counts,
        res_index + 1,
        current_resources,
        master_list
    )

def condition_is_met(game_state:GameState, condition:dict, player:PlayerState, coords:tuple) -> bool:
    """
    Check if the given condition is met for the player.
    The condition is a dictionary with the condition name and the parameters.
    The player is the player who triggered the event, if applicable.
    Assumes the condition is valid and can be checked.

    The conditions are basic game conditions that are not tied to a specific phase
    or action. They are used to check if a specific condition is met for the game state.
    The input coords is a tuple (cave:str, col:int) that specifies the location of the condition.
    """
    # check cases
    if "min_dragons_in_cave" in condition:
        # check if the player has at least the specified number of dragons in the cave
        cave_name = condition["min_dragons_in_cave"]["cave"]
        if cave_name == "this_cave":
            cave_name, col = coords
        include_text = condition["min_dragons_in_cave"].get("include", "any")
        count = 0
        for dragon_id in player.dragons_played[cave_name]:
            if dragon_id is not None:
                if include_text == "any":
                    count += 1
                    continue
                dragon = DRAGON_CARDS[dragon_id]
                if dragon["personality"] == include_text or dragon["size"] == include_text:
                    # check if the dragon is the same as the one in the condition
                    count += 1
        return (count >= condition["min_dragons_in_cave"]["amount"])
    elif "max_dragons_in_cave" in condition:
        # check if the player has at most the specified number of dragons in the cave
        cave_name = condition["max_dragons_in_cave"]["cave"]
        if cave_name == "this_cave":
            cave_name, col = coords
        include_text = condition["max_dragons_in_cave"].get("include", "any")
        count = 0
        for dragon_id in player.dragons_played[cave_name]:
            if dragon_id is not None:
                if include_text == "any":
                    count += 1
                    continue
                dragon = DRAGON_CARDS[dragon_id]
                if dragon["personality"] == include_text or dragon["size"] == include_text:
                    # check if the dragon is the same as the one in the condition
                    count += 1
        return (count <= condition["max_dragons_in_cave"]["amount"])
    elif "min_dragons_this_column" in condition:
        # check if the player has at least the specified number of dragons in the column
        this_cave, col = coords
        count = 0
        include_text = condition["min_dragons_this_column"].get("include", "any")
        for cave_name in CAVE_NAMES:
            dragon_id = player.dragons_played[cave_name][col]
            if dragon_id is not None:
                if include_text == "any":
                    count += 1
                    continue
                dragon = DRAGON_CARDS[dragon_id]
                if dragon["personality"] == include_text or dragon["size"] == include_text:
                    # check if the dragon is the same as the one in the condition
                    count += 1
        return (count >= condition["min_dragons_this_column"]["amount"])
    elif "min_spaces_excavated" in condition:
        # check if the player has at least the specified number of spaces excavated
        cave_name = condition["min_spaces_excavated"]["cave"]
        if cave_name == "this_cave":
            cave_name, col = coords
        count = sum(1 for i in player.caves_played[cave_name] if i is not None)
        return (count >= condition["min_spaces_excavated"]["amount"])
    elif "this_position" in condition:
        # check if the column is the same as the one in the condition
        return (coords[1] == condition["this_position"])
    elif "this_col_full" in condition:
        # check if the column is full
        this_cave, col = coords
        count = 0
        for cave_name in CAVE_NAMES:
            if player.dragons_played[cave_name][col] is not None:
                count += 1
        return (count == 3)
    elif "min_guild_markers" in condition:
        player_num = game_state.current_player
        guild_markers = game_state.board["guild"]["ability_uses"]
        num_marks = 0
        for lst in guild_markers.values():
            num_marks += lst.count(player_num)
            if num_marks >= condition["min_guild_markers"]:
                return True
        return False

    elif "and" in condition:
        return all(
            condition_is_met(game_state, cond, player, coords)
            for cond in condition["and"]
        )
    elif "or" in condition:
        return any(
            condition_is_met(game_state, cond, player, coords)
            for cond in condition["or"]
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

def handle_simple_event(game_state:GameState, event:dict, player:PlayerState=None) -> None:
    """
    NOTE: The events handled here are ones without the "adv_effects" key.
    
    Handle the given event in the game state with the target player.
    The event is a dictionary with the event name and the parameters.
    The player is the player who triggered the event, if applicable.
    Assumes the event is valid and can be handled.

    Events are basic game events that are not tied to a specific phase
    or action. They are used to modify the game state in a generic way.
    For example, discarding a dragon or cave, or gaining resources.
    Adding choices and things to the event queue could also happen here.
    The event is handled irrespective of the game phase.

    The states input are modified in place, so no return value is needed.
    """
    if "gain_resource" in event:
        # gain resources
        if event["gain_resource"]["type"] == "any":
            # CHOICE - choose to gain 1 of any resource
            # add the choice to the event queue
            new_event = {
                "choice": [{"gain_resource": {"type": resource}} for resource in RESOURCES]
            }
            game_state.event_queue.append({"adv_effects": new_event})
        else:
            resource = event["gain_resource"]["type"]
            player.resources[resource] += 1
            logger.info(f"> Player gains 1 {resource}")
    elif "make_payment" in event:
        # we make the payment for the one cost given
        cost = event["make_payment"]["cost"]
        new_action = event["make_payment"]["action"].copy()
        if any((k in RESOURCES) for k in cost):
            # pay any resource costs in the dictionary
            deduct_resources(player, cost)
            cost_name = "any_resource"
        elif "egg" in cost:
            for location in cost["egg"]:
                # pay each egg from the location
                pay_egg(player, location)
            cost_name = "egg"
        elif "dragon_card" in cost:
            # pay each dragon card from the hand
            for card in cost["dragon_card"]:
                discard_dragon(player, game_state, card)
            cost_name = "dragon_card"
        elif "cave_card" in cost:
            # pay each cave card from the hand
            for card in cost["cave_card"]:
                discard_cave(player, game_state, card)
            cost_name = "cave_card"
        elif "coin" in cost:
            # pay a coin
            player.coins -= cost["coin"]
            logger.info(f"> Player pays {cost['coin']} coin(s)")
            cost_name = "coin"
        # remove the cost from the action
        if cost_name in cost:
            new_action["cost"].pop(cost_name)
        elif cost_name == "any_resource":
            # remove any resource costs from the action
            for res in RESOURCES:
                if res in new_action["cost"]:
                    new_action["cost"].pop(res)
        elif "choice" in cost:
            # remove the choice from the action
            # NOTE - we assume that we are paying for an item in the choice
            new_action["cost"].pop("choice")
        # add the action to the event queue
        game_state.event_queue.append(new_action)
    elif "lay_egg" in event:
        # lay an egg
        handle_egg_laying(game_state, event, player)
    elif "gain_guild" in event:
        # NOTE - assume we have a SoloGameState for now
        pos = game_state.board["guild"]["player_position"]
        new_pos = (pos + 1) % 12
        game_state.board["guild"]["player_position"] = new_pos
        game_state.event_queue.append(GUILD_SPACE_EFFECTS[new_pos])
        logger.info(f"> Player moves their guild marker from position {pos} to {new_pos}")
    elif "refill_dragon_display" in event:
        # refill the dragon display
        dragon_id = event["refill_dragon_display"]["rand_outcome"]
        slot = event["refill_dragon_display"]["slot"]
        game_state.board["card_display"]["dragon_cards"][slot] = dragon_id
        game_state.dragon_deck.remove(dragon_id)
        logger.info(f">> Dragon display slot {slot} filled with dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']})")
        if len(game_state.dragon_deck) == 0:
            # we need to refill the dragon deck
            refresh_dragon_deck(game_state)
    elif "refill_cave_display" in event:
        # refill the cave display
        cave_id = event["refill_cave_display"]["rand_outcome"]
        slot = event["refill_cave_display"]["slot"]
        game_state.board["card_display"]["cave_cards"][slot] = cave_id
        game_state.cave_deck.remove(cave_id)
        logger.info(f">> Cave display slot {slot} filled with cave {cave_id} ({CAVE_CARDS[cave_id]['text']})")
        if len(game_state.cave_deck) == 0:
            # we need to refill the cave deck
            refresh_cave_deck(game_state)
    elif "cache_from" in event:
        # the player caches a resource, but we need
        # to check where this can happen
        handle_resource_caching(game_state, event, player)
    elif "tuck_from" in event:
        # tuck a dragon
        handle_tuck_dragon(game_state, event, player)
    elif "gain_cave" in event:
        # gain a cave card
        handle_gain_cave_card(game_state, event, player)
    elif "gain_dragon" in event:
        # gain a dragon card
        handle_gain_dragon_card(game_state, event, player)
    elif "gain_coin" in event:
        player.coins += event["gain_coin"]["amount"]
        logger.info(f"> Player gains {event['gain_coin']['amount']} coin(s)")
    elif "explore" in event:
        # the player has chosen to explore a cave
        handle_explore(game_state, event, player)
    elif "play_cave" in event:
        # playing a cave card from somewhere to the player's mat
        handle_play_cave(game_state, event, player)
    elif "play_dragon" in event:
        # playing a dragon card from somewhere to the player's mat
        handle_play_dragon(game_state, event, player)
    elif "end_game" in event:
        # TODO - effects that happen at the end of the game
        # this is a placeholder for now
        logger.info(f">> End of game effect triggered")
    elif "skip" in event:
        # the player chooses not to activate an
        # ability when given a choice
        logger.info(f">> Player chooses to skip the ability")
        pass
    elif "pass" in event:
        # the player chooses to pass for the round
        logger.info(f">>> Player chooses to pass for the round")
        player.passed_this_round = True
        game_state.phase = PHASE_END_TURN
    elif "brown_space" in event:
        # the player has landed on a brown space
        # and must choose a guild bonus
        handle_brown_space(game_state, event, player)
    elif "4th_space" in event:
        # the player has excavated the 4th cave space
        # and must choose whether to exchange for a coin or not
        handle_4th_space(game_state, player)
    elif "opr_option" in event:
        # the player has a choice to trigger a dragon's
        # Once Per Round ability
        handle_opr_option(game_state, event, player)
    elif "skip_opr" in event:
        # the player skips playing any more OPR abilities
        logger.info(f">> Player chooses to skip the remaining OPR abilities")
        game_state.board["round_tracker"]["finished_opr"] = True
    elif "top_deck_reveal" in event:
        # we reveal the top card and do something based on it
        handle_top_deck_reveal(game_state, event, player)
    elif "draw_decision" in event:
        # the player draws some number of cards
        # and chooses what to do with them
        handle_draw_decision(game_state, event, player)
    elif "other_ability_on_mat" in event:
        # the player chooses to activate another ability
        # of a dragon on their mat
        handle_other_ability_on_mat(game_state, event, player)
    elif "any_resource_decision" in event:
        # the player chooses some resources
        # and chooses what to do with them
        handle_any_resource_decision(game_state, event, player)
    elif "deduct_resources" in event:
        # deduct resources
        cost_dict = event["deduct_resources"]["cost"]
        deduct_resources(player, cost_dict)
    elif "discard_dragon" in event:
        # discard dragon
        dragon = event["discard_dragon"]["dragon"]
        discard_dragon(player, game_state, dragon)
    elif "discard_cave" in event:
        # discard cave
        cave = event["discard_cave"]["cave"]
        discard_cave(player, game_state, cave)
    elif "swap_dragons" in event:
        # player chooses two dragons on their mat to swap
        handle_swap_dragons(game_state, event, player)
    else:
        raise ValueError(f"Unknown event: {event}")

def handle_4th_space(game_state:GameState, player:PlayerState) -> None:
    """
    Handle the event where the player has excavated the 4th cave space
    and must choose whether to exchange for a coin or not.

    A player can choose to exchange any 3 resources, dragon cards,
    or cave cards in their supply for 1 coin.
    """
    # check what combinations the player can exchange
    if len(player.dragon_hand) + len(player.cave_hand) + sum(player.resources.values()) < 3:
        # the player cannot exchange anything, so nothing happens
        logger.info(f">> Player cannot afford a 4th space exchange.")
        return
    payment_amounts = [(3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(1,0,2),(0,2,1),(0,1,2),(1,1,1)]
    new_event = {"choice": [{"skip": True}]}
    gain_event = {"adv_effects": {"gain_coin": {"amount": 1}}, "cost": {"choice": []}}
    for n_resources, n_dragons, n_caves in payment_amounts:
        # check if the player can pay this cost
        if (n_resources <= sum(player.resources.values()) and
            n_dragons <= len(player.dragon_hand) and
            n_caves <= len(player.cave_hand)):
            # add the payment to the list of choices
            cost = {}
            if n_resources > 0:
                cost["any_resource"] = n_resources
            if n_dragons > 0:
                cost["dragon_card"] = n_dragons
            if n_caves > 0:
                cost["cave_card"] = n_caves
            gain_event["cost"]["choice"].append(cost)
    assert len(gain_event["cost"]["choice"]) > 0
    # combine the events
    new_event["choice"].append(gain_event)
    game_state.event_queue.append({"adv_effects": new_event})

def handle_opr_option(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the event where the player has a choice to trigger a dragon's
    Once Per Round ability. We have to keep track of which once per round
    abilities have been used, so we manage that here.
    """
    # get the dragon id and the ability
    dragon_id = event["opr_option"]["dragon_id"]
    dragon_coords = event["opr_option"]["coords"]
    # add coords to the ability
    opr_ability = copy.deepcopy(DRAGON_CARDS[dragon_id]["once_per_round"])
    opr_ability["coords"] = dragon_coords
    # remove this dragon from the list of available dragons
    game_state.board["round_tracker"]["opr_remaining"].remove((dragon_id, dragon_coords))
    game_state.event_queue.append(opr_ability)
    logger.info(f">> Player chooses to activate the OPR ability of dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) at {dragon_coords}")

def handle_explore(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the event where the player explores a cave.
    The event dict has the cave name and the index of the exploration.
    The player is the player who triggered the event.
    """
    event = full_event["explore"]
    explore_index = event["index"]
    cave_name = event["cave_name"]
    if explore_index == 0:
        # increment number of times explored
        player.times_explored[cave_name] += 1
        logger.info(f">> Player begins exploration number {player.times_explored[cave_name]} in their {cave_name}")
    # add correct event to the event queue
    if explore_index % 2 == 0:
        # event printed on the mat itself
        if explore_index != 8:
            # setup for next exploration event
            game_state.event_queue.append(
                {"explore": {
                    "cave_name": cave_name,
                    "index": explore_index + 1,
                }}
            )
        event_index = explore_index // 2
        game_state.event_queue.append(EXPLORE_CAVE_EFFECTS[cave_name][event_index])
        logger.info(f">> Player activates {cave_name} effect at index {explore_index}")
    else:
        # event printed on the dragon card (if one is present)
        event_index = (explore_index - 1) // 2
        dragon_id = player.dragons_played[cave_name][event_index]
        if dragon_id is not None:
            # setup for next exploration event
            game_state.event_queue.append(
                {"explore": {
                    "cave_name": cave_name,
                    "index": explore_index + 1,
                }}
            )
            # we have a dragon to activate
            dragon_card = DRAGON_CARDS[dragon_id]
            if "if_activated" in dragon_card:
                # add the event to the event queue
                event_to_add = copy.deepcopy(dragon_card["if_activated"])
                event_to_add["coords"] = (cave_name, event_index)
                game_state.event_queue.append(event_to_add)
                logger.info(f">> Player activates {dragon_card['name']} effect at index {explore_index}")

def handle_brown_space(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the event where the player has landed on a brown space
    on the guild track and must choose a guild bonus.
    """
    event_num = full_event["brown_space"]
    if event_num > 0:
        # we apply the chosen bonus
        player.guild_markers -= 1
        game_state.board["guild"]["ability_uses"][event_num].append(game_state.current_player)
        logger.info(f">> Player chooses guild bonus {event_num}")
        logger.info(f"- Current guild info: {game_state.board['guild']}")
        # add the event to the event queue
        if event_num != 5:
            guild_ability = GUILD_TILES[game_state.board["guild"]["guild_index"]]["abilities"][event_num-1]
            # add the event to the event queue
            game_state.event_queue.append(guild_ability["effect"])
        else:
            # the player immediately gains points
            num_uses = len(game_state.board["guild"]["ability_uses"][event_num])
            num_pts = 6 if (num_uses == 1) else (3 if (num_uses == 2) else 1)
            logger.info(f">> Player gains {num_pts} points")
            player.points += num_pts
        return
    # guild bonus not chosen yet
    logger.info(f"- Current guild info: {game_state.board['guild']}")
    if player.guild_markers == 0:
        # the player has no more guild markers, so nothing happens
        logger.info(f">> Player has no guild markers left, so nothing happens")
        return
    # check which bonuses are available
    # NOTE - assume this is a solo game for now
    new_event = {"choice": []}
    guild_info = GUILD_TILES[game_state.board["guild"]["guild_index"]]
    guild_ability_uses = game_state.board["guild"]["ability_uses"]
    for i, lst in guild_ability_uses.items():
        if i == 5:
            # special case for last ability
            ability_max_uses = 100
        else:
            ability_max_uses = guild_info["abilities"][i-1]["uses"]
        if len(lst) < ability_max_uses:
            # the player can use this ability
            new_event["choice"].append({"brown_space": i})

    if len(new_event["choice"]) > 1:
        # add the event to the event queue
        game_state.event_queue.append({"adv_effects": new_event})
    elif len(new_event["choice"]) == 1:
        # the player has only one choice, so we can add it directly
        game_state.event_queue.append(new_event["choice"][0])

def handle_swap_dragons(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the rare event for swapping two dragons on the player's mat.
    """
    event = full_event["swap_dragons"]
    # check if we have chosen dragons to swap
    if "coords1" in event:
        # we have chosen two dragons to swap
        coords1 = event["coords1"]
        coords2 = event["coords2"]
        dragon1 = player.dragons_played[coords1[0]][coords1[1]]
        dragon2 = player.dragons_played[coords2[0]][coords2[1]]
        # swap the dragons
        player.dragons_played[coords1[0]][coords1[1]] = dragon2
        player.dragons_played[coords2[0]][coords2[1]] = dragon1
        # all related items are also swapped
        item1,item2 = player.hatchling_grown[coords1[0]][coords1[1]], player.hatchling_grown[coords2[0]][coords2[1]]
        player.hatchling_grown[coords1[0]][coords1[1]] = item2
        player.hatchling_grown[coords2[0]][coords2[1]] = item1
        item1,item2 = player.cached_resources[coords1[0]][coords1[1]], player.cached_resources[coords2[0]][coords2[1]]
        player.cached_resources[coords1[0]][coords1[1]] = item2
        player.cached_resources[coords2[0]][coords2[1]] = item1
        item1,item2 = player.tucked_dragons[coords1[0]][coords1[1]], player.tucked_dragons[coords2[0]][coords2[1]]
        player.tucked_dragons[coords1[0]][coords1[1]] = item2
        player.tucked_dragons[coords2[0]][coords2[1]] = item1
        item1,item2 = player.nested_eggs[coords1[0]][coords1[1]], player.nested_eggs[coords2[0]][coords2[1]]
        player.nested_eggs[coords1[0]][coords1[1]] = item2
        player.nested_eggs[coords2[0]][coords2[1]] = item1
        logger.info(f">> Player swaps dragons {dragon1} and {dragon2} at {coords1} and {coords2}")
        return
    # else we need to create a choice event
    all_dragons = get_dragon_list(player, "any")
    new_event = {"choice": [{"skip": True}]}
    for i,(dragon_id1, coords1) in enumerate(all_dragons):
        for j in range(i+1, len(all_dragons)):
            dragon_id2, coords2 = all_dragons[j]
            # add the dragons to the list of choices
            new_event["choice"].append(
                {"swap_dragons": {
                    "coords1": coords1,
                    "coords2": coords2,
                }}
            )
    if len(new_event["choice"]) > 1:
        # add the event to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_other_ability_on_mat(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the event where the player chooses to activate another ability
    of a dragon on their mat. This is a special case where the player can
    choose to activate an ability of a dragon that is not the one they are
    currently playing. The event is a dictionary with the event name and
    the parameters.
    """
    event = full_event["other_ability_on_mat"]
    if "coords" in event:
        # we have chosen a specific dragon to activate
        dragon_id = player.dragons_played[event["coords"][0]][event["coords"][1]]
        event_ability = copy.deepcopy(DRAGON_CARDS[dragon_id][event["type"]])
        # add the coordinates to the event
        event_ability["coords"] = event["coords"]
        # add the event to the event queue
        game_state.event_queue.append(event_ability)
        logger.info(f">> Player chooses to activate the {event['type']} ability of dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) at {event['coords']}")
        return
    # else we need to create a choice event
    all_blacklists = {
        "if_activated": [73],
        "when_played": [76],
    }
    # get the list of dragons that can be activated
    all_dragons = get_dragon_list(player, "any")
    new_event = {"choice": []}
    this_blacklist = all_blacklists.get(event["type"], [])
    for dragon_id, coords in all_dragons:
        # check dragon validity
        dragon = DRAGON_CARDS[dragon_id]
        if event["type"] in dragon and dragon_id not in this_blacklist:
            # add the dragon to the list of choices
            new_event["choice"].append(
                {"other_ability_on_mat": {
                    "coords": coords,
                    "type": event["type"],
                }}
            )
    if len(new_event["choice"]) > 0:
        # add the event to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_top_deck_reveal(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handles the event where the top card of the deck is revealed and
    then something is done with it. This assumes that the randomly
    drawn card is already chosen in the event.
    """
    event = full_event["top_deck_reveal"]
    dragon_id = event["rand_outcome"]
    dragon_card = DRAGON_CARDS[dragon_id]
    game_state.dragon_deck.remove(dragon_id) # remove the dragon from the deck
    logger.info(f">> Player reveals dragon {dragon_id} ({dragon_card['name']}) from the top of the deck")
    if dragon_card["personality"] in event["tuck_targets"]:
        # we tuck the card at the specified location
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        # tuck the dragon at the specified location
        logger.info(">> The dragon is tucked!")
        tuck_dragon(player, game_state, dragon_id, full_event["coords"])
        return
    # else we do the specified effect of the dragon card
    fail_effect = event["fail_effect"]
    if fail_effect == "keep_card":
        # add card to player's hand
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.dragon_hand.append(dragon_id) # add the dragon to the player's hand
        logger.info(">> The dragon is added to the player's hand!")
    elif fail_effect == "gain_meat":
        # discard the dragon and gain 1 meat
        game_state.dragon_discard.append(dragon_id) # add the dragon to the discard pile
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.resources["meat"] += 1 # add 1 meat to the player's resources
        logger.info(">> The dragon is discarded and the player gains 1 meat!")
    elif fail_effect == "gain_milk":
        # discard the dragon and gain 1 milk
        game_state.dragon_discard.append(dragon_id) # add the dragon to the discard pile
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.resources["milk"] += 1 # add 1 milk to the player's resources
        logger.info(">> The dragon is discarded and the player gains 1 milk!")

def handle_draw_decision(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the "draw decision" event, where the player must draw some number of cards
    from the deck and then choose what to do with them. This could involve keeping them,
    tucking them under dragons, among other things.
    
    This will be a multi-stage event where we check for the decision to be made
    and then create the decision for the next choice, if needed.

    The possible choices will be handled in a certain order to reduce
    the number of combinations of choices possible for the player, simplifying
    the game tree. The order is:
    - tuck_any -> tuck_here -> discard -> keep
    """
    event = full_event["draw_decision"]
    ordered_choices = ["tuck_any", "tuck_here", "discard", "keep"]
    # first handle any chosen action from the event
    # NOTE - we assume the dragons have already been removed from the deck
    # for safety involving the dragon deck running out
    logger.info(">> Handling draw decision event...")
    if "chosen_id" in event:
        dragon_id = event["chosen_id"]
        new_limits = event["limits"].copy()
        # remove from list of dragons to choose from
        event["remaining_dragons"].remove(dragon_id)
        # find what to do with the dragon based on the order
        if "tuck_any" in event["limits"]:
            # tuck the dragon at the specified location
            tuck_dragon(player, game_state, dragon_id, full_event["coords"])
            if new_limits["tuck_any"] > 1:
                new_limits["tuck_any"] -= 1
            else:
                new_limits.pop("tuck_any")
        elif "tuck_here" in event["limits"]:
            # tuck the dragon at the specified location
            tuck_dragon(player, game_state, dragon_id, full_event["coords"])
            if new_limits["tuck_here"] > 1:
                new_limits["tuck_here"] -= 1
            else:
                new_limits.pop("tuck_here")
        elif "discard" in event["limits"]:
            # discard the dragon
            logger.info(f"> The player discards dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']})")
            game_state.dragon_discard.append(dragon_id)
            if new_limits["discard"] > 1:
                new_limits["discard"] -= 1
            else:
                new_limits.pop("discard")
        elif "keep" in event["limits"]:
            # add the dragon to the player's hand
            logger.info(f"> The player keeps dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) in their hand")
            player.dragon_hand.append(dragon_id)
            if new_limits["keep"] > 1:
                new_limits["keep"] -= 1
            else:
                new_limits.pop("keep")
    elif "skip_choice" in event:
        # we simply remove the current choice from the list of choices
        # and move on to the next choice
        for choice in ordered_choices:
            if choice in event["limits"]:
                new_limits = event["limits"].copy()
                new_limits.pop(choice)
                logger.info(f">> Player skips the choice: {choice}")
                break
    # check if we have any more dragons to choose from
    if len(event["remaining_dragons"]) == 0:
        # we are done with the event, so we can remove it from the queue
        return
    # we have more dragons to choose from, so we need to create the next event
    new_event = {"choice": []}
    for i,choice in enumerate(ordered_choices):
        if choice in new_limits:
            # we have a choice to make, so we need to create the event
            for dragon_id in event["remaining_dragons"]:
                if choice == "tuck_any":
                    # loop through all possible locations to tuck the dragon
                    all_dragons = get_dragon_list(player, "any")
                    for dragon, coords in all_dragons:
                        new_event["choice"].append(
                            {"draw_decision": {
                                "chosen_id": dragon_id,
                                "limits": new_limits,
                                "remaining_dragons": event["remaining_dragons"],
                                },
                                "coords": coords
                            }
                        )
                else:
                    # we have a specific location to tuck the dragon
                    new_event["choice"].append(
                        {"draw_decision": {
                            "chosen_id": dragon_id,
                            "limits": new_limits,
                            "remaining_dragons": event["remaining_dragons"],
                            },
                            "coords": full_event["coords"]
                        }
                    )
            # check if we can skip the current choice
            future_choice_amounts = 0
            for j in range(i+1, len(ordered_choices)):
                if ordered_choices[j] in new_limits:
                    future_choice_amounts += new_limits[ordered_choices[j]]
            if future_choice_amounts >= len(event["remaining_dragons"]):
                # add the option to skip the current choice
                new_event["choice"].append(
                    {"draw_decision": {
                        "skip_choice": True,
                        "limits": new_limits,
                        "remaining_dragons": event["remaining_dragons"],
                        },
                        "coords": full_event["coords"]
                    }
                )

    # add the new event to the event queue
    if len(new_event["choice"]) > 1:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})
    elif len(new_event["choice"]) == 1:
        # we have a single choice to make, so we can resolve it immediately
        # and remove the event from the queue
        game_state.event_queue.append(new_event["choice"][0])

def handle_any_resource_decision(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the "any resource decision" event, where the player must choose some number of
    resources and then choose what to do with them. This could involve keeping them or
    caching them onto different dragons.
    
    This will be a multi-stage event where we check for the decision to be made
    and then create the decision for the next choice, if needed.

    The possible choices will be handled in a certain order to reduce
    the number of combinations of choices possible for the player, simplifying
    the game tree. The order is:
    - cache_any -> cache_here -> keep
    """
    event = full_event["any_resource_decision"]
    ordered_choices = ["cache_any", "cache_here", "keep"]
    logger.info(">> Handling any resource decision event...")
    if event["remaining_resources"] == "agg_this_cave":
        # we must replace this value with the num of aggressive dragons in this cave
        count = 0
        dragons = get_dragon_list(player, full_event["coords"][0])
        for dragon_id, coords in dragons:
            if DRAGON_CARDS[dragon_id]["personality"] == "Aggressive":
                count += 1
        event["remaining_resources"] = count
    # first handle any chosen action from the event
    if "chosen_type" in event:
        resource_type = event["chosen_type"]
        new_limits = event["limits"].copy()
        # subtract from count of resources to choose from
        event["remaining_resources"] -= 1
        # find what to do with the resource based on the order
        if "cache_any" in event["limits"]:
            # cache the resource at the specified location
            cache_resource(player, game_state, resource_type, full_event["coords"])
            if new_limits["cache_any"] > 1:
                new_limits["cache_any"] -= 1
            else:
                new_limits.pop("cache_any")
        elif "cache_here" in event["limits"]:
            # cache the resource at the specified location
            cache_resource(player, game_state, resource_type, full_event["coords"])
            if new_limits["cache_here"] > 1:
                new_limits["cache_here"] -= 1
            else:
                new_limits.pop("cache_here")
        elif "keep" in event["limits"]:
            # add the resource to the player's supply
            logger.info(f"> The player adds 1 {resource_type} to their supply")
            player.resources[resource_type] += 1
            if new_limits["keep"] > 1:
                new_limits["keep"] -= 1
            else:
                new_limits.pop("keep")
    elif "skip_choice" in event:
        # we simply remove the current choice from the list of choices
        # and move on to the next choice
        for choice in ordered_choices:
            if choice in event["limits"]:
                new_limits = event["limits"].copy()
                new_limits.pop(choice)
                logger.info(f">> Player skips the choice: {choice}")
                break
    # check if we have any more resources to choose from
    if event["remaining_resources"] == 0:
        # we are done with the event, so we can remove it from the queue
        return
    # we have more dragons to choose from, so we need to create the next event
    new_event = {"choice": []}
    for i,choice in enumerate(ordered_choices):
        if choice in new_limits:
            # we have a choice to make, so we need to create the event
            for resource_type in RESOURCES:
                if choice == "cache_any":
                    # loop through all possible locations to cache
                    all_dragons = get_dragon_list(player, "any")
                    for dragon, coords in all_dragons:
                        new_event["choice"].append(
                            {"any_resource_decision": {
                                "chosen_type": resource_type,
                                "limits": new_limits,
                                "remaining_resources": event["remaining_resources"],
                                },
                                "coords": coords
                            }
                        )
                else:
                    # we have a specific location to tuck the dragon
                    new_event["choice"].append(
                        {"any_resource_decision": {
                            "chosen_type": resource_type,
                            "limits": new_limits,
                            "remaining_resources": event["remaining_resources"],
                            },
                            "coords": event["coords"]
                        }
                    )
            # check if we can skip the current choice
            future_choice_amounts = 0
            for j in range(i+1, len(ordered_choices)):
                if ordered_choices[j] in new_limits:
                    future_choice_amounts += new_limits[ordered_choices[j]]
            if future_choice_amounts >= event["remaining_resources"]:
                # add the option to skip the current choice
                new_event["choice"].append(
                    {"any_resource_decision": {
                        "skip_choice": True,
                        "limits": new_limits,
                        "remaining_resources": event["remaining_resources"],
                        },
                        "coords": event["coords"]
                    }
                )

    # add the new event to the event queue
    if len(new_event["choice"]) > 1:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})
    elif len(new_event["choice"]) == 1:
        # we have a single choice to make, so we can resolve it immediately
        # and remove the event from the queue
        game_state.event_queue.append(new_event["choice"][0])

def handle_play_cave(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the play cave event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we play the cave.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["play_cave"]
    # check if the event is a choice or random event
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific cave from the display to use
        cave_index = event["chosen_index"]
        cave_id = game_state.board["card_display"]["cave_cards"][cave_index]
        game_state.board["card_display"]["cave_cards"][cave_index] = None # remove the cave from the display
        # play the cave card
        logger.info(f">> Player plays cave {cave_id} from the display")
        excavate_cave(player, game_state, event, cave_id)
        return
    elif event.get("rand_outcome", None) is not None:
        # we have a random cave drawn from the deck
        cave_id = event["rand_outcome"]
        # remove the cave from the deck
        game_state.cave_deck.remove(cave_id)
        if len(game_state.cave_deck) == 0:
            # refresh the dragon deck
            refresh_cave_deck(game_state)
        # play the cave card
        logger.info(f">> Player plays cave {cave_id} randomly drawn from the deck")
        excavate_cave(player, game_state, event, cave_id)
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific cave to use from the hand
        cave_id = event["chosen_id"]
        # remove the cave from the hand
        player.cave_hand.remove(cave_id)
        # play the cave card
        logger.info(f">> Player plays cave {cave_id} from their hand")
        excavate_cave(player, game_state, event, cave_id)
        return
    
    valid_locations = []
    for cave_name in CAVE_NAMES:
        # check if the player can excavate the cave
        can_ex, cave_col = can_excavate_cave(player, cave_name, free=event.get("free", False))
        if can_ex:
            # add the cave to the list of valid locations
            valid_locations.append((cave_name, cave_col))
    # go through each possibility
    if event["source"] == "display":
        new_event = {"choice": []}
        for cave_index in range(3):
            # use a cave from the display
            if game_state.board["card_display"]["cave_cards"][cave_index] is not None:
                for cave_name, cave_col in valid_locations:
                    new_event["choice"].append(
                        {"play_cave": {
                            "source": "display",
                            "chosen_index": cave_index,
                            "free": event.get("free", False),
                            "cave_location": cave_name,
                            }
                        }
                    )
    elif event["source"] == "hand":
        new_event = {"choice": []}
        for cave_id in player.cave_hand:
            # use a cave from the hand
            for cave_name, cave_col in valid_locations:
                # NOTE - We will actually use the 'free' tag here
                # to add costs, as this is the only place that free=False
                # should be used in the game (?)
                pc_event = {
                    "play_cave": {
                        "source": "hand",
                        "chosen_id": cave_id,
                        "free": event.get("free", False),
                        "cave_location": cave_name,
                    }
                }
                if event.get("free", False):
                    # free is true, so we can use the cave
                    new_event["choice"].append(pc_event)
                else:
                    # we have a cost to pay, but we assume we can pay it
                    # for the valid locations previously checked
                    cost_event = {"adv_effects": pc_event, "cost": {"coin": 1}}
                    # check for later cave egg costs
                    if cave_col == 2:
                        cost_event["cost"]["egg"] = {"amount": 1, "location": "any"}
                    elif cave_col == 3:
                        cost_event["cost"]["egg"] = {"amount": 2, "location": "any"}
                    new_event["choice"].append(cost_event)

    elif event["source"] == "deck":
        # take a random dragon from the deck
        new_event = {"choice": []}
        for cave_name, cave_col in valid_locations:
            deck_outcomes = {"random": {
                "play_cave": {
                "source": "deck",
                "free": event.get("free", False),
                "cave_location": cave_name,
                "possible_outcomes": "cave_deck"
                }}
            }
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_gain_cave_card(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the gain cave event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we give the player a cave.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["gain_cave"]
    # check if the event is a choice or random event
    if event.get("chosen", None) is not None:
        # we have a specific cave to gain
        cave_index = event["chosen"]
        cave_id = game_state.board["card_display"]["cave_cards"][cave_index]
        game_state.board["card_display"]["cave_cards"][cave_index] = None # remove the cave from the display
        # add the cave to the player's hand
        logger.info(f">> Player gains cave {cave_id} ({CAVE_CARDS[cave_id]['text']}) from the display")
        player.cave_hand.append(cave_id)
    elif event.get("rand_outcome", None) is not None:
        # we have a random cave drawn from the deck
        cave_id = event["rand_outcome"]
        # remove the cave from the deck
        game_state.cave_deck.remove(cave_id)
        if len(game_state.cave_deck) == 0:
            # refresh the cave deck
            refresh_cave_deck(game_state)
        # add the cave to the player's hand
        logger.info(f">> Player gains cave {cave_id} ({CAVE_CARDS[cave_id]['text']}) randomly drawn from the deck")
        player.cave_hand.append(cave_id)
    else:
        # we have a choice to make - add the choice to the event queue
        new_event = {"choice": []}
        for cave_index in range(3):
            # take a cave from the display
            if game_state.board["card_display"]["cave_cards"][cave_index] is not None:
                new_event["choice"].append({"gain_cave": {"chosen": cave_index}})
        # take a random cave from the deck
        deck_outcomes = {"random": {"gain_cave": {"possible_outcomes": "cave_deck"}}}
        # add the random cave to the choice
        new_event["choice"].append(deck_outcomes)
        # add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_play_dragon(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the play dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we play the dragon.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["play_dragon"]
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific dragon from the display to use
        dragon_index = event["chosen_index"]
        dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
        game_state.board["card_display"]["dragon_cards"][dragon_index] = None # remove the dragon from the display
        # play the dragon card
        place_dragon(player, game_state, full_event["coords"], dragon_id)
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific dragon to use from the hand
        dragon_id = event["chosen_id"]
        # remove the dragon from the hand
        player.dragon_hand.remove(dragon_id)
        # play the dragon card
        place_dragon(player, game_state, full_event["coords"], dragon_id)
        return
    
    valid_locations = []
    for cave_name in CAVE_NAMES:
        # check if the player can entice any dragon in the cave
        # This is when the next slot is excavated and no dragon is there
        for col in range(4):
            if player.caves_played[cave_name][col] is not None:
                # this cave is excavated
                # check if there is a dragon in the cave
                if player.dragons_played[cave_name][col] is None:
                    # there is no dragon in the cave, so we can entice a dragon
                    valid_locations.append((cave_name, col))
                    break
            else:
                # cave is not excavated
                break
    new_event = {"choice": []}
    # go through each possibility
    if event["L1"] == "display":
        for dragon_index in range(3):
            # use a cave from the display
            dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
            if dragon_id is not None:
                costs = get_dragon_enticement_options(player, DRAGON_CARDS[dragon_id], discount=event.get("discount", "none"))
                for cave_name, col in valid_locations:
                    # check cave compatibility
                    if not DRAGON_CARDS[dragon_id][cave_name]:
                        continue
                    for cost in costs:
                        play_event = {"cost": cost}
                        play_event["adv_effects"] = {
                            "play_dragon": {
                                "L1": "display",
                                "L2": "any",
                                "discount": event.get("discount", "none"),
                                "chosen_index": dragon_index,
                            },
                            "coords": (cave_name, col),
                        }
                        new_event["choice"].append(play_event)

    elif event["L1"] == "hand":
        for dragon_id in player.dragon_hand:
            # use a dragon from the hand
            costs = get_dragon_enticement_options(player, DRAGON_CARDS[dragon_id], discount=event.get("discount", "none"))
            for cost in costs:
                for cave_name, col in valid_locations:
                    # check cave compatibility
                    if not DRAGON_CARDS[dragon_id][cave_name]:
                        continue
                    pd_event = {
                        "play_dragon": {
                            "L1": "hand",
                            "L2": "any",
                            "discount": event.get("discount", "none"),
                            "chosen_id": dragon_id,
                        },
                        "coords": (cave_name, col),
                    }
                    cost_event = {"adv_effects": pd_event, "cost": cost}
                    # main action requires 1 coin payment
                    # if event.get("discount", "none") == "none":
                    #     cost_event["cost"]["coin"] = cost_event["cost"].get("coin", 0) + 1
                    new_event["choice"].append(cost_event)
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_gain_dragon_card(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the gain dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we give the player a cave.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["gain_dragon"]
    # check if the event is a choice or random event
    if event.get("chosen", None) is not None:
        # we have a specific dragon to gain
        dragon_index = event["chosen"]
        dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
        game_state.board["card_display"]["dragon_cards"][dragon_index] = None # remove the dragon from the display
        # add the dragon to the player's hand
        logger.info(f">> Player gains dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) from the display")
        player.dragon_hand.append(dragon_id)
    elif event.get("rand_outcome", None) is not None:
        # we have a random dragon drawn from the deck
        dragon_id = event["rand_outcome"]
        # remove the dragon from the deck
        game_state.dragon_deck.remove(dragon_id)
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        # add the dragon to the player's hand
        logger.info(f">> Player gains dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) randomly drawn from the deck")
        player.dragon_hand.append(dragon_id)
    else:
        # we have a choice to make - add the choice to the event queue
        new_event = {"choice": []}
        if event["source"] == "any" or event["source"] == "display":
            for dragon_index in range(3):
                # take a dragon from the display
                if game_state.board["card_display"]["dragon_cards"][dragon_index] is not None:
                    new_event["choice"].append({"gain_dragon": {"chosen": dragon_index}})
        if event["source"] == "any" or event["source"] == "deck":
            # take a random dragon from the deck
            deck_outcomes = {"random": {"gain_dragon": {"possible_outcomes": "dragon_deck"}}}
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)

        # add the choice to the event queue
        if len(new_event["choice"]) > 0:
            game_state.event_queue.append({"adv_effects": new_event})

def handle_resource_caching(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the resource caching event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we perform the caching.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["cache_from"]
    # check if an outcome is already decided
    if "chosen_payment" in event:
        # we have a specific resource to cache
        if event["L1"] == "player_supply":
            # remove the cost from the player's supply
            deduct_resources(player, event["chosen_payment"])
        # cache the resource at the specified location
        cache_resource(player, game_state, event["type"], full_event["coords"])
        return
    valid_resource_costs = []
    valid_locations = []
    # find resources to cache, storing amounts in cost dictionaries
    if event["L1"] == "player_supply":
        # we find each payment for the resource, but we can substitute if necessary
        # e.g. we want to cache 1 meat, but we can pay 2 milk instead
        if event["type"] == "any":
            # we can cache any resource
            for resource in RESOURCES:
                cache_cost = {resource: 1}
                payments = get_all_payments(player, cache_cost)
                for payment in payments:
                    valid_resource_costs.append((payment, resource))
        else:
            # cache a specific resource, but we can substitute if necessary
            cache_cost = {event["type"]: 1}
            payments = get_all_payments(player, cache_cost)
            for payment in payments:
                valid_resource_costs.append((payment, event["type"]))
    elif event["L1"] == "general_supply":
        # we can cache any resource from the general supply
        if event["type"] == "any":
            for resource in RESOURCES:
                valid_resource_costs.append(({resource: 1}, resource))
        else:
            valid_resource_costs.append(({event["type"]: 1}, event["type"]))
    
    # now find locations to cache at
    if event["L2"] == "here":
        # assume coords have been added to the event
        valid_locations.append(full_event["coords"])
    else:
        # assume extra info is given in loc_info
        if event["L2"] == "this_column":
            event["loc_info"] = f"col{full_event['coords'][1]}"
        elif event["L2"] == "any":
            event["loc_info"] = "any"
        possible_dragons = get_dragon_list(player, event["loc_info"])
        valid_locations = [(cave_name, col) for (dragon, (cave_name, col)) in possible_dragons]
    # we need to add a choice to the event queue
    new_event = {
        "choice": [
            {
                "cache_from": {
                    "type": chosen_type,
                    "L1": event["L1"],
                    "L2": "here",
                    "chosen_payment": cost
                },
                "coords": coords,
            } for (cost, chosen_type) in valid_resource_costs for coords in valid_locations
        ]
    }
    if event["L1"] == "player_supply" and len(valid_resource_costs) > 0:
        # add a choice to skip the caching
        new_event["choice"].append({"skip": None})
    if len(new_event["choice"]) > 0:
        game_state.event_queue.append({"adv_effects": new_event})

def handle_tuck_dragon(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the tuck dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.

    If the event parameters are specific enough, we perform the tuck.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["tuck_from"]
    valid_locations = []
    # find what dragon(s) we can tuck
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific dragon to tuck from the display
        dragon_index = event["chosen_index"]
        dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
        game_state.board["card_display"]["dragon_cards"][dragon_index] = None # remove the dragon from the display
        # tuck the dragon
        logger.info(f">> Player tucks dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) from the display")
        tuck_dragon(player, game_state, dragon_id, full_event["coords"])
        return
    elif event.get("rand_outcome", None) is not None:
        # we have a random dragon drawn from the deck
        dragon_id = event["rand_outcome"]
        # remove the dragon from the deck
        game_state.dragon_deck.remove(dragon_id)
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        # tuck the dragon
        logger.info(f">> Player tucks dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) randomly drawn from the deck")
        tuck_dragon(player, game_state, dragon_id, full_event["coords"])
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific dragon to tuck from the hand
        dragon_id = event["chosen_id"]
        # remove the dragon from the hand
        player.dragon_hand.remove(dragon_id)
        # tuck the dragon
        logger.info(f">> Player tucks dragon {dragon_id} ({DRAGON_CARDS[dragon_id]['name']}) from their hand")
        tuck_dragon(player, game_state, dragon_id, full_event["coords"])
        # possible gain_from_cost
        if event.get("gain_from_cost", False):
            # we gain a resource from the cost of the dragon
            logger.info(f">> Player gains a resource from the cost of the dragon")
            new_event = {"choice": []}
            # check each resource type
            for resource in RESOURCES:
                if DRAGON_CARDS[dragon_id][f"{resource}_cost"] > 0:
                    new_event["choice"].append(
                        {"gain_resource": {
                            "type": resource
                            }
                        }
                    )
            if len(new_event["choice"]) > 1:
                # we have multiple choices to make, add the choice to the event queue
                game_state.event_queue.append({"adv_effects": new_event})
            elif len(new_event["choice"]) == 1:
                # we have a single choice to make, so we can resolve it immediately
                game_state.event_queue.append(new_event["choice"][0])
        return
    elif event.get("include", None) is not None:
        # there is one case where we tuck from the deck under all playful in the given cave
        logger.info(">> Player tucks from the deck under all playful in the given cave")
        cave_dragons = get_dragon_list(player, event["loc_info"])
        new_event = {
            "choice": [
                {"sequence": [
                    {
                        "tuck_from": {
                            "L1": "deck",
                            "L2": "here",
                        },
                        "coords": coords
                    } for dragon_id, coords in cave_dragons
                        if DRAGON_CARDS[dragon_id]["personality"] == event["include"]
                    ]
                },
                {"skip": None} # add a choice to skip the tucks
            ]
        }
        game_state.event_queue.append({"adv_effects": new_event})
        return
    # we have a choice to construct
    new_event = {"choice": [{"skip": None}]} # add a choice to skip the tuck
    # go through each possibility
    if event["L2"] == "here":
        valid_locations.append(full_event["coords"])
    elif event["L2"] == "this_column":
        # find all locations in the same column as coords given
        main_cave, main_col = full_event["coords"]
        valid_locations = [(cave_name, col_index) for (d, (cave_name, col_index)) in get_dragon_list(player, f"col{main_col}")]
    elif event["L2"] == "any":
        valid_locations = [(cave_name, col_index) for (d, (cave_name, col_index)) in get_dragon_list(player, "any")]
    # add valid dragon sources to the event
    if event["L1"] == "display":
        choice_event = {"choice": []}
        for dragon_index in range(3):
            # use a dragon from the display
            if game_state.board["card_display"]["dragon_cards"][dragon_index] is not None:
                for coords in valid_locations:
                    choice_event["choice"].append(
                        {
                            "tuck_from": {
                                "L1": "display",
                                "L2": "here",
                                "chosen_index": dragon_index,
                            },
                            "coords": coords
                        }
                    )
        if len(choice_event["choice"]) > 0:
            new_event["choice"].append(choice_event)
    elif event["L1"] == "hand":
        choice_event = {"choice": []}
        for dragon_id in player.dragon_hand:
            # use a dragon from the hand
            for coords in valid_locations:
                choice_event["choice"].append(
                    {
                        "tuck_from": {
                            "L1": "hand",
                            "L2": "here",
                            "chosen_id": dragon_id,
                            "gain_from_cost": event.get("gain_from_cost", False),
                        },
                        "coords": coords,
                    }
                )
        if len(choice_event["choice"]) > 0:
            new_event["choice"].append(choice_event)
    elif event["L1"] == "deck":
        # take a random dragon from the deck
        for coords in valid_locations:
            deck_outcomes = {
                "random": {
                    "tuck_from": {
                        "L1": "deck",
                        "L2": "here",
                        "possible_outcomes": "dragon_deck"
                    },
                    "coords": coords,
                }
            }
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_egg_laying(game_state:GameState, full_event:dict, player:PlayerState) -> None:
    """
    Handle the egg laying event, changing states in place.
    The event is a dictionary with the event name and the parameters.

    If the event parameters are specific enough, we perform the caching.
    Otherwise, we add a choice to the event queue.
    """
    event = full_event["lay_egg"]
    valid_locations = []
    # find locations to lay eggs at
    if event["location"] == "here":
        # assume coords have been added to the event
        # and it is a valid location to lay an egg at
        valid_locations.append(full_event["coords"])
    elif event["location"] == "any":
        # find all locations in the game board
        if player.egg_totals["mat_slots"] < 2:
            valid_locations.append(("mat_slots", 0))
        for cave_name in CAVE_NAMES:
            for col in range(4):
                coords = (cave_name, col)
                if can_lay_egg_at(player, coords):
                    valid_locations.append(coords)
    elif event["location"] == "this_cave":
        # find all locations in the same cave as coords given
        main_cave, main_col = full_event["coords"]
        for col in range(4):
            coords = (main_cave, col)
            if can_lay_egg_at(player, coords):
                valid_locations.append(coords)
    elif event["location"] == "this_column":
        # find all locations in the same column as coords given
        main_cave, main_col = full_event["coords"]
        for cave_name in CAVE_NAMES:
            coords = (cave_name, main_col)
            if can_lay_egg_at(player, coords):
                valid_locations.append(coords)
    elif event["location"] == "ortho":
        # find orthogonal locations to coords given
        main_cave, main_col = full_event["coords"]
        mci = CAVE_NAMES.index(main_cave)
        # check the orthogonal caves
        # upper cave
        if mci > 0:
            valid_locations.append((CAVE_NAMES[mci-1], main_col))
        # lower cave
        if mci < 2:
            valid_locations.append((CAVE_NAMES[mci+1], main_col))
        # left cave
        if main_col > 0:
            valid_locations.append((main_cave, main_col-1))
        # right cave
        if main_col < 3:
            valid_locations.append((main_cave, main_col+1))
        valid_locations = [coords for coords in valid_locations if can_lay_egg_at(player, coords)]
    # special "each" cases, we will lay the eggs now
    elif event["location"] == "each_this_cave":
        excluded_personality = event.get("exclude", "none")
        # find all locations in the same cave as coords given
        main_cave, main_col = full_event["coords"]
        for col in range(4):
            coords = (main_cave, col)
            if can_lay_egg_at(player, coords):
                # check if the coords are not the excluded personality
                d_info = DRAGON_CARDS[player.dragons_played[main_cave][col]]
                if d_info["personality"] != excluded_personality:
                    lay_egg(player, coords)
        return # we are done here
    elif event["location"] == "each_this_column":
        excluded_personality = event.get("exclude", "none")
        # find all locations in the same column as coords given
        main_cave, main_col = full_event["coords"]
        for cave_name in CAVE_NAMES:
            coords = (cave_name, main_col)
            if can_lay_egg_at(player, coords):
                # check if the coords are not the excluded personality
                d_info = DRAGON_CARDS[player.dragons_played[cave_name][main_col]]
                if d_info["personality"] != excluded_personality:
                    lay_egg(player, coords)
        return # we are done here

    # check if we have exactly one location where we can lay an egg now
    if len(valid_locations) == 1:
        # we can lay the egg automatically
        coords = valid_locations[0]
        lay_egg(player, coords)
    elif len(valid_locations) > 0:
        # we need to add a choice to the event queue
        new_event = {
            "choice": [
                {
                    "lay_egg": {
                        "location": "here"
                    },
                    "coords": coords
                } for coords in valid_locations]
        }
        game_state.event_queue.append({"adv_effects": new_event})

# Now we have functions to handle the game events
def get_next_state(game_state:GameState, chosen_input:Union[int,list]=None) -> GameState:
    """
    Get the next game state after applying the input, if any.
    A game state either has a current_choice, a current_random_event,
    or neither of the two.

    - current_choice means we need to input an index for one of the choices. We then
    progress the game state to the next state.
    - current_random_event means we need to input a random outcome. We also
    progress the game state to the next state.
    - If neither, we progress the current state until it has one of the two
    or the game is over.
    """
    logger.debug(f"*** Running get_next_state with chosen_input: {chosen_input}")
    # check if we have a current choice
    if game_state.current_choice is not None:
        # we have a choice to make
        # check if the input is valid
        logger.debug("\tChoice to make")
        if chosen_input is None:
            raise ValueError("No input given for the choice")
        if chosen_input < 0 or chosen_input >= len(game_state.current_choice):
            raise ValueError(f"Invalid input for the choice: {chosen_input}")
        # apply the choice
        new_state = copy.deepcopy(game_state)
        new_state.current_choice = None
        new_state.event_queue.append(game_state.current_choice[chosen_input])
    elif game_state.current_random_event is not None:
        # we have a random event to resolve
        # check if the input is valid
        logger.debug("\tRandom event to resolve")
        if chosen_input is None:
            raise ValueError("No input given for the random event")
        # apply the random event
        new_state = copy.deepcopy(game_state)
        new_state.current_random_event = None
        rand_event = game_state.current_random_event.copy()
        # add outcome to the event
        for key in rand_event.keys():
            if "draw_decision" in rand_event:
                rand_event[key]["remaining_dragons"] = chosen_input
            else:
                rand_event[key]["rand_outcome"] = chosen_input
            break
        new_state.event_queue.append(rand_event)
    else:
        new_state = game_state
        logger.debug("\tNo choice or random event to resolve")
    # we continue using the next event in the queue
    while not new_state.is_halted():
        logger.debug("- Not halted, checking event queue")
        logger.debug(f"Current event queue: {new_state.event_queue}")
        if len(new_state.event_queue) == 0:
            # we have no more events to process, progress the game
            new_state = progress_game(new_state)
        else:
            # we have an event to process
            event = new_state.event_queue.pop()
            new_state = apply_action(new_state, event)
    return new_state

def manually_progress_game():
    """
    Manually progress the game state, using command line input.
    This is useful for testing purposes.
    """
    game = SoloGameState()
    game.create_game()
    while game.phase != PHASE_END_GAME:
        # check if we have a choice or random event
        if game.current_choice is not None:
            print("Current choice:")
            for i, choice in enumerate(game.current_choice):
                print(f"{i}: {choice}")
            # get the input from the user
            chosen_input = int(input("Enter your choice: "))
            game = get_next_state(game, chosen_input)
        elif game.current_random_event is not None:
            print("Current random event:", game.current_random_event)
            print("Sample outcomes:")
            for i in range(5):
                print(get_random_outcome(game, game.current_random_event, game.current_player))
            # get the input from the user
            # it could be a list or integer
            chosen_input = input("Enter an outcome: ")
            try:
                chosen_input = int(chosen_input)
            except ValueError:
                # we have a list of outcomes
                chosen_input = [int(x) for x in chosen_input.split(",")]
            game = get_next_state(game, chosen_input)
        else:
            # progress the game
            game = get_next_state(game, chosen_input=None)

def randomly_progress_game():
    """
    Randomly progress the game state, using random inputs.
    This is useful for testing purposes.
    """
    game = SoloGameState()
    game.create_game()
    while game.phase != PHASE_END_GAME:
        # check if we have a choice or random event
        if game.current_choice is not None:
            # we have a choice to make
            chosen_input = random.randint(0, len(game.current_choice) - 1)
            game = get_next_state(game, chosen_input)
        elif game.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = get_random_outcome(game, game.current_random_event, game.current_player)
            game = get_next_state(game, chosen_input)
        else:
            # progress the game
            game = get_next_state(game, chosen_input=None)

if __name__ == "__main__":
    # Testing the functions
    randomly_progress_game()