from game_states import *

CaveName = str
DragonNumber = int
CaveNumber = int

# Constants
PHASE_SETUP = "setup"
PHASE_BEFORE_PASS = "before_pass"
PHASE_EXCAVATING = "excavating"
PHASE_ENTICING = "enticing"
PHASE_EXPLORING = "exploring"


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
        # turn deque into list
        "event_queue": list(game_state.event_queue),
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
    game_state.event_queue = collections.deque(data["event_queue"])
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
        "event_queue": list(solo_game_state.event_queue),
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
    solo_game_state.event_queue = collections.deque(data["event_queue"])
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
    for cave_name, dragon_list in player_state.caves_played.items():
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

def can_excavate_cave(player_state: PlayerState, cave_name: CaveName, free: bool=False) -> bool:
    """
    Check if the player can excavate a cave (a row on their mat).
    Does not check for where the cave card is being used from.
    If free is False (default), check if the player has enough resources for later caves.
    """
    cave_list = player_state.caves_played[cave_name]
    if cave_list[-1]: # check if last cave is already excavated
        return False
    if free: # check if the cave is free to excavate
        return True
    # not free - check if the player has enough resources for later caves
    if not cave_list[1]: # slot 2 has no extra cost
        return True
    # check cave slots 2 and 3
    num_eggs = get_total_eggs(player_state)
    if not cave_list[2]:
        return num_eggs >= 1
    # only slot 3 is available, since we saw this last slot is not excavated
    return num_eggs >= 2

def can_entice_dragon(player_state: PlayerState, dragon_info: dict, cave_name:CaveName, discount:str="none") -> bool:
    """
    Check if the player can entice a specific dragon in a specific cave. Does not check
    where this dragon is coming from. The discount is described by a string, which can be:
    "none", "free", "1off", "no_resources".
    """
    # check habitat compatibility
    if not dragon_info[cave_name]:
        return False
    # check if the player has enough resources for the dragon
    cost_dict = {}
    if discount != "free":
        # check coin cost
        if dragon_info["coin_cost"] > player_state.coins:
            return False
    # TODO should we actually return a list of costs?

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

def discard_dragon(player_state: PlayerState, game_state: GameState, dragon: DragonNumber) -> None:
    """
    Discard a dragon from the player's hand and add it to the discard pile.
    """
    player_state.dragon_hand.remove(dragon) # remove the dragon from the hand
    game_state.dragon_discard.append(dragon) # add the dragon to the discard pile

def discard_cave(player_state: PlayerState, game_state: GameState, cave: CaveNumber) -> None:
    """
    Discard a cave from the player's hand and add it to the discard pile.
    """
    player_state.cave_hand.remove(cave) # remove the cave from the hand
    game_state.cave_discard.append(cave) # add the cave to the discard pile

def gain_resources(player_state: PlayerState, amount_dict:dict) -> None:
    """
    Gain the specified amount of some resources given by the amount_dict.
    """
    for resource, amount in amount_dict.items():
        player_state.resources[resource] += amount

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

def cache_resource(player_state: PlayerState, resource:str, coords:tuple) -> None:
    """
    Cache a resource at the specified location.
    Does not affect the player's resources.

    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    player_state.cached_resources[cave_name][col][resource] += 1 # increment the cached resource
    
def tuck_dragon(player_state: PlayerState, dragon: DragonNumber, coords:tuple) -> None:
    """
    Tuck a dragon at the specified location.
    Does not affect the player's hand.

    The location is specified as a tuple (cave:str, col:int).
    The cave is a string representing the cave name.
    """
    cave_name, col = coords
    player_state.tucked_dragons[cave_name][col].append(dragon) # add the dragon to the tucked dragons

def refresh_cave_deck(game_state: GameState) -> None:
    """
    Refresh the cave deck by copying the discard pile back to the deck
    and clearing the discard pile.
    """
    game_state.cave_deck = game_state.cave_discard.copy()
    game_state.cave_discard.clear() # clear the discard pile

def refresh_dragon_deck(game_state: GameState) -> None:
    """
    Refresh the dragon deck by copying the discard pile back to the deck
    and clearing the discard pile.
    """
    game_state.dragon_deck = game_state.dragon_discard.copy()
    game_state.dragon_discard.clear() # clear the discard pile

# main game functions
def get_current_player(game_state:GameState) -> PlayerState:
    """
    Get the current player from the game state.
    The current player is the player whose turn it is.
    """
    if isinstance(game_state, SoloGameState):
        return game_state.player
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
    Progress the game state by resolving the next event in the queue.

    Returns a copy of the game state after the event is resolved.
    If the queue is empty, return the game state as is.
    """
    new_state = copy.deepcopy(game_state) # make a copy of the game state
    if new_state.event_queue:
        # there are events in the queue, resolve the next one
        event = new_state.event_queue.pop()
        # check if the event is a choice or random event
        if "adv_effects" in event:
            # this is a choice or random event, add it to the queue
            new_state.event_queue.append(event)
    else:
        # the queue is empty, we are in an active state
        # we can try to find the next choice or random event to resolve
        # or the game ends.
        pass

def get_available_actions(game_state:GameState) -> list[dict]:
    """
    Given a game_state, return a list of available actions.
    Each action is represented as a dictionary with the action name and the parameters.
    """
    actions = []
    player = get_current_player(game_state)

    # check for actions by phase
    current_phase = game_state.phase
    if current_phase == PHASE_SETUP:
        # setup phase - player must discard down to 4 total cards
        # and then choose any 3 resources
        if len(player.dragon_hand) + len(player.cave_hand) > 4:
            # discard dragons and caves
            for dragon in player.dragon_hand:
                actions.append({"discard_dragon": {"dragon": dragon}})
            for cave in player.cave_hand:
                actions.append({"discard_cave": {"cave": cave}})
            return actions
        # choose resources
        for resource in player.resources.keys():
            actions.append({"gain_resource": {"type": resource}})
    
    return actions

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
        if event["type"] == "any":
            # CHOICE - choose to gain 1 of any resource
            # add the choice to the event queue
            new_event = {
                "choice": [{"gain_resource": {"type": resource}} for resource in RESOURCES]
            }
            game_state.event_queue.append({"adv_effects": new_event})
        else:
            resource = event["gain resource"]["type"]
            player.resources[resource] += 1
    elif "lay_egg" in event:
        # lay an egg
        handle_egg_laying(game_state, event, player)
    elif "pay_egg" in event:
        # pay an egg
        coords = event["pay_egg"]["coords"]
        pay_egg(player, coords)
    elif "cache_from" in event:
        # the player caches a resource, but we need
        # to check where this can happen
        handle_resource_caching(game_state, event, player)
    elif "tuck_dragon" in event:
        # tuck a dragon
        dragon = event["tuck_dragon"]["dragon"]
        coords = event["tuck_dragon"]["coords"]
        tuck_dragon(player, dragon, coords)
    elif "gain_cave" in event:
        # gain a cave card
        handle_gain_cave_card(game_state, event, player)
    elif "gain_dragon" in event:
        # gain a dragon card
        handle_gain_dragon_card(game_state, event, player)
    elif "deduct_resources" in event:
        # deduct resources
        cost_dict = event["deduct_resources"]["cost"]
        deduct_resources(player, cost_dict)
    elif "discard_dragon" in event:
        # discard dragon
        dragon = event["discard dragon"]["dragon"]
        discard_dragon(player, game_state, dragon)
    elif "discard_cave" in event:
        # discard cave
        cave = event["discard cave"]["cave"]
        discard_cave(player, game_state, cave)

def handle_gain_cave_card(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the gain cave event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we give the player a cave.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    if event.get("chosen", None) is not None:
        # we have a specific cave to gain
        cave_index = event["chosen"]
        cave_id = game_state.board["card_display"]["cave_cards"][cave_index]
        game_state.board["card_display"]["cave_cards"][cave_index] = None # remove the cave from the display
        # add the cave to the player's hand
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
        player.cave_hand.append(cave_id)
    else:
        # we have a choice to make - add the choice to the event queue
        new_event = {"choice": []}
        for cave_index in range(3):
            # take a cave from the display
            if game_state.board["card_display"]["cave_cards"][cave_index] is not None:
                new_event["choice"].append({"gain_cave": {"chosen": cave_index}})
        # take a random cave from the deck
        deck_outcomes = {"random": []}
        for cave_id in game_state.cave_deck:
            deck_outcomes["random"].append({"gain_cave": {"rand_outcome": cave_id}})
        # add the random cave to the choice
        new_event["choice"].append(deck_outcomes)
        # add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_gain_dragon_card(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the gain dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we give the player a cave.
    Otherwise, we add a choice to the event queue.
    """
    pass

def handle_resource_caching(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the resource caching event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we perform the caching.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    valid_resources = []
    valid_locations = []
    # find resources to cache
    if event["type"] != "any":
        # specific resource - we assume the player has it
        valid_resources.append(event["type"])
    else:
        if event["L1"] == "general_supply":
            # caching from general supply is free for player
            valid_resources = [r for r in RESOURCES]
        elif event["L1"] == "player_supply":
            # check if the player has any resources to cache
            valid_resources = [r for r in player.resources.keys() if player.resources[r] > 0]
    # now find locations to cache at
    if event["L2"] == "here":
        # assume coords have been added to the event
        valid_locations.append(event["coords"])
    else:
        # assume extra info is given in loc_info
        possible_dragons = get_dragon_list(player, event["loc_info"])
        valid_locations = [(cave_name, col) for (dragon, (cave_name, col)) in possible_dragons]
    # check if we have exactly one resource and one location
    # where we can cache the resource now
    if (len(valid_resources) == 1 and len(valid_locations) == 1 and
            (event["L1"] == "general_supply" or "chosen" in event)):
        # we can cache the resource now, no need to add a choice
        resource = valid_resources[0]
        coords = valid_locations[0]
        cache_resource(player, resource, coords)
        if event["L1"] == "player_supply":
            # we remove the resource from the player's supply
            player.resources[resource] -= 1
            assert player.resources[resource] >= 0, f"Player has negative resources: {player.resources}"
    else:
        # we need to add a choice to the event queue
        new_event = {
            "choice": [
                {"cache_from": {
                    "type": resource,
                    "L1": event["L1"],
                    "L2": "here",
                    "coords": coords
                    }
                } for resource in valid_resources for coords in valid_locations]
        }
        if event["L1"] == "player_supply":
            # add a choice to skip the caching
            new_event["choice"].append({"skip": None})
        game_state.event_queue.append({"adv_effects": new_event})

def handle_egg_laying(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the egg laying event, changing states in place.
    The event is a dictionary with the event name and the parameters.

    If the event parameters are specific enough, we perform the caching.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    valid_locations = []
    # find locations to lay eggs at
    if event["location"] == "here":
        # assume coords have been added to the event
        # and it is a valid location to lay an egg at
        valid_locations.append(event["coords"])
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
        main_cave, main_col = event["coords"]
        for col in range(4):
            coords = (main_cave, col)
            if can_lay_egg_at(player, coords):
                valid_locations.append(coords)
    elif event["location"] == "this_column":
        # find all locations in the same column as coords given
        main_cave, main_col = event["coords"]
        for cave_name in CAVE_NAMES:
            coords = (cave_name, main_col)
            if can_lay_egg_at(player, coords):
                valid_locations.append(coords)
    elif event["location"] == "ortho":
        # find orthogonal locations to coords given
        main_cave, main_col = event["coords"]
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
        main_cave, main_col = event["coords"]
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
        main_cave, main_col = event["coords"]
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
    else:
        # we need to add a choice to the event queue
        new_event = {
            "choice": [
                {"lay_egg": {
                    "location": "here",
                    "coords": coords
                    }
                } for coords in valid_locations]
        }
        game_state.event_queue.append({"adv_effects": new_event})

def apply_action(game_state:GameState, action:dict) -> GameState:
    """
    Apply the given action to the game_state and return a new game state
    from after the action is applied.
    """
    new_game_state = copy.deepcopy(game_state) # make a copy of the game state
    player = get_current_player(new_game_state) # get the current player

    # check for actions by phase
    current_phase = new_game_state.phase
    if current_phase == PHASE_SETUP:
        # setup phase - player must discard down to 4 total cards
        # and then choose any 3 resources
        if "discard_dragon" in action:
            # discard dragon
            dragon = action["discard_dragon"]["dragon"]
            discard_dragon(player, new_game_state, dragon)
        elif "discard_cave" in action:
            # discard cave
            cave = action["discard_cave"]["cave"]
            discard_cave(player, new_game_state, cave)
        elif "gain_resource" in action:
            # gain resources
            resource = action["gain_resource"]["type"]
            player.resources[resource] += 1
            # check if the player has already chosen 3 resources
            if sum(player.resources.values()) >= 3:
                if isinstance(new_game_state, SoloGameState):
                    new_game_state.phase = PHASE_BEFORE_PASS
                else:
                    new_game_state.current_player += 1 # move to the next player
                    if new_game_state.current_player >= len(new_game_state.players):
                        # we are done with the setup phase
                        new_game_state.current_player = new_game_state.round_start_player
                        new_game_state.phase = PHASE_BEFORE_PASS

    return new_game_state

if __name__ == "__main__":
    # Testing the functions
    game = SoloGameState()
    game.create_game()
    while game.phase != PHASE_BEFORE_PASS:
        a = get_available_actions(game)
        print("Action list:")
        for action in a:
            print(f"\t{action}")
        print("Game state:")
        print(game)
        print("Player state:")
        print(game.player)
        print("Automa state:")
        print(game.automa)
        # run random action possible
        action = random.choice(a)
        print(f"Applying action: {action}")
        game = apply_action(game, action)

    # test gain_cave action
    action = {"gain_cave": {"source": "any"}}
    handle_gain_cave_card(game, action, game.player)
    print("\nEvent queue after processing:")
    print(game.event_queue)