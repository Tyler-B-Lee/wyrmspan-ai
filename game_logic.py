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

def can_entice_dragon(player_state: PlayerState, dragon_info: dict, cave_name:CaveName, discount:str="none") -> list[dict]:
    """
    Check if the player can entice a specific dragon in a specific cave. Does not check
    where this dragon is coming from. The discount is described by a string, which can be:
    "none", "free", "1off", "no_resources".

    Returns a list of cost dictionaries, each with a valid cost for the dragon.
    An empty dictionary means the player can entice the dragon for free.
    An empty list means the player cannot entice the dragon at all.
    """
    # check habitat compatibility
    if not dragon_info[cave_name]:
        return []
    costs = []
    resource_cost_dict = {}
    for resource_name in RESOURCES:
        cost_name = f"{resource_name}_cost"
        if dragon_info[cost_name] > 0:
            resource_cost_dict[resource_name] = dragon_info[cost_name]
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
    if not costs:
        return []
    
    # now we check other possible costs and ajust the current costs
    if discount != "free":
        # check coin cost
        if dragon_info["coin_cost"] > 0:
            if dragon_info["coin_cost"] > player_state.coins:
                return []
            for cost in costs:
                cost["coin"] = dragon_info["coin_cost"]
        # check egg cost
        if dragon_info["egg_cost"] > 0:
            if dragon_info["egg_cost"] > get_total_eggs(player_state):
                return []
            for cost in costs:
                cost["egg"] = {
                    "amount": dragon_info["egg_cost"],
                    "location": "any"
                }
    return costs

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

def excavate_cave(player_state: PlayerState, game_state:GameState, event:dict, cave_id:int) -> None:
    """
    Excavate a cave for the player specified by the event.
    """
    cave_location_name = event["play_cave"]["cave_location"]
    index_to_excavate = min(i for i, x in enumerate(player_state.caves_played[cave_location_name]) if x is None)
    # excavate the cave
    player_state.caves_played[cave_location_name][index_to_excavate] = cave_id
    # add effects from cave to the event queue
    cave_effect = CAVE_CARDS[cave_id]["when_played"]
    if index_to_excavate == 3:
        # we must check if the player can do the 4th space exchange
        if len(player.dragon_hand) + len(player.cave_hand) + sum(player.resources.values()) >= 3:
            # add the cave effect to the event queue
            new_event = {
                "adv_effects": 
                    {"choice": [
                        {"adv_effects": {"sequence":[{"4th_space": "any"}, cave_effect]}},
                        {"adv_effects": {"sequence":[cave_effect, {"4th_space": "any"}]}}
                    ]}
            }
            game_state.event_queue.append(new_event)
            return
    # add the cave effect to the event queue
    game_state.event_queue.append(cave_effect)

def place_dragon(player_state: PlayerState, game_state:GameState, event:dict, dragon_id:int) -> None:
    """
    Place a dragon for the player specified by the event.
    """
    cave_loc_name, index_to_place = event["play_dragon"]["coords"]
    # place the dragon
    # TODO - Guild of Highlands effect: play dragon on top of another dragon
    player_state.dragons_played[cave_loc_name][index_to_place] = dragon_id
    # add effects from dragon to the event queue
    dragon_effect = DRAGON_CARDS[dragon_id].get("when_played", None)
    if dragon_effect:
        game_state.event_queue.append(dragon_effect)

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
            resource = event["gain_resource"]["type"]
            player.resources[resource] += 1
    elif "lay_egg" in event:
        # lay an egg
        handle_egg_laying(game_state, event, player)
    elif "pay_egg" in event:
        # pay an egg
        coords = event["pay_egg"]["coords"]
        pay_egg(player, coords)
    elif "gain_guild" in event:
        # assume we have a SoloGameState for now
        pos = game_state.board["guild"]["player_position"]
        new_pos = (pos + 1) % 12
        game_state.board["guild"]["player_position"] = new_pos
        game_state.event_queue.append(GUILD_SPACE_EFFECTS[new_pos])
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
    elif "play_cave" in event:
        # playing a cave card from somewhere to the player's mat
        handle_play_cave(game_state, event, player)
    elif "top_deck_reveal" in event:
        # we reveal the top card and do something based on it
        handle_top_deck_reveal(game_state, event, player)
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

def handle_top_deck_reveal(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handles the event where the top card of the deck is revealed and
    then something is done with it. This assumes that the randomly
    drawn card is already chosen in the event.
    """
    dragon_id = event["top_deck_reveal"]["rand_outcome"]
    dragon_card = DRAGON_CARDS[dragon_id]
    game_state.dragon_deck.remove(dragon_id) # remove the dragon from the deck
    if dragon_card["personality"] in event["top_deck_reveal"]["tuck_targets"]:
        # we tuck the card at the specified location
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        # tuck the dragon at the specified location
        tuck_dragon(player, dragon_id, event["top_deck_reveal"]["coords"])
        return
    # else we do the specified effect of the dragon card
    fail_effect = event["top_deck_reveal"]["fail_effect"]
    if fail_effect == "keep_card":
        # add card to player's hand
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.dragon_hand.append(dragon_id) # add the dragon to the player's hand
    elif fail_effect == "gain_meat":
        # discard the dragon and gain 1 meat
        game_state.dragon_discard.append(dragon_id) # add the dragon to the discard pile
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.resources["meat"] += 1 # add 1 meat to the player's resources
    elif fail_effect == "gain_milk":
        # discard the dragon and gain 1 milk
        game_state.dragon_discard.append(dragon_id) # add the dragon to the discard pile
        if len(game_state.dragon_deck) == 0:
            # refresh the dragon deck
            refresh_dragon_deck(game_state)
        player.resources["milk"] += 1 # add 1 milk to the player's resources

def handle_play_cave(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the play cave event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we play the cave.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific cave from the display to use
        cave_index = event["chosen_index"]
        cave_id = game_state.board["card_display"]["cave_cards"][cave_index]
        game_state.board["card_display"]["cave_cards"][cave_index] = None # remove the cave from the display
        # play the cave card
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
        excavate_cave(player, game_state, event, cave_id)
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific cave to use from the hand
        cave_id = event["chosen_id"]
        # remove the cave from the hand
        player.cave_hand.remove(cave_id)
        # play the cave card
        excavate_cave(player, game_state, event, cave_id)
        return
    
    valid_locations = []
    for cave_name in CAVE_NAMES:
        # check if the player can excavate the cave
        if can_excavate_cave(player, cave_name, free=event.get("free", False)):
            # add the cave to the list of valid locations
            valid_locations.append(cave_name)
    # go through each possibility
    if event["source"] == "display":
        new_event = {"choice": []}
        for cave_index in range(3):
            # use a cave from the display
            if game_state.board["card_display"]["cave_cards"][cave_index] is not None:
                new_event["choice"].append(
                    {"play_cave": {
                        "source": "display",
                        "chosen_index": cave_index,
                        "free": event.get("free", False),
                        "cave_location": cave_name,
                        } for cave_name in valid_locations
                    }
                )
    elif event["source"] == "hand":
        new_event = {"choice": []}
        for cave_id in player.cave_hand:
            # use a cave from the hand
            new_event["choice"].append(
                {"play_cave": {
                        "source": "hand",
                        "chosen_id": cave_id,
                        "free": event.get("free", False),
                        "cave_location": cave_name,
                        } for cave_name in valid_locations
                    }
                )
    elif event["L1"] == "deck":
        # take a random dragon from the deck
        new_event = {"choice": []}
        for cave_name in valid_locations:
            deck_outcomes = {"random": {
                "event_name": "play_cave",
                "source": "deck",
                "free": event.get("free", False),
                "cave_location": cave_name,
                "possible_outcomes": "cave_deck"
                }
            }
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

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
        deck_outcomes = {"random": {"event_name": "gain_cave", "possible_outcomes": "cave_deck"}}
        # add the random cave to the choice
        new_event["choice"].append(deck_outcomes)
        # add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_play_dragon(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the play dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we play the dragon.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific dragon from the display to use
        cave_index = event["chosen_index"]
        cave_id = game_state.board["card_display"]["cave_cards"][cave_index]
        game_state.board["card_display"]["cave_cards"][cave_index] = None # remove the cave from the display
        # play the cave card
        excavate_cave(player, game_state, event, cave_id)
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific cave to use from the hand
        cave_id = event["chosen_id"]
        # remove the cave from the hand
        player.cave_hand.remove(cave_id)
        # play the cave card
        excavate_cave(player, game_state, event, cave_id)
        return
    
    valid_locations = []
    for cave_name in CAVE_NAMES:
        # check if the player can excavate the cave
        if can_excavate_cave(player, cave_name, free=event.get("free", False)):
            # add the cave to the list of valid locations
            valid_locations.append(cave_name)
    # go through each possibility
    if event["source"] == "display":
        new_event = {"choice": []}
        for cave_index in range(3):
            # use a cave from the display
            if game_state.board["card_display"]["cave_cards"][cave_index] is not None:
                new_event["choice"].append(
                    {"play_cave": {
                        "source": "display",
                        "chosen_index": cave_index,
                        "free": event.get("free", False),
                        "cave_location": cave_name,
                        } for cave_name in valid_locations
                    }
                )
    elif event["source"] == "hand":
        new_event = {"choice": []}
        for cave_id in player.cave_hand:
            # use a cave from the hand
            new_event["choice"].append(
                {"play_cave": {
                        "source": "hand",
                        "chosen_id": cave_id,
                        "free": event.get("free", False),
                        "cave_location": cave_name,
                        } for cave_name in valid_locations
                    }
                )
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
        game_state.event_queue.append({"adv_effects": new_event})

def handle_gain_dragon_card(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the gain dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.
    
    If the event parameters are specific enough, we give the player a cave.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    if event.get("chosen", None) is not None:
        # we have a specific dragon to gain
        dragon_index = event["chosen"]
        dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
        game_state.board["card_display"]["dragon_cards"][dragon_index] = None # remove the dragon from the display
        # add the dragon to the player's hand
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
            deck_outcomes = {"random": {"event_name": "gain_dragon", "possible_outcomes": "dragon_deck"}}
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)

        # add the choice to the event queue
        if len(new_event["choice"]) > 0:
            game_state.event_queue.append({"adv_effects": new_event})

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
                    "coords": coords,
                    "chosen": True
                    }
                } for resource in valid_resources for coords in valid_locations]
        }
        if event["L1"] == "player_supply" and len(valid_resources) > 0:
            # add a choice to skip the caching
            new_event["choice"].append({"skip": None})
        if len(new_event["choice"]) > 0:
            game_state.event_queue.append({"adv_effects": new_event})

def handle_tuck_dragon(game_state:GameState, event:dict, player:PlayerState) -> None:
    """
    Handle the tuck dragon event, changing states in place.
    The event is a dictionary with the event name and the parameters.

    If the event parameters are specific enough, we perform the tuck.
    Otherwise, we add a choice to the event queue.
    """
    # check if the event is a choice or random event
    valid_locations = []
    # find what dragon(s) we can tuck
    # first check if an outcome is already decided
    if event.get("chosen_index", None) is not None:
        # we have a specific dragon to tuck from the display
        dragon_index = event["chosen_index"]
        dragon_id = game_state.board["card_display"]["dragon_cards"][dragon_index]
        game_state.board["card_display"]["dragon_cards"][dragon_index] = None # remove the dragon from the display
        # tuck the dragon
        tuck_dragon(player, dragon_id, event["coords"])
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
        tuck_dragon(player, dragon_id, event["coords"])
        return
    elif event.get("chosen_id", None) is not None:
        # we have a specific dragon to tuck from the hand
        dragon_id = event["chosen_id"]
        # remove the dragon from the hand
        player.dragon_hand.remove(dragon_id)
        # tuck the dragon
        tuck_dragon(player, dragon_id, event["coords"])
        return
    elif event.get("include", None) is not None:
        # there is one case where we tuck from the deck under all playful in the given cave
        cave_dragons = get_dragon_list(player, event["loc_info"])
        new_event = {
            "choice": [
                {"sequence": [
                    {"tuck_from": {
                        "L1": "deck",
                        "L2": "here",
                        "coords": coords
                        }
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
        valid_locations.append(event["coords"])
    elif event["L2"] == "this_column":
        # find all locations in the same column as coords given
        main_cave, main_col = event["coords"]
        valid_locations = [(cave_name, col_index) for (d, (cave_name, col_index)) in get_dragon_list(player, f"col{main_col}")]
    elif event["L2"] == "any":
        valid_locations = [(cave_name, col_index) for (d, (cave_name, col_index)) in get_dragon_list(player, "any")]
    # add valid dragon sources to the event
    if event["L1"] == "display":
        choice_event = {"choice": []}
        for dragon_index in range(3):
            # use a dragon from the display
            if game_state.board["card_display"]["dragon_cards"][dragon_index] is not None:
                choice_event["choice"].append(
                        {"tuck_from": {
                            "L1": "display",
                            "L2": "here",
                            "chosen_index": dragon_index,
                            "coords": coords
                            } for coords in valid_locations
                        }
                    )
        if len(choice_event["choice"]) > 0:
            new_event["choice"].append(choice_event)
    elif event["L1"] == "hand":
        choice_event = {"choice": []}
        for dragon_id in player.dragon_hand:
            # use a dragon from the hand
            choice_event["choice"].append(
                    {"tuck_from": {
                        "L1": "hand",
                        "L2": "here",
                        "chosen_id": dragon_id,
                        "coords": coords
                        } for coords in valid_locations
                    }
                )
        if len(choice_event["choice"]) > 0:
            new_event["choice"].append(choice_event)
    elif event["L1"] == "deck":
        # take a random dragon from the deck
        for coords in valid_locations:
            deck_outcomes = {"random": {
                "event_name": "tuck_from",
                "L1": "deck",
                "L2": "here",
                "coords": coords,
                "possible_outcomes": "dragon_deck"
                }
            }
            # add the random dragon to the choice
            new_event["choice"].append(deck_outcomes)
    # add the choice to the event queue
    if len(new_event["choice"]) > 0:
        # we have multiple choices to make, add the choice to the event queue
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
    elif len(valid_locations) > 0:
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