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

def handle_event(game_state:GameState, event:dict, player:PlayerState=None) -> None:
    """
    Handle the given event in the game state with the target player.
    The event is a dictionary with the event name and the parameters.
    The player is the player who triggered the event, if applicable.

    Events are basic game events that are not tied to a specific phase
    or action. They are used to modify the game state in a generic way.
    There should be no further choices to be made for the event.
    For example, discarding a dragon or cave, or gaining resources.
    The event is handled irrespective of the game phase.

    The states input are modified in place, so no return value is needed.
    """
    if "discard_dragon" in event:
        # discard dragon
        dragon = event["discard dragon"]["dragon"]
        discard_dragon(player, game_state, dragon)
    elif "discard_cave" in event:
        # discard cave
        cave = event["discard cave"]["cave"]
        discard_cave(player, game_state, cave)
    elif "gain_resource" in event:
        # gain resources
        resource = event["gain resource"]["type"]
        player.resources[resource] += 1
    

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

        chosen_action = random.choice(a)
        print(f"Chosen action: {chosen_action}")
        game = apply_action(game, chosen_action)
        print("Game state after action:")
        print(game.dragon_discard)
        print(game.cave_discard)
        print(get_current_player(game).resources)
        print(get_current_player(game).dragon_hand)
        print(get_current_player(game).cave_hand)