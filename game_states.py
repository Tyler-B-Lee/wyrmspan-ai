import collections
import copy
import json
import random
import typing


# Constants for the game


# load the game data saved in multiple json files
with open('data/dragon_cards.json', 'r') as f:
    DRAGON_CARDS:list[dict] = json.load(f)
with open('data/cave_cards.json', 'r') as f:
    CAVE_CARDS:list[dict] = json.load(f)
with open('data/guild_tiles.json', 'r') as f:
    GUILD_TILES:list[dict] = json.load(f)
with open('data/objective_tiles.json', 'r') as f:
    OBJECTIVE_TILES:list[dict] = json.load(f)


def draw_random_objectives() -> typing.List[typing.Tuple[int, str]]:
    """
    Returns a random selection of four objectives from the available objectives.
    """
    remaining_tile_indices = list(range(10))
    selected_objectives = []
    while len(selected_objectives) < 4:
        tile_index = random.choice(remaining_tile_indices)
        side = random.choice(["side_a", "side_b"])
        # there is one objective that we cannot select
        if tile_index == 3 and side == "side_b":
            continue

        selected_objectives.append((tile_index, side))
        # remove the selected tile from the remaining options
        remaining_tile_indices.remove(tile_index)
        
    return selected_objectives


class PlayerState:
    def __init__(self):
        self.dragon_hand = set()
        self.cave_hand = set()
        self.resources = {"meat": 0, "gold": 0, "crystal": 0, "milk": 0}
        self.eggs = {"mat_slots": 0, "crimson_cavern": 0, "golden_grotto": 0, "amethyst_abyss": 0}
        self.num_dragons_played = {"crimson_cavern": 0, "golden_grotto": 0, "amethyst_abyss": 0}
        self.score = 0
        self.guild_markers = 4
        self.coins = 0
        self.caves_played = { # numbers of the caves played
            "crimson_cavern": [0, None, None, None],
            "golden_grotto": [0, None, None, None],
            "amethyst_abyss": [0, None, None, None],
        }
        self.dragons_played = { # numbers of the dragons played
            "crimson_cavern": [None, None, None, None],
            "golden_grotto": [None, None, None, None],
            "amethyst_abyss": [None, None, None, None],
        }
        self.times_explored = { # number of times the player explored in each cave
            "crimson_cavern": 0,
            "golden_grotto": 0,
            "amethyst_abyss": 0,
        }
        self.adventurer_position = None

class AutomaState:
    def __init__(self):
        self.dragons = set()
        self.caves = set()
        self.score = 0
        self.difficulty = 0


class GameState:
    """
    Represents the state of the game at any given time.
    This includes the current turn, phase, players, board state,
    deck, discard pile, and an event queue for handling game events.
    """
    def __init__(self):
        self.turn = 0
        self.phase = 0
        self.board = {}
        self.dragon_deck = set(range(1, 184))  # Set of dragon cards IDs
        self.cave_deck = set(range(1, 76)) # Cave cards IDs
        self.dragon_discard = set()
        self.cave_discard = set()
        self.event_queue = collections.deque()  # Use deque for the event queue

    def draw_random_dragon_cards(self, num:int=1) -> typing.List[int]:
        """
        Draws a specified number of random dragon cards from the deck.
        Removes drawn cards from the deck set.
        If the deck is empty, it refills from the discard pile.
        Returns a list of drawn card IDs.
        """
        if num > len(self.dragon_deck):
            # refill the deck from the discard pile
            self.dragon_deck = self.dragon_discard.copy()
            self.dragon_discard.clear()
            assert num <= len(self.dragon_deck), "Not enough cards in the deck to draw."
        
        drawn_cards = random.sample(self.dragon_deck, num)
        self.dragon_deck -= set(drawn_cards)
        return drawn_cards
    
    def draw_random_cave_cards(self, num:int=1) -> typing.List[int]:
        """
        Draws a specified number of random cave cards from the deck.
        Removes drawn cards from the deck set.
        If the deck is empty, it refills from the discard pile.
        Returns a list of drawn card IDs.
        """
        if num > len(self.cave_deck):
            # refill the deck from the discard pile
            self.cave_deck = self.cave_discard.copy()
            self.cave_discard.clear()
            assert num <= len(self.cave_deck), "Not enough cards in the deck to draw."
        
        drawn_cards = random.sample(self.cave_deck, num)
        self.cave_deck -= set(drawn_cards)
        return drawn_cards

    def create_game(self, num_players=2):
        "Initializes a new game state. Includes random generation of objectives."
        self.players = [PlayerState() for _ in range(num_players)]
        self.round_start_player = random.randint(0, num_players - 1)
        self.current_player = 0
        
        # initialize board parts
        # Guild Board
        chosen_guild = random.randint(0,3)
        self.board["guild"] = {
            "guild_index": chosen_guild,
            "ability_uses": {i: [] for i in range(1, 5)},
            "player_positions": [0] * num_players,
        }
        # Card Display Board
        board_d_cards = self.draw_random_dragon_cards(3)
        board_c_cards = self.draw_random_cave_cards(3)
        self.board["card_display"] = {
            "dragon_cards": board_d_cards,
            "cave_cards": board_c_cards,
        }
        # Round Tracker Board
        self.board["round_tracker"] = {
            "round": 1,
            "objectives": draw_random_objectives(),
            "scoring": [{"1st": [], "2nd": [], "3rd": [], "Other": []} for _ in range(3)]
        }
        for i in range(num_players):
            player = self.players[i]
            player.coins = 6
            player.eggs["mat_slots"] = 1
            player.dragon_hand = set(self.draw_random_dragon_cards(3))
            player.cave_hand = set(self.draw_random_cave_cards(3))


class SoloGameState(GameState):
    """
    Represents the state of a solo game.
    Inherits from GameState and adds an automa state to track and use.
    """
    ignore_cards = {47,103,114,168,170}

    def __init__(self):
        super().__init__()
        self.automa = AutomaState()
        self.automa_score = 0
        self.automa_bonuses = []
        self.automa_difficulty = 0


def game_state_to_dict(game_state):
    """
    Converts a GameState object to a dictionary representation.
    """
    return {
        "turn": game_state.turn,
        "phase": game_state.phase,
        "players": [player_state_to_dict(p) for p in game_state.players],
        "board": game_state.board,
        "deck": game_state.deck,
        "discard": game_state.discard,
        "event_queue": list(game_state.event_queue)  # Convert deque to list
    }

def dict_to_game_state(state_dict):
    """
    Creates a GameState object from a dictionary representation.
    """
    game_state = GameState()
    game_state.turn = state_dict["turn"]
    game_state.phase = state_dict["phase"]
    game_state.players = [dict_to_player_state(p) for p in state_dict["players"]]
    game_state.board = state_dict["board"]
    game_state.deck = state_dict["deck"]
    game_state.discard = state_dict["discard"]
    game_state.event_queue = collections.deque(state_dict["event_queue"]) # Convert list to deque
    return game_state

def player_state_to_dict(player_state):
    """
    Converts a PlayerState object to a dictionary.
    """
    return {
        "hand": player_state.hand,
        "resources": player_state.resources,
        "eggs": player_state.eggs,
        "num_dragons_played": player_state.num_dragons_played,
        "score": player_state.score,
        "bonuses": player_state.bonuses,
        "food_tokens": player_state.food_tokens,
        "habitat_state": player_state.habitat_state
    }

def dict_to_player_state(state_dict):
    """
    Creates a PlayerState object from a dictionary.
    """
    player_state = PlayerState()
    player_state.hand = state_dict["hand"]
    player_state.resources = state_dict["resources"]
    player_state.eggs = state_dict["eggs"]
    player_state.num_dragons_played = state_dict["num_dragons_played"]
    player_state.score = state_dict["score"]
    player_state.bonuses = state_dict["bonuses"]
    player_state.food_tokens = state_dict["food_tokens"]
    player_state.habitat_state = state_dict["habitat_state"]
    return player_state


if __name__ == "__main__":
    # Test objectives drawing
    for i in range(10):
        obj = draw_random_objectives()
        for (idx, side) in obj:
            print(f"Objective Tile {idx} Side {side}:")
            print(OBJECTIVE_TILES[idx][side])
        print()