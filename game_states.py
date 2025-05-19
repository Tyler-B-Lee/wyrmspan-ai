import collections
import copy
import json
import random
import typing
import logging

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for the game
PHASE_SETUP = "setup"
PHASE_ROUND_START = "round_start"
PHASE_BEFORE_ACTION = "before_pass"
PHASE_END_ROUND = "end_round"
PHASE_MAIN_ACTION = "main_action"
PHASE_END_TURN = "end_turn"
PHASE_FINAL_SCORING = "final_scoring"
PHASE_END_GAME = "end_of_game"

RESOURCES = ["meat", "gold", "crystal", "milk"]
CAVE_NAMES = ["crimson_cavern", "golden_grotto", "amethyst_abyss"]
DRAGON_SIZES = ["Hatchling", "Small", "Medium", "Large"]
DRAGON_PERSONALITIES = ["Helpful", "Shy", "Playful", "Aggressive"]
ABILITY_TYPES = ["once_per_round", "if_activated", "end_game", "when_played"]

GUILD_SPACE_EFFECTS = [
    {"brown_space": -1}, # space 0
    {"lay_egg": {"location": "any"}}, # space 1
    {"gain_resource": {"type": "meat"}}, # space 2
    {"gain_dragon": {"source": "any"}}, # space 3
    {"gain_cave": {"source": "any"}}, # space 4
    {"gain_resource": {"type": "crystal"}}, # space 5
    {"brown_space": -1}, # space 6
    {"lay_egg": {"location": "any"}}, # space 7
    {"gain_resource": {"type": "gold"}}, # space 8
    {"gain_dragon": {"source": "any"}}, # space 9
    {"gain_coin": {"amount": 1}}, # space 10
    {"gain_resource": {"type": "milk"}}, # space 11
]

OBJECTIVE_POSITION_SCORES = [
    {"1st": 4, "2nd": 1, "3rd": 0, "Other": 0}, # round 1
    {"1st": 5, "2nd": 2, "3rd": 1, "Other": 0}, # round 2
    {"1st": 6, "2nd": 3, "3rd": 2, "Other": 0}, # round 3
    {"1st": 7, "2nd": 4, "3rd": 3, "Other": 0}, # round 4
]

EXPLORE_CAVE_EFFECTS = {
    "crimson_cavern": {
        0: {"gain_resource": {"type": "any"}},
        1: {"gain_guild": {"source": "any"}},
        2: {"gain_resource": {"type": "any"}},
        3: {"lay_egg": {"location": "any"}},
        4: {"adv_effects": {
            "cache_from": {
                "type": "any",
                "L1": "player_supply",
                "L2": "any"
            }},
            "max_uses": 2
        }
    },
    "golden_grotto": {
        0: {"gain_dragon": {"source": "any"}},
        1: {"gain_guild": {"source": "any"}},
        2: {"gain_dragon": {"source": "any"}},
        3: {"lay_egg": {"location": "any"}},
        4: {"adv_effects": {
            "tuck_from": {
                "L1": "hand",
                "L2": "any"
            }},
            "max_uses": 2
        }
    },
    "amethyst_abyss": {
        0: {"gain_cave": {"source": "any"}},
        1: {"gain_guild": {"source": "any"}},
        2: {"gain_cave": {"source": "any"}},
        3: {"lay_egg": {"location": "any"}},
        4: {"adv_effects": {
            "sequence": [
                {"lay_egg": {"location": "any"}},
                {"lay_egg": {"location": "any"}}
            ]},
            "max_uses": 2,
            "cost": {"cave_card": 1}
        }
    }
}

# load the game data saved in multiple json files
with open('data/dragon_cards.json', 'r') as f:
    DRAGON_CARDS:list[dict] = json.load(f)
with open('data/cave_cards.json', 'r') as f:
    CAVE_CARDS:list[dict] = json.load(f)
with open('data/guild_tiles.json', 'r') as f:
    GUILD_TILES:list[dict] = json.load(f)
with open('data/objective_tiles.json', 'r') as f:
    OBJECTIVE_TILES:list[dict] = json.load(f)


def draw_random_objectives(is_solo:bool) -> typing.List[typing.Tuple[int, str]]:
    """
    Returns a random selection of four objectives from the available objectives.
    """
    remaining_tile_indices = list(range(10))
    selected_objectives = []
    while len(selected_objectives) < 4:
        tile_index = random.choice(remaining_tile_indices)
        side = random.choice(["side_a", "side_b"])
        # there is one objective that we cannot select for solo play
        if is_solo and tile_index == 3 and side == "side_b":
            continue

        selected_objectives.append((tile_index, side))
        # remove the selected tile from the remaining options
        remaining_tile_indices.remove(tile_index)
        
    return selected_objectives


class PlayerState:
    def __init__(self):
        self.dragon_hand = []
        self.cave_hand = []
        self.resources = {"meat": 0, "gold": 0, "crystal": 0, "milk": 0}
        self.egg_totals = {"mat_slots": 0, "crimson_cavern": 0, "golden_grotto": 0, "amethyst_abyss": 0}
        self.num_dragons_played = {"crimson_cavern": 0, "golden_grotto": 0, "amethyst_abyss": 0}
        self.score = 0
        self.guild_markers = 4
        self.coins = 0
        self.caves_played = { # numbers of the caves played
            "crimson_cavern": [-1, None, None, None],
            "golden_grotto": [-1, None, None, None],
            "amethyst_abyss": [-1, None, None, None],
        }
        self.dragons_played = { # numbers of the dragons played
            "crimson_cavern": [None, None, None, None],
            "golden_grotto": [None, None, None, None],
            "amethyst_abyss": [None, None, None, None],
        }
        self.hatchling_grown = { # hatchlings grown in each cave
            "crimson_cavern": [None, None, None, None],
            "golden_grotto": [None, None, None, None],
            "amethyst_abyss": [None, None, None, None],
        }
        self.cached_resources = { # cached resources for the player
            "crimson_cavern": [collections.defaultdict(int) for _ in range(4)],
            "golden_grotto": [collections.defaultdict(int) for _ in range(4)],
            "amethyst_abyss": [collections.defaultdict(int) for _ in range(4)],
        }
        self.tucked_dragons = { # dragons tucked in each cave
            "crimson_cavern": [[] for _ in range(4)],
            "golden_grotto": [[] for _ in range(4)],
            "amethyst_abyss": [[] for _ in range(4)],
        }
        self.nested_eggs = { # eggs nested in each cave
            "crimson_cavern": [[0, 0] for _ in range(4)], # (# eggs, # slots)
            "golden_grotto": [[0, 0] for _ in range(4)],
            "amethyst_abyss": [[0, 0] for _ in range(4)],
        }
        self.times_explored = { # number of times the player explored in each cave this round
            "crimson_cavern": 0,
            "golden_grotto": 0,
            "amethyst_abyss": 0,
        }
        self.adventurer_position = None
        self.passed_this_round = False
        self.const_end_game_abilities = []
    
    def __str__(self):
        """Returns a string representation of the player state, with newlines and tabs for formatting."""
        dragon_names = [(dragon_id, DRAGON_CARDS[dragon_id]["name"]) for dragon_id in self.dragon_hand]
        cave_info = [(cave_id, CAVE_CARDS[cave_id]["text"]) for cave_id in self.cave_hand]
        return (
            f"Player State:\n"
            f"\tDragon Hand: {dragon_names}\n"
            f"\tCave Hand: {cave_info}\n"
            f"\tResources: {self.resources}\n"
            f"\tEgg Totals: {self.egg_totals}\n"
            f"\tDragons Played: {self.num_dragons_played}\n"
            f"\tScore: {self.score}\n"
            f"\tGuild Markers: {self.guild_markers}\n"
            f"\tCoins: {self.coins}\n"
            f"\tCaves Played: {self.caves_played}\n"
            f"\tDragons Played: {self.dragons_played}\n"
            f"\tCached Resources: {self.cached_resources}\n"
            f"\tTucked Dragons: {self.tucked_dragons}\n"
            f"\tNested Eggs: {self.nested_eggs}\n"
            f"\tTimes Explored: {self.times_explored}\n"
        )

class AutomaState:
    """
    Represents the state of the automa player in a solo game.
    This includes the automa's dragons, caves, score, and difficulty level.
    """
    # class variables for automa difficulty levels
    difficulty_names = {
        0: "Automa Level 1 (Easy)",
        1: "Automa Level 2 (Medium)",
        2: "Automa Level 3 (Hard)",
        # alternate 'Ravel' solo mode
        3: "Ravel Level 1 (Medium?)",
        4: "Ravel Level 2 (Hard?)",
        5: "Ravel Level 3 (Very Hard?)",
    }
    difficulty_card_decks = {
        0: [0,1,2,3,4,5,6,7],
        1: [8,9,10,3,4,5,6,7],
        2: [0,1,2,8,9,10,6,7],
        3: [12,13,14,3,4,5,6,15],
        4: [0,1,2,12,13,14,6,15],
        5: [12,13,14,9,10,11,6,15],
    }

    def __init__(self, difficulty:int=0):
        self.dragons = []
        self.caves = []
        self.score = 0
        self.difficulty = difficulty
        self.reset_decision_deck()
        self.passed_this_round = False
        self.const_end_game_abilities = []

    def __str__(self):
        """Returns a string representation of the automa state, with newlines and tabs for formatting."""
        dragon_names = [(dragon_id, DRAGON_CARDS[dragon_id]["name"]) for dragon_id in self.dragons]
        return (
            f"Automa State:\n"
            f"\tDragons: {dragon_names}\n"
            f"\tCaves: {self.caves}\n"
            f"\tScore: {self.score}\n"
            f"\tDifficulty: {AutomaState.difficulty_names[self.difficulty]}\n"
        )

    def reset_decision_deck(self):
        """
        Resets the automa's deck based on the current difficulty level.
        """
        self.decision_deck = copy.deepcopy(AutomaState.difficulty_card_decks[self.difficulty])
        

class GameState:
    """
    Represents the state of the game at any given time.
    This includes the current turn, phase, players, board state,
    deck, discard pile, and an event queue for handling game events.
    """
    def __init__(self, allowed_guilds=(0,1,3), ending_round=4):
        self.turn = 0
        self.phase = PHASE_SETUP
        self.board = {}
        self.dragon_deck = list(range(1, 184))  # Set of dragon cards IDs
        self.cave_deck = list(range(1, 76)) # Cave cards IDs
        self.dragon_discard = []  # Discard pile for dragon cards
        self.cave_discard = []  # Discard pile for cave cards
        self.event_queue = []  # Use list for the event queue
        self.current_choice = None  # Current choice for the player
        self.current_random_event = None  # Current random event for the game
        self.allowed_guilds = allowed_guilds
        self.ending_round = ending_round
        self.finished_end_game_abilities = False
    
    def make_copy(self) -> 'GameState':
        """
        Returns a deep copy of the current game state.
        This is useful for undo/redo functionality or saving the game state.
        """
        return copy.deepcopy(self)
    
    def is_halted(self) -> bool:
        "Checks if the game is halted. Either we have a choice, a random event, or the game is over."
        return self.current_choice is not None or self.current_random_event is not None or self.phase == PHASE_END_GAME

    def draw_random_dragon_cards(self, num:int=1) -> typing.List[int]:
        """
        Draws a specified number of random dragon cards from the deck.
        Removes drawn cards from the deck list.
        If the deck is empty, it refills from the discard pile.
        Returns a list of drawn card IDs.
        """
        if num > len(self.dragon_deck):
            # refill the deck from the discard pile
            self.dragon_deck = self.dragon_discard.copy()
            self.dragon_discard.clear()
            assert num <= len(self.dragon_deck), "Not enough cards in the deck to draw."
        
        drawn_cards = random.sample(self.dragon_deck, num)
        # remove drawn cards from the deck list
        for card in drawn_cards:
            self.dragon_deck.remove(card)
        return drawn_cards
    
    def draw_random_cave_cards(self, num:int=1) -> typing.List[int]:
        """
        Draws a specified number of random cave cards from the deck.
        Removes drawn cards from the deck list.
        If the deck is empty, it refills from the discard pile.
        Returns a list of drawn card IDs.
        """
        if num > len(self.cave_deck):
            # refill the deck from the discard pile
            self.cave_deck = self.cave_discard.copy()
            self.cave_discard.clear()
            assert num <= len(self.cave_deck), "Not enough cards in the deck to draw."
        
        drawn_cards = random.sample(self.cave_deck, num)
        # remove drawn cards from the deck list
        for card in drawn_cards:
            self.cave_deck.remove(card)
        return drawn_cards
    
    def all_players_passed(self) -> bool:
        """
        Checks if all players have passed their turn in the current round.
        Returns True if all players have passed, False otherwise.
        """
        return all(player.passed_this_round for player in self.players)

    def log_card_display(self):
        """
        Logs the current state of the card display.
        This includes the dragon and cave cards on display.
        """
        logger.info(self.get_card_display_string())

    def get_card_display_string(self) -> str:
        """
        Returns a string representation of the card display.
        This includes the dragon and cave cards on display.
        """
        out = "--- Card Display ---\n"
        display_list = []
        for card in self.board['card_display']['dragon_cards']:
            if card is not None:
                display_list.append((card, DRAGON_CARDS[card]['name']))
            else:
                display_list.append((card, "---"))
        out += f"- Dragon Cards: {display_list}\n"
        display_list = []
        for card in self.board['card_display']['cave_cards']:
            if card is not None:
                display_list.append((card, CAVE_CARDS[card]['text']))
            else:
                display_list.append((card, "---"))
        out += f"- Cave Cards: {display_list}\n"
        out += "---------------------"
        return out

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
            "round": 0,
            "objectives": draw_random_objectives(is_solo=False),
            # objectives are a list of tuples (tile_index, side)
            "scoring": [{"1st": [], "2nd": [], "3rd": [], "Other": []} for _ in range(4)],
            "finished_once_per_round": [False] * num_players,
        }
        for i in range(num_players):
            player = self.players[i]
            player.coins = 6
            player.egg_totals["mat_slots"] = 1
            player.dragon_hand = self.draw_random_dragon_cards(3)
            player.cave_hand = self.draw_random_cave_cards(3)


class SoloGameState(GameState):
    """
    Represents the state of a solo game.
    Inherits from GameState and adds an automa state to track and use.
    """
    ignore_cards = [47,103,114,168,170]

    def __init__(self, automa_difficulty=0, allowed_guilds=(0,1,3), ending_round=4):
        super().__init__(allowed_guilds, ending_round)
        self.automa_difficulty = automa_difficulty
        self.automa = AutomaState(self.automa_difficulty)

    def __str__(self):
        "Returns a string representation of the game state, with newlines and tabs for formatting."
        chosen_guild = self.board["guild"]["guild_index"]
        return (
            f"Game State:\n"
            f"\tTurn: {self.turn}\n"
            f"\tPhase: {self.phase}\n"
            f"\tDragon Discard: {self.dragon_discard}\n"
            f"\tCave Discard: {self.cave_discard}\n"
            f"\tEvent Queue: {self.event_queue}\n"
            f"\tGuild Board: {self.board['guild']} ({GUILD_TILES[chosen_guild]['name']})\n"
            f"{self.player}\n{self.automa}"
        )

    def all_players_passed(self):
        return self.player.passed_this_round and self.automa.passed_this_round
    
    def create_game(self, num_players=2):
        "Initializes a new automa game state. Includes random generation of objectives."
        logger.warning("Creating a new SoloGameState with 2 players.")
        self.player = PlayerState()
        self.round_start_player = 0
        self.current_player = 0
        
        # initialize board parts
        # Guild Board
        chosen_guild = random.choice(self.allowed_guilds)
        logger.info(f"Chosen guild: {chosen_guild} ({GUILD_TILES[chosen_guild]['name']})")
        self.board["guild"] = {
            "guild_index": chosen_guild,
            "ability_uses": {i: [] for i in range(1, 6)},
            "player_position": 0,
            "automa_position": 0,
            "automa_markers_ready": 1
        }
        # Remove ignored cards from the deck
        for card in SoloGameState.ignore_cards:
            self.dragon_deck.remove(card)
        # Card Display Board
        board_d_cards = self.draw_random_dragon_cards(3)
        board_c_cards = self.draw_random_cave_cards(3)
        self.board["card_display"] = {
            "dragon_cards": board_d_cards,
            "cave_cards": board_c_cards,
        }
        self.log_card_display()
        # Round Tracker Board
        logger.info("Drawing objectives:")
        objectives = draw_random_objectives(is_solo=True)
        for (idx, side) in objectives:
            logger.info(f"Objective Tile {idx} Side {side}:\n\t{OBJECTIVE_TILES[idx][side]['text']}")
        self.board["round_tracker"] = {
            "round": 0,
            "objectives": objectives,
            # objectives are a list of tuples (tile_index, side)
            "scoring": [{"1st": [], "2nd": [], "Other": []} for _ in range(4)],
            "automa_bonus": [0, 0, 0, 0],
            "finished_opr": False,
            "opr_remaining": None,
            "egg_given": False,
        }
        
        self.player.coins = 6
        self.player.egg_totals["mat_slots"] = 1
        self.player.dragon_hand = self.draw_random_dragon_cards(3)
        self.player.cave_hand = self.draw_random_cave_cards(3)
        logger.info(self.player)
    

if __name__ == "__main__":
    # Test objectives drawing
    # for i in range(10):
    #     obj = draw_random_objectives()
    #     for (idx, side) in obj:
    #         print(f"Objective Tile {idx} Side {side}:")
    #         print(OBJECTIVE_TILES[idx][side])
    #     print()
    
    # Test card drawing
    for i in range(10):
        game = GameState()
        game.create_game(2)
        print(f"Round {i+1}:")
        print("Dragon Deck:", game.dragon_deck)
        print("Cave Deck:", game.cave_deck)
        print("Dragon Discard:", game.dragon_discard)
        print("Cave Discard:", game.cave_discard)
        print("Players:")
        for player in game.players:
            print(player.dragon_hand, player.cave_hand, player.resources, player.egg_totals, player.num_dragons_played, player.score, player.guild_markers, player.coins, player.caves_played, player.dragons_played, player.times_explored)
        print()
        print("Board:")
        print(game.board)
        print()