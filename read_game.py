# Read a game 'saved' in a json file
# the name of the file contains the seed and automa difficulty
# the file contains a list of actions taken in the game

import logging
logging.basicConfig(
    filename='read_game.log',
    level=logging.WARNING,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES, AUTOMA_CARDS
from playout_compare import simulate_game, simulate_multiple_games, RNGOrder
import numpy as np
from collections import defaultdict

import json
import random
from pathlib import Path

def read_game_sequence(file_path: str) -> list:
    """
    Read the game sequence from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        list: The list of actions taken in the game.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def log_game_state(game_state: GameState):
    """
    Log the given game state object using the logger
    and the level 'warning'.
    """
    logger.warning(f"\n{game_state}")
    logger.warning(f"\n{game_state.get_card_display_string()}")
    logger.warning(f"\n{game_state.board['round_tracker']}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Score: Player = {game_state.player.score} | Automa = {game_state.automa.score}")

def full_log(message: str):
    """
    Log a message using the logger at level 'warning' and
    print it to the console for visibility.
    """
    logger.warning(message)
    print(message)  # Also print to console for visibility

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a game of Wyrmspan from a saved JSON file.")
    parser.add_argument("--file", type=str, default=None, help="Path to the JSON file containing the game sequence")
    args = parser.parse_args()

    game_sequence = read_game_sequence(args.file)
    
    file_name_parts = args.file.split('_')
    SEED = int(file_name_parts[2])
    AUTOMA_DIFFICULTY = int(file_name_parts[3])

    # Example usage of NRPA
    random.seed(SEED)
    game = SoloGameState(automa_difficulty=AUTOMA_DIFFICULTY)
    game.create_game()
    rng = RNGOrder(game_state=game)
    full_log(f"Running game with seed {SEED} and automa difficulty {AUTOMA_DIFFICULTY}")
    # Log the initial game state
    objectives = game.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx, side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(game)
    # Simulate a game until the end
    while game.phase != logic.PHASE_END_GAME:
        if game.current_choice is not None:
            # Get the next action from the game_sequence
            if not game_sequence:
                full_log("No more actions left in the game sequence.")
                break
            next_action = game_sequence.pop(0)
            full_log(f"Round {game.board['round_tracker']['round'] + 1} | Player Coins: {game.player.coins} | Automa Decisions: {game.automa.num_decisions_left()}")
            full_log(f"> Points: Player = {game.player.score} | Automa = {game.automa.score}")
            full_log(f"Current choice: {game.current_choice}")
            full_log(f"Chosen action from sequence: {next_action}")
            full_log("-" * 20)
            full_log(f"Action taken: {game.current_choice[next_action]}")
            full_log("-" * 20)
            game = logic.get_next_state(game, next_action)
            log_game_state(game)
        elif game.current_random_event is not None:
            full_log(f"Round {game.board['round_tracker']['round'] + 1} | Player Coins: {game.player.coins} | Automa Decisions: {game.automa.num_decisions_left()}")
            full_log(f"Current random event: {game.current_random_event}")
            chosen_input = rng.get_random_outcome(game, game.current_random_event, game.player)
            full_log("#" * 20)
            if "automa_action" in game.current_random_event:
                automa_string = AUTOMA_CARDS[chosen_input]["corner_id"]
                full_log(f"Chosen automa card number: {automa_string}")
            else:
                full_log(f"Chosen random outcome: {chosen_input}")
            full_log("#" * 20)
            full_log("Resetting policy after random event.")
            game = logic.get_next_state(game, chosen_input)
            log_game_state(game)
        else:
            game = logic.get_next_state(game, None)
    log_game_state(game)
    full_log(f"Game ended. Final score: Player = {game.player.score} | Automa = {game.automa.score}")
