# Nested Rollout Policy Adaptation (NRPA) for a game environment
import logging
logging.basicConfig(
    filename='game_nrpa_no_rand.log',
    level=logging.WARNING,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES, AUTOMA_CARDS
from playout_compare import simulate_game, simulate_multiple_games, RNGOrder
import random
import time
import numpy as np
from collections import defaultdict

NUM_ITERATIONS = 10  # Number of iterations for NRPA
ALPHA = 1  # Alpha value for NRPA, used in policy adaptation
NRPA_LEVEL = 3  # Level of NRPA to run, can be adjusted for deeper searches

SEED = 8082025
AUTOMA_DIFFICULTY = 1 # Difficulty level for the automa, can be adjusted

def softmax(values, temperature=1.0):
    """
    Softmax function to convert values into probabilities.
    Args:
        values (list): List of values to convert.
        temperature (float): Temperature parameter for softmax.
    Returns:
        np.ndarray: Array of probabilities corresponding to the input values.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    exp_values = np.exp(np.array(values) / temperature)
    return exp_values / np.sum(exp_values)

def show_final_policy(policy: defaultdict):
    """
    Show the final policy in a readable format.
    This function prints the policy mapping actions to weights.
    Args:
        policy (defaultdict): The policy mapping actions to weights.
    """
    logger.warning("Final Policy:")
    logger.warning("----------------")
    all_items = [(value, action) for action, value in policy.items()]
    all_items.sort(reverse=True, key=lambda x: x[0])
    disp = all_items[:10] + all_items[-10:]  # Show top and bottom 10 items
    for value, action in disp:
        if abs(value) < 0.01:
            continue
        logger.warning(f"{value:.4f} -> {action}")

def get_action_code(action_index: int, game_state: GameState) -> str:
    """
    Get the action code for a given action index in the current game state.
    We can incorporate details about the state to determine the action code
    (such as round number, number of coins) and return it as a string.
    We assume that there is a current choice available in the game state.
    Args:
        action_index (int): The index of the action to get the code for.
        game_state (GameState): The current game state.
    Returns:
        str: The action code corresponding to the action index.
    """
    # if game_state.current_choice is None or action_index >= len(game_state.current_choice):
    #     raise ValueError("Invalid action index or no current choice available.")
    action = game_state.current_choice[action_index]
    # Convert the action to a string representation
    action_code = str(action)
    # Optionally, we can include more details from the game state
    round_number = game_state.board['round_tracker']['round']
    coins = game_state.player.coins
    resources = tuple(amt for amt in game_state.player.resources.values())
    disp = tuple(game_state.board["card_display"]["dragon_cards"] + 
                 game_state.board["card_display"]["cave_cards"])
    return f"{action_code} r{round_number} c{coins} res{resources} disp{disp}"

def adapt_policy(policy: defaultdict, sequence: list) -> defaultdict:
    """
    Adapt the policy based on the sequence of actions taken.
    This function updates the policy by increasing the weights of the actions taken
    and decreasing the weights of the other actions.
    Args:
        policy (defaultdict): The current policy mapping actions to weights.
        sequence (list): A list of tuples where each tuple contains the action index and the game state.
    Returns:
        defaultdict: The updated policy mapping actions to weights.
    """
    new_policy = policy.copy()
    for action_index, gs in sequence:
        action_strings = [get_action_code(a, gs) for a in range(len(gs.current_choice))]
        action_name = action_strings[action_index]
        new_policy[action_name] += ALPHA
        z = sum(np.exp(policy[a]) for a in action_strings)
        for a in action_strings:
            new_policy[a] -= ALPHA * np.exp(policy[a]) / z
    return new_policy

def NRPA_no_rand(level: int, policy: defaultdict, root: GameState, rng: RNGOrder) -> tuple:
    """
    Nested Rollout Policy Algorithm (NRPA).
    This function performs a search at the given level in the game tree.
    If level is 0, it performs a base rollout policy.
    If level is greater than 0, it performs a nested search.
    Args:
        level (int): The level of the search.
        policy: The policy to use for rollouts.
        root (Node): The root node of the search.
    Returns:
        tuple: A tuple containing the best score, the sequence of actions taken,
                and the updated policy.
    """
    if level == 0:
        # perform base rollout policy
        gs = root
        this_rng = rng.get_copy()  # Get a copy of the RNGOrder for this simulation
        sequence = []
        # outcome_prob = 1.0  # Probability of the outcome
        # logger.warning("- Starting simulation from root node:")
        while gs.phase != logic.PHASE_END_GAME:
            # print(f"{gs.board['round_tracker']['round']} - {gs.player.coins} | ", end="")
            # check if we have a choice or random event
            if gs.current_choice is not None:
                # find weights of choices using the policy
                weights = [policy.get(get_action_code(i, gs), 0) for i in range(len(gs.current_choice))]
                # apply softmax to get probabilities
                probabilities = softmax(weights, temperature=1.0)
                # choose an action based on the probabilities
                chosen_action_index = np.random.choice(len(gs.current_choice), p=probabilities)
                sequence.append((chosen_action_index, gs))
                # progress the game
                gs = logic.get_next_state(gs, chosen_action_index)
            elif gs.current_random_event is not None:
                # we have a random event to resolve
                # num_outcomes = logic.get_num_random_outcomes(gs, gs.current_random_event, gs.player)
                # outcome_prob *= 1.0 / num_outcomes  # Update the probability of the outcome
                chosen_input = this_rng.get_random_outcome(gs, gs.current_random_event, gs.player)
                gs = logic.get_next_state(gs, chosen_input)
            else:
                # progress the game
                gs = logic.get_next_state(gs, chosen_input=None)
        # we reached a terminal state, return the score
        # Standard Score: Match or exceed Automa score
        # - Secondary: Get as many points as possible
        score = (gs.player.score ** 2) / 40000
        if gs.player.score >= gs.automa.score:
            score += 0.75

        # Other Objectives to Maximize
        # get as many cached resources as possible
        # score = 0
        # for cache_list in gs.player.cached_resources.values():
        #     score += sum(num_caches for cache_dict in cache_list for num_caches in cache_dict.values())
        return score, sequence, policy
    else:
        # perform nested search
        best_score = float('-inf')
        sequence = []
        for i in range(NUM_ITERATIONS):
            if level == NRPA_LEVEL and i % 5 == 0:
                logger.warning(f">>> Iteration {i+1}/{NUM_ITERATIONS} at level {level}")
            
            result, new_seq, _new_policy = NRPA_no_rand(level - 1, policy, root, rng)
            
            if result >= best_score:
                if level == NRPA_LEVEL:
                    logger.warning(f"\tNew best score found: {result} (Iteration {i+1})")
                # logger.warning(f"\tNew sequence: {new_seq}")
                best_score = result
                sequence = new_seq
            policy = adapt_policy(policy, sequence)

            if level == NRPA_LEVEL and i % 10 == 0 and i > 0:
                date_str = time.strftime("%Y%m%d-%H%M%S")
                save_sequence(sequence, f"sequence_{SEED}_{AUTOMA_DIFFICULTY}_{date_str}_{best_score}.json")
        return best_score, sequence, policy

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

def save_sequence(sequence: list, filename: str):
    """
    Save the sequence of actions to a file.
    Args:
        sequence (list): The sequence of actions to save.
        filename (str): The name of the file to save the sequence to.
    """
    import json
    sequence = [action_index for action_index, _ in sequence]  # Extract only action indices
    with open('saved_sequences/' + filename, 'w') as f:
        json.dump(sequence, f, indent=4)

if __name__ == "__main__":
    # Example usage of NRPA
    random.seed(SEED)
    game = SoloGameState(automa_difficulty=AUTOMA_DIFFICULTY)
    game.create_game()
    rng = RNGOrder(game_state=game)

    policy = defaultdict(int)  # Start with a uniform policy
    full_log(f"Running game with seed {SEED} and NRPA level {NRPA_LEVEL}")
    # Log the initial game state
    objectives = game.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx,side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(game)
    # Simulate a game until the end
    while game.phase != logic.PHASE_END_GAME:
        # Simulate a game until the end
        if game.current_choice is not None:
            # we have a choice to make
            full_log(f"Round {game.board['round_tracker']['round'] + 1} | Player Coins: {game.player.coins} | Automa Decisions: {game.automa.num_decisions_left()}")
            full_log(f"> Points: Player = {game.player.score} | Automa = {game.automa.score}")
            full_log(f"Current choice: {game.current_choice}")
            # run NRPA to find the best move
            best_score, sequence, policy = NRPA_no_rand(NRPA_LEVEL, policy, game, rng)
            show_final_policy(policy)
            full_log(f"Sequence of actions taken: {sequence}")
            # save the sequence to a file
            date_str = time.strftime("%Y%m%d-%H%M%S")
            save_sequence(sequence, f"sequence_{SEED}_{AUTOMA_DIFFICULTY}_{date_str}_{best_score}.json")
            # get probabilities for the actions in root node
            action_strings = [get_action_code(a, game) for a in range(len(game.current_choice))] if game.current_choice else []
            action_weights = [policy.get(action, 0) for action in action_strings]
            probabilities = softmax(action_weights, temperature=1.0)
            logger.warning("Action probabilities:")
            for action, prob in zip(action_strings, probabilities):
                logger.warning(f"{action}: {prob:.6f}")
            # choose the best action based on the policy
            best_move = np.argmax(action_weights)

            full_log("-" * 20)
            full_log(f"Best move: {game.current_choice[best_move]}")
            full_log("-" * 20)
            game = logic.get_next_state(game, best_move)  # Get the next state after the best move
            log_game_state(game)
            # input("Press Enter to continue...")
        elif game.current_random_event is not None:
            # we have a random event to resolve
            full_log(f"Round {game.board['round_tracker']['round'] + 1} | Player Coins: {game.player.coins} | Automa Decisions: {game.automa.num_decisions_left()}")
            full_log(f"Current random event: {game.current_random_event}")
            # chosen_input = logic.get_random_outcome(gs, gs.current_random_event, gs.player)
            chosen_input = rng.get_random_outcome(game, game.current_random_event, game.player)
            full_log("#" * 20)
            if "automa_action" in game.current_random_event:
                # get the number on the card drawn
                automa_string = AUTOMA_CARDS[chosen_input]["corner_id"]
                full_log(f"Chosen automa card number: {automa_string}")
            else:
                full_log(f"Chosen random outcome: {chosen_input}")
            full_log("#" * 20)
            # reset the policy after a random event
            policy = defaultdict(int)  # Reset the policy after a random event
            full_log("Resetting policy after random event.")
            game = logic.get_next_state(game, chosen_input)
            log_game_state(game)
            # input("Press Enter to continue...")
        else:
            game = logic.get_next_state(game, None)  # Get the next halted state
    log_game_state(game)  # Log the final game state
    full_log(f"Game ended. Final score: Player = {game.player.score} | Automa = {game.automa.score}")
