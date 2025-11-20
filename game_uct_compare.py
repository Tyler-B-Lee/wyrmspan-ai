import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES, AUTOMA_CARDS
from playout_compare import simulate_game, RNGOrder
import random
import math
import logging

MAX_DEPTH = 35
SIMS_PER_MOVE = 250
MIN_BUDGET = SIMS_PER_MOVE * 20
MAX_BUDGET = 35 * SIMS_PER_MOVE
ENDING_ROUND = 4
UCT_CONSTANT = 1
AUTOMA_DIFFICULTY = 2

def get_num_simulations(game_state: GameState) -> int:
    """
    Get the number of simulations to run based on the game state.
    This function will return a number between MIN_BUDGET and MAX_BUDGET.
    """
    return max(min(len(game_state.current_choice) * SIMS_PER_MOVE, MAX_BUDGET), MIN_BUDGET)

class Node:
    """
    A node in the game tree.
    Each node represents a game state and a score.
    This connects to possible moves and their outcomes.
    """
    def __init__(self, game_state: GameState, parent=None, action=None, uct_constant=UCT_CONSTANT):
        self.game_state = game_state
        self.is_terminal = (game_state.phase == logic.PHASE_END_GAME)
        self.uct_constant = uct_constant
        if game_state.current_choice is not None:
            self.is_random = False
            self.num_children = len(self.game_state.current_choice)
            self.unchosen_actions = [action for action in range(self.num_children)]
        elif game_state.current_random_event is not None:
            self.is_random = True
            self.num_children = logic.get_num_random_outcomes(game_state, game_state.current_random_event, game_state.player)
            self.unchosen_actions = []
        else:
            self.is_random = False
            self.num_children = 0
            self.unchosen_actions = []
        self.parent = parent
        self.action = action # The action that led to this state
        self.children = {} # The children of this node, mapping action to Node
        self.score = 0 # The score of this node, if applicable
        self.visits = 0

    def __repr__(self):
        return (
            f"Node: action={self.action}, score={self.score}, visits={self.visits})\n"
            f"GameState(phase={self.game_state.phase}, player={self.game_state.current_player})"
            f"\tChildren: {len(self.children)}\tUnchosen actions: {len(self.unchosen_actions)}\n"
            f"\tIs random: {self.is_random}"
        )
    
    def update_score(self, score: int):
        """
        Update the score of this node.
        This function will also increment the visit count.
        """
        self.score += score
        self.visits += 1
    
    def is_fully_expanded(self):
        """
        Check if this node has been fully expanded.
        This means that all children are either terminal or have been visited.
        """
        if not self.is_random:
            return len(self.children) == self.num_children
        else:
            return True
    
    def prune_other_branches(self, keep_action: int):
        """
        Prune all other branches of this node except for the one with the given action.
        This is used to keep the tree small and manageable.
        """
        to_remove = [action for action in self.children if action != keep_action]
        for action in to_remove:
            child = self.children[action]
            child.parent = None
            del self.children[action]
    
    def select_child(self) -> 'Node':
        """
        Select a child node based on the UCT (Upper Confidence Bound for Trees) algorithm.
        This function will return the child with the highest UCT value.
        """
        if self.is_terminal:
            return self
        if self.is_random:
            # we randomly choose a child node
            rand_event = self.game_state.current_random_event
            rand_outcome = logic.get_random_outcome(self.game_state, rand_event, self.game_state.player)
            if rand_outcome in self.children:
                return self.children[rand_outcome]
            else:
                # If the random outcome is not in the children, we need to create it
                new_game_state = logic.get_next_state(self.game_state, rand_outcome)
                child_node = Node(game_state=new_game_state, parent=self, action=rand_outcome, uct_constant=self.uct_constant)
                self.children[rand_outcome] = child_node
                return child_node
        # Not random
        # we must try each child node at least once
        if len(self.children) < self.num_children:
            action = random.choice(self.unchosen_actions)
            self.unchosen_actions.remove(action)
            # we create a new child node
            new_game_state = logic.get_next_state(self.game_state, action)
            child_node = Node(game_state=new_game_state, parent=self, action=action, uct_constant=self.uct_constant)
            self.children[action] = child_node
            return child_node
        # If all children are present, we use UCT to select the best one
        best_child = None
        best_value = float('-inf')
        for action, child in self.children.items():
            uct_value = child.score / (child.visits) + self.uct_constant * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child
    

def traverse(node: Node, max_depth: int, depth: int) -> Node:
    """
    Traverse the tree to find a leaf node.
    This function will select the child with the highest score.
    """
    while node.is_fully_expanded() and not node.is_terminal:
        node = node.select_child()
        if not node.is_random:
            depth += 1
        if depth >= max_depth:
            break
    
    return node

def backpropagate(node: Node, result: int) -> None:
    """
    Backpropagate the result of a simulation up the tree.
    """
    node.update_score(result)
    if node.parent is not None:
        # backpropagate(node.parent, result * DISCOUNT_FACTOR)  # Discount the score for the parent node
        backpropagate(node.parent, result)  # Discount the score for the parent node

def best_action(node: Node) -> int:
    """
    Select the best action from the root node based on the scores of the children.
    """
    print(f"\nUCT = {node.uct_constant} | Selecting best action from given node:")
    action_values = [(action, child.score / child.visits) for action, child in node.children.items()]
    action_values.sort(key=lambda x: x[1], reverse=True)
    for action, value in action_values:
        print(f"UCT = {node.uct_constant} | * action: {node.game_state.current_choice[action]}")
        print(f"\t{node.uct_constant} | -> value: {value}")
    return action_values[0][0] if action_values else None

# Update the MCTS function to use the helper methods
def MCTS(root_node: Node, max_depth: int, num_simulations: int) -> int:
    """
    Perform Monte Carlo Tree Search (MCTS) on the game tree.
    This function will simulate random games from the current state
    and update the tree with the results.
    """
    for sim in range(num_simulations):
        p = (sim % 500 == 0)
        if p:
            current_scores = [(action, child.score / child.visits) for action, child in root_node.children.items()]
            current_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"\nUCT = {root_node.uct_constant} | Simulation {sim + 1}/{num_simulations}")
            print(f"\t{root_node.uct_constant} | Top 3 actions: {current_scores[:3]}")
        # Selection
        node = traverse(root_node, max_depth, depth=0)
        # if p: print(f"Selected node: {node}")

        # Expansion
        if not node.is_terminal:
            node = node.select_child()
            # if p: print(f"Expanded node: {node}")

        # Simulation
        result, _, _ = simulate_game(
            game_state=node.game_state,
            algo_name="play_dragon_cave",
            algo_kwargs={
                "entice_prob": 0.97,
                "excavate_prob": 0.75
                # "entice_prob": 0.825,
                # "excavate_prob": 0.85
            },
            display_name="pdc"
        )
        # if p: print(f"Simulation result: {result}")

        # Backpropagation
        backpropagate(node, result)
    return best_action(root_node)

def run_mcts(root_node:Node, num_simulations: int) -> int:
    """
    Run MCTS on the given game state.
    This function will create a root node and perform MCTS on it.
    """
    best_action = MCTS(root_node, MAX_DEPTH, num_simulations)
    return best_action

def get_next_node(node: Node, action: int) -> Node:
    """
    Get the next node in the tree based on the action taken.
    This function will create a new child node if it does not exist.
    """
    if action in node.children:
        # we have already visited this child
        print(f"> Child node already exists for outcome {action}")
        ret = node.children[action]
        print(f"\t> Number of visits: {ret.visits}")
        ret.parent = None
        return ret
    else:
        # we need to create a new child node
        print(f"> Creating new child node for outcome {action}")
        new_game_state = logic.get_next_state(node.game_state, action)
        child_node = Node(game_state=new_game_state, parent=None, action=action, uct_constant=node.uct_constant)
        # node.children[action] = child_node
        return child_node

def full_log(message: str, logger: logging.Logger):
    """
    Log a message using the logger at level 'warning' and
    print it to the console for visibility.
    """
    logger.warning(message)
    print(message)  # Also print to console for visibility

def log_game_state(game_state: GameState, logger: logging.Logger):
    """
    Log the given game state object using the logger
    and the level 'warning'.
    """
    logger.warning(f"\n{game_state}")
    logger.warning(f"\n{game_state.get_card_display_string()}")
    logger.warning(f"\n{game_state.board['round_tracker']}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Player Score: {game_state.player.score} | Automa Score: {game_state.automa.score}")


def run_game(seed=None, uct_constant=UCT_CONSTANT, log_filename='game_uct_compare.log'):
    # set seed for reproducibility
    random.seed(seed)

    logging.basicConfig(
        filename=("logs/" + log_filename),
        # level=logging.DEBUG,
        # level=logging.INFO,
        level=logging.WARNING,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    gs = SoloGameState(automa_difficulty=AUTOMA_DIFFICULTY, ending_round=ENDING_ROUND)
    gs.create_game()
    rng = RNGOrder(game_state=gs)
    full_log(f"Running game with seed {seed} and UCT constant {uct_constant}", logger)
    # Log the initial game state
    objectives = gs.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx,side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(gs, logger)
    current_node = Node(game_state=gs, uct_constant=uct_constant)  # Create the root node

    # Simulate a game until the end
    while gs.phase != logic.PHASE_END_GAME:
        # Simulate a game until the end
        if gs.current_choice is not None:
            # we have a choice to make
            full_log(f"UCT = {uct_constant} | Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}", logger)
            # print(f"Current choice: {gs.current_choice}")
            best_move = run_mcts(current_node, get_num_simulations(gs))
            full_log("-" * 20, logger)
            full_log(f"UCT = {uct_constant} | Best move: {gs.current_choice[best_move]}", logger)
            full_log("-" * 20, logger)
            current_node.prune_other_branches(best_move)
            current_node = get_next_node(current_node, best_move)
            gs = current_node.game_state
            log_game_state(gs, logger)
            # input("Press Enter to continue...")
        elif gs.current_random_event is not None:
            # we have a random event to resolve
            # print(f"UCT = {uct_constant} | Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}")
            full_log(f"Current random event: {gs.current_random_event}", logger)
            # get a random outcome from the predetermined RNGOrder
            chosen_input = rng.get_random_outcome(gs, gs.current_random_event, gs.player)
            full_log("#" * 20, logger)
            if "automa_action" in gs.current_random_event:
                # get the number on the card drawn
                automa_string = AUTOMA_CARDS[chosen_input]["corner_id"]
                full_log(f"Chosen automa card number: {automa_string}", logger)
            else:
                full_log(f"Chosen random outcome: {chosen_input}", logger)
            full_log("#" * 20, logger)
            current_node.prune_other_branches(chosen_input)
            current_node = get_next_node(current_node, chosen_input)
            gs = current_node.game_state
            log_game_state(gs, logger)
            # input("Press Enter to continue...")
        else:
            gs = logic.get_next_state(gs, None)  # Get the next halted state
            new_node = Node(game_state=gs, uct_constant=uct_constant)
            new_node.action = current_node.action
            current_node = new_node
    log_game_state(gs, logger)  # Log the final game state
    full_log(f"UCT = {uct_constant} >> Game ended. Final score: Player = {gs.player.score} | Automa = {gs.automa.score}", logger)
    return gs.player.score, gs.automa.score

def _run_game_worker(args):
    """Helper for parallel execution."""
    seed, uct_constant, log_filename = args
    try:
        player_score, automa_score = run_game(seed=seed, uct_constant=uct_constant, log_filename=log_filename)
        return {
            "uct_constant": uct_constant,
            "seed": seed,
            "player_score": player_score,
            "automa_score": automa_score,
            "log_filename": log_filename
        }
    except Exception as e:
        return {
            "uct_constant": uct_constant,
            "seed": seed,
            "error": str(e),
            "log_filename": log_filename
        }

def run_multiple_games_parallel(uct_constants, base_seeds=None):
    """
    Run multiple games in parallel, one for each uct_constant (optionally multiple runs per constant).
    Each run will have its own log file.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    tasks = []
    for seed in base_seeds:
        for uct in uct_constants:
            date = time.strftime("%Y%m%d-%H%M%S")
            log_filename = f"compare_uct_c-{uct}_seed-{seed}_{date}.log"
            tasks.append((seed, uct, log_filename))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_game_worker, t) for t in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(f"Completed: {result}")
            results.append(result)
    return results

# python game_uct_compare.py --seed 1 --uct_constants 0.5 1 1.414213562 2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a game of Wyrmspan with MCTS.")
    parser.add_argument("--seeds", type=int, nargs='*', default=None, help="List of seeds to run per UCT constant")
    parser.add_argument("--uct_constants", type=float, nargs='*', default=None, help="List of UCT constants to try in parallel")
    args = parser.parse_args()
    
    if args.uct_constants is None:
        args.uct_constants = [0.5, 1]
    if args.seeds is None:
        args.seeds = [None]  # Default seeds if none provided
    
    run_multiple_games_parallel(
        uct_constants=args.uct_constants,
        base_seeds=args.seeds
    )