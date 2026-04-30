import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES, AUTOMA_CARDS
from playout_compare import simulate_game, RNGOrder
import random
import math
import logging
from typing import Dict, Optional, Tuple, Union

MAX_DEPTH = 35
SIMS_PER_MOVE = 125
MIN_BUDGET = SIMS_PER_MOVE * 10
MAX_BUDGET = 40 * SIMS_PER_MOVE
AUTOMA_DIFFICULTY = 1
ENDING_ROUND = 4
UCT_CONSTANT = 1
BASE_RANDOM_CHILDREN = 15
MAX_RANDOM_CHILDREN = 30
RANDOM_CHILD_GROWTH_INTERVAL = 150
MCTS_DEBUG_INTERVAL = 300

ROLLOUT_ALGO_NAME = "strategic_objective_aware"
ROLLOUT_ALGO_KWARGS = {'dragon_weight': 12.441, 'cave_weight': 7.055, 'explore_weight': 4.927, 'pass_penalty': 1.541, 'tie_threshold': 1.535}

def set_simulation_budget(sims_per_move: int) -> None:
    """
    Update simulation budget globals from a single sims-per-move setting.
    """
    global SIMS_PER_MOVE, MIN_BUDGET, MAX_BUDGET
    SIMS_PER_MOVE = sims_per_move
    MIN_BUDGET = SIMS_PER_MOVE * 10
    MAX_BUDGET = 40 * SIMS_PER_MOVE

def set_search_config(
    sims_per_move: int,
    base_random_children: int,
    max_random_children: int,
    random_growth_interval: int,
) -> None:
    """
    Apply all runtime-tunable search parameters in one place.
    """
    global BASE_RANDOM_CHILDREN, MAX_RANDOM_CHILDREN, RANDOM_CHILD_GROWTH_INTERVAL
    set_simulation_budget(sims_per_move)
    BASE_RANDOM_CHILDREN = base_random_children
    MAX_RANDOM_CHILDREN = max_random_children
    RANDOM_CHILD_GROWTH_INTERVAL = random_growth_interval

def get_num_simulations(game_state: GameState) -> int:
    """
    Get the number of simulations to run based on the game state.
    This function will return a number between MIN_BUDGET and MAX_BUDGET.
    """
    # check if we are in setup phase
    if game_state.phase == logic.PHASE_SETUP:
        return MAX_BUDGET
    return max(min(len(game_state.current_choice) * SIMS_PER_MOVE, MAX_BUDGET), MIN_BUDGET)

class Node:
    """
    A node in the game tree.
    Each node represents a game state and a score.
    This connects to possible moves and their outcomes.
    """
    def __init__(
        self,
        game_state: GameState,
        parent=None,
        action=None,
        uct_constant=UCT_CONSTANT,
        max_random_children=MAX_RANDOM_CHILDREN,
        base_random_children=BASE_RANDOM_CHILDREN,
    ):
        self.game_state = game_state
        self.is_terminal = (game_state.phase == logic.PHASE_END_GAME)
        self.uct_constant = uct_constant
        self.max_random_children = max_random_children
        self.base_random_children = base_random_children
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
        self.outcome_visits: Dict[Union[int, Tuple[int, ...]], int] = {}
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

    def _create_child(self, action: Union[int, Tuple[int, ...]]) -> 'Node':
        new_game_state = logic.get_next_state(self.game_state, action)
        child_node = Node(
            game_state=new_game_state,
            parent=self,
            action=action,
            uct_constant=self.uct_constant,
            max_random_children=self.max_random_children,
            base_random_children=self.base_random_children,
        )
        self.children[action] = child_node
        if self.is_random:
            self.outcome_visits[action] = 0
        return child_node

    def get_random_cap(self) -> int:
        """
        Chance-node cap grows with node visits, inspired by ZeusAI section B.
        """
        growth_steps = self.visits // RANDOM_CHILD_GROWTH_INTERVAL
        cap = self.base_random_children + growth_steps
        cap = min(cap, self.max_random_children)
        return min(cap, self.num_children)
    
    def is_fully_expanded(self):
        """
        Check if this node has been fully expanded.
        This means that all children are either terminal or have been visited.
        """
        if self.is_terminal:
            return True
        if not self.is_random:
            return len(self.unchosen_actions) == 0
        return len(self.children) >= self.get_random_cap()
    
    def prune_other_branches(self, keep_action: int):
        """
        Prune all other branches of this node except for the one with the given action.
        This is used to keep the tree small and manageable.
        """
        to_remove = [action for action in self.children if action != keep_action]
        for action in to_remove:
            child = self.children[action]
            child.parent = None
            if action in self.outcome_visits:
                del self.outcome_visits[action]
            del self.children[action]

    def expand_one_child(self) -> Optional['Node']:
        """
        Expand at most one child from this node.
        """
        if self.is_terminal:
            return None

        if self.is_random:
            cap = self.get_random_cap()
            if len(self.children) >= cap:
                return None

            rand_event = self.game_state.current_random_event
            # Retry a few times to find an unseen outcome before giving up.
            for _ in range(8):
                rand_outcome = logic.get_random_outcome(self.game_state, rand_event, self.game_state.player)
                if rand_outcome not in self.children:
                    return self._create_child(rand_outcome)

            return None

        if self.unchosen_actions:
            action = random.choice(self.unchosen_actions)
            self.unchosen_actions.remove(action)
            return self._create_child(action)

        return None

    def select_child(self) -> Optional['Node']:
        """
        Selection policy for descending through existing nodes.
        """
        if self.is_terminal:
            return None
        if self.is_random:
            if not self.children:
                return None
            return random.choice(list(self.children.values()))

        if self.unchosen_actions:
            return None

        best_child = None
        best_value = float('-inf')
        parent_visits_for_log = max(1, self.visits)
        for action in sorted(self.children.keys()):
            child = self.children[action]
            if child.visits == 0:
                uct_value = float('inf')
            else:
                exploitation = child.score / child.visits
                exploration = self.uct_constant * math.sqrt(math.log(parent_visits_for_log) / child.visits)
                uct_value = exploitation + exploration
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child


def traverse(node: Node, max_depth: int, depth: int) -> Tuple[Node, int]:
    """
    Traverse the tree to find a leaf node.
    This function will select the child with the highest score.
    """
    while not node.is_terminal and depth < max_depth:
        next_node = node.select_child()
        if next_node is None:
            break

        node = next_node
        if not node.is_random:
            depth += 1

    return node, depth

def backpropagate(node: Node, result: int) -> None:
    """
    Backpropagate the result of a simulation up the tree.
    """
    while node is not None:
        node.update_score(result)
        parent = node.parent
        if parent is not None and parent.is_random and node.action in parent.outcome_visits:
            parent.outcome_visits[node.action] += 1
        node = parent

def best_action(node: Node) -> int:
    """
    Select the best action from the root node based on the scores of the children.
    """
    if not node.children:
        return None

    # Deterministic root policy: max visits, then max average value, then lowest action id.
    ranked = []
    for action, child in node.children.items():
        avg_value = (child.score / child.visits) if child.visits > 0 else float('-inf')
        ranked.append((action, child.visits, avg_value))

    ranked.sort(key=lambda row: (-row[1], -row[2], row[0]))
    return ranked[0][0]

# Update the MCTS function to use the helper methods
def MCTS(root_node: Node, max_depth: int, num_simulations: int, debug_interval: int = MCTS_DEBUG_INTERVAL) -> int:
    """
    Perform Monte Carlo Tree Search (MCTS) on the game tree.
    This function will simulate random games from the current state
    and update the tree with the results.
    """
    for sim in range(num_simulations):
        if debug_interval > 0 and sim > 0 and sim % debug_interval == 0:
            current_scores = []
            for action, child in root_node.children.items():
                if child.visits > 0:
                    current_scores.append((action, child.score / child.visits, child.visits))
            current_scores.sort(key=lambda x: (-x[2], -x[1]))
            print(f"UCT = {root_node.uct_constant} | Simulation {sim}/{num_simulations} | Top actions: {current_scores[:3]}")
            # print child node visits and types
            # for action, child in root_node.children.items():
            #     print(f"Action {action}: visits={child.visits}, score={child.score}, is_random={child.is_random}, cap={child.get_random_cap() if child.is_random else 'N/A'}")
        # Selection
        node, depth = traverse(root_node, max_depth, depth=0)

        # Expansion
        if not node.is_terminal and depth < max_depth:
            expanded = node.expand_one_child()
            if expanded is not None:
                node = expanded

        # Simulation
        result, _, _ = simulate_game(
            game_state=node.game_state,
            algo_name=ROLLOUT_ALGO_NAME,
            algo_kwargs=ROLLOUT_ALGO_KWARGS,
            display_name="mcts_rollout"
        )
        # result, _, _ = simulate_game(
        #     game_state=node.game_state,
        #     algo_name="play_dragon_cave",
        #     algo_kwargs={
        #         "entice_prob": 0.97,
        #         "excavate_prob": 0.75
        #         # "entice_prob": 0.825,
        #         # "excavate_prob": 0.85
        #     },
        #     display_name="pdc"
        # )

        # Backpropagation
        backpropagate(node, result)
    return best_action(root_node)

def run_mcts(root_node:Node, num_simulations: int) -> int:
    """
    Run MCTS on the given game state.
    This function will create a root node and perform MCTS on it.
    """
    return MCTS(root_node, MAX_DEPTH, num_simulations)

def get_next_node(node: Node, action: int) -> Node:
    """
    Get the next node in the tree based on the action taken.
    This function will create a new child node if it does not exist.
    """
    if action in node.children:
        ret = node.children[action]
        ret.parent = None
        return ret
    else:
        new_game_state = logic.get_next_state(node.game_state, action)
        return Node(
            game_state=new_game_state,
            parent=None,
            action=action,
            uct_constant=node.uct_constant,
            max_random_children=node.max_random_children,
            base_random_children=node.base_random_children,
        )

def full_log(message: str, logger: logging.Logger, echo: bool = True):
    """
    Log a message using the logger at level 'warning' and
    print it to the console for visibility.
    """
    logger.warning(message)
    if echo:
        print(message)  # Also print to console for visibility

def log_game_state(game_state: GameState, logger: logging.Logger, echo: bool = True):
    """
    Log the given game state object using the logger
    and the level 'warning'.
    """
    logger.warning(f"\n{game_state}")
    logger.warning(f"\n{game_state.get_card_display_string()}")
    logger.warning(f"\n{game_state.board['round_tracker']}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Player Score: {game_state.player.score} | Automa Score: {game_state.automa.score}")


def run_game(seed=None, uct_constant=UCT_CONSTANT, log_filename='game_uct_compare.log', echo=True):
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
    full_log(f"Running game with seed {seed} and UCT constant {uct_constant}", logger, echo=True)
    full_log(f"Simulation budget: {SIMS_PER_MOVE} sims per legal action | Random child cap: {BASE_RANDOM_CHILDREN} base, up to {MAX_RANDOM_CHILDREN} with growth {RANDOM_CHILD_GROWTH_INTERVAL}", logger, echo=True)
    # Log the initial game state
    objectives = gs.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx,side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(gs, logger, echo=echo)
    current_node = Node(
        game_state=gs,
        uct_constant=uct_constant,
        max_random_children=MAX_RANDOM_CHILDREN,
        base_random_children=BASE_RANDOM_CHILDREN,
    )

    # Simulate a game until the end
    while gs.phase != logic.PHASE_END_GAME:
        # Simulate a game until the end
        if gs.current_choice is not None:
            # we have a choice to make
            full_log(f"UCT = {uct_constant} | Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}", logger, echo=True)
            # print(f"Current choice: {gs.current_choice}")
            best_move = run_mcts(current_node, get_num_simulations(gs))
            full_log("-" * 20, logger, echo=echo)
            full_log(f"UCT = {uct_constant} | Best move: {gs.current_choice[best_move]}", logger, echo=echo)
            full_log("-" * 20, logger, echo=echo)
            current_node.prune_other_branches(best_move)
            current_node = get_next_node(current_node, best_move)
            gs = current_node.game_state
            log_game_state(gs, logger, echo=echo)
            # input("Press Enter to continue...")
        elif gs.current_random_event is not None:
            # we have a random event to resolve
            # print(f"UCT = {uct_constant} | Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}")
            full_log(f"Current random event: {gs.current_random_event}", logger, echo=True)
            # get a random outcome from the predetermined RNGOrder
            chosen_input = rng.get_random_outcome(gs, gs.current_random_event, gs.player)
            full_log("#" * 20, logger, echo=echo)
            if "automa_action" in gs.current_random_event:
                # get the number on the card drawn
                automa_string = AUTOMA_CARDS[chosen_input]["corner_id"]
                full_log(f"Chosen automa card number: {automa_string}", logger, echo=echo)
            else:
                full_log(f"Chosen random outcome: {chosen_input}", logger, echo=echo)
            full_log("#" * 20, logger, echo=echo)
            current_node.prune_other_branches(chosen_input)
            current_node = get_next_node(current_node, chosen_input)
            gs = current_node.game_state
            log_game_state(gs, logger, echo=echo)
            # input("Press Enter to continue...")
        else:
            gs = logic.get_next_state(gs, None)  # Get the next halted state
            new_node = Node(
                game_state=gs,
                uct_constant=uct_constant,
                max_random_children=MAX_RANDOM_CHILDREN,
                base_random_children=BASE_RANDOM_CHILDREN,
            )
            new_node.action = current_node.action
            current_node = new_node
    log_game_state(gs, logger, echo=echo)  # Log the final game state
    full_log(f"UCT = {uct_constant} >> Game ended. Final score: Player = {gs.player.score} | Automa = {gs.automa.score}", logger, echo=True)
    return gs.player.score, gs.automa.score

def _run_game_worker(args):
    """Helper for parallel execution."""
    seed, uct_constant, log_filename, sims_per_move, base_random_children, max_random_children, random_growth_interval = args
    try:
        set_search_config(
            sims_per_move=sims_per_move,
            base_random_children=base_random_children,
            max_random_children=max_random_children,
            random_growth_interval=random_growth_interval,
        )
        player_score, automa_score = run_game(seed=seed, uct_constant=uct_constant, log_filename=log_filename, echo=False)
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

def run_multiple_games_parallel(
    uct_constants,
    base_seeds=None,
    sims_per_move: int = SIMS_PER_MOVE,
    base_random_children: int = BASE_RANDOM_CHILDREN,
    max_random_children: int = MAX_RANDOM_CHILDREN,
    random_growth_interval: int = RANDOM_CHILD_GROWTH_INTERVAL,
):
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
            log_filename = f"compare_uct_v2_c-{uct}_seed-{seed}_{date}.log"
            tasks.append(
                (
                    seed,
                    uct,
                    log_filename,
                    sims_per_move,
                    base_random_children,
                    max_random_children,
                    random_growth_interval,
                )
            )

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_game_worker, t) for t in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(f"Completed: {result}")
            results.append(result)
    return results

# python game_uct_compare_v2.py --uct_constants 0.5 1 --seeds 42 43 44 --sims_per_move 50 --max_random_children 30 --random_growth_interval 75

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a game of Wyrmspan with MCTS.")
    parser.add_argument("--seeds", type=int, nargs='*', default=None, help="List of seeds to run per UCT constant")
    parser.add_argument("--uct_constants", type=float, nargs='*', default=None, help="List of UCT constants to try in parallel")
    parser.add_argument("--sims_per_move", type=int, default=SIMS_PER_MOVE, help="Base simulations per legal action")
    parser.add_argument("--base_random_children", type=int, default=BASE_RANDOM_CHILDREN, help="Starting chance-node child cap")
    parser.add_argument("--max_random_children", type=int, default=MAX_RANDOM_CHILDREN, help="Maximum chance-node child cap")
    parser.add_argument("--random_growth_interval", type=int, default=RANDOM_CHILD_GROWTH_INTERVAL, help="Visits per +1 chance-node cap")
    args = parser.parse_args()
    
    if args.uct_constants is None:
        args.uct_constants = [0.5, 1]
    if args.seeds is None:
        args.seeds = [None]  # Default seeds if none provided

    if args.sims_per_move < 1:
        raise ValueError("--sims_per_move must be >= 1")
    if args.base_random_children < 1:
        raise ValueError("--base_random_children must be >= 1")
    if args.max_random_children < args.base_random_children:
        raise ValueError("--max_random_children must be >= --base_random_children")
    if args.random_growth_interval < 1:
        raise ValueError("--random_growth_interval must be >= 1")

    set_search_config(
        sims_per_move=args.sims_per_move,
        base_random_children=args.base_random_children,
        max_random_children=args.max_random_children,
        random_growth_interval=args.random_growth_interval,
    )
    
    run_multiple_games_parallel(
        uct_constants=args.uct_constants,
        base_seeds=args.seeds,
        sims_per_move=args.sims_per_move,
        base_random_children=args.base_random_children,
        max_random_children=args.max_random_children,
        random_growth_interval=args.random_growth_interval,
    )