import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES
import random
import math

MAX_SCORE = 80
MAX_DEPTH = 35
# MCTS_SIMS_PER_MOVE = 250
UCB_SIMS_PER_MOVE = 1000
DELTA_0 = 1.0
DISCOUNT_FACTOR = 1
ENDING_ROUND = 4

class Node:
    """
    A node in the game tree.
    Each node represents a game state and a score.
    This connects to possible moves and their outcomes.
    """
    def __init__(self, game_state: GameState, parent=None, action=None):
        self.game_state = game_state
        self.is_terminal = (game_state.phase == logic.PHASE_END_GAME)
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
    
    def load_all_children(self):
        """
        Load all children of this node.
        This function will create a child node for each action.
        """
        assert not self.is_terminal, "Cannot load children of a terminal node"
        assert not self.is_random, "Cannot load children of a random node"
        # we need to create a child node for each action
        for action in self.unchosen_actions:
            new_game_state = logic.get_next_state(self.game_state, action)
            child_node = Node(game_state=new_game_state, parent=self, action=action)
            self.children[action] = child_node
        # we remove the unchosen actions
        self.unchosen_actions = []
    
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
                child_node = Node(game_state=new_game_state, parent=self, action=rand_outcome)
                self.children[rand_outcome] = child_node
                return child_node
        # Not random
        # we must try each child node at least once
        if len(self.children) < self.num_children:
            action = random.choice(self.unchosen_actions)
            self.unchosen_actions.remove(action)
            # we create a new child node
            new_game_state = logic.get_next_state(self.game_state, action)
            child_node = Node(game_state=new_game_state, parent=self, action=action)
            self.children[action] = child_node
            return child_node
        # If all children are present, we use UCT to select the best one
        best_child = None
        best_value = float('-inf')
        for action, child in self.children.items():
            uct_value = child.score / (child.visits) + math.sqrt(2 * math.log(self.visits) / child.visits)
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

def simulate_game(game_state: GameState) -> int:
    """
    Simulate a random game from the given game state until a terminal state is reached.
    Returns a score for the simulation.
    """
    while game_state.phase != logic.PHASE_END_GAME:
        # check if we have a choice or random event
        if game_state.current_choice is not None:
            # we have a choice to make
            chosen_input = random.randint(0, len(game_state.current_choice) - 1)
            game_state = logic.get_next_state(game_state, chosen_input)
        elif game_state.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = logic.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            game_state = logic.get_next_state(game_state, chosen_input)
        else:
            # progress the game
            game_state = logic.get_next_state(game_state, chosen_input=None)
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 50) / 100  # Normalize the score to be between 0 and 1
    return 1 if game_state.player.score > 30 else 0

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
    print("\nSelecting best action from given node:")
    action_values = [(action, child.score / child.visits) for action, child in node.children.items()]
    action_values.sort(key=lambda x: x[1], reverse=True)
    for action, value in action_values:
        print(f"* action: {node.game_state.current_choice[action]}")
        print(f"\t-> value: {value}")
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
        if p: print(f"\nSimulation {sim + 1}/{num_simulations}")
        # Selection
        node = traverse(root_node, max_depth, depth=0)
        if p: print(f"Selected node: {node}")

        # Expansion
        if not node.is_terminal:
            node = node.select_child()
            if p: print(f"Expanded node: {node}")

        # Simulation
        result = simulate_game(node.game_state)
        if p: print(f"Simulation result: {result}")

        # Backpropagation
        backpropagate(node, result)
    return best_action(root_node)

def ucb_elim(root_node: Node, num_simulations: int) -> int:
    """
    Perform the 'Improved UCB' or 'Elimination UCB' algorithm as
    described in 'UCB Revisited: Improved Regret Bounds for the
    Stochastic Multi-Armed Bandit Problem' by Auer and Ortner (2011).

    This function will run a number of simulations and return the
    best action based on the results.

    Returns the best action to take from the root node.
    """
    root_node.load_all_children()
    delta = DELTA_0
    remaining_candidates = [i for i in range(root_node.num_children)]
    num_rounds = math.floor(0.5 * math.log2(num_simulations / math.e))
    total_sims_run = 0
    get_action_visits = lambda action: root_node.children[action].visits
    for rnd in range(num_rounds + 1):
        print(f"\n>>> Round {rnd + 1}/{num_rounds + 1}")
        d2 = delta ** 2
        runs_per_action = math.ceil((2*math.log(num_simulations*d2)) / d2)
        print(f"\t- Runs per action: {runs_per_action}")
        actions_left_this_round = [a for a in remaining_candidates if get_action_visits(a) < runs_per_action]
        print(f"\t- Actions left this round: {actions_left_this_round}")
        # run each arm until it has been chosen enough times
        print("\n>> Running simulations...")
        while len(actions_left_this_round) > 0:
            for action in actions_left_this_round:
                child_node = root_node.children[action]
                # run a simulation specifically for this child node
                node = traverse(child_node, max_depth=MAX_DEPTH, depth=0)
                # expansion
                if not node.is_terminal:
                    node = node.select_child()
                # simulation
                result = simulate_game(node.game_state)
                # backprop
                backpropagate(node, result)
                total_sims_run += 1
                if total_sims_run >= num_simulations:
                    # we have hit the simulation limit
                    print(f"\n\t> Hit simulation limit: {num_simulations} <")
                    return best_action(root_node)
                if total_sims_run % 500 == 0:
                    current_scores = [(child_node.action, (child_node.score / child_node.visits)) for child_node in root_node.children.values()]
                    best_info = max(current_scores, key=lambda x: x[1])
                    print(f"\t- Total simulations run: {total_sims_run} --- Current Best: {best_info}")
                if child_node.visits >= runs_per_action:
                    # this action has been run enough times this round
                    actions_left_this_round.remove(action)
                    print(f"Action {action} finished |", end="")
        # arm elimination
        print("\n>> Running arm elimination...")
        adjustment = math.sqrt(math.log(num_simulations * d2) / (2 * runs_per_action))
        print(f"\t- Adjustment: {adjustment}")
        # get average score of each action
        candidate_scores = []
        for action in remaining_candidates:
            child_node = root_node.children[action]
            score = (child_node.score / child_node.visits)
            candidate_scores.append((action, score))
        # remove actions that are below the cutoff score
        cutoff_score = max([s[1] for s in candidate_scores]) - adjustment
        print(f"\t- Cutoff score: {cutoff_score}")
        removal_list = [s[0] for s in candidate_scores if ((s[1] + adjustment) < cutoff_score)]
        print(f"\n> Actions to remove: {removal_list}")
        for action in removal_list:
            print(f"-{action}: {root_node.game_state.current_choice[action]}")
            print(f"\t-> Value: {root_node.children[action].score / root_node.children[action].visits}")
        remaining_candidates = [a for a in remaining_candidates if a not in removal_list]
        print(f">> Remaining candidates: {remaining_candidates}")
        # if we have only one candidate left, we can stop
        if len(remaining_candidates) == 1:
            print(f"Only one candidate left: {remaining_candidates[0]}")
            return best_action(root_node)
        # otherwise, we need to continue
        delta *= 0.5
    # we have run all rounds, we need to select the best action
    print(f"Finished all rounds. Remaining candidates: {remaining_candidates}")
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
        child_node = Node(game_state=new_game_state, parent=None, action=action)
        # node.children[action] = child_node
        return child_node

def log_game_state(game_state: GameState):
    """
    Log the given game state object using the logger
    and the level 'warning'.
    """
    logger = logging.getLogger(__name__)
    logger.warning(f"\n{game_state}")
    logger.warning(f"\n{game_state.get_card_display_string()}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Score: {game_state.player.score}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename='game_tree.log',
        # level=logging.DEBUG,
        # level=logging.INFO,
        level=logging.WARNING,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    # Example usage

    gs = SoloGameState(ending_round=ENDING_ROUND)  # Initialize the game state
    gs.create_game()
    # Log the initial game state
    objectives = gs.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx,side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(gs)
    current_node = Node(game_state=gs)  # Create the root node
    # Simulate a game until the end
    while gs.phase != logic.PHASE_END_GAME:
        # Simulate a game until the end
        if gs.current_choice is not None:
            # we have a choice to make
            print(f"Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins}")
            print(f"Current choice: {gs.current_choice}")
            # best_move = run_mcts(current_node, MCTS_SIMS_PER_MOVE * len(gs.current_choice))
            best_move = ucb_elim(current_node, UCB_SIMS_PER_MOVE * len(gs.current_choice))
            print("-" * 20)
            print(f"Best move: {gs.current_choice[best_move]}")
            print("-" * 20)
            current_node.prune_other_branches(best_move)
            current_node = get_next_node(current_node, best_move)
            gs = current_node.game_state
            log_game_state(gs)
            # input("Press Enter to continue...")
        elif gs.current_random_event is not None:
            # we have a random event to resolve
            print(f"Round {gs.board['round_tracker']['round'] + 1}")
            print(f"Current random event: {gs.current_random_event}")
            chosen_input = logic.get_random_outcome(gs, gs.current_random_event, gs.player)
            print(f"Chosen random outcome: {chosen_input}")
            current_node.prune_other_branches(chosen_input)
            current_node = get_next_node(current_node, chosen_input)
            gs = current_node.game_state
            log_game_state(gs)
            # input("Press Enter to continue...")
        else:
            gs = logic.get_next_state(gs, None)  # Get the next halted state
            new_node = Node(game_state=gs)
            new_node.action = current_node.action
            current_node = new_node
    log_game_state(gs)  # Log the final game state
    print(f"Game ended. Final score: Player = {gs.player.score} | Automa = {gs.automa.score}")