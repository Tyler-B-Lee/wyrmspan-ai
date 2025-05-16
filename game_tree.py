import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES
import random
import math

MAX_SCORE = 40
MAX_DEPTH = 50
SIMS_PER_MOVE = 1000
DISCOUNT_FACTOR = 0.99
ENDING_ROUND = 2

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
    
    def select_child(self):
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
    

def traverse(node: Node, max_depth: int, depth: int):
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

def simulate_game(game_state: GameState):
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
    return game_state.player.score / MAX_SCORE

def backpropagate(node: Node, result: int):
    """
    Backpropagate the result of a simulation up the tree.
    """
    node.update_score(result)
    if node.parent is not None:
        backpropagate(node.parent, result * DISCOUNT_FACTOR)  # Discount the score for the parent node

# Update the MCTS function to use the helper methods
def MCTS(root_node: Node, max_depth: int, num_simulations: int):
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

def best_action(node: Node):
    """
    Select the best action from the root node based on the scores of the children.
    """
    print("\nSelecting best action from given node:")
    best_action = None
    best_value = float('-inf')
    action_values = [(action, child.score / child.visits) for action, child in node.children.items()]
    action_values.sort(key=lambda x: x[1], reverse=True)
    for action, value in action_values:
        print(f"* action: {node.game_state.current_choice[action]}")
        print(f"\t-> value: {value}")
        if value > best_value:
            best_value = value
            best_action = action
    return best_action

def run_mcts(game_state: GameState, num_simulations: int):
    """
    Run MCTS on the given game state.
    This function will create a root node and perform MCTS on it.
    """
    root_node = Node(game_state)
    best_action = MCTS(root_node, MAX_DEPTH, num_simulations)
    return best_action

def log_game_state(game_state: GameState):
    """
    Log the given game state object using the logger
    and the level 'warning'.
    """
    logger = logging.getLogger(__name__)
    logger.warning(f"GameState: {game_state}")
    logger.warning(game_state.get_card_display_string())
    logger.warning(f"Phase: {game_state.phase}")
    logger.warning(f"Score: {game_state.player.score}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename='file.log',
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
    # Simulate a game until the end
    while gs.phase != logic.PHASE_END_GAME:
        # Simulate a game until the end
        if gs.current_choice is not None:
            # we have a choice to make
            print(f"Current choice: {gs.current_choice}")
            best_move = run_mcts(gs, SIMS_PER_MOVE)
            print(f"Best move: {gs.current_choice[best_move]}")
            gs = logic.get_next_state(gs, best_move)
            # wait for operator input
            log_game_state(gs)
            input("Press Enter to continue...")
        elif gs.current_random_event is not None:
            # we have a random event to resolve
            print(f"Current random event: {gs.current_random_event}")
            chosen_input = logic.get_random_outcome(gs, gs.current_random_event, gs.player)
            print(f"Chosen random outcome: {chosen_input}")
            gs = logic.get_next_state(gs, chosen_input)
            # wait for operator input
            log_game_state(gs)
            input("Press Enter to continue...")
        else:
            gs = logic.get_next_state(gs, None)  # Get the next halted state
    log_game_state(gs)  # Log the final game state
    print(f"Game ended. Final score: {gs.player.score}")