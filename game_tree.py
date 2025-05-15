import game_logic as logic
from game_states import GameState
import random

class Node:
    """
    A node in the game tree.
    Each node represents a game state and a score.
    This connects to possible moves and their outcomes.
    """
    def __init__(self, game_state: GameState, parent=None, action=None):
        self.game_state = game_state
        if game_state.current_choice is not None:
            self.is_random = False
            self.num_children = len(self.game_state.current_choice)
        elif game_state.current_random_event is not None:
            self.is_random = True
            self.num_children = logic.get_num_random_outcomes(game_state, game_state.current_random_event, game_state.current_player)
        self.parent = parent
        self.action = action # The action that led to this state
        self.children = {} # The children of this node, mapping action to Node
        self.score = 0 # The score of this node, if applicable
        self.visits = 0
        self.is_terminal = False

    def __repr__(self):
        return f"Node(action={self.action}, score={self.score})"
    
    def update_score(self, score: int):
        """
        Update the score of this node.
        This function will also increment the visit count.
        """
        self.score += score
        self.visits += 1
    
    def fully_expanded(self):
        """
        Check if all children of this node have been expanded.
        This function will return True if all children are terminal or have been visited.
        """
        return all(child.is_terminal or child.visits > 0 for child in self.children.values())
    
def pick_unvisited_child(node: Node):
    """
    Pick a child node that has not been visited yet.
    This function will return the first unvisited child.
    """
    for action, child in node.children.items():
        if not child.is_terminal:
            return child
    return None

def traverse(node: Node):
    """
    Traverse the tree to find a leaf node.
    This function will select the child with the highest score.
    """
    while node.fully_expanded():
        node = node.select_child()
    
    return pick_unvisited_child(node) or node

def expand_node(node: Node):
    """
    Expand the current node by adding all possible actions as children.
    """
    possible_actions = logic.get_main_action_choice(node.game_state)
    for action in possible_actions.get("choice", []):
        new_game_state = logic.apply_action(node.game_state, action)
        child_node = Node(game_state=new_game_state, parent=node, action=action)
        node.children[action] = child_node

def simulate_game(game_state: GameState):
    """
    Simulate a random game from the given game state until a terminal state is reached.
    Returns a score for the simulation.
    """
    while not game_state.is_terminal():
        possible_actions = logic.get_main_action_choice(game_state)
        action = random.choice(possible_actions.get("choice", []))
        game_state = logic.apply_action(game_state, action)
    return game_state.get_score()

def backpropagate(node: Node, result: int):
    """
    Backpropagate the result of a simulation up the tree.
    """
    while node is not None:
        node.update_score(result)
        backpropagate(node.parent, result)

# Update the MCTS function to use the helper methods
def MCTS(root_node: Node, max_depth: int, num_simulations: int):
    """
    Perform Monte Carlo Tree Search (MCTS) on the game tree.
    This function will simulate random games from the current state
    and update the tree with the results.
    """
    for _ in range(num_simulations):
        # Selection
        node = traverse(root_node)

        # Expansion
        if not node.is_terminal:
            expand_node(node)

        # Simulation
        result = simulate_game(node.game_state)

        # Backpropagation
        backpropagate(node, result)
    return best_action(root_node)

def best_action(node: Node):
    """
    Select the best action from the root node based on the scores of the children.
    """
    if not node.children:
        return None
    best_action = max(node.children, key=lambda action: node.children[action].visits)
    return best_action