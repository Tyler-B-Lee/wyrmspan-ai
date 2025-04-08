from game_states import GameState
import game_logic as logic

class Node:
    """
    A node in the game tree.
    Each node represents a game state and a score.
    This connects to possible moves and their outcomes.
    """
    def __init__(self, game_state: GameState, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action # The action that led to this state
        self.children = []
        self.score = 0 # The score of this node, if applicable
        self.is_terminal = False

    def __repr__(self):
        return f"Node(action={self.action}, score={self.score})"