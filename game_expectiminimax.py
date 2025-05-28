import game_logic as logic
from game_states import GameState, SoloGameState, OBJECTIVE_TILES, DRAGON_CARDS
import random
import math
import time
import itertools

MAX_SCORE = 100
MAX_BUDGET = 12500
SIMS_PER_MOVE = 500
ENDING_ROUND = 4

# Constants for the expectiminimax algorithm
MAX_DEPTH = 5
BEST_ACTION_SO_FAR = None
EVAL_SCORE_SO_FAR = -math.inf
MAX_RANDOM_OUTCOMES = 50

def get_num_simulations(game_state: GameState) -> int:
    """
    Get the number of simulations to run based on the game state.
    This function will return a number between 1 and MAX_BUDGET.
    """
    return min(len(game_state.current_choice) * SIMS_PER_MOVE, MAX_BUDGET)

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
    # return 1 if game_state.player.score > 30 else 0
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    if game_state.player.score >= game_state.automa.score:
        return (5000 + (game_state.player.score - game_state.automa.score) ** 2) / 10000
    else:
        return (5000 - (game_state.automa.score - game_state.player.score) ** 2) / 10000

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
    logger.warning(f"\n{game_state.board['round_tracker']}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Score: {game_state.player.score}")

# expectiminimax algorithm with Monte Carlo Tree Search (MCTS)
def score_heuristic(game_state: GameState) -> float:
    """
    Calculate a heuristic score for the given game state.
    This function will return a score between 0 and 1.
    """
    # Example heuristic: score based on player score and automa score
    if game_state.phase == logic.PHASE_END_GAME:
        return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    else:
        return (game_state.player.score - 100 + 70) / 140  # Normalize the score to be between 0 and 1

def expectiminimax_value(node: Node, current_depth: int, max_depth: int) -> float:
    """
    Expectiminimax algorithm to calculate the value of the game state.
    This function will return the expected value of the game state.
    """
    game_state = node.game_state
    if current_depth >= max_depth or game_state.phase == logic.PHASE_END_GAME:
        return score_heuristic(game_state)

    if game_state.current_choice is not None:
        # we have a choice to make
        return get_max_value(node, current_depth, max_depth)
    elif game_state.current_random_event is not None:
        # we have a random event to resolve
        return get_expected_value(node, current_depth, max_depth)
    else:
        # progress the game
        new_game_state = logic.get_next_state(game_state, None)
        new_node = Node(game_state=new_game_state, parent=node.parent, action=node.action)
        return expectiminimax_value(new_node, current_depth + 1, max_depth)

def get_max_value(node: Node, current_depth: int, max_depth: int):
    """
    Get the maximum value of the game state.
    This function will return the maximum score for the player of each
    future state reachable by the actions at this state.
    """
    v = -math.inf
    game_state = node.game_state
    actions = game_state.current_choice
    
    # we update the best action so far if we are at depth 0
    if current_depth == 0:
        global BEST_ACTION_SO_FAR, EVAL_SCORE_SO_FAR
        local_best_action = None

        for i, action in enumerate(actions):
            print(f"  Evaluating action {i}: {action}", end=' ')
            # check if child node already exists
            if i not in node.children:
                # we create a new child node for this action
                next_game_state = logic.get_next_state(game_state, i)
                child_node = Node(game_state=next_game_state, parent=node, action=i)
                node.children[i] = child_node
            else:
                child_node = node.children[i]
            # we get the value of this child node
            child_value = expectiminimax_value(child_node, current_depth + 1, max_depth)
            print(f"-> value: {child_value}")

            if child_value > v:
                v = child_value
                local_best_action = i
        # we update the best action so far
        if v > EVAL_SCORE_SO_FAR:
            BEST_ACTION_SO_FAR = local_best_action
            EVAL_SCORE_SO_FAR = v
            print(f"  > New best action: {actions[local_best_action]} with value {v}\n")
        return v
    # non-root node
    for i, action in enumerate(actions):
        # we create a new child node for this action
        if i not in node.children:
            next_game_state = logic.get_next_state(game_state, i)
            child_node = Node(game_state=next_game_state, parent=node, action=i)
            node.children[i] = child_node
        else:
            child_node = node.children[i]
        
        # we get the value of this child node
        child_value = expectiminimax_value(child_node, current_depth + 1, max_depth)
        v = max(v, child_value)
    return v

def get_expected_value(node: Node, current_depth: int, max_depth: int) -> float:
    """
    Calculate the expected value of a chance node (random event).
    """
    expected_value = 0.0
    game_state = node.game_state
    event = game_state.current_random_event
    
    num_outcomes = logic.get_num_random_outcomes(game_state, game_state.current_random_event, game_state.player)
    if num_outcomes > MAX_RANDOM_OUTCOMES:
        outcome_list = []
        for _ in range(MAX_RANDOM_OUTCOMES):
            chosen_input = logic.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            outcome_list.append(chosen_input)
        num_outcomes = MAX_RANDOM_OUTCOMES
    elif "automa_action" in event:
        outcome_list = game_state.automa.decision_deck
    elif ("top_deck_reveal" in event) or ("refill_dragon_display" in event) or ("gain_dragon" in event) or ("tuck_from" in event):
        # we return one card sampled from the dragon deck
        outcome_list = game_state.dragon_deck
    elif ("refill_cave_display" in event) or ("play_cave" in event) or ("gain_cave" in event):
        # we return one card sampled from the cave deck
        outcome_list = game_state.cave_deck
    elif "draw_decision" in event:
        # we return a number of cards sampled from the dragon deck
        num_cards = event["draw_decision"]["amount"]
        if num_cards == "shy_this_cave":
            # we find the amount to draw
            cave_name, col = event["coords"]
            num_cards = 0
            for col in range(4):
                dragon_id = game_state.player.dragons_played[cave_name][col]
                if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == "Shy":
                    num_cards += 1
        outcome_list = itertools.combinations(game_state.dragon_deck, num_cards)
    
    for outcome in outcome_list:
        # we create a new child node for this outcome
        if outcome not in node.children:
            next_game_state = logic.get_next_state(game_state, outcome)
            child_node = Node(game_state=next_game_state, parent=node, action=outcome)
            node.children[outcome] = child_node
        else:
            child_node = node.children[outcome]
        
        # we get the value of this child node
        child_value = expectiminimax_value(child_node, current_depth + 1, max_depth)
        expected_value += child_value / num_outcomes
    return expected_value

def iterative_deepening_expectiminimax(root_node: Node, max_depth: int) -> int:
    """
    Iterative deepening expectiminimax algorithm to find the best action.
    This function will return the best action to take from the root node.
    """
    global BEST_ACTION_SO_FAR, EVAL_SCORE_SO_FAR
    BEST_ACTION_SO_FAR = None
    EVAL_SCORE_SO_FAR = -math.inf

    for depth in range(1, max_depth + 1):
        print(f"@ Running expectiminimax at depth {depth}...")
        expectiminimax_value(root_node, 0, depth)
    
    return BEST_ACTION_SO_FAR

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename='game_expectiminimax.log',
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
            print(f"Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}")
            print(f"Current choice: {gs.current_choice}")
            best_move = iterative_deepening_expectiminimax(current_node, MAX_DEPTH)
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
            print(f"Round {gs.board['round_tracker']['round'] + 1} | Player Coins: {gs.player.coins} | Automa Decisions: {gs.automa.num_decisions_left()}")
            print(f"Current random event: {gs.current_random_event}")
            chosen_input = logic.get_random_outcome(gs, gs.current_random_event, gs.player)
            print("#" * 20)
            print(f"Chosen random outcome: {chosen_input}")
            print("#" * 20)
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
