import copy
import logging
import random
import time
from game_states import GameState, SoloGameState, PlayerState, DRAGON_CARDS
import game_logic as logic

# File for comparing different algorithms for simulating game playouts in Wyrmspan.

DEFAULT_BATCH_SIZE = 32


def _configure_worker_logging(level=logging.WARNING):
    """Keep worker processes quiet so logging does not dominate simulation time."""
    logging.getLogger().setLevel(level)


def _chunk_count(total_items: int, batch_size: int):
    batch_size = max(1, batch_size)
    for start in range(0, total_items, batch_size):
        yield start, min(start + batch_size, total_items)


def _run_seeded_game_batch(game_state: GameState, algo_name, algo_kwargs, display_name, seeds, worker_log_level=logging.WARNING):
    """Run several simulations from the same starting state inside one worker."""
    _configure_worker_logging(worker_log_level)
    total_score = 0
    total_time = 0.0
    for seed in seeds:
        score, _, elapsed = simulate_game(copy.deepcopy(game_state), algo_name, algo_kwargs, display_name, seed)
        total_score += score
        total_time += elapsed
    return total_score, display_name, total_time, len(seeds)


def _run_simulation_task_batch(task_batch, worker_log_level=logging.WARNING):
    """Run a batch of independent simulation tasks inside one worker."""
    _configure_worker_logging(worker_log_level)
    results = []
    for game_state, algo_name, algo_kwargs, display_name, seed in task_batch:
        results.append(simulate_game(game_state, algo_name, algo_kwargs, display_name, seed))
    return results

class RNGOrder:
    """
    A class to manage the order of random events in a game state.
    This is used to ensure that the same random events are chosen in the same order
    during actual games, which is important for reproducibility.

    The order is maintained as lists of outcomes.
    We can create this object at the start of a game along with the game state,
    then use it whenever a random event needs to be resolved.
    """
    def __init__(self, game_state: GameState):
        # assume the game_state has just been initialized,
        # create_game has been run, and nothing has happened yet
        self.order = []
        self.game_state = game_state
        # Initialize the RNG order with the current game state
        # setup the dragon and cave decks
        dragon_deck = game_state.dragon_deck.copy()
        random.shuffle(dragon_deck)
        cave_deck = game_state.cave_deck.copy()
        random.shuffle(cave_deck)
        # setup the automa decks for each round
        automa_decks = []
        for r in range(4):
            rnd_automa_deck = game_state.automa.decision_deck.copy()
            random.shuffle(rnd_automa_deck)
            rnd_automa_deck.pop()  # remove the last card, as only 7 are used in each round
            automa_decks.extend(rnd_automa_deck)

        # Keep deck orders immutable and only mutate tiny cursor indices.
        self.dragon_deck = tuple(dragon_deck)
        self.cave_deck = tuple(cave_deck)
        self.automa_decks = tuple(automa_decks)
        self._dragon_idx = len(self.dragon_deck) - 1
        self._cave_idx = len(self.cave_deck) - 1
        self._automa_idx = len(self.automa_decks) - 1

    @classmethod
    def _from_existing(cls, source: "RNGOrder") -> "RNGOrder":
        """Fast copy constructor that shares immutable deck order data."""
        new_obj = cls.__new__(cls)
        new_obj.order = source.order
        new_obj.game_state = source.game_state
        new_obj.dragon_deck = source.dragon_deck
        new_obj.cave_deck = source.cave_deck
        new_obj.automa_decks = source.automa_decks
        new_obj._dragon_idx = source._dragon_idx
        new_obj._cave_idx = source._cave_idx
        new_obj._automa_idx = source._automa_idx
        return new_obj

    def _draw_from_deck(self, deck_name: str):
        if deck_name == "dragon":
            if self._dragon_idx < 0:
                return None
            ret = self.dragon_deck[self._dragon_idx]
            self._dragon_idx -= 1
            return ret
        if deck_name == "cave":
            if self._cave_idx < 0:
                return None
            ret = self.cave_deck[self._cave_idx]
            self._cave_idx -= 1
            return ret
        if self._automa_idx < 0:
            return None
        ret = self.automa_decks[self._automa_idx]
        self._automa_idx -= 1
        return ret

    def get_copy(self):
        """
        Get a copy of the RNGOrder object.
        This is useful for simulating multiple games from the same initial state.
        """
        return RNGOrder._from_existing(self)

    def get_random_outcome(self, game_state: GameState, event:dict, player:PlayerState):
        """
        Get the next random outcome for a given event.
        This will return the next outcome in the order and remove the outcome from the list.
        """
        # check the event type
        if "automa_action" in event:
            # we return one card sampled from the automa's decision deck
            return self._draw_from_deck("automa")
        elif ("top_deck_reveal" in event) or ("refill_dragon_display" in event) or ("gain_dragon" in event) or ("tuck_from" in event):
            # we return one card sampled from the dragon deck
            return self._draw_from_deck("dragon")
        elif ("refill_cave_display" in event) or ("play_cave" in event) or ("gain_cave" in event):
            # we return one card sampled from the cave deck
            return self._draw_from_deck("cave")
        elif "draw_decision" in event:
            # we return a number of cards sampled from the dragon deck
            num_cards = event["draw_decision"]["amount"]
            if num_cards == "shy_this_cave":
                # we find the amount to draw
                cave_name, col = event["coords"]
                num_cards = 0
                for col in range(4):
                    dragon_id = player.dragons_played[cave_name][col]
                    if dragon_id is not None and DRAGON_CARDS[dragon_id]["personality"] == "Shy":
                        num_cards += 1
            ret = []
            for _ in range(num_cards):
                ret.append(self._draw_from_deck("dragon"))
            return tuple(ret)
        raise ValueError(f"Invalid random event: {event}")

def alg_uniform_random(game_state: GameState) -> int:
    """
    Choose an action to play during the simulation, assuming the game state
    has a choice (game_state.current_choice is a list of actions/events).
    This is an evenly random choice, which is not optimal for Wyrmspan.

    Returns the index of the action to take from the current choice list.
    """
    return random.randint(0, len(game_state.current_choice) - 1)

def alg_non_pass(game_state: GameState) -> int:
    """
    Choose an action to play during the simulation, assuming the game state
    has a choice (game_state.current_choice is a list of actions/events).
    By default, you can choose a random action, but this is not optimal for Wyrmspan.

    Returns the index of the action to take from the current choice list.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        # we only have one action to choose from
        return 0
    # assume we have more than 1 action now
    pass_index = None
    skip_index = None
    # iterate from the end because pass is often near the end of the list
    for i in range(len(current_choice) - 1, -1, -1):
        action = current_choice[i]
        if "pass" in action:
            pass_index = i
            break
        elif "skip" in action or "skip_opr" in action:
            skip_index = i
            break
    if pass_index is not None:
        # we have a pass action
        if game_state.player.coins <= 4 and random.random() < (1 / (game_state.player.coins * 100)):
            # we can very rarely pass if we have fewer coins
            return pass_index
        # otherwise choose a non-pass action
        return random.choice([i for i in range(len(current_choice)) if i != pass_index])
    elif skip_index is not None:
        # we have a skip action
        # skip less often than uniform random
        if random.random() < (1 / (len(current_choice) * 2)):
            return skip_index
        return random.choice([i for i in range(len(current_choice)) if i != skip_index])
    else:
        return random.randint(0, len(current_choice) - 1)

def alg_play_dragon_cave(game_state: GameState, entice_prob=0.7, excavate_prob=0.7) -> int:
    """
    This algorithm specifically tries to play dragons and caves more often
    when they are available in the current choice. Otherwise, it defers to another algorithm.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        # we only have one action to choose from
        return 0
    # assume we have more than 1 action now
    entice_index = None
    excavate_index = None
    for i, action in enumerate(current_choice):
        if "play_cave" in action:
            excavate_index = i
        elif "play_dragon" in action:
            entice_index = i
        if entice_index is not None and excavate_index is not None:
            break
    if entice_index is not None and random.random() < entice_prob:
        # play dragon if available and with some probability
        return entice_index
    elif excavate_index is not None and random.random() < excavate_prob:
        # play cave if available and with some probability
        return excavate_index
    else:
        # defer to another algorithm
        return alg_uniform_random(game_state)
        # return alg_non_pass(game_state)

# original weights were 3.2, 2.8, 2.1 with pass_penalty 1.5 and tie_threshold 0.35
def alg_greedy_action_priority(game_state: GameState,
                               dragon_weight=2.845,
                               cave_weight=2.056,
                               explore_weight=1.431,
                               pass_penalty=1.5,
                               tie_threshold=0.35) -> int:
    """
    Fast heuristic playout policy.

    It scores each available action with lightweight state features and picks
    the best option, with top-2 tie randomness for rollout diversity.
    """
    current_choice = game_state.current_choice
    if len(current_choice) == 1:
        return 0

    player = game_state.player
    coins = player.coins
    total_eggs = sum(player.egg_totals.values())
    resources = player.resources
    total_resources = sum(resources.values())
    round_num = game_state.board["round_tracker"]["round"]

    def get_effect(action: dict) -> dict:
        """Extract the immediate action payload if wrapped in adv_effects."""
        if "adv_effects" in action and isinstance(action["adv_effects"], dict):
            return action["adv_effects"]
        return action

    def get_cost(action: dict) -> dict:
        if "cost" in action and isinstance(action["cost"], dict):
            return action["cost"]
        return {}

    best_idx = 0
    best_score = float("-inf")
    second_idx = None
    second_score = float("-inf")

    for i, action in enumerate(current_choice):
        effect = get_effect(action)
        cost = get_cost(action)

        if "play_dragon" in effect:
            score = dragon_weight + 0.35
            if len(player.dragon_hand) == 0:
                score -= 3.0
            if coins <= 1:
                score -= 0.2
            if round_num <= 2:
                score += 0.25
        elif "play_cave" in effect:
            score = cave_weight
            if len(player.cave_hand) == 0:
                score -= 3.0
            cave_payload = effect.get("play_cave", {})
            if isinstance(cave_payload, dict) and cave_payload.get("free", False):
                score += 0.45
            if round_num <= 2:
                score += 0.2
        elif "explore" in effect:
            score = explore_weight
            if round_num <= 2:
                score += 0.45
        elif "gain_resource" in effect:
            score = 1.7
        elif "lay_egg" in effect:
            score = 1.4
        elif "draw_decision" in effect or "gain_dragon" in effect or "gain_cave" in effect:
            score = 1.2
        elif "pass" in effect:
            score = -pass_penalty
            if coins > 0:
                score -= 0.6
        elif "skip" in effect or "skip_opr" in effect:
            score = -(pass_penalty * 0.7)
        else:
            score = 0.8

        if score > best_score:
            second_score, second_idx = best_score, best_idx
            best_score, best_idx = score, i
        elif score > second_score:
            second_score, second_idx = score, i

    if second_idx is not None and abs(best_score - second_score) <= tie_threshold:
        return random.choice([best_idx, second_idx])
    return best_idx

def get_sim_algo(algo_name, algo_kwargs):
    """
    Dispatcher for simulation algorithms.
    """
    if algo_name == "uniform_random":
        return alg_uniform_random
    elif algo_name == "non_pass":
        return alg_non_pass
    elif algo_name == "play_dragon_cave":
        def algo(gs):
            return alg_play_dragon_cave(gs, **algo_kwargs)
        return algo
    elif algo_name == "greedy_action_priority":
        def algo(gs):
            return alg_greedy_action_priority(gs, **algo_kwargs)
        return algo
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def simulate_game(game_state: GameState, algo_name, algo_kwargs, display_name, seed=None) -> tuple:
    """
    Simulate a random game from the given game state until a terminal state is reached.
    Returns a score for the simulation, the name of the algorithm used, and the time taken to simulate.
    """
    if seed is not None:
        random.seed(seed)
    sim_algo = get_sim_algo(algo_name, algo_kwargs)
    start_time = time.time()
    while game_state.phase != logic.PHASE_END_GAME:
        # check if we have a choice or random event
        if game_state.current_choice is not None:
            # we have a choice to make
            chosen_input = sim_algo(game_state)
            game_state = logic.get_next_state(game_state, chosen_input)
        elif game_state.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = logic.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            game_state = logic.get_next_state(game_state, chosen_input)
        else:
            # progress the game
            game_state = logic.get_next_state(game_state, chosen_input=None)
    end_time = time.time()
    # return 1 if game_state.player.score > 30 else 0
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    # if game_state.player.score >= game_state.automa.score:
    #     return (5000 + (game_state.player.score - game_state.automa.score) ** 2) / 10000, display_name, end_time - start_time
    # else:
    #     return (5000 - (game_state.automa.score - game_state.player.score) ** 2) / 10000, display_name, end_time - start_time
    
    score_based_reward = (game_state.player.score ** 2) / 40_000
    if game_state.player.score >= game_state.automa.score:
        return score_based_reward + 0.75, display_name, end_time - start_time
    else:
        return score_based_reward, display_name, end_time - start_time

    # Other Objectives to Maximize
    #get as many cached resources as possible
    # score = 0
    # for cache_list in game_state.player.cached_resources.values():
    #     score += sum(num_caches for cache_dict in cache_list for num_caches in cache_dict.values())
    # return score / 50, display_name, end_time - start_time


def simulate_multiple_games(game_state: GameState, algo_name, algo_kwargs, display_name, num_simulations, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple:
    """
    Simulate multiple games from the given game state using the specified algorithm.
    Returns the average score for the simulations, the name of the algorithm used, and the time taken to simulate.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    total_score = 0
    start_time = time.time()
    base_seed = int(time.time() * 1000)  # Use current time in milliseconds as a base seed
    batch_size = max(1, batch_size)
    with ProcessPoolExecutor() as executor:
        futures = []
        for start, end in _chunk_count(num_simulations, batch_size):
            seeds = [base_seed + s for s in range(start, end)]
            futures.append(
                executor.submit(
                    _run_seeded_game_batch,
                    game_state,
                    algo_name,
                    algo_kwargs,
                    display_name,
                    seeds,
                )
            )
        # wait for all futures to complete and sum the scores
        for future in as_completed(futures):
            score, _, _, _ = future.result()
            total_score += score
    end_time = time.time()
    return total_score, display_name, end_time - start_time

def simulate_game_given_rng(game_state: GameState, algo_name, algo_kwargs, display_name, rng: RNGOrder=None) -> tuple:
    """
    Simulate a game following the given RNG order.
    Returns a score for the simulation, the name of the algorithm used, and the time taken to simulate.
    """
    sim_algo = get_sim_algo(algo_name, algo_kwargs)
    start_time = time.time()
    this_rng = rng.get_copy()
    while game_state.phase != logic.PHASE_END_GAME:
        # check if we have a choice or random event
        if game_state.current_choice is not None:
            # we have a choice to make
            chosen_input = sim_algo(game_state)
            game_state = logic.get_next_state(game_state, chosen_input)
        elif game_state.current_random_event is not None:
            # we have a random event to resolve
            chosen_input = this_rng.get_random_outcome(game_state, game_state.current_random_event, game_state.player)
            game_state = logic.get_next_state(game_state, chosen_input)
        else:
            # progress the game
            game_state = logic.get_next_state(game_state, chosen_input=None)
    end_time = time.time()
    # return 1 if game_state.player.score > 30 else 0
    # return game_state.player.score / MAX_SCORE
    # return (game_state.player.score - game_state.automa.score + 70) / 140  # Normalize the score to be between 0 and 1
    # if game_state.player.score >= game_state.automa.score:
    #     return (5000 + (game_state.player.score - game_state.automa.score) ** 2) / 10000, display_name, end_time - start_time
    # else:
    #     return (5000 - (game_state.automa.score - game_state.player.score) ** 2) / 10000, display_name, end_time - start_time
    score_based_reward = (game_state.player.score ** 2) / 40000
    if game_state.player.score >= game_state.automa.score:
        return score_based_reward + 0.75, display_name, end_time - start_time
    else:
        return score_based_reward, display_name, end_time - start_time

def compare_algorithms(game_state: GameState = None, num_simulations: int = 500, algos=None, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Compare the performance of different simulation algorithms by running
    multiple simulations and returning the average score for each algorithm.
    """
    run_results = {}
    if algos is None:
        algos = []
        for entice_prob in [0.6, 0.75, 0.9]:
            for excavate_prob in [0.6, 0.75, 0.9]:
                algo_kwargs = {'entice_prob': entice_prob, 'excavate_prob': excavate_prob}
                algos.append(("play_dragon_cave", algo_kwargs, f"play_dragon_cave_{entice_prob:.4f}_{excavate_prob:.4f}"))
        # define some random play_dragon_cave probabilities
        # for i in range(6):
        #     entice_prob = random.random()
        #     excavate_prob = random.random()
        #     algo_kwargs = {'entice_prob': entice_prob, 'excavate_prob': excavate_prob}
        #     algos.append(("play_dragon_cave", algo_kwargs, f"play_dragon_cave_{entice_prob:.4f}_{excavate_prob:.4f}"))
        # constant algorithms
        # algos += [("uniform_random", {}, "uniform_random"), ("non_pass", {}, "non_pass")]
    for _, _, name in algos:
        run_results[name] = {'total_score': 0, 'num_simulations': 0, 'total_time': 0.0}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    batch_size = max(1, batch_size)
    # run each algorithm in parallel num_simulations times
    with ProcessPoolExecutor() as executor:
        futures = []
        task_batch = []
        task_seed = int(time.time() * 1000)
        for _ in range(num_simulations):
            if game_state is None:
                this_game_state = SoloGameState(automa_difficulty=1)
                this_game_state.create_game()
            else:
                this_game_state = game_state
            for algo_name, algo_kwargs, display_name in algos:
                # batch the simulation tasks to reduce executor overhead
                task_batch.append((copy.deepcopy(this_game_state), algo_name, algo_kwargs, display_name, task_seed))
                task_seed += 1
                if len(task_batch) >= batch_size:
                    futures.append(executor.submit(_run_simulation_task_batch, task_batch))
                    task_batch = []

        if task_batch:
            futures.append(executor.submit(_run_simulation_task_batch, task_batch))

        for i, future in enumerate(as_completed(futures)):
            for score, algo_name, time_taken in future.result():
                run_results[algo_name]['total_score'] += score
                run_results[algo_name]['num_simulations'] += 1
                run_results[algo_name]['total_time'] += time_taken
            if i % 500 == 0:
                print(f"\nCompleted {i+1} simulations")
                for display_name, results in run_results.items():
                    if results['num_simulations'] > 0:
                        print(f"\t{display_name}: {results['total_score'] / results['num_simulations']:.4f} (Total: {results['total_score']}, Simulations: {results['num_simulations']}, Time: {results['total_time']:.2f} seconds)")
    # calculate the average scores
    print(f"\nSimulations completed per algorithm: {num_simulations}")
    for display_name, results in run_results.items():
        if results['num_simulations'] > 0:
            average_score = results['total_score'] / results['num_simulations']
            print(f"{display_name}: {average_score:.4f} (Total: {results['total_score']}, Simulations: {results['num_simulations']}, Time: {results['total_time']:.2f} seconds)")
        else:
            print(f"{display_name}: No simulations run.")
    
    return run_results

def evolutionary_compare_algorithms(game_state: GameState = None, num_simulations: int = 200):
    """
    Find the best hyperparameters for the play_dragon_cave algorithm using an evolutionary algorithm.
    Starting with a set of random hyperparameters, evolve them over multiple generations
    by selecting the best performing ones and mutating them slightly.
    """
    import numpy as np

    # Define the initial population size and number of generations
    population_size = 10
    num_generations = 30

    # Initialize a random population of hyperparameters
    population = [(random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)) for _ in range(population_size)]
    # population.append((0.8718, 0.7396)) # add previous best hyperparameters if available
    population.append((0.97, 0.75)) # add previous best hyperparameters if available

    for generation in range(num_generations):
        print(f"\n\n*> Generation {generation + 1}/{num_generations}")
        print(f"Current population of {len(population)}: {population}")
        # generate possible offspring from the current population
        offspring = []
        remaining_parents = population.copy()
        while len(remaining_parents) > 1:
            # randomly pair two individuals from the population
            parent1_index = random.choice(range(len(remaining_parents)))
            parent1 = remaining_parents[parent1_index]
            remaining_parents.pop(parent1_index)
            parent2_index = random.choice(range(len(remaining_parents)))
            parent2 = remaining_parents[parent2_index]
            remaining_parents.pop(parent2_index)
            print(f"Pairing parents: {parent1} and {parent2}")
            # create a new individual by averaging the parents' hyperparameters
            new_individual = (
                (parent1[0] + parent2[0]) / 2 + np.random.normal(0, 0.05),  # entice_prob
                (parent1[1] + parent2[1]) / 2 + np.random.normal(0, 0.05)   # excavate_prob
            )
            # ensure the new individual is within bounds
            new_individual = (max(0.1, min(1.0, new_individual[0])), max(0.1, min(1.0, new_individual[1])))
            # add the new individual to the offspring
            offspring.append(new_individual)
        # Use the full population for the next generation
        population = offspring + population
        print(f"\nOffspring generated: {offspring}")
        print("\nEvaluating offspring...")
        algos = [("play_dragon_cave", {'entice_prob': ep, 'excavate_prob': ex}, f"play_dragon_cave_{ep:.4f}_{ex:.4f}") for ep, ex in population]
        results = compare_algorithms(game_state=game_state, num_simulations=num_simulations, algos=algos)

        # Extract scores for the current population
        scores = []
        for _, _, display_name in algos:
            if display_name in results:
                scores.append(results[display_name]['total_score'] / results[display_name]['num_simulations'])
            else:
                scores.append(0)
        # Sort the population based on scores
        sorted_population = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)

        # Keep the top 'population_size' individuals for the next generation
        population = [ind for ind, score in sorted_population[:population_size]]

    # Return the best hyperparameters found
    best_entice_prob, best_excavate_prob = sorted_population[0][0]
    print(f"\n\n>> Best hyperparameters: entice_prob={best_entice_prob:.4f}, excavate_prob={best_excavate_prob:.4f}")

if __name__ == "__main__":
    algos = [
        ("uniform_random", {}, "uniform_random"),
        ("non_pass", {}, "non_pass"),
        ("play_dragon_cave", {'entice_prob': 0.8718, 'excavate_prob': 0.7396}, "play_dragon_cave_0.8718_0.7396"),
        ("greedy_action_priority", {'dragon_weight': 3.2, 'cave_weight': 2.8, 'explore_weight': 2.1}, "greedy_action_priority_original"),
        ("greedy_action_priority", {'dragon_weight': 2.845, 'cave_weight': 2.056, 'explore_weight': 1.431}, "greedy_action_priority_tuned?")
    ]
    compare_algorithms(num_simulations=1000, algos=algos)
    
    # evolutionary_compare_algorithms(num_simulations=300)

    # import logging
    # logging.basicConfig(
    #     filename='playout_compare.log',
    #     level=logging.DEBUG,
    #     # level=logging.INFO,
    #     # level=logging.WARNING,
    #     format='%(asctime)s:%(levelname)s:%(message)s',
    #     filemode='w'
    # )
    # logger = logging.getLogger(__name__)
    # game = SoloGameState()
    # game.create_game()
    # # simulate_game(game, "play_dragon_cave", {'entice_prob': 0.8718, 'excavate_prob': 0.7396}, "play_dragon_cave_0.8718_0.7396")

    # rng = RNGOrder(game)
    # print(f"Opening hand: {game.player.dragon_hand}")
    # print(any(d in rng.dragon_deck for d in game.player.dragon_hand))
    # print(f"Dragon Deck order: {rng.dragon_deck}")
    # print(f"Cave deck order: {rng.cave_deck}")
    # print(f"Automa decks: {rng.automa_decks}")