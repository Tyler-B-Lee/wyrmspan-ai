import random
import time
from game_states import GameState, SoloGameState
import game_logic as logic

# File for comparing different algorithms for simulating game playouts in Wyrmspan.

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
    # reversed because pass is often at the end of the list
    for i, action in enumerate(reversed(current_choice)):
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
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def simulate_game(game_state: GameState, algo_name, algo_kwargs, display_name) -> tuple:
    """
    Simulate a random game from the given game state until a terminal state is reached.
    Returns a score for the simulation, the name of the algorithm used, and the time taken to simulate.
    """
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
    if game_state.player.score >= game_state.automa.score:
        return (game_state.player.score / 250 + 0.5), display_name, end_time - start_time
    else:
        return game_state.player.score / 250, display_name, end_time - start_time
    
def compare_algorithms(game_state: GameState = None, num_simulations: int = 500, algos=None):
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
    # run each algorithm in parallel num_simulations times
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(num_simulations):
            if game_state is None:
                this_game_state = SoloGameState()
                this_game_state.create_game()
            else:
                this_game_state = game_state
            for algo_name, algo_kwargs, display_name in algos:
                # submit the simulation task for each algorithm
                futures.append(executor.submit(simulate_game, this_game_state, algo_name, algo_kwargs, display_name))

        for i, future in enumerate(as_completed(futures)):
            score, algo_name, time_taken = future.result()
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
    population.append((0.8718, 0.7396)) # add previous best hyperparameters if available

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
    # evolutionary_compare_algorithms(num_simulations=300)

    import logging
    logging.basicConfig(
        filename='playout_compare.log',
        level=logging.DEBUG,
        # level=logging.INFO,
        # level=logging.WARNING,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    game = SoloGameState()
    game.create_game()
    simulate_game(game, "play_dragon_cave", {'entice_prob': 0.8718, 'excavate_prob': 0.7396}, "play_dragon_cave_0.8718_0.7396")