import logging
import random
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from game_states import SoloGameState
from playout_compare import simulate_game

# Parameters to tune (starting from known baseline)
DEFAULT_PARAMS = {
    "dragon_weight": 11.649,
    "cave_weight": 6.571,
    "explore_weight": 4.927,
    "pass_penalty": 1.57,
    "tie_threshold": 1.496
}
# DEFAULT_PARAMS = {
#     "dragon_weight": 2.9,
#     "cave_weight": 2.35,
#     "explore_weight": 1.7,
#     "pass_penalty": 1.5,
#     "tie_threshold": 0.3,
# }

# Tuning configuration
POPULATION_SIZE = 12
NUM_GENERATIONS = 10
ELITE_FRACTION = 0.25
SIMS_PER_CONFIG = 50  # Simulations per configuration to estimate performance
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.7
MUTATION_SCALE = {
    "dragon_weight": 0.35,
    "cave_weight": 0.35,
    "explore_weight": 0.25,
    "pass_penalty": 0.25,
    "tie_threshold": 0.08,
}
SEARCH_BOUNDS = {
    "dragon_weight": (5.0, 20.0),
    "cave_weight": (2.0, 15.0),
    "explore_weight": (1.5, 12.0),
    "pass_penalty": (0.0, 5.0),
    "tie_threshold": (0.01, 4.0),
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def random_param_value(name, center=None, spread=None):
    lower, upper = SEARCH_BOUNDS[name]
    if center is None:
        return round(random.uniform(lower, upper), 3)
    spread = spread if spread is not None else (upper - lower) * 0.25
    sampled = random.gauss(center, spread)
    return round(clamp(sampled, lower, upper), 3)


def random_config():
    return {name: random_param_value(name) for name in DEFAULT_PARAMS}


def mutate_config(config, mutation_rate=MUTATION_RATE):
    mutated = config.copy()
    for name, value in mutated.items():
        if random.random() < mutation_rate:
            mutated[name] = random_param_value(name, center=value, spread=MUTATION_SCALE[name])
    return mutated


def crossover_configs(parent_a, parent_b):
    child = {}
    for name in DEFAULT_PARAMS:
        if random.random() < 0.5:
            child[name] = parent_a[name]
        else:
            child[name] = parent_b[name]
    return child

def evaluate_config(config, num_sims=SIMS_PER_CONFIG):
    """
    Evaluates a specific parameter configuration by running simulations.
    Disables logging within workers to ensure performance.
    """
    logging.getLogger().setLevel(logging.WARNING)
    
    total_score = 0
    wins = 0
    
    # We use a fixed seed base for the batch to make comparisons fairer between configs
    # while still allowing intra-batch variety.
    seed_base = 42 
    
    for i in range(num_sims):
        # Create a fresh game state for each simulation
        gs = SoloGameState(automa_difficulty=1)
        gs.create_game()
        
        # simulate_game handles the deep copy or fresh state usage
        # reward is normalized [0, 1] in the updated playout_compare.py
        result = simulate_game(
            game_state=gs,
            algo_name="strategic_objective_aware",
            algo_kwargs=config,
            display_name="tuning_worker",
            seed=seed_base + i
        )
        
        # Depending on playout_compare implementation, result might be (reward, name, time)
        reward = result[0]
        total_score += reward
        
        # We can also track if player score exceeded automa score if desired,
        # but the reward function usually captures this.
        
    return total_score / num_sims, config


def tournament_select(scored_population, tournament_size=TOURNAMENT_SIZE):
    contenders = random.sample(scored_population, k=min(tournament_size, len(scored_population)))
    contenders.sort(key=lambda item: item[0], reverse=True)
    return contenders[0][1]


def build_next_generation(scored_population, population_size=POPULATION_SIZE):
    scored_population = sorted(scored_population, key=lambda item: item[0], reverse=True)
    elite_count = max(1, int(round(population_size * ELITE_FRACTION)))
    elites = [config.copy() for _, config in scored_population[:elite_count]]

    next_population = [elite.copy() for elite in elites]
    while len(next_population) < population_size:
        if random.random() < 0.25:
            candidate = random.choice(elites).copy()
        else:
            parent_a = tournament_select(scored_population)
            parent_b = tournament_select(scored_population)
            candidate = crossover_configs(parent_a, parent_b)

        candidate = mutate_config(candidate)
        next_population.append(candidate)

    return next_population[:population_size]


def tune_evolutionary_search(
    generations=NUM_GENERATIONS,
    population_size=POPULATION_SIZE,
    sims_per_config=SIMS_PER_CONFIG,
):
    """
    Evolutionary search strategy with elitism, crossover, and mutation.
    """
    print(
        f"Starting evolutionary tuning with {generations} generations, "
        f"population {population_size}, {sims_per_config} sims each..."
    )

    best_avg_reward = -1.0
    best_params = DEFAULT_PARAMS.copy()

    results_dir = "tuning_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    population = [DEFAULT_PARAMS.copy()]
    while len(population) < population_size:
        population.append(random_config())

    with ProcessPoolExecutor() as executor:
        for generation in range(1, generations + 1):
            futures = [executor.submit(evaluate_config, config, sims_per_config) for config in population]
            scored_population = []

            for i, future in enumerate(as_completed(futures)):
                avg_reward, config = future.result()
                scored_population.append((avg_reward, config))
                print(
                    f"[gen {generation}/{generations} | {i+1}/{len(population)}] "
                    f"Reward: {avg_reward:.4f} | Params: {config}"
                )

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_params = config.copy()
                    print(f"*** New best found! Reward: {best_avg_reward:.4f}")

                    with open(os.path.join(results_dir, "best_params.json"), "w") as f:
                        json.dump(
                            {
                                "reward": best_avg_reward,
                                "params": best_params,
                                "generation": generation,
                            },
                            f,
                            indent=4,
                        )

            generation_best = max(scored_population, key=lambda item: item[0])
            generation_mean = sum(score for score, _ in scored_population) / len(scored_population)
            print(
                f"Generation {generation} summary: best={generation_best[0]:.4f} "
                f"mean={generation_mean:.4f}"
            )

            if generation < generations:
                population = build_next_generation(scored_population, population_size=population_size)

    print("\n" + "=" * 30)
    print("TUNING COMPLETE")
    print(f"Best Average Reward: {best_avg_reward:.4f}")
    print(f"Best Parameters: {best_params}")
    print("=" * 30)

if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    # and logging is suppressed
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Tune strategic_objective_aware rollout parameters.")
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--population-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--sims-per-config", type=int, default=SIMS_PER_CONFIG)
    args = parser.parse_args()

    SIMS_PER_CONFIG = args.sims_per_config
    tune_evolutionary_search(
        generations=args.generations,
        population_size=args.population_size,
        sims_per_config=args.sims_per_config,
    )
