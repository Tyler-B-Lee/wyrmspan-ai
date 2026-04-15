import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_states import SoloGameState
from playout_compare import simulate_game, simulate_multiple_games

DEFAULT_ALGO = "greedy_action_priority"
DEFAULT_ALGO_KWARGS = {
    "dragon_weight": 2.845,
    "cave_weight": 2.056,
    "explore_weight": 1.431,
}


def build_game(seed=None):
    if seed is not None:
        random.seed(seed)
    game = SoloGameState(automa_difficulty=1)
    game.create_game()
    return game


def profile_single_game(game, algo_name, algo_kwargs, display_name, seed, top_n):
    profiler = cProfile.Profile()
    profiler.enable()
    result = simulate_game(game.make_copy(), algo_name, algo_kwargs, display_name, seed=seed)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(top_n)
    print(stream.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark simulation throughput and profile a single rollout.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the benchmark game setup")
    parser.add_argument("--sims", type=int, default=128, help="Number of simulations for the batched benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for process-pool dispatch")
    parser.add_argument("--algo", default=DEFAULT_ALGO, help="Rollout policy to benchmark")
    parser.add_argument("--profile-top", type=int, default=25, help="Number of cProfile rows to print")
    parser.add_argument("--skip-profile", action="store_true", help="Skip cProfile and only time the batched run")
    args = parser.parse_args()

    algo_kwargs = dict(DEFAULT_ALGO_KWARGS)
    display_name = f"benchmark_{args.algo}"
    game = build_game(seed=args.seed)

    if not args.skip_profile:
        print("Single-game profile:\n")
        score, _, elapsed = profile_single_game(
            game,
            args.algo,
            algo_kwargs,
            display_name,
            seed=args.seed,
            top_n=args.profile_top,
        )
        print(f"Single game score: {score:.4f} | elapsed: {elapsed:.4f}s\n")

    print("Batched throughput run:\n")
    start = time.time()
    total_score, _, elapsed = simulate_multiple_games(
        game.make_copy(),
        args.algo,
        algo_kwargs,
        display_name,
        num_simulations=args.sims,
        batch_size=args.batch_size,
    )
    wall_time = time.time() - start
    print(f"Total score: {total_score:.4f}")
    print(f"Reported simulation time: {elapsed:.4f}s")
    print(f"Wall time: {wall_time:.4f}s")
    print(f"Simulations: {args.sims} | Batch size: {args.batch_size}")


if __name__ == "__main__":
    main()
