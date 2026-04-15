import argparse
import copy
from contextlib import contextmanager
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_states import GameState, SoloGameState
from playout_compare import simulate_game

DEFAULT_ALGO = "greedy_action_priority"
DEFAULT_ALGO_KWARGS = {
    "dragon_weight": 2.845,
    "cave_weight": 2.056,
    "explore_weight": 1.431,
}


def build_game(seed: int):
    random.seed(seed)
    game = SoloGameState(automa_difficulty=1)
    game.create_game()
    return game


def copy_with_make_copy(game):
    return game.make_copy()


def copy_with_deepcopy(game):
    return copy.deepcopy(game)


@contextmanager
def patched_make_copy_with_deepcopy():
    """Temporarily force GameState.make_copy to use deepcopy for A/B testing."""
    original_make_copy = GameState.make_copy
    try:
        GameState.make_copy = lambda self: copy.deepcopy(self)
        yield
    finally:
        GameState.make_copy = original_make_copy


def benchmark_copy_only(game, copier, iterations: int):
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        cloned = copier(game)
        checksum += len(cloned.dragon_deck) + len(cloned.cave_deck)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_rollouts(game, copier, sims: int, seed_base: int, algo_name: str, algo_kwargs: dict):
    start = time.perf_counter()
    total_score = 0.0
    for i in range(sims):
        cloned = copier(game)
        score, _, _ = simulate_game(
            cloned,
            algo_name=algo_name,
            algo_kwargs=algo_kwargs,
            display_name="copy-benchmark",
            seed=seed_base + i,
        )
        total_score += score
    elapsed = time.perf_counter() - start
    return elapsed, total_score


def parity_check(game, sims: int, seed_base: int, algo_name: str, algo_kwargs: dict, tolerance: float = 1e-12):
    mismatches = []
    for i in range(sims):
        sim_seed = seed_base + i
        make_copy_score, _, _ = simulate_game(
            game.make_copy(),
            algo_name=algo_name,
            algo_kwargs=algo_kwargs,
            display_name="make_copy",
            seed=sim_seed,
        )
        deepcopy_score, _, _ = simulate_game(
            copy.deepcopy(game),
            algo_name=algo_name,
            algo_kwargs=algo_kwargs,
            display_name="deepcopy",
            seed=sim_seed,
        )
        if abs(make_copy_score - deepcopy_score) > tolerance:
            mismatches.append((sim_seed, make_copy_score, deepcopy_score))
    return mismatches


def format_ratio(make_copy_time: float, deepcopy_time: float):
    if make_copy_time == 0 or deepcopy_time == 0:
        return "n/a"
    speedup = deepcopy_time / make_copy_time
    return f"{speedup:.3f}x"


def benchmark_transition_copy_mode(game, sims: int, seed_base: int, algo_name: str, algo_kwargs: dict):
    """
    Compare transition performance with current GameState.make_copy vs forced deepcopy.
    Uses deepcopy for the root simulation copy in both cases to isolate get_next_state cost.
    """
    fast_time, fast_total = benchmark_rollouts(
        game,
        copy_with_deepcopy,
        sims,
        seed_base,
        algo_name,
        algo_kwargs,
    )
    with patched_make_copy_with_deepcopy():
        slow_time, slow_total = benchmark_rollouts(
            game,
            copy_with_deepcopy,
            sims,
            seed_base,
            algo_name,
            algo_kwargs,
        )
    return fast_time, fast_total, slow_time, slow_total


def main():
    parser = argparse.ArgumentParser(description="Compare GameState.make_copy() vs copy.deepcopy() across seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to benchmark")
    parser.add_argument("--copy-iters", type=int, default=200, help="Copy-only iterations per seed")
    parser.add_argument("--sims", type=int, default=16, help="Rollouts per seed for timing")
    parser.add_argument("--parity-sims", type=int, default=8, help="Rollouts per seed for score parity check")
    parser.add_argument("--skip-parity", action="store_true", help="Skip score parity check")
    parser.add_argument("--algo", default=DEFAULT_ALGO, help="Rollout policy name")
    parser.add_argument("--transition-compare", action="store_true", help="Benchmark get_next_state copy mode (make_copy vs forced deepcopy)")
    parser.add_argument("--transition-sims", type=int, default=8, help="Rollouts per seed for transition copy mode benchmark")
    args = parser.parse_args()

    algo_kwargs = dict(DEFAULT_ALGO_KWARGS)

    copy_times_make = []
    copy_times_deep = []
    rollout_times_make = []
    rollout_times_deep = []

    print("Benchmarking make_copy vs deepcopy")
    print(f"Seeds: {args.seeds}")
    print(f"Copy iterations/seed: {args.copy_iters}")
    print(f"Rollouts/seed: {args.sims}\n")

    for seed in args.seeds:
        game = build_game(seed)

        make_copy_copy_time, checksum_a = benchmark_copy_only(game, copy_with_make_copy, args.copy_iters)
        deepcopy_copy_time, checksum_b = benchmark_copy_only(game, copy_with_deepcopy, args.copy_iters)

        if checksum_a != checksum_b:
            raise RuntimeError(f"Checksum mismatch for seed {seed}: {checksum_a} != {checksum_b}")

        make_copy_rollout_time, make_copy_total = benchmark_rollouts(
            game, copy_with_make_copy, args.sims, seed * 100000, args.algo, algo_kwargs
        )
        deepcopy_rollout_time, deepcopy_total = benchmark_rollouts(
            game, copy_with_deepcopy, args.sims, seed * 100000, args.algo, algo_kwargs
        )

        copy_times_make.append(make_copy_copy_time)
        copy_times_deep.append(deepcopy_copy_time)
        rollout_times_make.append(make_copy_rollout_time)
        rollout_times_deep.append(deepcopy_rollout_time)

        print(f"Seed {seed}:")
        print(
            f"  Copy-only: make_copy={make_copy_copy_time:.4f}s | deepcopy={deepcopy_copy_time:.4f}s"
            f" | speedup={format_ratio(make_copy_copy_time, deepcopy_copy_time)}"
        )
        print(
            f"  Rollouts:  make_copy={make_copy_rollout_time:.4f}s | deepcopy={deepcopy_rollout_time:.4f}s"
            f" | speedup={format_ratio(make_copy_rollout_time, deepcopy_rollout_time)}"
        )
        print(
            f"  Total score: make_copy={make_copy_total:.4f} | deepcopy={deepcopy_total:.4f}"
        )

        if not args.skip_parity:
            mismatches = parity_check(
                game,
                sims=args.parity_sims,
                seed_base=seed * 1000000,
                algo_name=args.algo,
                algo_kwargs=algo_kwargs,
            )
            if mismatches:
                print(f"  Parity: FAIL ({len(mismatches)} mismatches, first={mismatches[0]})")
            else:
                print("  Parity: PASS")

        if args.transition_compare:
            transition_fast, transition_fast_total, transition_slow, transition_slow_total = benchmark_transition_copy_mode(
                game,
                sims=args.transition_sims,
                seed_base=seed * 10000000,
                algo_name=args.algo,
                algo_kwargs=algo_kwargs,
            )
            print(
                "  Transition copy mode: "
                f"make_copy={transition_fast:.4f}s | forced-deepcopy={transition_slow:.4f}s "
                f"| speedup={format_ratio(transition_fast, transition_slow)}"
            )
            print(
                f"  Transition totals: make_copy={transition_fast_total:.4f} | "
                f"forced-deepcopy={transition_slow_total:.4f}"
            )
        print()

    print("Aggregate:")
    print(
        f"  Copy-only avg: make_copy={statistics.mean(copy_times_make):.4f}s | "
        f"deepcopy={statistics.mean(copy_times_deep):.4f}s | "
        f"speedup={format_ratio(statistics.mean(copy_times_make), statistics.mean(copy_times_deep))}"
    )
    print(
        f"  Rollouts avg:  make_copy={statistics.mean(rollout_times_make):.4f}s | "
        f"deepcopy={statistics.mean(rollout_times_deep):.4f}s | "
        f"speedup={format_ratio(statistics.mean(rollout_times_make), statistics.mean(rollout_times_deep))}"
    )


if __name__ == "__main__":
    main()
