import argparse
import copy
import csv
import io
import statistics
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import game_uct_compare as uct
from game_states import GameState


@contextmanager
def patched_make_copy_with_deepcopy(enabled: bool):
    """Temporarily force GameState.make_copy to use deepcopy for A/B testing."""
    if not enabled:
        yield
        return

    original_make_copy = GameState.make_copy
    try:
        GameState.make_copy = lambda self: copy.deepcopy(self)
        yield
    finally:
        GameState.make_copy = original_make_copy


@contextmanager
def temporary_uct_settings(sims_per_move=None, max_depth=None, max_random_children=None):
    """Temporarily override game_uct_compare global tuning constants."""
    original = {
        "SIMS_PER_MOVE": uct.SIMS_PER_MOVE,
        "MIN_BUDGET": uct.MIN_BUDGET,
        "MAX_BUDGET": uct.MAX_BUDGET,
        "MAX_DEPTH": uct.MAX_DEPTH,
        "MAX_RANDOM_CHILDREN": uct.MAX_RANDOM_CHILDREN,
    }
    try:
        if sims_per_move is not None:
            uct.SIMS_PER_MOVE = sims_per_move
            uct.MIN_BUDGET = sims_per_move * 10
            uct.MAX_BUDGET = 40 * sims_per_move
        if max_depth is not None:
            uct.MAX_DEPTH = max_depth
        if max_random_children is not None:
            uct.MAX_RANDOM_CHILDREN = max_random_children
        yield
    finally:
        uct.SIMS_PER_MOVE = original["SIMS_PER_MOVE"]
        uct.MIN_BUDGET = original["MIN_BUDGET"]
        uct.MAX_BUDGET = original["MAX_BUDGET"]
        uct.MAX_DEPTH = original["MAX_DEPTH"]
        uct.MAX_RANDOM_CHILDREN = original["MAX_RANDOM_CHILDREN"]


@contextmanager
def maybe_silence_stdout(enabled: bool):
    """Suppress heavy debug prints from UCT internals during benchmarks."""
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()):
        yield


def run_uct_batch(seeds, uct_constants, force_deepcopy_mode: bool, sims_per_move=None, max_depth=None, max_random_children=None, suppress_prints=True):
    rows = []
    with temporary_uct_settings(sims_per_move, max_depth, max_random_children):
        with patched_make_copy_with_deepcopy(force_deepcopy_mode):
            for seed in seeds:
                for uct_constant in uct_constants:
                    log_filename = f"bench_uct_seed-{seed}_c-{uct_constant}_{'deepcopy' if force_deepcopy_mode else 'makecopy'}.log"
                    start = time.perf_counter()
                    with maybe_silence_stdout(suppress_prints):
                        player_score, automa_score = uct.run_game(
                            seed=seed,
                            uct_constant=uct_constant,
                            log_filename=log_filename,
                            echo=False,
                        )
                    elapsed = time.perf_counter() - start
                    rows.append(
                        {
                            "mode": "forced_deepcopy" if force_deepcopy_mode else "make_copy",
                            "seed": seed,
                            "uct_constant": uct_constant,
                            "player_score": player_score,
                            "automa_score": automa_score,
                            "score_delta": player_score - automa_score,
                            "elapsed_sec": elapsed,
                            "sims_per_move": uct.SIMS_PER_MOVE,
                            "max_depth": uct.MAX_DEPTH,
                            "max_random_children": uct.MAX_RANDOM_CHILDREN,
                        }
                    )
    return rows


def summarize(rows):
    if not rows:
        return None
    return {
        "avg_elapsed": statistics.mean(r["elapsed_sec"] for r in rows),
        "median_elapsed": statistics.median(r["elapsed_sec"] for r in rows),
        "avg_delta": statistics.mean(r["score_delta"] for r in rows),
        "avg_player": statistics.mean(r["player_score"] for r in rows),
        "avg_automa": statistics.mean(r["automa_score"] for r in rows),
    }


def key_for_row(row):
    return (row["seed"], row["uct_constant"])


def compare_modes(make_copy_rows, forced_deepcopy_rows):
    make_map = {key_for_row(r): r for r in make_copy_rows}
    deep_map = {key_for_row(r): r for r in forced_deepcopy_rows}

    keys = sorted(set(make_map.keys()) & set(deep_map.keys()))
    comparisons = []
    mismatches = []

    for key in keys:
        make_row = make_map[key]
        deep_row = deep_map[key]
        speedup = (
            deep_row["elapsed_sec"] / make_row["elapsed_sec"]
            if make_row["elapsed_sec"] > 0
            else float("inf")
        )
        comparisons.append(
            {
                "seed": key[0],
                "uct_constant": key[1],
                "make_copy_elapsed_sec": make_row["elapsed_sec"],
                "forced_deepcopy_elapsed_sec": deep_row["elapsed_sec"],
                "speedup": speedup,
                "make_copy_score_delta": make_row["score_delta"],
                "forced_deepcopy_score_delta": deep_row["score_delta"],
            }
        )

        if (
            make_row["player_score"] != deep_row["player_score"]
            or make_row["automa_score"] != deep_row["automa_score"]
        ):
            mismatches.append(
                {
                    "seed": key[0],
                    "uct_constant": key[1],
                    "make_copy_scores": (make_row["player_score"], make_row["automa_score"]),
                    "forced_deepcopy_scores": (deep_row["player_score"], deep_row["automa_score"]),
                }
            )

    return comparisons, mismatches


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark UCT runtime impact of make_copy transition cloning vs forced deepcopy mode."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Seeds to test")
    parser.add_argument("--uct-constants", type=float, nargs="+", default=[0.5, 1.0], help="UCT constants to test")
    parser.add_argument("--sims-per-move", type=int, default=None, help="Override SIMS_PER_MOVE for this benchmark run")
    parser.add_argument("--max-depth", type=int, default=None, help="Override MAX_DEPTH for this benchmark run")
    parser.add_argument("--max-random-children", type=int, default=None, help="Override MAX_RANDOM_CHILDREN for this benchmark run")
    parser.add_argument("--skip-deepcopy-mode", action="store_true", help="Only run current make_copy mode")
    parser.add_argument("--no-suppress-prints", action="store_true", help="Show full UCT prints while benchmarking")
    parser.add_argument("--out-prefix", default=None, help="Optional prefix for CSV outputs")
    args = parser.parse_args()

    suppress_prints = not args.no_suppress_prints

    print("UCT impact benchmark")
    print(f"Seeds: {args.seeds}")
    print(f"UCT constants: {args.uct_constants}")
    print(f"SIMS_PER_MOVE override: {args.sims_per_move}")
    print(f"MAX_DEPTH override: {args.max_depth}")
    print(f"MAX_RANDOM_CHILDREN override: {args.max_random_children}")
    print()

    make_rows = run_uct_batch(
        seeds=args.seeds,
        uct_constants=args.uct_constants,
        force_deepcopy_mode=False,
        sims_per_move=args.sims_per_move,
        max_depth=args.max_depth,
        max_random_children=args.max_random_children,
        suppress_prints=suppress_prints,
    )

    deep_rows = []
    if not args.skip_deepcopy_mode:
        deep_rows = run_uct_batch(
            seeds=args.seeds,
            uct_constants=args.uct_constants,
            force_deepcopy_mode=True,
            sims_per_move=args.sims_per_move,
            max_depth=args.max_depth,
            max_random_children=args.max_random_children,
            suppress_prints=suppress_prints,
        )

    make_summary = summarize(make_rows)
    print("make_copy mode summary:")
    print(
        f"  avg_elapsed={make_summary['avg_elapsed']:.3f}s | median_elapsed={make_summary['median_elapsed']:.3f}s | "
        f"avg_delta={make_summary['avg_delta']:.3f} | avg_player={make_summary['avg_player']:.3f} | avg_automa={make_summary['avg_automa']:.3f}"
    )

    if deep_rows:
        deep_summary = summarize(deep_rows)
        print("forced_deepcopy mode summary:")
        print(
            f"  avg_elapsed={deep_summary['avg_elapsed']:.3f}s | median_elapsed={deep_summary['median_elapsed']:.3f}s | "
            f"avg_delta={deep_summary['avg_delta']:.3f} | avg_player={deep_summary['avg_player']:.3f} | avg_automa={deep_summary['avg_automa']:.3f}"
        )

        comparisons, mismatches = compare_modes(make_rows, deep_rows)
        if comparisons:
            avg_speedup = statistics.mean(row["speedup"] for row in comparisons)
            median_speedup = statistics.median(row["speedup"] for row in comparisons)
            print("mode comparison:")
            print(f"  avg_speedup={avg_speedup:.3f}x | median_speedup={median_speedup:.3f}x")

        if mismatches:
            print(f"parity: FAIL ({len(mismatches)} mismatches). Example: {mismatches[0]}")
        else:
            print("parity: PASS")

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        write_csv(str(prefix) + "_make_copy.csv", make_rows)
        if deep_rows:
            write_csv(str(prefix) + "_forced_deepcopy.csv", deep_rows)
            comparisons, _ = compare_modes(make_rows, deep_rows)
            if comparisons:
                write_csv(str(prefix) + "_comparison.csv", comparisons)
        print(f"CSV written with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
