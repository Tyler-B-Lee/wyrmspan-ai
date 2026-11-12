"""Run MCTS with compact rollouts and large-model leaf evaluation."""

from __future__ import annotations

import argparse

import game_uct_compare_v2 as search
from compact_rollout_model import load_compact_checkpoint
from hybrid_rollout import load_large_value_evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--compact-checkpoint", required=True)
    parser.add_argument("--large-checkpoint", required=True)
    parser.add_argument("--large-architecture", choices=("auto", "semantic", "legacy"), default="auto")
    parser.add_argument("--sims-per-move", type=int, default=10)
    parser.add_argument("--uct-constant", type=float, default=search.UCT_CONSTANT)
    parser.add_argument("--log-filename", default="hybrid_uct.log")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--echo", action="store_true")
    args = parser.parse_args()

    compact_policy = load_compact_checkpoint(args.compact_checkpoint, device=args.device)
    large_evaluator = load_large_value_evaluator(
        args.large_checkpoint,
        device=args.device,
        architecture=args.large_architecture,
        value_scale=0.5,
        value_bias=0.0,
        value_clip=(0.0,1.0)
    )
    search.set_search_config(
        sims_per_move=args.sims_per_move,
        base_random_children=search.BASE_RANDOM_CHILDREN,
        max_random_children=search.MAX_RANDOM_CHILDREN,
        random_growth_interval=search.RANDOM_CHILD_GROWTH_INTERVAL,
    )
    search.configure_hybrid_rollout(compact_policy, large_evaluator)
    try:
        player_score, automa_score = search.run_game(
            seed=args.seed,
            uct_constant=args.uct_constant,
            log_filename=args.log_filename,
            echo=args.echo,
        )
    finally:
        search.configure_hybrid_rollout(None, None)
    print(f"Hybrid game complete: player_score={player_score}, automa_score={automa_score}")


if __name__ == "__main__":
    main()
