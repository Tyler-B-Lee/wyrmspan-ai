"""Collect deterministic heuristic-labeled data for the compact rollout policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import game_logic as logic
from compact_rollout_model import RolloutFeatureAdapter, SCHEMA_VERSION
from game_states import SoloGameState
from playout_compare import RNGOrder, get_sim_algo


DATASET_VERSION = 1


def collect_episode(seed: int, teacher_name: str = "strategic_objective_aware") -> list[dict[str, Any]]:
    """Collect choice-node features from one deterministic game episode."""
    random.seed(seed)
    game_state = SoloGameState(automa_difficulty=1, ending_round=4)
    game_state.create_game()
    rng = RNGOrder(game_state)
    teacher = get_sim_algo(
        teacher_name,
        {"tie_threshold": -1.0},
    )
    adapter = RolloutFeatureAdapter()
    records: list[dict[str, Any]] = []
    step = 0

    while game_state.phase != logic.PHASE_END_GAME:
        if game_state.current_choice is not None:
            snapshot = game_state.make_copy()
            encoded = adapter.encode_choices(snapshot)
            label = int(teacher(snapshot))
            action_count = len(snapshot.current_choice)
            if not 0 <= label < action_count:
                raise RuntimeError(
                    f"Teacher returned invalid action index {label} for {action_count} choices"
                )
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "teacher": teacher_name,
                    "label": label,
                    "action_count": action_count,
                    "state_features": encoded.state_features.tolist(),
                    "action_features": encoded.action_features.tolist(),
                    "action_summaries": list(encoded.summaries),
                    "overflow_count": encoded.overflow_count,
                }
            )
            game_state = logic.get_next_state(game_state, label)
        elif game_state.current_random_event is not None:
            outcome = rng.get_random_outcome(
                game_state,
                game_state.current_random_event,
                game_state.player,
            )
            game_state = logic.get_next_state(game_state, outcome)
        else:
            game_state = logic.get_next_state(game_state, chosen_input=None)
        step += 1

    return records


def collect_dataset(
    seeds: list[int],
    teacher_name: str = "strategic_objective_aware",
) -> dict[str, Any]:
    """Collect and package records from independent seeded episodes."""
    records: list[dict[str, Any]] = []
    for seed in seeds:
        records.extend(collect_episode(seed, teacher_name=teacher_name))
    return {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "teacher": teacher_name,
        "seeds": seeds,
        "records": records,
    }


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--teacher",
        default="strategic_objective_aware",
        choices=("greedy_action_priority", "strategic_objective_aware"),
    )
    args = parser.parse_args()
    dataset = collect_dataset(args.seeds, teacher_name=args.teacher)
    save_dataset(dataset, args.output)
    print(f"Saved {len(dataset['records'])} choice records to {args.output}")


if __name__ == "__main__":
    main()
