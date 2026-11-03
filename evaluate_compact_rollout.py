"""Evaluate compact rollout checkpoints on imitation and matched seeded games."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

import game_logic as logic
from compact_rollout_batch import simulate_compact_games_batched
from compact_rollout_model import load_compact_checkpoint
from game_states import SoloGameState
from playout_compare import RNGOrder, simulate_game_given_rng
from train_compact_rollout import load_dataset, split_records


def evaluate_records(policy, records: list[dict[str, Any]], top_k: int = 3) -> dict[str, float]:
    top1 = 0
    topk = 0
    total = len(records)
    with torch.inference_mode():
        for record in records:
            state = torch.tensor(record["state_features"], dtype=torch.float32, device=policy.device)
            actions = torch.tensor(record["action_features"], dtype=torch.float32, device=policy.device)
            scores = policy.model(state, actions)
            label = int(record["label"])
            ranking = torch.argsort(scores, descending=True)
            top1 += int(int(ranking[0]) == label)
            topk += int(label in ranking[: min(top_k, len(ranking))].tolist())
    return {
        "records": total,
        "top1_accuracy": top1 / total if total else 0.0,
        "topk_accuracy": topk / total if total else 0.0,
    }


def build_game(seed: int) -> SoloGameState:
    random.seed(seed)
    game = SoloGameState(automa_difficulty=1, ending_round=4)
    game.create_game()
    return game


def benchmark_games(checkpoint_path: str, seeds: list[int], device: str, backend: str) -> dict[str, dict[str, float]]:
    policy = load_compact_checkpoint(checkpoint_path, device=device, backend=backend)
    algorithms = {
        "greedy": ("greedy_action_priority", {"tie_threshold": -1.0}),
        "strategic": ("strategic_objective_aware", {"tie_threshold": -1.0}),
        "compact": ("compact_rollout", {"policy": policy}),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (algorithm, kwargs) in algorithms.items():
        scores = []
        elapsed = []
        for seed in seeds:
            game = build_game(seed)
            rng = RNGOrder(game)
            random_state = random.getstate()
            start = time.perf_counter()
            score, _, reported_elapsed = simulate_game_given_rng(
                game.make_copy(),
                algorithm,
                kwargs,
                f"evaluate_{name}",
                rng=rng,
                random_state=random_state,
            )
            elapsed.append(time.perf_counter() - start)
            scores.append(score)
        results[name] = {
            "mean_score": sum(scores) / len(scores),
            "mean_seconds": sum(elapsed) / len(elapsed),
            "games_per_second": len(seeds) / sum(elapsed),
            "reported_mean_seconds": sum(elapsed) / len(elapsed),
        }

    batched_games = []
    batched_rngs = []
    batched_random_states = []
    for seed in seeds:
        game = build_game(seed)
        batched_games.append(game)
        batched_rngs.append(RNGOrder(game))
        batched_random_states.append(random.getstate())
    start = time.perf_counter()
    batched_results = simulate_compact_games_batched(
        batched_games,
        policy,
        batched_rngs,
        random_states=batched_random_states,
    )
    elapsed = time.perf_counter() - start
    results["compact_batched"] = {
        "mean_score": sum(result[0] for result in batched_results) / len(batched_results),
        "mean_seconds": elapsed / len(batched_results),
        "games_per_second": len(batched_results) / elapsed,
        "reported_mean_seconds": sum(result[1] for result in batched_results) / len(batched_results),
    }
    return results


def evaluate_checkpoint(checkpoint_path: str, dataset_path: str, device: str, backend: str) -> dict[str, Any]:
    policy = load_compact_checkpoint(checkpoint_path, device=device, backend=backend)
    dataset = load_dataset(dataset_path)
    _, validation_records = split_records(dataset["records"])
    return {
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "teacher": dataset.get("teacher"),
        "validation": evaluate_records(policy, validation_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_checkpoint(str(args.checkpoint), str(args.dataset), args.device, args.backend)
    report["backend"] = args.backend
    report["games"] = benchmark_games(str(args.checkpoint), args.seeds, args.device, args.backend)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
