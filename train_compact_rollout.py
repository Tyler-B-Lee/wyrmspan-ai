"""Train the compact rollout scorer from heuristic-labeled JSON records."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from compact_rollout_model import (
    ACTION_FEATURE_DIM,
    SCHEMA_VERSION,
    STATE_FEATURE_DIM,
    CompactRolloutScorer,
    save_compact_checkpoint,
)


@dataclass
class TrainingResult:
    train_records: int
    validation_records: int
    best_validation_loss: float
    checkpoint_path: str


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset.get("feature_schema_version") != SCHEMA_VERSION:
        raise ValueError("Dataset feature schema does not match the compact model")
    records = dataset.get("records")
    if not records:
        raise ValueError("Dataset contains no records")
    return dataset


def split_records(records: list[dict[str, Any]], validation_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = sorted({int(record["seed"]) for record in records})
    if len(seeds) < 2:
        raise ValueError("At least two episode seeds are required for a seed-level validation split")
    validation_count = max(1, round(len(seeds) * validation_fraction))
    validation_seeds = set(seeds[-validation_count:])
    train = [record for record in records if int(record["seed"]) not in validation_seeds]
    validation = [record for record in records if int(record["seed"]) in validation_seeds]
    if not train or not validation:
        raise ValueError("Seed split produced an empty train or validation set")
    return train, validation


def _record_loss(model: CompactRolloutScorer, record: dict[str, Any], device: torch.device) -> torch.Tensor:
    state = torch.tensor(record["state_features"], dtype=torch.float32, device=device)
    actions = torch.tensor(record["action_features"], dtype=torch.float32, device=device)
    label = torch.tensor([int(record["label"])], dtype=torch.long, device=device)
    scores = model(state, actions).reshape(1, -1)
    return nn.functional.cross_entropy(scores, label)


def _evaluate(model: CompactRolloutScorer, records: list[dict[str, Any]], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for record in records:
            losses.append(float(_record_loss(model, record, device).item()))
    return sum(losses) / len(losses)


def train(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingResult:
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(dataset_path)
    train_records, validation_records = split_records(dataset["records"], validation_fraction)
    device_t = torch.device(device)
    model = CompactRolloutScorer(
        state_dim=STATE_FEATURE_DIM,
        action_dim=ACTION_FEATURE_DIM,
        hidden_dim=hidden_dim,
    ).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        random.shuffle(train_records)
        for record in train_records:
            optimizer.zero_grad(set_to_none=True)
            loss = _record_loss(model, record, device_t)
            loss.backward()
            optimizer.step()
        validation_loss = _evaluate(model, validation_records, device_t)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    save_compact_checkpoint(model, str(checkpoint_path))
    return TrainingResult(
        train_records=len(train_records),
        validation_records=len(validation_records),
        best_validation_loss=best_validation_loss,
        checkpoint_path=str(checkpoint_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    print(
        f"Saved {result.checkpoint_path}; train={result.train_records}, "
        f"validation={result.validation_records}, "
        f"best_validation_loss={result.best_validation_loss:.4f}"
    )


if __name__ == "__main__":
    main()
