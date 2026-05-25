#!/usr/bin/env python3
"""Evaluate a selected model across many seeds and report the positive runs.

By default this treats "positive" as a positive final episode reward, which is
the same scalar returned by `WyrmspanEnv.step()`. A margin-based mode is
available if you want to filter by player score minus automa score instead.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent
from test_models import OBS_BOOL_KEYS, OBS_LONG_KEYS


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    steps: int
    total_reward: float
    final_player_score: int
    final_automa_score: int
    score_margin: int
    positive: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_torch(obs, device):
    out = {}
    for key, value in obs.items():
        if key in OBS_LONG_KEYS:
            dtype = torch.long
        elif key in OBS_BOOL_KEYS:
            dtype = torch.bool
        else:
            dtype = torch.float32
        out[key] = torch.as_tensor(value, dtype=dtype, device=device).unsqueeze(0)
    return out


def load_agent(model_path: str, env: WyrmspanEnv, device: torch.device) -> WyrmspanAgent:
    agent = WyrmspanAgent(
        main_emb_dim=256,
        fusion_dim=256,
        action_vocab_size=env.action_token_vocab_size,
        action_pad_id=env.pad_token_id,
        max_action_tokens=env.max_action_tokens,
        max_queue_size=env.max_queue_size,
        max_hand_size=env.max_hand_size,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def parse_seed_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


def build_seeds(seed_start: int, num_seeds: int, seed_list: Optional[str]) -> List[int]:
    parsed = parse_seed_list(seed_list)
    if parsed is not None:
        return parsed
    return list(range(seed_start, seed_start + num_seeds))


def run_episode(env: WyrmspanEnv, agent: WyrmspanAgent, seed: int, device: torch.device, max_steps: int) -> EpisodeResult:
    set_seed(seed)

    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        obs_t = obs_to_torch(obs, device)
        with torch.no_grad():
            state_embedding, _ = agent.forward(obs_t)
            action_scores = agent.score_actions(
                state_embedding,
                obs_t["action_token_ids"],
                obs_t["action_token_mask"],
                obs_t["action_mask"],
            )[0].detach().cpu().numpy()

        legal_mask = obs["action_mask"].astype(bool)
        masked_scores = np.where(legal_mask, action_scores, -1e9)
        chosen_action = int(np.argmax(masked_scores))

        obs, reward, terminated, truncated, _ = env.step(chosen_action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        steps += 1

    score_margin = int(env.game_state.player.score - env.game_state.automa.score)
    return EpisodeResult(
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        final_player_score=int(env.game_state.player.score),
        final_automa_score=int(env.game_state.automa.score),
        score_margin=score_margin,
        positive=False,
    )


def result_value(result: EpisodeResult, metric: str) -> float:
    if metric == "reward":
        return result.total_reward
    if metric == "margin":
        return float(result.score_margin)
    raise ValueError(f"Unsupported metric: {metric}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on many seeds and report which runs are positive."
    )
    parser.add_argument("--model-path", required=True, help="Path to a PPO checkpoint")
    parser.add_argument("--seed-start", type=int, default=1, help="Start of the seed range (inclusive)")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of sequential seeds to test")
    parser.add_argument("--seed-list", type=str, default=None, help="Optional comma-separated explicit seed list")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum environment steps per seed")
    parser.add_argument(
        "--metric",
        choices=("reward", "margin"),
        default="reward",
        help="Metric used to decide whether a run is positive",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum metric value required for a seed to count as positive",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the full sweep summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = build_seeds(args.seed_start, args.num_seeds, args.seed_list)
    env = WyrmspanEnv()
    agent = load_agent(args.model_path, env, device)

    print(f"Evaluating model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Metric: {args.metric} threshold>={args.threshold}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)" if len(seeds) > 1 else f"Seeds: {seeds}")

    results: List[EpisodeResult] = []
    positive_results: List[EpisodeResult] = []

    for seed in seeds:
        result = run_episode(env, agent, seed, device, args.max_steps)
        value = result_value(result, args.metric)
        is_positive = value >= args.threshold
        result = EpisodeResult(
            seed=result.seed,
            steps=result.steps,
            total_reward=result.total_reward,
            final_player_score=result.final_player_score,
            final_automa_score=result.final_automa_score,
            score_margin=result.score_margin,
            positive=is_positive,
        )
        results.append(result)
        if is_positive:
            positive_results.append(result)
        print(
            f"seed={seed} metric={value:+.4f} reward={result.total_reward:+.4f} "
            f"margin={result.score_margin:+d} P={result.final_player_score} A={result.final_automa_score} "
            f"{'POS' if is_positive else 'neg'}"
        )

    print("\nSummary")
    print(f"Positive runs: {len(positive_results)}/{len(results)}")
    if positive_results:
        print("Seeds with positive results:")
        print(", ".join(str(result.seed) for result in positive_results))
    else:
        print("No seeds met the positive threshold.")

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("logs", f"positive_seeds_{stamp}.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "device": str(device),
        "metric": args.metric,
        "threshold": args.threshold,
        "seeds": seeds,
        "positive_seeds": [result.seed for result in positive_results],
        "results": [asdict(result) for result in results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())