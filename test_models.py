import argparse
import json
import os
import random
from datetime import datetime
import logging

import numpy as np
import torch

from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent
from game_states import OBJECTIVE_TILES

OBS_LONG_KEYS = {
    "card_display_dragons",
    "card_display_caves",
    "hand_card_ids",
    "slot_types",
    "dragons_on_slots",
    "other_indices",
    "queue_tokens",
    "action_token_ids",
}

OBS_BOOL_KEYS = {
    "hand_card_mask",
    "queue_pad_mask",
    "queue_slot_mask",
    "action_token_mask",
    "action_mask",
}


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
    state_dict = ckpt.get("model", ckpt)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def decode_tokens(env: WyrmspanEnv, token_ids):
    tokens = []
    for tid in token_ids:
        tid = int(tid)
        if tid == env.pad_token_id:
            continue
        if 0 <= tid < len(env.token_strings):
            tokens.append(env.token_strings[tid])
        else:
            tokens.append("<oov>")
    return tokens


def log_game_state(game_state, logger, step_idx):
    logger.warning(f"\n=== Step {step_idx} ===")
    logger.warning(f"\n{game_state}")
    logger.warning(f"\n{game_state.get_card_display_string()}")
    logger.warning(f"\n{game_state.board['round_tracker']}")
    logger.warning(f"> Phase: {game_state.phase}")
    logger.warning(f">>> Player Score: {game_state.player.score} | Automa Score: {game_state.automa.score}")

def score_logger(logger, obs, env: WyrmspanEnv, scores=None, probs=None, top_k: int = 5):
    logger.warning("\n>>> Action Scores:")
    legal = np.flatnonzero(obs["action_mask"])
    ranked = legal[np.argsort(-scores[legal])]
    logger.warning("Top actions:")
    for rank, idx in enumerate(ranked[:top_k], start=1):
        token_ids = obs["action_token_ids"][idx]
        logger.warning(
            f"  {rank}. idx={idx} logit={scores[idx]:.4f} prob={probs[idx]:.4f} tokens={decode_tokens(env, token_ids)}"
        )
    logger.warning(">>>\n")


def run(model_path: str, seed: int, output_name: str, device: str = "auto", max_steps: int = 500):
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    
    logging.basicConfig(
        filename=(output_name + ".log"),
        # level=logging.DEBUG,
        # level=logging.INFO,
        level=logging.WARNING,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

        
    set_seed(seed)
    env = WyrmspanEnv()
    agent = load_agent(model_path, env, device_t)

    obs, _ = env.reset(seed=seed)
    done = False
    step_idx = 0
    total_reward = 0.0
    logs = []

    print(f"Running model test with model_path={model_path}, seed={seed}, output_name={output_name}, device={device_t}, max_steps={max_steps}")
    logger.warning(f"Running model test with model_path={model_path}, seed={seed}, output_name={output_name}, device={device_t}, max_steps={max_steps}")
    
    objectives = env.game_state.board["round_tracker"]["objectives"]
    logger.warning("> Objectives Drawn:")
    for i, (idx,side) in enumerate(objectives):
        logger.warning(f"Round {i + 1}: {OBJECTIVE_TILES[idx][side]['text']}\n")
    logger.warning("\n> Initial Game State:")
    log_game_state(env.game_state, logger, 0)

    while not done and step_idx < max_steps:
        obs_t = obs_to_torch(obs, device_t)
        with torch.no_grad():
            state_embedding, state_value = agent.forward(obs_t)
            action_scores = agent.score_actions(
                state_embedding,
                obs_t["action_token_ids"],
                obs_t["action_token_mask"],
                obs_t["action_mask"],
            )[0].detach().cpu().numpy()

        legal_mask = obs["action_mask"].astype(bool)
        masked_scores = np.where(legal_mask, action_scores, -1e9)
        max_logit = np.max(masked_scores)
        exp_scores = np.exp(masked_scores - max_logit)
        exp_scores[~legal_mask] = 0.0
        prob_sum = float(exp_scores.sum())
        probs = exp_scores / max(prob_sum, 1e-8)

        log_game_state(env.game_state, logger, step_idx=step_idx)
        score_logger(logger, obs, env, scores=action_scores, probs=probs, top_k=5)
        logger.warning(f"* State Value: {state_value.item():.3f}")

        chosen_action = int(np.argmax(masked_scores))
        logs.append(
            {
                "step": step_idx,
                "state_value": float(state_value.item()),
                "chosen_action": chosen_action,
                "chosen_logit": float(action_scores[chosen_action]),
                "chosen_prob": float(probs[chosen_action]),
                "top_actions": [
                    {
                        "action_index": int(i),
                        "logit": float(action_scores[i]),
                        "prob": float(probs[i]),
                        "tokens": decode_tokens(env, obs["action_token_ids"][i]),
                    }
                    for i in np.flatnonzero(legal_mask)[np.argsort(-action_scores[legal_mask])][:5]
                ],
            }
        )

        obs, reward, terminated, truncated, _ = env.step(chosen_action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        step_idx += 1

    # log final game state
    log_game_state(env.game_state, logger, step_idx=step_idx)
    logger.warning(f"Episode finished after {step_idx} steps with total_reward={total_reward:.3f}")

    result = {
        "model_path": model_path,
        "seed": seed,
        "steps": step_idx,
        "total_reward": total_reward,
        "final_player_score": env.game_state.player.score,
        "final_automa_score": env.game_state.automa.score,
        "logs": logs,
    }

    os.makedirs(os.path.dirname(output_name + ".json") or ".", exist_ok=True)
    with open(output_name + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to {output_name}")
    print(
        f"Final score: player={env.game_state.player.score} automa={env.game_state.automa.score} total_reward={total_reward:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Load a PPO-trained Wyrmspan model and log a seeded run.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a PPO checkpoint")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the environment and RNGs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for run results")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum number of env steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join("logs", f"model_test_{stamp}_seed{args.seed}")
    run(args.model_path, args.seed, output, device=args.device, max_steps=args.max_steps)
