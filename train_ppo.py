import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent
from playout_compare import RolloutCache, get_sim_algo


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


def stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    stacked = {}
    for key in obs_list[0].keys():
        stacked[key] = np.stack([obs[key] for obs in obs_list], axis=0)
    return stacked


def obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        if key in OBS_LONG_KEYS:
            dtype = torch.long
        elif key in OBS_BOOL_KEYS:
            dtype = torch.bool
        else:
            dtype = torch.float32
        out[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return out


class VectorEnv:
    def __init__(self, num_envs: int, seed: Optional[int] = None):
        self.num_envs = num_envs
        self.seed = seed
        self.envs = [WyrmspanEnv() for _ in range(num_envs)]

    def reset(self) -> Dict[str, np.ndarray]:
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = None if self.seed is None else self.seed + i
            obs, _ = env.reset(seed=env_seed)
            obs_list.append(obs)
        return stack_obs(obs_list)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(int(actions[i]))
            done = bool(terminated or truncated)
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards[i] = reward
            dones[i] = done
            infos.append(info)
        return stack_obs(obs_list), rewards, dones, infos


class RolloutBuffer:
    def __init__(self, rollout_length: int, num_envs: int, obs_example: Dict[str, np.ndarray]):
        self.rollout_length = rollout_length
        self.num_envs = num_envs

        self.obs: Dict[str, torch.Tensor] = {}
        for key, value in obs_example.items():
            shape = (rollout_length, num_envs) + value.shape[1:]
            if key in OBS_LONG_KEYS:
                dtype = torch.long
            elif key in OBS_BOOL_KEYS:
                dtype = torch.bool
            else:
                dtype = torch.float32
            self.obs[key] = torch.zeros(shape, dtype=dtype)

        self.actions = torch.zeros((rollout_length, num_envs), dtype=torch.long)
        self.logprobs = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.values = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((rollout_length, num_envs), dtype=torch.bool)

    def add(
        self,
        step: int,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for key, value in obs.items():
            self.obs[key][step].copy_(value.cpu())
        self.actions[step].copy_(actions.cpu())
        self.logprobs[step].copy_(logprobs.cpu())
        self.values[step].copy_(values.cpu())
        self.rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32))
        self.dones[step].copy_(torch.as_tensor(dones, dtype=torch.bool))

    def flatten_obs(self) -> Dict[str, torch.Tensor]:
        flat = {}
        for key, value in self.obs.items():
            flat[key] = value.reshape((-1,) + value.shape[2:])
        return flat


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rollout_length, num_envs = rewards.shape
    advantages = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
    last_advantage = torch.zeros(num_envs, dtype=torch.float32)

    for t in reversed(range(rollout_length)):
        next_non_terminal = 1.0 - dones[t].float()
        next_values = next_value if t == rollout_length - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def evaluate_agent(model: WyrmspanAgent, device: torch.device, episodes: int, seed: Optional[int] = None) -> dict:
    env = WyrmspanEnv()
    wins = 0
    score_diffs: List[float] = []
    returns: List[float] = []

    for ep in range(episodes):
        env_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=env_seed)
        done = False
        ep_return = 0.0

        while not done:
            obs_batch = {k: v[None, ...] for k, v in obs.items()}
            obs_t = obs_to_torch(obs_batch, device)
            with torch.no_grad():
                logits, _ = model.policy_value(obs_t)
            action = int(torch.argmax(logits, dim=1).item())
            obs, reward, done, _, _ = env.step(action)
            ep_return += float(reward)

        score_diff = float(env.game_state.player.score - env.game_state.automa.score)
        wins += 1 if score_diff >= 0 else 0
        score_diffs.append(score_diff)
        returns.append(ep_return)

    return {
        "win_rate": wins / episodes,
        "score_diff": float(np.mean(score_diffs)),
        "return": float(np.mean(returns)),
    }


def evaluate_baseline(
    episodes: int,
    policy: str,
    seed: Optional[int] = None,
    algo_name: Optional[str] = None,
    algo_kwargs: Optional[dict] = None,
) -> dict:
    env = WyrmspanEnv()
    wins = 0
    score_diffs: List[float] = []
    returns: List[float] = []

    sim_algo = None
    rollout_cache = None
    if policy == "heuristic":
        if algo_name is None:
            raise ValueError("algo_name required for heuristic evaluation")
        sim_algo = get_sim_algo(algo_name, algo_kwargs or {})
        if algo_name == "strategic_objective_aware":
            rollout_cache = RolloutCache()

    for ep in range(episodes):
        env_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=env_seed)
        done = False
        ep_return = 0.0

        while not done:
            if policy == "random":
                legal = np.flatnonzero(obs["action_mask"])  # 1 for legal actions
                action = int(np.random.choice(legal))
            elif policy == "heuristic":
                action = int(sim_algo(env.game_state, rollout_cache))
            else:
                raise ValueError(f"Unknown policy: {policy}")

            obs, reward, done, _, _ = env.step(action)
            ep_return += float(reward)

        score_diff = float(env.game_state.player.score - env.game_state.automa.score)
        wins += 1 if score_diff >= 0 else 0
        score_diffs.append(score_diff)
        returns.append(ep_return)

    return {
        "win_rate": wins / episodes,
        "score_diff": float(np.mean(score_diffs)),
        "return": float(np.mean(returns)),
    }


@dataclass
class PPOConfig:
    seed: int = 123
    total_updates: int = 200
    rollout_length: int = 128
    num_envs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    eval_interval: int = 20
    eval_episodes: int = 10
    save_interval: int = 50
    save_dir: str = "checkpoints/ppo"
    device: str = "auto"


def train(cfg: PPOConfig) -> None:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    set_seed(cfg.seed)

    envs = VectorEnv(cfg.num_envs, seed=cfg.seed)
    obs = envs.reset()

    env_spec = envs.envs[0]
    model = WyrmspanAgent(
        main_emb_dim=256,
        fusion_dim=256,
        action_vocab_size=env_spec.action_token_vocab_size,
        action_pad_id=env_spec.pad_token_id,
        max_action_tokens=env_spec.max_action_tokens,
        max_queue_size=env_spec.max_queue_size,
        max_hand_size=env_spec.max_hand_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    batch_size = cfg.rollout_length * cfg.num_envs
    if batch_size % cfg.minibatch_size != 0:
        raise ValueError("minibatch_size must divide rollout_length * num_envs")

    os.makedirs(cfg.save_dir, exist_ok=True)

    global_step = 0
    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
    recent_returns: List[float] = []
    recent_lengths: List[int] = []

    for update in range(1, cfg.total_updates + 1):
        model.train()
        buffer = RolloutBuffer(cfg.rollout_length, cfg.num_envs, obs)

        for step in range(cfg.rollout_length):
            obs_t = obs_to_torch(obs, device)

            with torch.no_grad():
                logits, values = model.policy_value(obs_t)
                if not torch.all(obs_t["action_mask"].any(dim=1)):
                    raise RuntimeError("No legal actions available for at least one env")
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, _ = envs.step(actions_np)

            buffer.add(
                step,
                obs_t,
                actions,
                logprobs,
                values.squeeze(-1),
                rewards,
                dones,
            )

            episode_returns += rewards
            episode_lengths += 1
            for i, done in enumerate(dones):
                if done:
                    recent_returns.append(float(episode_returns[i]))
                    recent_lengths.append(int(episode_lengths[i]))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0

            obs = next_obs
            global_step += cfg.num_envs

        with torch.no_grad():
            next_obs_t = obs_to_torch(obs, device)
            _, next_values = model.policy_value(next_obs_t)

        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values,
            next_values.squeeze(-1).cpu(),
            cfg.gamma,
            cfg.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = buffer.flatten_obs()
        flat_actions = buffer.actions.reshape(-1)
        flat_logprobs = buffer.logprobs.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        b_inds = np.arange(batch_size)
        clip_fracs = []
        approx_kls = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_inds = b_inds[start : start + cfg.minibatch_size]

                mb_obs = {key: value[mb_inds].to(device) for key, value in flat_obs.items()}
                mb_actions = flat_actions[mb_inds].to(device)
                mb_logprobs = flat_logprobs[mb_inds].to(device)
                mb_returns = flat_returns[mb_inds].to(device)
                mb_advantages = flat_advantages[mb_inds].to(device)

                logits, values = model.policy_value(mb_obs)
                values = values.squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - mb_logprobs)
                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_loss = 0.5 * F.mse_loss(values, mb_returns)
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                approx_kl = (mb_logprobs - new_logprobs).mean().item()
                approx_kls.append(approx_kl)
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                clip_fracs.append(clip_frac)

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

        avg_return = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
        avg_length = float(np.mean(recent_lengths[-100:])) if recent_lengths else 0.0
        print(
            f"update={update} step={global_step} "
            f"return={avg_return:.3f} len={avg_length:.1f} "
            f"clip={np.mean(clip_fracs):.3f} kl={np.mean(approx_kls):.4f}"
        )

        if cfg.eval_interval and update % cfg.eval_interval == 0:
            model.eval()
            agent_metrics = evaluate_agent(model, device, cfg.eval_episodes, seed=cfg.seed + 1000)
            random_metrics = evaluate_baseline(cfg.eval_episodes, policy="random", seed=cfg.seed + 2000)
            heuristic_metrics = evaluate_baseline(
                cfg.eval_episodes,
                policy="heuristic",
                seed=cfg.seed + 3000,
                algo_name="greedy_action_priority",
                algo_kwargs={},
            )
            print(
                "eval_agent "
                f"win={agent_metrics['win_rate']:.2f} diff={agent_metrics['score_diff']:.2f} ret={agent_metrics['return']:.3f}"
            )
            print(
                "eval_random "
                f"win={random_metrics['win_rate']:.2f} diff={random_metrics['score_diff']:.2f} ret={random_metrics['return']:.3f}"
            )
            print(
                "eval_greedy "
                f"win={heuristic_metrics['win_rate']:.2f} diff={heuristic_metrics['score_diff']:.2f} ret={heuristic_metrics['return']:.3f}"
            )
            model.train()

        if cfg.save_interval and update % cfg.save_interval == 0:
            checkpoint_path = os.path.join(cfg.save_dir, f"ppo_update_{update}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "config": cfg.__dict__,
                },
                checkpoint_path,
            )


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train Wyrmspan agent with PPO")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    return PPOConfig(
        seed=args.seed,
        total_updates=args.total_updates,
        rollout_length=args.rollout_length,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
