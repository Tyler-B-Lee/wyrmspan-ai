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
        self.final_game_states = [None] * num_envs  # Store final game state before reset

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
                # Store final game state before reset
                self.final_game_states[i] = env.game_state.make_copy() if hasattr(env.game_state, 'make_copy') else env.game_state
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


def evaluate_agent(model: WyrmspanAgent, device: torch.device, episodes: int, seed: Optional[int] = None, num_envs: int = 8) -> dict:
    """Vectorized batch evaluation: run multiple episodes in parallel using VectorEnv."""
    num_envs = min(num_envs, episodes)
    envs = VectorEnv(num_envs, seed=seed)
    obs = envs.reset()

    wins = 0
    score_diffs: List[float] = []
    returns: List[float] = []
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_count = 0

    while episode_count < episodes:
        obs_t = obs_to_torch(obs, device)
        with torch.no_grad():
            logits, _ = model.policy_value(obs_t)
            action_mask = obs_t["action_mask"].bool()
            if logits.shape != action_mask.shape:
                raise RuntimeError(f"logits/action_mask shape mismatch in eval: {logits.shape} vs {action_mask.shape}")
            masked_logits = logits.masked_fill(~action_mask, -1e9)
            actions = torch.argmax(masked_logits, dim=1)

        actions_np = actions.cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(actions_np)
        episode_returns += rewards

        for i, done in enumerate(dones):
            if done:
                final_state = envs.final_game_states[i]
                if final_state is not None:
                    score_diff = float(final_state.player.score - final_state.automa.score)
                    wins += 1 if score_diff >= 0 else 0
                    score_diffs.append(score_diff)
                returns.append(float(episode_returns[i]))
                episode_count += 1
                episode_returns[i] = 0.0

        obs = next_obs

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
    num_envs: int = 8,
) -> dict:
    """Vectorized batch evaluation for baseline policies (random or heuristic)."""
    num_envs = min(num_envs, episodes)
    envs = VectorEnv(num_envs, seed=seed)
    obs = envs.reset()

    wins = 0
    score_diffs: List[float] = []
    returns: List[float] = []
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_count = 0

    sim_algo = None
    rollout_cache = None
    if policy == "heuristic":
        if algo_name is None:
            raise ValueError("algo_name required for heuristic evaluation")
        sim_algo = get_sim_algo(algo_name, algo_kwargs or {})
        if algo_name == "strategic_objective_aware":
            rollout_cache = RolloutCache()

    while episode_count < episodes:
        actions = np.zeros(num_envs, dtype=np.int32)
        for i, env in enumerate(envs.envs):
            if policy == "random":
                legal = np.flatnonzero(obs["action_mask"][i])
                actions[i] = int(np.random.choice(legal))
            elif policy == "heuristic":
                actions[i] = int(sim_algo(env.game_state, rollout_cache))
            else:
                raise ValueError(f"Unknown policy: {policy}")

        next_obs, rewards, dones, _ = envs.step(actions)
        episode_returns += rewards

        for i, done in enumerate(dones):
            if done:
                final_state = envs.final_game_states[i]
                if final_state is not None:
                    score_diff = float(final_state.player.score - final_state.automa.score)
                    wins += 1 if score_diff >= 0 else 0
                    score_diffs.append(score_diff)
                returns.append(float(episode_returns[i]))
                episode_count += 1
                episode_returns[i] = 0.0

        obs = next_obs

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
    dropout: float = 0.0
    eval_interval: int = 20
    eval_episodes: int = 10
    save_interval: int = 50
    save_dir: str = "checkpoints/ppo"
    device: str = "auto"
    resume: bool = False
    resume_path: Optional[str] = None


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
        dropout=cfg.dropout,
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

    start_update = 1

    # Resume from checkpoint if requested
    if getattr(cfg, "resume", False) or getattr(cfg, "resume_path", None):
        ckpt_path = getattr(cfg, "resume_path", None)
        if ckpt_path is None:
            if os.path.isdir(cfg.save_dir):
                files = [f for f in os.listdir(cfg.save_dir) if f.startswith("ppo_update_") and f.endswith(".pt")]
                if files:
                    files.sort()
                    ckpt_path = os.path.join(cfg.save_dir, files[-1])
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model", {}))
            if "optimizer" in ckpt and optimizer is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state: {e}")
            start_update = ckpt.get("update", 0) + 1
            global_step = ckpt.get("global_step", 0)
            if "config" in ckpt:
                print("Checkpoint config:", ckpt["config"]) 
        else:
            print("No checkpoint found to resume; starting from scratch.")

    for update in range(start_update, cfg.total_updates + 1):
        print(f"Starting update {update}/{cfg.total_updates} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        model.train()
        buffer = RolloutBuffer(cfg.rollout_length, cfg.num_envs, obs)
        rollout_illegal_samples = 0

        for step in range(cfg.rollout_length):
            obs_t = obs_to_torch(obs, device)

            with torch.no_grad():
                logits, values = model.policy_value(obs_t)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits detected during rollout collection")
                if not torch.isfinite(values).all():
                    raise RuntimeError("Non-finite value predictions detected during rollout collection")

                action_mask = obs_t["action_mask"].bool()
                if logits.shape != action_mask.shape:
                    raise RuntimeError(f"logits/action_mask shape mismatch: {logits.shape} vs {action_mask.shape}")
                if not torch.all(action_mask.any(dim=1)):
                    raise RuntimeError("No legal actions available for at least one env")

                masked_logits = logits.masked_fill(~action_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

                sampled_is_legal = action_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
                rollout_illegal_samples += int((~sampled_is_legal).sum().item())

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
                    print(f"Env {i} finished episode with return {episode_returns[i]:.3f} and length {episode_lengths[i]}")
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
        policy_losses = []
        value_losses = []
        entropies = []
        grad_norms = []
        logits_max = []
        logits_min = []
        update_illegal_actions = 0

        print(f"Updating policy for {cfg.update_epochs} epochs with batch size {cfg.minibatch_size}...")
        for epoch in range(cfg.update_epochs):
            print(f"> Epoch {epoch + 1}/{cfg.update_epochs}...")
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
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits detected during PPO update")
                if not torch.isfinite(values).all():
                    raise RuntimeError("Non-finite value predictions detected during PPO update")

                action_mask = mb_obs["action_mask"].bool()
                if logits.shape != action_mask.shape:
                    raise RuntimeError(f"minibatch logits/action_mask shape mismatch: {logits.shape} vs {action_mask.shape}")
                if not torch.all(action_mask.any(dim=1)):
                    raise RuntimeError("No legal actions available for at least one minibatch sample")

                masked_logits = logits.masked_fill(~action_mask, -1e9)
                logits_max.append(float(masked_logits.max().item()))
                logits_min.append(float(masked_logits.min().item()))

                chosen_legal = action_mask.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                update_illegal_actions += int((~chosen_legal).sum().item())

                dist = torch.distributions.Categorical(logits=masked_logits)
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                grad_norms.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))

                approx_kl = (mb_logprobs - new_logprobs).mean().item()
                approx_kls.append(approx_kl)
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                clip_fracs.append(clip_frac)

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    print(f"\t> Early stopping at epoch {epoch}, step {start} due to reaching target KL: {approx_kl:.4f} > {cfg.target_kl}")
                    break

        avg_return = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
        avg_length = float(np.mean(recent_lengths[-100:])) if recent_lengths else 0.0
        print(
            f"update={update} step={global_step} "
            f"return={avg_return:.3f} len={avg_length:.1f} "
            f"clip={np.mean(clip_fracs):.3f} kl={np.mean(approx_kls):.4f}"
        )
        print(
            f"diag policy={np.mean(policy_losses):.4f} value={np.mean(value_losses):.4f} "
            f"entropy={np.mean(entropies):.4f} grad={np.mean(grad_norms):.4f} "
            f"logit_min={np.min(logits_min):.2f} logit_max={np.max(logits_max):.2f} "
            f"illegal_rollout={rollout_illegal_samples} illegal_update={update_illegal_actions}"
        )

        if cfg.eval_interval and update % cfg.eval_interval == 0:
            print(f"Evaluating agent at update {update}...")
            model.eval()
            eval_seed = random.randint(0, 100_000_000)
            agent_metrics = evaluate_agent(model, device, cfg.eval_episodes, seed=eval_seed, num_envs=8)
            print(
                "eval_agent "
                f"win={agent_metrics['win_rate']:.2f} diff={agent_metrics['score_diff']:.2f} ret={agent_metrics['return']:.3f}"
            )
            random_metrics = evaluate_baseline(cfg.eval_episodes, policy="random", seed=eval_seed, num_envs=8)
            print(
                "eval_random "
                f"win={random_metrics['win_rate']:.2f} diff={random_metrics['score_diff']:.2f} ret={random_metrics['return']:.3f}"
            )
            heuristic_metrics = evaluate_baseline(
                cfg.eval_episodes,
                policy="heuristic",
                seed=eval_seed,
                algo_name="greedy_action_priority",
                algo_kwargs={},
                num_envs=8,
            )
            print(
                "eval_greedy "
                f"win={heuristic_metrics['win_rate']:.2f} diff={heuristic_metrics['score_diff']:.2f} ret={heuristic_metrics['return']:.3f}"
            )
            model.train()

        if cfg.save_interval and update % cfg.save_interval == 0:
            checkpoint_path = os.path.join(cfg.save_dir, f"ppo_update_{update}_{global_step}_eval_{agent_metrics['return']:.3f}.pt")
            print(f"Saving checkpoint at update {update} to {checkpoint_path}...")
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
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in save-dir if available")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--dropout", type=float, default=0.1)

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
        dropout=args.dropout,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        resume=args.resume,
        resume_path=args.resume_path,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
