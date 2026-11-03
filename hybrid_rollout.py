"""Bridge between compact rollout action selection and the large state critic."""

from __future__ import annotations

from typing import Any, Optional

import torch

from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent
from model_arch_legacy import WyrmspanAgent as LegacyWyrmspanAgent


OBS_LONG_KEYS = {
    "card_display_dragons",
    "card_display_caves",
    "hand_card_ids",
    "card_display_dragon_tokens",
    "card_display_cave_tokens",
    "hand_card_tokens",
    "slot_card_tokens",
    "slot_types",
    "dragons_on_slots",
    "other_indices",
    "queue_tokens",
    "action_token_ids",
}

OBS_BOOL_KEYS = {
    "hand_card_mask",
    "card_display_dragon_token_mask",
    "card_display_cave_token_mask",
    "hand_card_token_mask",
    "slot_card_token_mask",
    "queue_pad_mask",
    "queue_slot_mask",
    "action_token_mask",
    "action_mask",
}


class LargeModelValueEvaluator:
    """Evaluate a raw GameState with the large model's critic head."""

    def __init__(
        self,
        model: Any,
        env: Optional[WyrmspanEnv] = None,
        device: str = "cpu",
        value_scale: float = 1.0,
        value_bias: float = 0.0,
        value_clip: Optional[tuple[float, float]] = None,
    ):
        self.model = model.to(device).eval()
        self.env = env or WyrmspanEnv()
        self.device = torch.device(device)
        self.value_scale = value_scale
        self.value_bias = value_bias
        self.value_clip = value_clip

    def _obs_to_torch(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        tensors = {}
        for key, value in observation.items():
            if key in OBS_LONG_KEYS:
                dtype = torch.long
            elif key in OBS_BOOL_KEYS:
                dtype = torch.bool
            else:
                dtype = torch.float32
            tensors[key] = torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)
        return tensors

    @torch.inference_mode()
    def evaluate(self, game_state: Any) -> float:
        self.env.game_state = game_state.make_copy()
        observation = self.env._get_obs()
        _, state_value = self.model(self._obs_to_torch(observation))
        value = float(state_value.reshape(-1)[0].item())
        value = value * self.value_scale + self.value_bias
        if self.value_clip is not None:
            value = max(self.value_clip[0], min(self.value_clip[1], value))
        return value

    def __call__(self, game_state: Any) -> float:
        return self.evaluate(game_state)


def load_large_value_evaluator(
    checkpoint_path: str,
    device: str = "cpu",
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    value_clip: Optional[tuple[float, float]] = None,
    architecture: str = "auto",
) -> LargeModelValueEvaluator:
    """Build the semantic large model from an existing PPO checkpoint."""
    env = WyrmspanEnv()
    device_t = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_t)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if architecture == "auto":
        architecture = "semantic" if any(key.startswith("semantic_card_encoder.") for key in state_dict) else "legacy"
    if architecture not in {"semantic", "legacy"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    agent_class = WyrmspanAgent if architecture == "semantic" else LegacyWyrmspanAgent
    model_kwargs = {
        "main_emb_dim": 256,
        "fusion_dim": 256,
        "action_vocab_size": env.action_token_vocab_size,
        "action_pad_id": env.pad_token_id,
        "max_action_tokens": env.max_action_tokens,
        "max_queue_size": env.max_queue_size,
        "max_hand_size": env.max_hand_size,
        "dropout": 0.0,
    }
    if architecture == "semantic":
        model_kwargs.update(
            card_vocab_size=env.card_token_vocab_size,
            max_card_tokens=env.max_card_tokens,
        )
    model = agent_class(**model_kwargs).to(device_t)
    model.load_state_dict(state_dict, strict=True)
    return LargeModelValueEvaluator(
        model,
        env=env,
        device=device,
        value_scale=value_scale,
        value_bias=value_bias,
        value_clip=value_clip,
    )
