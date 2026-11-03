"""Compact CPU-first policy for ranking already-legal rollout actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn
from collections import OrderedDict

from game_states import CAVE_NAMES, OBJECTIVE_TILES


SCHEMA_VERSION = 2
ACTION_FEATURE_DIM = 64
STATE_FEATURE_DIM = 64

ACTION_TYPES = (
    "play_dragon",
    "play_cave",
    "explore",
    "gain_resource",
    "lay_egg",
    "gain_dragon",
    "gain_cave",
    "draw_decision",
    "discard_dragon",
    "discard_cave",
    "tuck_from",
    "make_payment",
    "choice",
    "sequence",
    "random",
    "pass",
    "skip",
    "skip_opr",
)

RESOURCE_TYPES = (
    "meat",
    "milk",
    "grain",
    "fruit",
    "crystal",
    "gold",
    "egg",
    "coin",
)

_WRAPPER_TYPES = ("make_payment", "adv_effects", "choice", "sequence", "random")


@dataclass(frozen=True)
class EncodedChoices:
    """Features aligned one-to-one with a state's current legal choices."""

    state_features: torch.Tensor
    action_features: torch.Tensor
    action_indices: tuple[int, ...]
    summaries: tuple[str, ...]
    overflow_count: int


class RolloutFeatureAdapter:
    """Convert raw game states and nested legal actions into fixed features."""

    schema_version = SCHEMA_VERSION
    action_feature_dim = ACTION_FEATURE_DIM
    state_feature_dim = STATE_FEATURE_DIM

    def __init__(self, action_cache_size: int = 8192):
        self.action_cache_size = max(0, action_cache_size)
        self._action_feature_cache: OrderedDict[str, tuple[torch.Tensor, str, int]] = OrderedDict()

    def encode_choices(self, game_state: Any, device: Optional[torch.device] = None) -> EncodedChoices:
        choices = game_state.current_choice or []
        state_features = self.encode_state(game_state, device=device)
        rows = []
        summaries = []
        overflow_count = 0
        for index, action in enumerate(choices):
            features, summary, overflow = self.encode_action(action, device=device)
            rows.append(features)
            summaries.append(summary)
            overflow_count += overflow

        if rows:
            action_features = torch.stack(rows, dim=0)
        else:
            action_features = torch.empty((0, self.action_feature_dim), dtype=torch.float32, device=device)
        return EncodedChoices(
            state_features=state_features,
            action_features=action_features,
            action_indices=tuple(range(len(choices))),
            summaries=tuple(summaries),
            overflow_count=overflow_count,
        )

    def encode_state(self, game_state: Any, device: Optional[torch.device] = None) -> torch.Tensor:
        player = game_state.player
        features = torch.zeros(self.state_feature_dim, dtype=torch.float32, device=device)
        resources = getattr(player, "resources", {}) or {}
        for offset, name in enumerate(RESOURCE_TYPES):
            features[offset] = self._bounded(resources.get(name, 0), 10.0)
        features[8] = self._bounded(getattr(player, "coins", 0), 20.0)
        features[9] = self._bounded(getattr(player, "score", 0), 100.0)
        features[10] = self._bounded(sum(getattr(player, "egg_totals", {}).values()), 40.0)
        features[11] = self._bounded(len(getattr(player, "dragon_hand", ())), 15.0)
        features[12] = self._bounded(len(getattr(player, "cave_hand", ())), 15.0)
        automa = getattr(game_state, "automa", None)
        features[13] = self._bounded(getattr(automa, "score", 0), 100.0)
        features[14] = self._bounded(len(getattr(game_state, "dragon_deck", ())), 183.0)
        features[15] = self._bounded(len(getattr(game_state, "cave_deck", ())), 75.0)
        features[16] = self._bounded(len(getattr(game_state, "event_queue", ())), 10.0)
        features[17] = self._bounded(len(game_state.current_choice or ()), 180.0)
        features[18] = float(bool(getattr(player, "passed_this_round", False)))
        features[19] = self._bounded(getattr(game_state, "turn", 0), 30.0)
        features[20] = self._bounded(game_state.board.get("round_tracker", {}).get("round", 0), 4.0)
        features[21] = float(getattr(game_state, "phase", "") == "end_game")
        objective_progress = self._objective_progress(game_state)
        features[22] = objective_progress[0]
        features[23] = objective_progress[1]

        dragons_played = getattr(player, "dragons_played", {}) or {}
        caves_played = getattr(player, "caves_played", {}) or {}
        explored = getattr(player, "times_explored", {}) or {}
        for cave_index, cave_name in enumerate(CAVE_NAMES[:4]):
            dragon_count = sum(card is not None for card in dragons_played.get(cave_name, ()))
            cave_count = sum(card not in (None, -1) for card in caves_played.get(cave_name, ()))
            features[24 + cave_index] = self._bounded(dragon_count, 4.0)
            features[28 + cave_index] = self._bounded(cave_count, 4.0)
            features[32 + cave_index] = self._bounded(explored.get(cave_name, 0), 10.0)
        return features

    def encode_action(
        self,
        action: Mapping[str, Any],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, str, int]:
        cache_key = repr(action)
        cached = self._action_feature_cache.get(cache_key)
        if cached is not None:
            self._action_feature_cache.move_to_end(cache_key)
            cached_features, summary, overflow = cached
            features = cached_features if device is None else cached_features.to(device)
            return features, summary, overflow

        features = torch.zeros(self.action_feature_dim, dtype=torch.float32, device=device)
        keys = tuple(self._collect_keys(action))
        key_set = set(keys)
        known_keys = set(ACTION_TYPES)
        for key in keys:
            if key in known_keys:
                features[ACTION_TYPES.index(key)] = 1.0
        for offset, wrapper in enumerate(_WRAPPER_TYPES):
            features[18 + offset] = float(wrapper in keys)

        costs = self._collect_costs(action)
        for offset, resource in enumerate(RESOURCE_TYPES):
            features[23 + offset] = -self._bounded(costs.get(resource, 0), 10.0)
        features[31] = -self._bounded(costs.get("any_resource", 0), 10.0)
        features[32] = -self._bounded(costs.get("dragon_card", 0), 10.0)
        features[33] = -self._bounded(costs.get("cave_card", 0), 10.0)
        features[34] = float(bool(costs))

        card_ids = self._collect_card_ids(action)
        features[35] = self._bounded(card_ids["dragon"], 183.0)
        features[36] = self._bounded(card_ids["cave"], 75.0)
        features[37] = self._bounded(len(card_ids["all"]), 8.0)
        features[38] = self._bounded(self._count_key_values(action, "choice"), 10.0)
        features[39] = self._bounded(self._count_key_values(action, "sequence"), 10.0)
        features[40] = self._bounded(self._depth(action), 8.0)
        features[41] = float("pass" in key_set)
        features[42] = float("skip" in key_set or "skip_opr" in key_set)
        features[43] = float("random" in key_set)
        features[44] = float("coords" in key_set)
        features[45] = float("chosen_id" in key_set)
        features[46] = float("free" in key_set)
        features[47] = float("discount" in key_set)
        features[48] = self._bounded(len(keys), 32.0)
        location_values = self._collect_string_values(action, {"cave_name", "cave_location", "location"})
        for location_index, cave_name in enumerate(CAVE_NAMES):
            if cave_name in location_values:
                features[49 + location_index] = 1.0

        summary = "/".join(keys[:6]) or "empty"
        overflow = max(0, len(keys) - 32) + max(0, self._depth(action) - 8)
        if self.action_cache_size:
            self._action_feature_cache[cache_key] = (features.detach().cpu(), summary, overflow)
            self._action_feature_cache.move_to_end(cache_key)
            if len(self._action_feature_cache) > self.action_cache_size:
                self._action_feature_cache.popitem(last=False)
        return features, summary, overflow

    @staticmethod
    def _bounded(value: Any, scale: float) -> float:
        try:
            return max(-1.0, min(1.0, float(value) / scale))
        except (TypeError, ValueError):
            return 0.0

    def _collect_keys(self, value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                key = str(key)
                if key not in found:
                    found.append(key)
                found.extend(item for item in self._collect_keys(child) if item not in found)
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.extend(item for item in self._collect_keys(child) if item not in found)
        return found

    def _collect_costs(self, value: Any) -> dict[str, float]:
        costs: dict[str, float] = {}
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "cost" and isinstance(child, Mapping):
                    for resource, amount in child.items():
                        if isinstance(amount, Mapping):
                            amount = amount.get("amount", 0)
                        elif isinstance(amount, (list, tuple)):
                            amount = len(amount)
                        try:
                            costs[str(resource)] = costs.get(str(resource), 0) + float(amount)
                        except (TypeError, ValueError):
                            continue
                nested = self._collect_costs(child)
                for resource, amount in nested.items():
                    costs[resource] = costs.get(resource, 0) + amount
        elif isinstance(value, (list, tuple)):
            for child in value:
                nested = self._collect_costs(child)
                for resource, amount in nested.items():
                    costs[resource] = costs.get(resource, 0) + amount
        return costs

    def _collect_card_ids(self, value: Any, card_context: Optional[str] = None) -> dict[str, Any]:
        result = {"dragon": 0, "cave": 0, "all": []}
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_context = key if key in {"play_dragon", "gain_dragon", "discard_dragon", "play_cave", "gain_cave", "discard_cave"} else card_context
                if key in {"dragon_id", "dragon_id1", "dragon_id2"} or (key == "chosen_id" and child_context not in {"play_cave", "gain_cave", "discard_cave"}):
                    try:
                        result["dragon"] = max(result["dragon"], int(child))
                        result["all"].append(int(child))
                    except (TypeError, ValueError):
                        pass
                elif key in {"cave_id", "cave_id1", "cave_id2"} or (key == "chosen_id" and child_context in {"play_cave", "gain_cave", "discard_cave"}):
                    try:
                        result["cave"] = max(result["cave"], int(child))
                        result["all"].append(int(child))
                    except (TypeError, ValueError):
                        pass
                nested = self._collect_card_ids(child, child_context)
                result["dragon"] = max(result["dragon"], nested["dragon"])
                result["cave"] = max(result["cave"], nested["cave"])
                result["all"].extend(nested["all"])
        elif isinstance(value, (list, tuple)):
            for child in value:
                nested = self._collect_card_ids(child, card_context)
                result["dragon"] = max(result["dragon"], nested["dragon"])
                result["cave"] = max(result["cave"], nested["cave"])
                result["all"].extend(nested["all"])
        return result

    def _collect_string_values(self, value: Any, field_names: set[str]) -> set[str]:
        values: set[str] = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in field_names and isinstance(child, str):
                    values.add(child)
                values.update(self._collect_string_values(child, field_names))
        elif isinstance(value, (list, tuple)):
            for child in value:
                values.update(self._collect_string_values(child, field_names))
        return values

    def _objective_progress(self, game_state: Any) -> tuple[float, float]:
        objectives = game_state.board.get("round_tracker", {}).get("objectives", ())
        if not objectives:
            return 0.0, 0.0
        player = game_state.player
        progress_values = []
        positive_count = 0
        for objective_index, side in objectives:
            try:
                objective = OBJECTIVE_TILES[objective_index][side]
                objective_type = objective.get("for_each", {}).get("type")
            except (KeyError, IndexError, TypeError):
                continue
            if objective_type == "eggs":
                count, scale = sum(getattr(player, "egg_totals", {}).values()), 20.0
            elif objective_type == "cave_cards_played":
                count = sum(
                    card not in (None, -1)
                    for cave in getattr(player, "caves_played", {}).values()
                    for card in cave
                )
                scale = 16.0
            elif objective_type == "cached_resources":
                count = sum(
                    sum(slot.values())
                    for cave in getattr(player, "cached_resources", {}).values()
                    for slot in cave
                )
                scale = 20.0
            elif objective_type == "guild_position":
                count, scale = getattr(player, "guild_markers", 0), 5.0
            else:
                count, scale = 0, 1.0
            progress_values.append(self._bounded(count, scale))
            positive_count += int(count > 0)
        if not progress_values:
            return 0.0, 0.0
        return sum(progress_values) / len(progress_values), positive_count / len(progress_values)

    def _count_key_values(self, value: Any, target: str) -> int:
        if isinstance(value, Mapping):
            return sum((len(child) if key == target and isinstance(child, (list, tuple, Mapping)) else 0) + self._count_key_values(child, target) for key, child in value.items())
        if isinstance(value, (list, tuple)):
            return sum(self._count_key_values(child, target) for child in value)
        return 0

    def _depth(self, value: Any) -> int:
        if isinstance(value, Mapping):
            return 1 + max((self._depth(child) for child in value.values()), default=0)
        if isinstance(value, (list, tuple)):
            return 1 + max((self._depth(child) for child in value), default=0)
        return 0


class CompactRolloutScorer(nn.Module):
    """Small state/action scorer intended for CPU rollout inference."""

    def __init__(self, state_dim: int = STATE_FEATURE_DIM, action_dim: int = ACTION_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU())
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(state_features)
        actions = self.action_encoder(action_features)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        state = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        fused = torch.cat((state, actions, state * actions), dim=-1)
        scores = self.scorer(fused).squeeze(-1)
        return scores.squeeze(0) if scores.shape[0] == 1 else scores


class NumpyCompactRolloutScorer:
    """NumPy inference path for small CPU batches where Torch dispatch dominates."""

    def __init__(self, model: CompactRolloutScorer):
        self.state_weight = model.state_encoder[0].weight.detach().cpu().numpy().T
        self.state_bias = model.state_encoder[0].bias.detach().cpu().numpy()
        self.action_weight = model.action_encoder[0].weight.detach().cpu().numpy().T
        self.action_bias = model.action_encoder[0].bias.detach().cpu().numpy()
        self.fusion_weight = model.scorer[0].weight.detach().cpu().numpy().T
        self.fusion_bias = model.scorer[0].bias.detach().cpu().numpy()
        self.output_weight = model.scorer[2].weight.detach().cpu().numpy().T
        self.output_bias = model.scorer[2].bias.detach().cpu().numpy()

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    def __call__(self, state_features: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        state = self._relu(state_features @ self.state_weight + self.state_bias)
        actions = self._relu(action_features @ self.action_weight + self.action_bias)
        fused = np.concatenate((
            np.broadcast_to(state, (actions.shape[0], state.shape[0])),
            actions,
            np.broadcast_to(state * actions, (actions.shape[0], state.shape[0])),
        ), axis=1)
        hidden = self._relu(fused @ self.fusion_weight + self.fusion_bias)
        return (hidden @ self.output_weight + self.output_bias).reshape(-1)


class CompactRolloutPolicy:
    """Adapter-backed policy wrapper matching the rollout dispatcher contract."""

    def __init__(
        self,
        model: CompactRolloutScorer,
        adapter: Optional[RolloutFeatureAdapter] = None,
        device: str = "cpu",
        cpu_threads: Optional[int] = 1,
        backend: str = "torch",
    ):
        if backend not in {"torch", "numpy"}:
            raise ValueError(f"Unsupported compact rollout backend: {backend}")
        self.model = model.to(device).eval()
        self.adapter = adapter or RolloutFeatureAdapter()
        self.device = torch.device(device)
        self.backend = backend
        self.numpy_model = NumpyCompactRolloutScorer(model) if backend == "numpy" else None
        if self.device.type == "cpu" and cpu_threads is not None and cpu_threads > 0:
            torch.set_num_threads(cpu_threads)

    @torch.inference_mode()
    def score_actions(self, game_state: Any) -> torch.Tensor:
        encoded = self.adapter.encode_choices(game_state, device=self.device)
        if encoded.action_features.shape[0] == 0:
            return torch.empty(0, device=self.device)
        if self.numpy_model is not None:
            scores = self.numpy_model(
                encoded.state_features.detach().cpu().numpy(),
                encoded.action_features.detach().cpu().numpy(),
            )
            return torch.from_numpy(scores).to(self.device)
        return self.model(encoded.state_features, encoded.action_features)

    @torch.inference_mode()
    def score_action_batches(self, game_states: list[Any]) -> list[torch.Tensor]:
        """Score multiple independent choice states with one padded model call."""
        encoded_states = [self.adapter.encode_choices(game_state, device=self.device) for game_state in game_states]
        if not encoded_states:
            return []
        counts = [encoded.action_features.shape[0] for encoded in encoded_states]
        if any(count == 0 for count in counts):
            return [self.score_actions(game_state) for game_state in game_states]

        if self.numpy_model is not None:
            return [self.score_actions(game_state) for game_state in game_states]

        max_actions = max(counts)
        action_batch = torch.zeros(
            (len(encoded_states), max_actions, self.adapter.action_feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        state_batch = torch.stack([encoded.state_features for encoded in encoded_states], dim=0)
        for batch_index, encoded in enumerate(encoded_states):
            action_batch[batch_index, : encoded.action_features.shape[0]] = encoded.action_features
        score_batch = self.model(state_batch, action_batch)
        if score_batch.ndim == 1:
            score_batch = score_batch.unsqueeze(0)
        return [score_batch[index, :count] for index, count in enumerate(counts)]

    @torch.inference_mode()
    def choose_action(self, game_state: Any) -> int:
        if not game_state.current_choice:
            raise ValueError("CompactRolloutPolicy requires at least one current choice")
        scores = self.score_actions(game_state)
        return int(torch.argmax(scores).item())

    def __call__(self, game_state: Any, rollout_cache: Any = None) -> int:
        return self.choose_action(game_state)


def build_compact_policy(hidden_dim: int = 64, device: str = "cpu", backend: str = "torch") -> CompactRolloutPolicy:
    return CompactRolloutPolicy(CompactRolloutScorer(hidden_dim=hidden_dim), device=device, backend=backend)


def save_compact_checkpoint(model: CompactRolloutScorer, path: str) -> None:
    """Save model weights together with the feature contract metadata."""
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_feature_dim": STATE_FEATURE_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "hidden_dim": model.state_encoder[0].out_features,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_compact_checkpoint(path: str, device: str = "cpu", backend: str = "torch") -> CompactRolloutPolicy:
    """Load a compact policy and reject incompatible feature schemas."""
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported compact feature schema: {checkpoint.get('schema_version')}"
        )
    if checkpoint.get("state_feature_dim") != STATE_FEATURE_DIM or checkpoint.get("action_feature_dim") != ACTION_FEATURE_DIM:
        raise ValueError("Compact checkpoint dimensions do not match the current feature schema")
    model = CompactRolloutScorer(hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return CompactRolloutPolicy(model, device=device, backend=backend)
