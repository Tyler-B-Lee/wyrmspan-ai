from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

import torch
from torch import nn


TRIGGER_KEYS = {
    "if_activated": "trigger_if_activated",
    "when_played": "trigger_when_played",
    "once_per_round": "trigger_once_per_round",
    "end_game": "trigger_end_game",
    # "on_feed": "trigger_on_feed",
    # "on_grow_up": "trigger_on_grow_up",
}

EFFECT_KEY_MAP = {
    "gain_resource": "effect_gain_resource",
    "gain_coin": "effect_gain_resource",
    "gain_vp": "effect_gain_vp",
    "gain_cave": "effect_gain_cave",
    "lay_egg": "effect_lay_egg",
    "cache_from": "effect_cache_from",
    "tuck": "effect_tuck",
    "discard": "effect_discard",
    "draw": "effect_draw",
    "score": "effect_score",
    "heal": "effect_heal",
    "move": "effect_move",
    "multiply": "effect_multiply",
    "convert": "effect_convert",

    "play_dragon": "effect_play_dragon",
    "play_cave": "effect_play_cave",
    "gain_guild": "effect_gain_guild",
    "gain_dragon": "effect_gain_dragon",
    "tuck_from": "effect_tuck_from",
    "make_payment": "effect_make_payment",
    "deduct_resources": "effect_deduct_resources",
    "discard_dragon": "effect_discard_dragon",
    "discard_cave": "effect_discard_cave",
    "swap_dragons": "effect_swap_dragons",
    "skip": "effect_skip",
    "skip_opr": "effect_skip_opr",
    "skip_choice": "effect_skip_choice", # for draw/any resource decisions
    "brown_space": "effect_brown_space",
    "4th_space": "effect_4th_space",
    "automa_action": "effect_automa_action",
    "automa_guild_move": "effect_automa_guild_move",
    "top_deck_reveal": "effect_top_deck_reveal",
    "draw_decision": "effect_draw_decision",
    "other_ability_on_mat": "effect_other_ability_on_mat",
    "any_resource_decision": "effect_any_resource_decision",
    "end_game": "effect_end_game",
    "opr_option": "effect_opr_option",
}

SIZE_MAP = {
    "Hatchling": "size_hatchling",
    "Small": "size_small",
    "Medium": "size_medium",
    "Large": "size_large",
}

PERSONALITY_MAP = {
    "Aggressive": "personality_aggressive",
    "Helpful": "personality_helpful",
    "Shy": "personality_shy",
    "Playful": "personality_playful",
}


def _add_tokens(tokens: list[str], *values: str | None) -> None:
    for value in values:
        if not value:
            continue
        tokens.append(value)


def _word_tokens_from_text(text: str) -> list[str]:
    """Tokenize ability text without using the structured ability fields."""
    tokens: list[str] = []
    parts = text.split(".")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = re.findall(r"[A-Za-z0-9\-\_]+", part)
        for word in words:
            tokens.append(f"word_{word.lower()}")
        if words:
            tokens.append("word_period")
    return tokens


def _ability_text(card: Mapping) -> str:
    return str(card.get("ability_text", card.get("text", "")))


def _cost_tokens(card: Mapping) -> list[str]:
    tokens: list[str] = []
    for field_name in ["meat_cost", "gold_cost", "crystal_cost", "coin_cost", "milk_cost", "egg_cost"]:
        value = int(card.get(field_name, 0) or 0)
        if value <= 0:
            continue
        token = f"cost_{field_name.replace('_cost', '')}_{value}"
        _add_tokens(tokens, token)
    return tokens


def _board_affinity_tokens(card: Mapping) -> list[str]:
    tokens: list[str] = []
    if bool(card.get("crimson_cavern")):
        _add_tokens(tokens, "board_crimson_cavern")
    if bool(card.get("golden_grotto")):
        _add_tokens(tokens, "board_golden_grotto")
    if bool(card.get("amethyst_abyss")):
        _add_tokens(tokens, "board_amethyst_abyss")
    return tokens


def serialize_dragon_card(card: Mapping) -> list[str]:
    tokens: list[str] = ["card_type_dragon"]

    size_name = card.get("size")
    if size_name:
        _add_tokens(tokens, SIZE_MAP.get(size_name, f"size_{str(size_name).lower()}"))

    personality = card.get("personality")
    if personality:
        _add_tokens(tokens, PERSONALITY_MAP.get(personality, f"personality_{str(personality).lower()}"))

    if card.get("VP") is not None:
        _add_tokens(tokens, f"vp_{int(card['VP'])}")
    if card.get("capacity") is not None:
        _add_tokens(tokens, f"capacity_{int(card['capacity'])}")

    tokens.append("start_board_affinities")
    tokens.extend(_board_affinity_tokens(card))

    tokens.append("start_costs")
    _add_tokens(tokens, *_cost_tokens(card))

    # find main trigger and add text after it
    for k,v in TRIGGER_KEYS.items():
        if k in card:
            tokens.append(v)
            break
    tokens.extend(_word_tokens_from_text(_ability_text(card)))

    return tokens


def serialize_cave_card(card: Mapping) -> list[str]:
    tokens: list[str] = ["card_type_cave", "trigger_when_played"]

    tokens.extend(_word_tokens_from_text(_ability_text(card)))

    return tokens


class SemanticCardSequenceEncoder(nn.Module):
    """Pool a semantic token sequence into a single card embedding."""

    def __init__(self, vocab_size: int = 512, embedding_dim: int = 64, max_tokens: int = 32, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_tokens, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(f"Expected token_ids shape [batch, max_tokens], received {tuple(token_ids.shape)}")

        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_tokens:
            raise ValueError(f"Sequence length {seq_len} exceeds max_tokens {self.max_tokens}")
        if token_mask is None:
            token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        else:
            token_mask = token_mask.to(device=token_ids.device, dtype=torch.bool)

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)

        valid_mask = token_mask & (token_ids != self.pad_id)
        pooled_rows = []

        for i in range(batch_size):
            row_mask = valid_mask[i]
            if not row_mask.any():
                pooled_rows.append(torch.zeros(self.embedding_dim, device=token_ids.device, dtype=embeddings.dtype))
                continue

            row_emb = embeddings[i, row_mask]
            row_weights = row_mask[row_mask].to(dtype=row_emb.dtype).unsqueeze(-1)
            pooled_row = (row_emb * row_weights).sum(dim=0) / row_weights.sum(dim=0).clamp_min(1.0)
            pooled_rows.append(pooled_row)

        pooled = torch.stack(pooled_rows, dim=0)
        output = self.output_proj(pooled)
        return output


def _token_to_id_map() -> dict[str, int]:
    token_map: dict[str, int] = {"<pad>": 0}
    counter = 1
    for card in []:
        _ = card
    tokens = [
        "start_costs",
        "start_board_affinities",
        "card_type_dragon",
        "card_type_cave",
        "cost_meat",
        "cost_gold",
        "cost_crystal",
        "cost_coin",
        "cost_milk",
        "cost_egg",
    ]
    tokens.extend(SIZE_MAP.values())
    tokens.extend(PERSONALITY_MAP.values())
    tokens.extend(TRIGGER_KEYS.values())
    # tokens.extend(EFFECT_KEY_MAP.values())

    for token_name in tokens:
        token_map.setdefault(token_name, counter)
        counter += 1
    return token_map


def build_token_map(token_sequences: Iterable[Iterable[str]]) -> dict[str, int]:
    """Build a stable vocabulary containing metadata and observed word tokens."""
    token_map = _token_to_id_map()
    observed_tokens = {token for sequence in token_sequences for token in sequence}
    for token in sorted(observed_tokens):
        if token not in token_map:
            token_map[token] = len(token_map)
    return token_map


def card_tokens_to_ids(token_sequence: Iterable[str], token_map: dict[str, int] | None = None) -> list[int]:
    token_sequence = list(token_sequence)
    if token_map is None:
        token_map = build_token_map([token_sequence])
    ids: list[int] = []
    for token in token_sequence:
        ids.append(token_map.get(token, token_map.get("<pad>", 0)))
    return ids
