import torch

from card_tokenizer import (
    SemanticCardSequenceEncoder,
    build_token_map,
    card_tokens_to_ids,
    serialize_cave_card,
    serialize_dragon_card,
)


def test_serialize_dragon_card_returns_mechanics_tokens():
    card = {
        "number": 10,
        "name": "Melodious Firedragon",
        "ability_text": "Tuck [DragonCard] from your hand here. On_feed gain [CaveCard]. When the 3rd [DragonCard] is tucked, also excavate 1 space for free with [CaveCard] from your hand.",
        "VP": 2,
        "size": "Hatchling",
        "capacity": 0,
        "personality": "Helpful",
        "meat_cost": 0,
        "gold_cost": 0,
        "crystal_cost": 0,
        "coin_cost": 0,
        "crimson_cavern": True,
        "golden_grotto": False,
        "amethyst_abyss": True,
        "egg_cost": 1,
        "milk_cost": 1,
        "if_activated": {
            "tuck_from": {
                "L1": "hand",
                "L2": "here"
            }
        },
        "on_feed": {
            "type": "tuck",
            "effect": {"gain_cave": {"source": "any"}}
        },
        "on_grow_up": {
            "play_cave": {"source": "hand", "free": True}
        }
    }

    tokens = serialize_dragon_card(card)
    print(tokens)

    assert tokens
    assert "cost_egg_1" in tokens and "cost_milk_1" in tokens
    assert "word_dragoncard" in tokens
    assert "word_played" in tokens
    assert "word_gain" in tokens
    assert "size_hatchling" in tokens
    assert "personality_helpful" in tokens
    assert "board_crimson_cavern" in tokens
    assert "trigger_when_played" in tokens
    assert "trigger_on_grow_up" in tokens
    assert "trigger_on_feed" in tokens
    assert "effect_gain_resource" not in tokens


def test_serialize_cave_card_returns_mechanics_tokens():
    card = {
        "number": 7,
        "name": "Test Cave",
        "text": "When played: gain [Coin].",
        "VP": 2,
        "meat_cost": 1,
        "gold_cost": 0,
        "crystal_cost": 0,
        "coin_cost": 0,
        "egg_cost": 0,
        "milk_cost": 0,
    }

    tokens = serialize_cave_card(card)
    print(tokens)

    assert tokens
    assert "card_type_cave" in tokens
    assert "cost_meat" in tokens
    assert "word_when" in tokens
    assert "word_played" in tokens
    assert "word_gain" in tokens
    assert "word_coin" in tokens
    assert "effect_gain_resource" not in tokens
    assert "trigger_when_played" not in tokens


def test_semantic_card_encoder_shapes():
    encoder = SemanticCardSequenceEncoder(vocab_size=256, embedding_dim=32, max_tokens=18)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)

    out = encoder(input_ids, mask)

    assert out.shape == (1, 32)
    assert not torch.isnan(out).any()


def test_word_tokens_receive_non_padding_ids():
    token_sequences = [["card_type_dragon", "word_gain", "word_coin"]]
    token_map = build_token_map(token_sequences)

    ids = card_tokens_to_ids(token_sequences[0], token_map)

    assert ids[0] != 0
    assert ids[1] != 0
    assert ids[2] != 0
