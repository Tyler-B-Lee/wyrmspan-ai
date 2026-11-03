from types import SimpleNamespace
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from collect_rollout_dataset import collect_dataset, collect_episode
from compact_rollout_batch import simulate_compact_games_batched
from compact_rollout_model import (
    ACTION_FEATURE_DIM,
    CompactRolloutPolicy,
    CompactRolloutScorer,
    RolloutFeatureAdapter,
    SCHEMA_VERSION,
    STATE_FEATURE_DIM,
    load_compact_checkpoint,
    save_compact_checkpoint,
)
from playout_compare import get_sim_algo
from playout_compare import RNGOrder, simulate_game_given_rng
from game_states import SoloGameState
import game_logic as logic
from train_compact_rollout import train


def make_state(choices):
    player = SimpleNamespace(
        resources={"meat": 2, "milk": 1},
        coins=3,
        score=4,
        egg_totals={"crimson_cavern": 2},
        dragon_hand=[1],
        cave_hand=[2],
        passed_this_round=False,
        dragons_played={},
        caves_played={},
        times_explored={},
    )
    return SimpleNamespace(
        player=player,
        automa=SimpleNamespace(score=0),
        dragon_deck=list(range(10)),
        cave_deck=list(range(5)),
        event_queue=[],
        current_choice=choices,
        board={"round_tracker": {"round": 1}},
        turn=1,
        phase="player_turn",
    )


def test_adapter_preserves_choice_order_and_shapes():
    state = make_state([
        {"pass": {}},
        {"explore": {"cave_name": "crimson_cavern", "index": 0}},
    ])

    encoded = RolloutFeatureAdapter().encode_choices(state)

    assert encoded.action_indices == (0, 1)
    assert encoded.action_features.shape == (2, 64)
    assert encoded.state_features.shape == (64,)
    assert encoded.summaries[0].startswith("pass")


def test_nested_payment_features_preserve_payment_difference():
    adapter = RolloutFeatureAdapter()
    first, _, _ = adapter.encode_action({
        "make_payment": {
            "cost": {"meat": 1},
            "action": {"adv_effects": {"play_dragon": {"chosen_id": 81}}},
        }
    })
    second, _, _ = adapter.encode_action({
        "make_payment": {
            "cost": {"milk": 2},
            "action": {"adv_effects": {"play_dragon": {"chosen_id": 81}}},
        }
    })

    assert not torch.equal(first, second)
    assert first[23] != second[23]
    assert first[24] != second[24]


def test_action_type_wrapper_and_card_context_features_do_not_collide():
    adapter = RolloutFeatureAdapter()
    skip, _, _ = adapter.encode_action({"skip_opr": {}})
    payment, _, _ = adapter.encode_action({"make_payment": {"cost": {"meat": 1}}})
    dragon, _, _ = adapter.encode_action({"play_dragon": {"chosen_id": 81}})
    cave, _, _ = adapter.encode_action({"play_cave": {"chosen_id": 17}})

    assert skip[17] == 1.0
    assert skip[18] == 0.0
    assert payment[18] == 1.0
    assert dragon[35] > 0.0
    assert dragon[36] == 0.0
    assert cave[36] > 0.0
    assert cave[35] == 0.0


def test_policy_returns_legal_index_and_handles_empty_choices():
    policy = CompactRolloutPolicy(CompactRolloutScorer())
    state = make_state([{"pass": {}}, {"skip": {}}])

    assert policy.choose_action(state) in (0, 1)

    empty_state = make_state([])
    assert policy.score_actions(empty_state).shape == (0,)


def test_numpy_backend_matches_torch_backend():
    model = CompactRolloutScorer()
    state = make_state([{"pass": {}}, {"play_dragon": {"chosen_id": 81}}])
    torch_policy = CompactRolloutPolicy(model, backend="torch")
    numpy_policy = CompactRolloutPolicy(model, backend="numpy")

    assert torch.allclose(torch_policy.score_actions(state), numpy_policy.score_actions(state), atol=1e-6)
    assert torch_policy.choose_action(state) == numpy_policy.choose_action(state)


def test_batch_scores_match_individual_scores():
    policy = CompactRolloutPolicy(CompactRolloutScorer())
    states = [
        make_state([{"pass": {}}, {"skip": {}}]),
        make_state([{"play_dragon": {"chosen_id": 81}}]),
    ]

    batch_scores = policy.score_action_batches(states)
    individual_scores = [policy.score_actions(state) for state in states]

    assert len(batch_scores) == len(individual_scores)
    assert all(torch.allclose(batch, individual) for batch, individual in zip(batch_scores, individual_scores))


def test_batched_rollouts_match_sequential_rollouts():
    def build_game(seed):
        random.seed(seed)
        game = SoloGameState(automa_difficulty=1, ending_round=4)
        game.create_game()
        return logic.get_next_state(game, None)

    base_games = [build_game(seed) for seed in (71, 72)]
    policy = CompactRolloutPolicy(CompactRolloutScorer())
    rngs = []
    random_states = []
    for game in base_games:
        rngs.append(RNGOrder(game))
        random_states.append(random.getstate())
    sequential = [
        simulate_game_given_rng(
            game.make_copy(),
            "compact_rollout",
            {"policy": policy},
            "sequential",
            rng=rng,
            random_state=random_state,
        )[0]
        for game, rng, random_state in zip(base_games, rngs, random_states)
    ]
    batched = simulate_compact_games_batched(
        [game.make_copy() for game in base_games],
        policy,
        rngs,
        random_states=random_states,
    )

    assert [result[0] for result in batched] == sequential


def test_dispatcher_requires_explicit_compact_policy():
    try:
        get_sim_algo("compact_rollout", {})
    except ValueError as error:
        assert "CompactRolloutPolicy" in str(error)
    else:
        raise AssertionError("compact_rollout must reject an uninitialized policy")

    policy = CompactRolloutPolicy(CompactRolloutScorer())
    algorithm = get_sim_algo("compact_rollout", {"policy": policy})
    assert algorithm(make_state([{"pass": {}}])) == 0


def test_dataset_collection_is_deterministic_and_labels_are_legal():
    first = collect_dataset([1], teacher_name="greedy_action_priority")
    second = collect_dataset([1], teacher_name="greedy_action_priority")

    assert first == second
    assert first["records"]
    assert all(0 <= row["label"] < row["action_count"] for row in first["records"])


def test_strategic_teacher_handles_late_round_objective_index():
    records = collect_episode(33, teacher_name="strategic_objective_aware")

    assert records
    assert all(0 <= row["label"] < row["action_count"] for row in records)


def test_checkpoint_round_trip_preserves_weights():
    model = CompactRolloutScorer()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "compact.pt"
        save_compact_checkpoint(model, str(path))
        loaded = load_compact_checkpoint(str(path))

    assert all(torch.equal(value, original[key]) for key, value in loaded.model.state_dict().items())


def test_training_writes_checkpoint_from_seed_split():
    records = []
    for seed in (1, 2):
        for step in range(2):
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "label": step,
                    "action_count": 2,
                    "state_features": [0.0] * STATE_FEATURE_DIM,
                    "action_features": [
                        [1.0] + [0.0] * (ACTION_FEATURE_DIM - 1),
                        [0.0, 1.0] + [0.0] * (ACTION_FEATURE_DIM - 2),
                    ],
                }
            )
    with TemporaryDirectory() as directory:
        dataset_path = Path(directory) / "dataset.json"
        checkpoint_path = Path(directory) / "compact.pt"
        dataset_path.write_text(
            json.dumps(
                {
                    "dataset_version": 1,
                    "feature_schema_version": SCHEMA_VERSION,
                    "records": records,
                }
            ),
            encoding="utf-8",
        )
        result = train(dataset_path, checkpoint_path, epochs=1, seed=3)
        assert checkpoint_path.exists()

    assert result.train_records == 2
    assert result.validation_records == 2
