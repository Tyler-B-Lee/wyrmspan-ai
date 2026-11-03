"""Batched compact-policy rollout execution for independent game states."""

from __future__ import annotations

import time
import random
from typing import Any, Optional

import game_logic as logic
from compact_rollout_model import CompactRolloutPolicy
from playout_compare import RNGOrder


def rollout_reward(game_state: Any) -> float:
    score_based_reward = (game_state.player.score ** 2) / 40_000
    if game_state.player.score >= game_state.automa.score:
        return score_based_reward + 0.75
    return score_based_reward


def simulate_compact_games_batched(
    game_states: list[Any],
    policy: CompactRolloutPolicy,
    rngs: Optional[list[RNGOrder]] = None,
    random_states: Optional[list[object]] = None,
) -> list[tuple[float, float]]:
    """Run independent games while batching all available choice states per step.

    Returns ``(final_reward, elapsed_seconds)`` in the same order as ``game_states``.
    The supplied game states are advanced; callers should pass copies when needed.
    """
    if rngs is None:
        rngs = [RNGOrder(game_state) for game_state in game_states]
    if len(rngs) != len(game_states):
        raise ValueError("There must be one RNGOrder per game state")
    if random_states is None:
        random_states = [random.getstate() for _ in game_states]
    if len(random_states) != len(game_states):
        raise ValueError("There must be one Python random state per game state")

    def advance(index: int, chosen_input: Any) -> None:
        caller_state = random.getstate()
        random.setstate(random_states[index])
        try:
            game_states[index] = logic.get_next_state(game_states[index], chosen_input)
            random_states[index] = random.getstate()
        finally:
            random.setstate(caller_state)

    start_times = [time.perf_counter() for _ in game_states]
    active = list(range(len(game_states)))
    while active:
        choice_indices = [
            index for index in active
            if game_states[index].phase != logic.PHASE_END_GAME
            and game_states[index].current_choice is not None
        ]
        if choice_indices:
            choice_states = [game_states[index] for index in choice_indices]
            chosen_actions = policy.score_action_batches(choice_states)
            for index, scores in zip(choice_indices, chosen_actions):
                advance(index, int(scores.argmax().item()))

        for index in list(active):
            game_state = game_states[index]
            if game_state.phase == logic.PHASE_END_GAME:
                active.remove(index)
                continue
            if game_state.current_choice is not None:
                continue
            if game_state.current_random_event is not None:
                outcome = rngs[index].get_random_outcome(
                    game_state,
                    game_state.current_random_event,
                    game_state.player,
                )
                advance(index, outcome)
            else:
                advance(index, chosen_input=None)

    return [
        (rollout_reward(game_state), time.perf_counter() - start_times[index])
        for index, game_state in enumerate(game_states)
    ]
