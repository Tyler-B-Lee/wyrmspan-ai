import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_logic import get_next_state, get_random_outcome
from game_states import CAVE_NAMES, SoloGameState


def build_game(seed: int, automa_difficulty: int) -> SoloGameState:
    random.seed(seed)
    game = SoloGameState(automa_difficulty=automa_difficulty)
    game.create_game()
    return game


def reset_to_injected_event(game: SoloGameState, free_play: bool) -> SoloGameState:
    game.event_queue = []
    game.current_choice = None
    game.current_random_event = None
    game.event_queue.append({"play_cave": {"source": "deck", "free": free_play}})
    return game


def cave_is_on_player_mat(game: SoloGameState, cave_id: int) -> bool:
    for cave_name in CAVE_NAMES:
        for placed in game.player.caves_played[cave_name]:
            if placed == cave_id:
                return True
    return False


def assert_random_first(game: SoloGameState) -> SoloGameState:
    next_state = get_next_state(game, None)
    assert next_state.current_random_event is not None, "Expected random event to be halted first"
    assert next_state.current_choice is None, "Expected no choice before random cave draw"
    assert "play_cave" in next_state.current_random_event, "Expected play_cave random event"
    return next_state


def assert_choice_after_draw(game: SoloGameState, drawn_cave_id: int) -> SoloGameState:
    next_state = get_next_state(game, drawn_cave_id)
    assert next_state.current_random_event is None, "Expected random event to be resolved"
    assert next_state.current_choice is not None, "Expected placement choice after random cave draw"
    assert len(next_state.current_choice) > 0, "Expected at least one placement choice"

    for i, choice in enumerate(next_state.current_choice):
        assert "play_cave" in choice, f"Choice {i} missing play_cave payload"
        payload = choice["play_cave"]
        assert payload.get("source") == "deck", f"Choice {i} expected source=deck"
        assert payload.get("chosen_id") == drawn_cave_id, f"Choice {i} expected chosen_id={drawn_cave_id}"
        assert "cave_location" in payload, f"Choice {i} missing cave_location"

    return next_state


def assert_placement_resolution(game: SoloGameState, drawn_cave_id: int, choice_index: int = 0) -> SoloGameState:
    next_state = get_next_state(game, choice_index)
    assert cave_is_on_player_mat(next_state, drawn_cave_id), "Drawn cave was not placed on player mat"
    assert drawn_cave_id not in next_state.cave_deck, "Drawn cave should no longer be in deck"
    assert drawn_cave_id not in next_state.player.cave_hand, "Deck-drawn cave should not enter player hand"
    return next_state


def run_trial(seed: int, automa_difficulty: int, free_play: bool) -> tuple[bool, str]:
    game = build_game(seed=seed, automa_difficulty=automa_difficulty)
    game = reset_to_injected_event(game, free_play=free_play)

    phase_one = assert_random_first(game)
    drawn_cave_id = get_random_outcome(phase_one, phase_one.current_random_event, phase_one.player)
    phase_two = assert_choice_after_draw(phase_one, drawn_cave_id)
    _ = assert_placement_resolution(phase_two, drawn_cave_id)

    return True, f"seed={seed} drawn={drawn_cave_id} choices={len(phase_two.current_choice)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify deck cave-play ordering: draw random cave first, then choose placement location."
    )
    parser.add_argument("--seed-start", type=int, default=1, help="Start of seed range (inclusive)")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of sequential seeds to test")
    parser.add_argument("--difficulty", type=int, default=2, help="Automa difficulty used when creating games")
    parser.add_argument("--free-play", dest="free_play", action="store_true", help="Inject free deck cave play event")
    parser.add_argument("--paid-play", dest="free_play", action="store_false", help="Inject paid deck cave play event")
    parser.set_defaults(free_play=True)
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failing seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    print("Running cave draw ordering helper")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)")
    print(f"Automa difficulty: {args.difficulty}")
    print(f"Injected event: source=deck free={args.free_play}")

    passed = 0
    failed = 0

    for seed in seeds:
        try:
            ok, detail = run_trial(seed=seed, automa_difficulty=args.difficulty, free_play=args.free_play)
            if ok:
                passed += 1
                print(f"PASS {detail}")
        except Exception as exc:
            failed += 1
            print(f"FAIL seed={seed} error={exc}")
            if args.fail_fast:
                break

    print("Summary")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
