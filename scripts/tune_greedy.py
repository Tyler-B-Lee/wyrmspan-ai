import sys
import pathlib
import random
import time
import multiprocessing

# ensure project root is on sys.path when running as a script from scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from playout_compare import simulate_multiple_games
from game_states import SoloGameState

# Simple random search tuner for greedy_action_priority weights

NUM_CANDIDATES = 20
SIMS_PER_CANDIDATE = 250

def main():
    random.seed(42)

    results = []

    base_state = SoloGameState(automa_difficulty=3)
    base_state.create_game()

    start_all = time.time()
    for i in range(NUM_CANDIDATES):
        dragon_w = random.uniform(2.0, 4.0)
        cave_w = random.uniform(1.5, 3.5)
        explore_w = random.uniform(1.0, 3.0)
        algo_kwargs = {
            'dragon_weight': dragon_w,
            'cave_weight': cave_w,
            'explore_weight': explore_w,
            'pass_penalty': 1.5,
            'tie_threshold': 0.35,
        }
        display_name = f"greedy_{i:02d}_{dragon_w:.3f}_{cave_w:.3f}_{explore_w:.3f}"
        print(f"Candidate {i+1}/{NUM_CANDIDATES}: {display_name}")
        total_score, _, total_time = simulate_multiple_games(base_state, 'greedy_action_priority', algo_kwargs, display_name, SIMS_PER_CANDIDATE)
        avg_score = total_score / SIMS_PER_CANDIDATE if SIMS_PER_CANDIDATE>0 else 0
        print(f" -> avg_score={avg_score:.6f}, time={total_time:.2f}s")
        results.append((avg_score, total_score, total_time, algo_kwargs, display_name))

    end_all = time.time()
    results.sort(reverse=True, key=lambda x: x[0])
    print("\nTop 5 candidates (avg_score, time, weights):")
    for r in results[:5]:
        avg, tot, ttime, kws, name = r
        print(f"{name}: avg={avg:.6f}, time={ttime:.2f}s, dragon={kws['dragon_weight']:.3f}, cave={kws['cave_weight']:.3f}, explore={kws['explore_weight']:.3f}")
    print(f"Total tuning time: {end_all-start_all:.2f}s")


if __name__ == '__main__':
    # Required on Windows to safely spawn worker processes
    multiprocessing.freeze_support()
    main()