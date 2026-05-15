#!/usr/bin/env python3
"""
Test script to validate the new multi-component reward scheme.
Runs a few episodes and logs reward components to verify they're reasonable.
"""

import random
import json
from game_env import WyrmspanEnv
from game_logic import get_next_state, get_random_outcome
from game_states import PHASE_END_GAME

def run_test_episode(seed=42):
    """Run a single episode and log detailed reward information."""
    env = WyrmspanEnv()
    obs, _ = env.reset(seed=seed)
    
    done = False
    step_count = 0
    total_reward = 0.0
    reward_components = {
        'margin': [],
        'points': [],
        'round_bonus': [],
        'end_game': [],
        'total': []
    }
    
    print(f"\n{'='*70}")
    print(f"Episode Test (seed={seed})")
    print(f"{'='*70}")
    print(f"Initial state - Player: {env.game_state.player.score}, Automa: {env.game_state.automa.score}")
    
    while not done:
        legal_actions = obs["action_mask"].sum()
        if legal_actions == 0:
            print("ERROR: No legal actions available!")
            break
            
        chosen_action = random.randint(0, int(legal_actions) - 1)
        prev_player = env.game_state.player.score
        prev_automa = env.game_state.automa.score
        
        obs, reward, done, _, info = env.step(chosen_action)
        step_count += 1
        total_reward += reward
        
        # Compute individual components for logging (mimicking the step() logic)
        current_margin = env.game_state.player.score - env.game_state.automa.score
        prev_margin = prev_player - prev_automa
        margin_delta = current_margin - prev_margin
        reward_margin = margin_delta / env.REWARD_CONFIG['margin_scaling']
        
        point_delta = env.game_state.player.score - prev_player
        reward_points = (point_delta / 100.0) * env.REWARD_CONFIG['point_weight']
        
        reward_components['margin'].append(reward_margin)
        reward_components['points'].append(reward_points)
        reward_components['total'].append(reward)
        
        # Log every 10 steps or at end
        if step_count % 10 == 0 or done:
            margin_str = f"margin_delta={margin_delta:+.1f}" if margin_delta != 0 else "margin_delta=0"
            points_str = f"pts_delta={point_delta:+.1f}" if point_delta != 0 else "pts_delta=0"
            print(f"Step {step_count:3d}: {margin_str:20s} | {points_str:20s} | "
                  f"reward={reward:+.4f} | P:{env.game_state.player.score:3d} A:{env.game_state.automa.score:3d}")
    
    print(f"\n{'─'*70}")
    print(f"Episode finished after {step_count} steps")
    print(f"Final Score - Player: {env.game_state.player.score}, Automa: {env.game_state.automa.score}")
    print(f"Result: {'WIN' if env.game_state.player.score >= env.game_state.automa.score else 'LOSS'}")
    print(f"Total Reward: {total_reward:.4f}")
    
    # Statistics
    if reward_components['margin']:
        print(f"\nReward Component Statistics:")
        print(f"  Margin component:    mean={sum(reward_components['margin'])/len(reward_components['margin']):.4f}, "
              f"min={min(reward_components['margin']):.4f}, max={max(reward_components['margin']):.4f}")
        print(f"  Points component:    mean={sum(reward_components['points'])/len(reward_components['points']):.4f}, "
              f"min={min(reward_components['points']):.4f}, max={max(reward_components['points']):.4f}")
        print(f"  Total reward/step:   mean={sum(reward_components['total'])/len(reward_components['total']):.4f}, "
              f"min={min(reward_components['total']):.4f}, max={max(reward_components['total']):.4f}")
    
    return total_reward, env.game_state.player.score >= env.game_state.automa.score

if __name__ == "__main__":
    print("Testing multi-component reward scheme...")
    print(f"REWARD_CONFIG: {json.dumps(WyrmspanEnv.REWARD_CONFIG, indent=2)}\n")
    
    num_episodes = 5
    win_count = 0
    total_rewards = []
    
    for i in range(num_episodes):
        reward, won = run_test_episode(seed=42 + i)
        total_rewards.append(reward)
        if won:
            win_count += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {num_episodes} episodes")
    print(f"{'='*70}")
    print(f"Wins: {win_count}/{num_episodes} ({100*win_count/num_episodes:.1f}%)")
    print(f"Avg total reward: {sum(total_rewards)/len(total_rewards):.4f}")
    print(f"Min total reward: {min(total_rewards):.4f}")
    print(f"Max total reward: {max(total_rewards):.4f}")
    print("\n✓ Reward scheme validation complete!")
