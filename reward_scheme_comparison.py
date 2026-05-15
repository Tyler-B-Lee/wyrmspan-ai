#!/usr/bin/env python3
"""
Comparison of old vs. new reward schemes.
Shows how rewards differ for different game scenarios.
"""

def old_reward_scheme(score_delta, player_final_score, automa_final_score, is_game_end=False):
    """Original reward scheme: +0.01 per point, +8 for winning."""
    reward = (score_delta / 100.0)
    if is_game_end and player_final_score >= automa_final_score:
        reward += 8
    return reward

def new_reward_scheme(
    score_delta, 
    player_score, automa_score, 
    prev_player_score, prev_automa_score,
    is_round_end=False, 
    is_game_end=False,
    margin_scaling=100.0, point_weight=0.3,
    round_bonus_threshold=-5, round_bonus_amount=0.15,
    target_score=75,
    win_bonus_base=2.0, win_bonus_per_margin=0.05,
    target_bonus_max=1.0, loss_penalty=-0.5
):
    """New multi-component reward scheme."""
    # Margin shaping
    current_margin = player_score - automa_score
    prev_margin = prev_player_score - prev_automa_score
    margin_delta = current_margin - prev_margin
    reward_margin = margin_delta / margin_scaling
    
    # Point accumulation
    reward_points = (score_delta / 100.0) * point_weight
    
    # Round bonus
    reward_round = 0.0
    if is_round_end and current_margin >= round_bonus_threshold:
        reward_round = round_bonus_amount
    
    # End-game
    reward_end = 0.0
    if is_game_end:
        if player_score >= automa_score:
            margin_at_end = player_score - automa_score
            margin_bonus = min(margin_at_end * win_bonus_per_margin, target_bonus_max)
            reward_end = win_bonus_base + margin_bonus
            if player_score >= target_score:
                target_bonus = min((player_score - target_score) / 50.0, target_bonus_max)
                reward_end += target_bonus
        else:
            reward_end = loss_penalty
    
    return reward_margin + reward_points + reward_round + reward_end

print("=" * 80)
print("REWARD SCHEME COMPARISON: Old vs. New")
print("=" * 80)

scenarios = [
    {
        "name": "Mid-game: Player gains 2 points, stays behind (-10 margin)",
        "score_delta": 2,
        "player_score": 20,
        "automa_score": 30,
        "prev_player_score": 18,
        "prev_automa_score": 30,
        "is_round_end": False,
        "is_game_end": False,
    },
    {
        "name": "Round end: Player at -5 (competitive threshold)",
        "score_delta": 0,
        "player_score": 25,
        "automa_score": 30,
        "prev_player_score": 25,
        "prev_automa_score": 30,
        "is_round_end": True,
        "is_game_end": False,
    },
    {
        "name": "Mid-game: Player closes gap (+1 margin improvement)",
        "score_delta": 2,
        "player_score": 28,
        "automa_score": 29,
        "prev_player_score": 26,
        "prev_automa_score": 31,
        "is_round_end": False,
        "is_game_end": False,
    },
    {
        "name": "Game end: Player WINS with 80 pts vs Automa 75",
        "score_delta": 5,
        "player_score": 80,
        "automa_score": 75,
        "prev_player_score": 75,
        "prev_automa_score": 75,
        "is_round_end": False,
        "is_game_end": True,
    },
    {
        "name": "Game end: Player LOSES with 65 pts vs Automa 75",
        "score_delta": 0,
        "player_score": 65,
        "automa_score": 75,
        "prev_player_score": 65,
        "prev_automa_score": 70,
        "is_round_end": False,
        "is_game_end": True,
    },
    {
        "name": "Game end: Player wins DECISIVELY with 85 pts vs Automa 70",
        "score_delta": 5,
        "player_score": 85,
        "automa_score": 70,
        "prev_player_score": 80,
        "prev_automa_score": 70,
        "is_round_end": False,
        "is_game_end": True,
    },
]

for scenario in scenarios:
    old_reward = old_reward_scheme(
        scenario["score_delta"],
        scenario["player_score"],
        scenario["automa_score"],
        scenario["is_game_end"]
    )
    new_reward = new_reward_scheme(
        scenario["score_delta"],
        scenario["player_score"],
        scenario["automa_score"],
        scenario["prev_player_score"],
        scenario["prev_automa_score"],
        scenario["is_round_end"],
        scenario["is_game_end"]
    )
    
    improvement = "✓" if new_reward > old_reward else ("↔" if abs(new_reward - old_reward) < 0.01 else "✗")
    
    print(f"\n{scenario['name']}")
    print(f"  Old reward: {old_reward:+.4f}")
    print(f"  New reward: {new_reward:+.4f}")
    print(f"  Difference: {new_reward - old_reward:+.4f} {improvement}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)
print("""
1. MID-GAME IMPROVEMENTS:
   - New scheme rewards closing the gap (margin improvement) even with same points
   - Creates intermediate signal about competitive position
   - Old scheme only cares about raw points gained
   
2. ROUND BOUNDARIES:
   - New scheme: +0.15 bonus if competitive at round end (encourages strategic checkpoints)
   - Old scheme: 0 (no signal)
   
3. GAME END - WINNING:
   - Old scheme: Flat +8 bonus for any win
   - New scheme: 2.0–3.0 proportional to margin (incentivizes larger victories)
   - Example: 80 vs 75 (5 pt lead) gives +2.25 vs fixed +8
   - Example: 85 vs 70 (15 pt lead) gives +2.75 vs fixed +8
   
4. GAME END - LOSING:
   - Old scheme: 0 (no penalty, agent might ignore losses)
   - New scheme: -0.5 (light penalty encourages trying to win)
   
5. POINT ACCUMULATION WEIGHT:
   - Old: 0.01 per point (dominant signal in early training)
   - New: 0.003 per point (reduced, emphasizes margin over hoarding points)

TRAINING IMPLICATION:
  Agent learns to COMPETE (beat automa) rather than just accumulate points.
  Early on, margin tracking + round bonuses provide guidance.
  Late game, winning becomes clear win condition with proportional bonus.
""")
