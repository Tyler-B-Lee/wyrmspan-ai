#!/usr/bin/env python3
"""Verify all fixed examples work correctly."""

from game_env import WyrmspanEnv
from debug_utils import ActionDecoder, visualize_action_encoding

env = WyrmspanEnv()
decoder = ActionDecoder(env)

# All complex examples from debug_examples.py
actions = [
    {'adv_effects': {'play_dragon': {'L1': 'hand', 'L2': 'any', 'discount': 'none', 'chosen_id': 109}, 'coords': ('crimson_cavern', 0)}, 'cost': {'meat': 1, 'gold': 1, 'coin': 1}, 'coords': None},
    {'adv_effects': {'play_cave': {'source': 'hand', 'chosen_id': 17, 'free': False, 'cave_location': 'golden_grotto'}}, 'cost': {'coin': 1}, 'coords': None},
    {'adv_effects': {'explore': {'cave_name': 'crimson_cavern', 'index': 0}}, 'cost': {'coin': 1}, 'coords': None},
    {'make_payment': {'cost': {'milk': 1}, 'action': {'adv_effects': {'random': {'draw_decision': {'amount': 3, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}}}}, 'cost': {'any_resource': 1}, 'coords': ('crimson_cavern', 0)}}, 'coords': None},
    {'gain_dragon': {'chosen': 2}, 'coords': None},
    {'draw_decision': {'chosen_id': 64, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}, 'remaining_dragons': [64, 70, 175]}, 'coords': ('crimson_cavern', 0)},
    {"make_payment": {"cost": {"meat": 3}, "action": {"play_cave": {"source": "hand", "chosen_id": 5}}}},
]

names = [
    "play_dragon with coords",
    "play_cave with nested cave_location",
    "explore",
    "make_payment with nested coords",
    "gain_dragon",
    "draw_decision with dragon",
    "make_payment with play_cave",
]

print("=" * 70)
print("TESTING COMPLEX ACTION EXTRACTION")
print("=" * 70)

for i, (action, name) in enumerate(zip(actions, names)):
    print(f"\n{i+1}. {name}")
    print("-" * 70)
    
    vec = env.featurize_json(action)
    decoded = decoder.decode_vector(vec)
    
    cave = decoded["location"].get("cave", "MISSING")
    dragon_norm = decoded["card_ids"]["dragon_id_norm"]
    action_types = decoded["action_types"]
    
    print(f"   Location: {cave}")
    print(f"   Dragon (norm): {dragon_norm:.4f}")
    print(f"   Action types: {action_types}")
    
    # Verify each one
    if i == 1:  # play_cave with nested cave_location
        assert cave == "golden_grotto", f"Expected 'golden_grotto', got '{cave}'"
        print("   [OK] cave_location extracted correctly")
    elif i == 5:  # draw_decision
        assert cave == "crimson_cavern", f"Expected 'crimson_cavern', got '{cave}'"
        assert abs(dragon_norm - 64/183.0) < 0.01, f"Expected dragon_norm ~0.3497, got {dragon_norm}"
        print("   [OK] draw_decision dragon extracted correctly")

print("\n" + "=" * 70)
print("ALL COMPLEX EXAMPLES WORKING CORRECTLY!")
print("=" * 70)
