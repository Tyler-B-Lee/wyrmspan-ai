#!/usr/bin/env python3
"""Test nested coordinate and dragon extraction."""

from game_env import WyrmspanEnv
from debug_utils import ActionDecoder

env = WyrmspanEnv()
decoder = ActionDecoder(env)

# Test example 2: play_cave with nested cave_location
print("="*70)
print("TEST 1: play_cave with nested cave_location")
print("="*70)
action2 = {'adv_effects': {'play_cave': {'source': 'hand', 'chosen_id': 17, 'free': False, 'cave_location': 'golden_grotto'}}, 'cost': {'coin': 1}, 'coords': None}
vec2 = env.featurize_json(action2)
decoded2 = decoder.decode_vector(vec2)

print(f'Decoded location cave: {decoded2["location"].get("cave")}')
print(f'Expected: golden_grotto')
assert decoded2["location"].get("cave") == "golden_grotto", "FAILED: cave_location not extracted"
print("✓ PASS\n")

# Test example 6: draw_decision with dragons
print("="*70)
print("TEST 2: draw_decision with chosen dragon")
print("="*70)
action6 = {'draw_decision': {'chosen_id': 64, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}, 'remaining_dragons': [64, 70, 175]}, 'coords': ('crimson_cavern', 0)}
vec6 = env.featurize_json(action6)
decoded6 = decoder.decode_vector(vec6)

print(f'Decoded location cave: {decoded6["location"].get("cave")}')
print(f'Decoded dragon_id_norm: {decoded6["card_ids"]["dragon_id_norm"]:.4f}')
print(f'Expected cave: crimson_cavern')
print(f'Expected dragon (64/183 norm): {64/183:.4f}')

assert decoded6["location"].get("cave") == "crimson_cavern", "FAILED: location not found"
# Check dragon ID (64/183 ≈ 0.3497)
expected_norm = 64 / 183.0
actual_norm = decoded6["card_ids"]["dragon_id_norm"]
assert abs(actual_norm - expected_norm) < 0.01, f"FAILED: dragon norm {actual_norm} != {expected_norm}"
print("✓ PASS\n")

print("="*70)
print("✓ ALL NESTED EXTRACTION TESTS PASSED")
print("="*70)
