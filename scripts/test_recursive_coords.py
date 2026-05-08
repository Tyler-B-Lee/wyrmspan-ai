#!/usr/bin/env python3
"""Test recursive coordinate extraction."""

from game_env import WyrmspanEnv
from debug_utils import ActionDecoder

env = WyrmspanEnv()
decoder = ActionDecoder(env)

# Test example 4: make_payment with nested coords in adv_effects
print("="*70)
print("TEST: make_payment with nested coords")
print("="*70)
action4 = {
    'make_payment': {
        'cost': {'milk': 1}, 
        'action': {
            'adv_effects': {
                'random': {
                    'draw_decision': {
                        'amount': 3, 
                        'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}
                    }
                }
            }, 
            'cost': {'any_resource': 1}, 
            'coords': ('crimson_cavern', 0)
        }
    }, 
    'coords': None
}

vec4 = env.featurize_json(action4)
decoded4 = decoder.decode_vector(vec4)

cave = decoded4['location'].get('cave')
print(f'Decoded location: {cave}')
print(f'Expected: crimson_cavern')
if cave == 'crimson_cavern':
    print('✓ PASS - nested coordinates found!')
else:
    print('✗ FAIL - nested coordinates NOT found')
    
print("\n" + "="*70)
print("Batch test all complex examples:")
print("="*70)

actions = [
    ('play_dragon with coords', {'adv_effects': {'play_dragon': {'L1': 'hand', 'L2': 'any', 'discount': 'none', 'chosen_id': 109}, 'coords': ('crimson_cavern', 0)}, 'cost': {'meat': 1, 'gold': 1, 'coin': 1}, 'coords': None}, 'crimson_cavern'),
    ('play_cave with cave_location', {'adv_effects': {'play_cave': {'source': 'hand', 'chosen_id': 17, 'free': False, 'cave_location': 'golden_grotto'}}, 'cost': {'coin': 1}, 'coords': None}, 'golden_grotto'),
    ('make_payment nested coords', action4, 'crimson_cavern'),
    ('draw_decision', {'draw_decision': {'chosen_id': 64, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}, 'remaining_dragons': [64, 70, 175]}, 'coords': ('crimson_cavern', 0)}, 'crimson_cavern'),
]

passed = 0
for name, action, expected_cave in actions:
    vec = env.featurize_json(action)
    decoded = decoder.decode_vector(vec)
    actual_cave = decoded['location'].get('cave')
    result = "✓" if actual_cave == expected_cave else "✗"
    print(f"{result} {name}: {actual_cave}")
    if actual_cave == expected_cave:
        passed += 1

print(f"\nResult: {passed}/{len(actions)} passed")
