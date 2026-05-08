#!/usr/bin/env python3
"""
COORDINATE AND DRAGON EXTRACTION FIXES - FINAL VERIFICATION
===========================================================

This script comprehensively tests all three fixes implemented:
1. Nested cave_location extraction (in play_cave)
2. Draw_decision dragon ID extraction  
3. Recursive coordinate search (through make_payment/adv_effects wrappers)
"""

from game_env import WyrmspanEnv
from debug_utils import ActionDecoder

def test_fixes():
    env = WyrmspanEnv()
    decoder = ActionDecoder(env)
    
    print("="*80)
    print("COORDINATE & DRAGON EXTRACTION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_cases = [
        {
            "name": "Fix #1: Nested cave_location in play_cave",
            "action": {
                'adv_effects': {
                    'play_cave': {
                        'source': 'hand',
                        'chosen_id': 17,
                        'free': False,
                        'cave_location': 'golden_grotto'
                    }
                },
                'cost': {'coin': 1},
                'coords': None
            },
            "expected_cave": "golden_grotto",
            "expected_dragon": 0.0,
            "description": "cave_location nested inside play_cave (not at top level)"
        },
        {
            "name": "Fix #2: Dragon extraction from draw_decision",
            "action": {
                'draw_decision': {
                    'chosen_id': 64,
                    'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1},
                    'remaining_dragons': [64, 70, 175]
                },
                'coords': ('crimson_cavern', 0)
            },
            "expected_cave": "crimson_cavern",
            "expected_dragon": 64/183.0,
            "description": "Dragon ID extracted from draw_decision's chosen_id"
        },
        {
            "name": "Fix #3: Recursive coords in nested make_payment",
            "action": {
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
            },
            "expected_cave": "crimson_cavern",
            "expected_dragon": 0.0,
            "description": "Coordinates found through make_payment → adv_effects → random"
        },
        {
            "name": "Integration: play_dragon with top-level coords",
            "action": {
                'adv_effects': {
                    'play_dragon': {
                        'L1': 'hand',
                        'L2': 'any',
                        'discount': 'none',
                        'chosen_id': 109
                    },
                    'coords': ('crimson_cavern', 0)
                },
                'cost': {'meat': 1, 'gold': 1, 'coin': 1},
                'coords': None
            },
            "expected_cave": "crimson_cavern",
            "expected_dragon": 109/183.0,
            "description": "Traditional coords at adv_effects level"
        },
        {
            "name": "Integration: explore action",
            "action": {
                'adv_effects': {
                    'explore': {
                        'cave_name': 'crimson_cavern',
                        'index': 0
                    }
                },
                'cost': {'coin': 1},
                'coords': None
            },
            "expected_cave": "crimson_cavern",
            "expected_dragon": 0.0,
            "description": "explore action with cave_name field"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"  Description: {test['description']}")
        print("-" * 80)
        
        try:
            vec = env.featurize_json(test['action'])
            decoded = decoder.decode_vector(vec)
            
            actual_cave = decoded['location'].get('cave', 'OTHER')
            actual_dragon = decoded['card_ids']['dragon_id_norm']
            
            cave_match = actual_cave == test['expected_cave']
            dragon_match = abs(actual_dragon - test['expected_dragon']) < 0.01
            
            print(f"  Location: {actual_cave}")
            print(f"    Expected: {test['expected_cave']}")
            print(f"    Match: {'✓' if cave_match else '✗'}")
            
            print(f"  Dragon (normalized): {actual_dragon:.4f}")
            print(f"    Expected: {test['expected_dragon']:.4f}")
            print(f"    Match: {'✓' if dragon_match else '✗'}")
            
            if cave_match and dragon_match:
                print("  Result: ✓ PASS")
                passed += 1
            else:
                print("  Result: ✗ FAIL")
                failed += 1
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print("  Result: ✗ FAIL (Exception)")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*80)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED - All coordinate and dragon extraction fixes working!")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = test_fixes()
    exit(0 if success else 1)
