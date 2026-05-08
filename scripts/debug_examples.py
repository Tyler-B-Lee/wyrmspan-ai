#!/usr/bin/env python3
"""
Quick Start Guide for Debugging Utilities

This script demonstrates how to use the debugging utilities to verify and understand
the Wyrmspan AI action encoding system.
"""

import torch
import numpy as np
from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent
from debug_utils import (
    ActionDecoder,
    visualize_action_encoding,
    test_action_batch,
    test_embedding_integration,
    get_action_type_from_vector,
    get_action_summary
)


def example_1_decode_single_action():
    """Example 1: Decode and inspect a single encoded action."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Decode a Single Action")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Create a test action
    action = {
        "play_dragon": {
            "L1": "hand",
            "L2": "here",
            "chosen_id": 42,
            "discount": "free"
        },
        "coords": ("golden_grotto", 3),
        "cost": {"meat": 2, "gold": 1}
    }
    
    # Encode to 128-dim vector
    vec = env.featurize_json(action)
    
    # Decode back to human-readable format
    decoder = ActionDecoder(env)
    decoded = decoder.decode_vector(vec)
    
    print(f"\nAction JSON: {action}")
    print(f"Vector shape: {vec.shape}")
    print(f"Non-zero elements: {np.count_nonzero(vec)}/128")
    print(f"\nDecoded info:")
    print(f"  Action types: {decoded['action_types']}")
    print(f"  Location: {decoded['location']}")
    print(f"  Costs: {decoded['costs']}")
    print(f"  Dragon ID (normalized): {decoded['card_ids']['dragon_id_norm']:.4f}")


def example_2_visualize_encoding():
    """Example 2: Visualize which dimensions are active in an encoding."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Visualize Action Encoding")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Create actions with different structures
    actions = [
        {'make_payment': {'cost': {'milk': 1}, 'action': {'adv_effects': {'random': {'draw_decision': {'amount': 3, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}}}}, 'cost': {'any_resource': 1}, 'coords': ('crimson_cavern', 0)}}, 'coords': None},
        {'random': {'gain_dragon': {'possible_outcomes': 'dragon_deck'}, 'coords': None}, 'coords': None},
        {'draw_decision': {'chosen_id': 64, 'limits': {'keep': 1, 'discard': 1, 'tuck_here': 1}, 'remaining_dragons': [64, 70, 175]}, 'coords': ('crimson_cavern', 0)},
        {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 30, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)},
        {
            "adv_effects": {
                "choice": [
                    {"adv_effects": {"sequence":[{"gain_dragon": {"source": "any"}}, {"gain_guild": {"source": "any"}}]}},
                    {"adv_effects": {"sequence":[{"gain_dragon": {"source": "any"}}, {"gain_cave": {"source": "any"}}]}},
                    {"adv_effects": {"sequence":[{"gain_dragon": {"source": "any"}}, {"gain_resource": {"type": "any"}}]}},
                    {"adv_effects": {"sequence":[{"gain_cave": {"source": "any"}}, {"gain_guild": {"source": "any"}}]}},
                    {"adv_effects": {"sequence":[{"gain_cave": {"source": "any"}}, {"gain_resource": {"type": "any"}}]}},
                    {"adv_effects": {"sequence":[{"gain_resource": {"type": "any"}}, {"gain_guild": {"source": "any"}}]}}
                ]
            },
            "opponent_effect": {
                "adv_effects": {}
            }
        },
        {'choice': [{'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 30, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)}, {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 175, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)}, {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 162, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)}, {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 120, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)}, {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 126, 'gain_from_cost': False}, 'coords': ('golden_grotto', 0)}], 'coords': None}
    ]
    
    for i, action in enumerate(actions):
        visualize_action_encoding(action, env, f"Test Action {i+1}")


def example_3_batch_validation():
    """Example 3: Validate a batch of actions."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Validation")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Generate diverse test actions
    actions = [
        {"play_dragon": {"L1": "hand", "L2": "here", "chosen_id": dragon_id}}
        for dragon_id in range(10, 50)
    ]
    
    # Validate the entire batch
    stats = test_action_batch(actions, env)
    
    print(f"\nValidation Results:")
    print(f"  ✓ All actions validated")
    print(f"  Mean non-zeros per vector: {stats['statistics']['mean_non_zeros']:.2f}")
    print(f"  Value range: [{stats['statistics']['min_value']:.4f}, {stats['statistics']['max_value']:.4f}]")


def example_4_embedding_verification():
    """Example 4: Verify embedding integration in the agent."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Embedding Verification")
    print("="*70)
    
    # Create agent with new architecture
    agent = WyrmspanAgent(
        embedding_dim=768,
        state_dim=256,
        action_scalar_dim=128,
        dragon_embed_dim=16,
        cave_embed_dim=12,
    )
    agent.eval()
    
    # Create dummy action batch
    batch_size = 2
    num_actions = 16
    action_batch = torch.randn(batch_size, num_actions, 128)  # Scalar features
    card_ids = torch.randint(1, 184, (batch_size, num_actions, 2))  # Dragon and cave IDs
    
    # Test embedding integration
    with torch.no_grad():
        report = test_embedding_integration(agent, action_batch, card_ids)
    
    print(f"\nEmbedding Integration:")
    print(f"  Dragon embedding: {report['dragon_embedding']['shape']}")
    print(f"  Cave embedding: {report['cave_embedding']['shape']}")
    print(f"  Total concatenated dims: {report['concatenation']['total_after_concat']}")


def example_5_action_summary():
    """Example 5: Get quick summaries of encoded actions."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Action Summaries")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Create diverse actions
    actions = [
        {"play_dragon": {"L1": "hand", "L2": "here", "chosen_id": 10}},
        {"explore": {"cave_name": "crimson_cavern"}},
        {"pass": {}},
        {"lay_egg": {"location": "here"}},
        {
            "play_cave": {
                "source": "display",
                "chosen_id": 20,
                "coords": ("golden_grotto", 1)
            }
        }
    ]
    
    print("\nAction Summaries:")
    for i, action in enumerate(actions):
        vec = env.featurize_json(action)
        summary = get_action_summary(vec, env)
        action_types = get_action_type_from_vector(vec, env)
        print(f"  {i+1}. {summary}")
        print(f"     Types: {action_types}\n")


def example_6_compare_encodings():
    """Example 6: Compare encodings of similar actions."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Comparing Action Encodings")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Create two similar actions
    action1 = {
        "play_dragon": {"L1": "hand", "L2": "here", "chosen_id": 50, "discount": "none"}
    }
    action2 = {
        "play_dragon": {"L1": "hand", "L2": "here", "chosen_id": 51, "discount": "none"}
    }
    
    vec1 = env.featurize_json(action1)
    vec2 = env.featurize_json(action2)
    
    # Compute similarity
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    l2_dist = np.linalg.norm(vec1 - vec2)
    
    print(f"\nAction 1: Dragon ID 50")
    print(f"Action 2: Dragon ID 51")
    print(f"\nSimilarity Metrics:")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  L2 distance: {l2_dist:.4f}")
    print(f"  Hamming distance (non-zero diff): {np.count_nonzero(vec1 != vec2)}")
    
    # Highlight differences
    decoder = ActionDecoder(env)
    dec1 = decoder.decode_vector(vec1)
    dec2 = decoder.decode_vector(vec2)
    
    print(f"\nKey differences:")
    print(f"  Dragon ID (norm): {dec1['card_ids']['dragon_id_norm']:.4f} → {dec2['card_ids']['dragon_id_norm']:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("WYRMSPAN AI DEBUGGING UTILITIES - QUICK START GUIDE")
    print("="*70)
    
    print("""
This guide demonstrates the debugging utilities for the Wyrmspan action encoding system.

Key Components:
1. ActionDecoder - Decode 128-dim vectors to human-readable format
2. visualize_action_encoding - Show active dimensions in an encoding
3. test_action_batch - Validate multiple actions for errors
4. test_embedding_integration - Verify embedding correctness in agent
5. get_action_summary - Quick one-line action description
6. get_action_type_from_vector - Extract action types

Running examples...
""")
    
    try:
        example_1_decode_single_action()
        example_2_visualize_encoding()
        example_3_batch_validation()
        example_4_embedding_verification()
        example_5_action_summary()
        example_6_compare_encodings()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("""
Next steps:
- Examine debug_utils.py for complete documentation
- Integrate debugging calls into your training loop
- Monitor action encodings during learning
- Use ActionDecoder to understand model decisions
""")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
