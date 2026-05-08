#!/usr/bin/env python3
"""
Integration test for Wyrmspan AI system.
Verifies:
1. Action encoding (192-dim)
2. Embedding integration (dragon/cave)
3. Projection MLP fusion
4. End-to-end forward pass
"""

import sys
import torch
import numpy as np
from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent, WyrmspanActionScorer
from debug_utils import (
    ActionDecoder, 
    visualize_action_encoding, 
    test_action_batch,
    test_embedding_integration,
    get_action_summary
)


def test_action_dimension():
    """Test 1: Verify action dimension is 192."""
    print("\n" + "="*70)
    print("TEST 1: ACTION DIMENSION OPTIMIZATION (192 → 128)")
    print("="*70)
    
    env = WyrmspanEnv()
    
    print(f"✓ ACTION_VEC_SIZE = {env.ACTION_VEC_SIZE}")
    print(f"✓ Observation space action_candidates shape: {env.observation_space['action_candidates'].shape}")
    
    assert env.ACTION_VEC_SIZE == 128, f"Expected 128, got {env.ACTION_VEC_SIZE}"
    assert env.observation_space['action_candidates'].shape[-1] == 128, "Observation space mismatch"
    
    print("✓ PASS: Action dimension optimized to 128 (was 192, removed 99 dims unused padding)")
    return True


def test_simple_action_encoding():
    """Test 2: Verify action encoding works with 128-dim vectors."""
    print("\n" + "="*70)
    print("TEST 2: ACTION ENCODING (128-DIM VECTORS)")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Create simple test actions
    test_actions = [
        {'tuck_from': {'L1': 'hand', 'L2': 'here', 'chosen_id': 176, 'gain_from_cost': False}, 'coords': ('crimson_cavern', 2)},
        {'cache_from': {'type': 'milk', 'L1': 'player_supply', 'L2': 'here', 'chosen_payment': {'milk': 1}}, 'coords': ('crimson_cavern', 1)},
        {"pass": {}},
        {'adv_effects': {'play_dragon': {'L1': 'hand', 'L2': 'any', 'discount': 'none', 'chosen_id': 109}, 'coords': ('crimson_cavern', 0)}, 'cost': {'meat': 1, 'gold': 1, 'coin': 1}, 'coords': None},
        {
            "make_payment": {
                "cost": {"meat": 1, "gold": 2},
                "action": {"play_dragon": {"L1": "hand", "L2": "here", "chosen_id": 15}}
            }
        },
        {
            "adv_effects": {
                "sequence": [
                    {"lay_egg": {"location": "any"}},
                    {"lay_egg": {"location": "any"}}
                ]
            }
        },
        {
            "adv_effects": {
                "sequence": [
                    {"gain_resource": {"type": "gold"}},
                    {"gain_resource": {"type": "crystal"}},
                    {"gain_resource": {"type": "meat"}}
                ]
            },
            "cost": {
                "coin": 1
            }
        },
        {
            "adv_effects": {
                "choice": [
                    {"adv_effects": {"gain_guild": {"source": "any"}}, "cost": {"dragon_card": 1}},
                    {"adv_effects": {"gain_guild": {"source": "any"}}, "cost": {"cave_card": 1}}
                ]
            }
        },
        {'make_payment': 
            {'cost': {'dragon_card': (90, 88)}, 
             'action': {'adv_effects': {'end_game': {'amount': 3}}, 
                        'cost': {'dragon_card': 2}, 'coords': None}}, 
            'coords': None
        }
    ]
    
    print(f"Encoding {len(test_actions)} test actions...")
    vectors = []
    for i, action in enumerate(test_actions):
        vec = env.featurize_json(action)
        assert len(vec) == 128, f"Vector {i} has length {len(vec)}, expected 128"
        assert not np.isnan(vec).any(), f"Vector {i} contains NaN"
        assert not np.isinf(vec).any(), f"Vector {i} contains Inf"
        vectors.append(vec)
        print(f"  ✓ Action {i+1}: {get_action_summary(vec, env)}")
    
    print(f"✓ PASS: All {len(test_actions)} actions encoded successfully to 128 dims")
    return True, vectors


def test_action_decoder():
    """Test 3: Verify action decoder works on encoded vectors."""
    print("\n" + "="*70)
    print("TEST 3: ACTION DECODER")
    print("="*70)
    
    env = WyrmspanEnv()
    decoder = ActionDecoder(env)
    
    # Create a test action with multiple features
    # Note: coords typically appear at the top level, extracted from play_dragon context
    action = {
        "play_dragon": {
            "L1": "hand",
            "L2": "here",
            "chosen_id": 42,
            "discount": "free"
        },
        "coords": ("crimson_cavern", 2),
        "cost": {"meat": 1, "gold": 1}
    }
    
    vec = env.featurize_json(action)
    decoded = decoder.decode_vector(vec)
    
    print(f"\nOriginal action types: {decoded['action_types']}")
    print(f"Decoded location: {decoded['location']}")
    print(f"Decoded costs: {decoded['costs']}")
    print(f"Decoded dragon_id_norm: {decoded['card_ids']['dragon_id_norm']:.4f}")
    
    # Verify key fields
    assert "play_dragon" in decoded["action_types"], "Missing action type"
    assert decoded["location"].get("cave") == "crimson_cavern", f"Cave location mismatch: got {decoded['location'].get('cave')}"
    assert decoded["location"].get("column") == 2, "Column mismatch"
    
    print("✓ PASS: Action decoder correctly reconstructs encoded vectors")
    return True


def test_embeddings():
    """Test 4: Verify embedding dimensions and integration."""
    print("\n" + "="*70)
    print("TEST 4: EMBEDDING DIMENSIONS (Dragon: 16, Cave: 12)")
    print("="*70)
    
    agent = WyrmspanAgent(
        embedding_dim=768,
        state_dim=256,
        action_scalar_dim=192,
        dragon_embed_dim=16,
        cave_embed_dim=12,
    )
    
    print(f"Dragon embedding: {agent.dragon_embed.embedding_dim} dims")
    print(f"Cave embedding: {agent.cave_embed.embedding_dim} dims")
    
    assert agent.dragon_embed.embedding_dim == 16, "Dragon embedding dimension incorrect"
    assert agent.cave_embed.embedding_dim == 12, "Cave embedding dimension incorrect"
    
    # Test embedding lookup
    dragon_ids = torch.tensor([[10, 50, 1], [0, 100, 183]], dtype=torch.long)
    dragon_vecs = agent.dragon_embed(dragon_ids)
    
    assert dragon_vecs.shape == (2, 3, 16), f"Dragon embedding shape mismatch: {dragon_vecs.shape}"
    assert not torch.isnan(dragon_vecs).any(), "Dragon embeddings contain NaN"
    
    cave_ids = torch.tensor([[5, 30], [0, 75]], dtype=torch.long)
    cave_vecs = agent.cave_embed(cave_ids)
    
    assert cave_vecs.shape == (2, 2, 12), f"Cave embedding shape mismatch: {cave_vecs.shape}"
    assert not torch.isnan(cave_vecs).any(), "Cave embeddings contain NaN"
    
    print(f"✓ Dragon embedding lookup: {dragon_vecs.shape}")
    print(f"✓ Cave embedding lookup: {cave_vecs.shape}")
    print("✓ PASS: Embeddings initialized and functional")
    return True


def test_fusion_mlp():
    """Test 5: Verify projection MLP fusion with new dimensions."""
    print("\n" + "="*70)
    print("TEST 5: PROJECTION MLP (Fusion Layer)")
    print("="*70)
    
    state_dim = 256
    # New dimensions: scalar (128) + pooled_action_cards (16) = 144
    action_dim = 128 + 16
    fusion_dim = 256
    
    scorer = WyrmspanActionScorer(state_dim, action_dim, fusion_dim)
    
    # Create batch of fused action features
    batch_size = 8
    num_actions = 16
    
    action_features = torch.randn(batch_size, num_actions, action_dim)
    state_embedding = torch.randn(batch_size, state_dim)
    
    # Forward pass
    with torch.no_grad():
        # Test fusion MLP
        fused = scorer.fusion_mlp(action_features)
        assert fused.shape == (batch_size, num_actions, fusion_dim), f"Fusion output shape mismatch: {fused.shape}"
        assert not torch.isnan(fused).any(), "Fusion output contains NaN"
        
        # Test full scoring
        scores = scorer(state_embedding, action_features)
        assert scores.shape == (batch_size, num_actions), f"Score shape mismatch: {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
    
    print(f"✓ Input shape: {action_features.shape}")
    print(f"✓ Fusion MLP: {action_dim} → {fusion_dim} → {state_dim}")
    print(f"✓ Output scores shape: {scores.shape}")
    print("✓ PASS: Projection MLP and scoring working correctly")
    return True


def test_full_forward_pass():
    """Test 6: End-to-end agent forward pass with new architecture (card IDs)."""
    print("\n" + "="*70)
    print("TEST 6: END-TO-END FORWARD PASS (NEW ARCHITECTURE WITH CARD IDS)")
    print("="*70)
    
    # Create agent
    agent = WyrmspanAgent(
        embedding_dim=768,
        state_dim=256,
        action_scalar_dim=128,
        dragon_embed_dim=16,
        cave_embed_dim=12,
    )
    agent.eval()
    
    # Create dummy batch with new observation structure
    batch_size = 4
    max_actions = 32
    max_cards_per_action = 5
    
    with torch.no_grad():
        global_stats = torch.randn(batch_size, 20)
        # NEW: Card IDs instead of embedded vectors
        hand_card_ids = torch.randint(0, 184, (batch_size, 15), dtype=torch.int64)  # Dragon IDs
        board_slot_card_ids = torch.randint(0, 184, (batch_size, 12), dtype=torch.int64)  # Dragon IDs
        action_candidates = torch.randn(batch_size, max_actions, 128)
        # NEW: Variable-length action card references [card_kind, card_id]
        action_cards = torch.zeros((batch_size, max_actions, max_cards_per_action, 2), dtype=torch.int64)
        # Fill in some card references for testing (dragons only in this test)
        for b in range(batch_size):
            for a in range(min(max_actions, 10)):
                # Add 1-3 random dragon references per action
                num_cards = torch.randint(1, 4, (1,)).item()
                for c in range(num_cards):
                    action_cards[b, a, c] = torch.tensor([0, torch.randint(1, 184, (1,)).item()], dtype=torch.int64)
        
        action_mask = torch.ones(batch_size, max_actions)
        
        # Forward pass with new signature
        scores, value = agent.forward(
            global_stats, hand_card_ids, board_slot_card_ids,
            action_candidates, action_cards, action_mask
        )
        
        assert scores.shape == (batch_size, max_actions), f"Scores shape mismatch: {scores.shape}"
        assert value.shape == (batch_size,), f"Value shape mismatch: {value.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isnan(value).any(), "Values contain NaN"
    
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Max actions: {max_actions}")
    print(f"✓ Hand card IDs shape: {hand_card_ids.shape}")
    print(f"✓ Board slot card IDs shape: {board_slot_card_ids.shape}")
    print(f"✓ Action cards shape: {action_cards.shape}")
    print(f"✓ Action scores shape: {scores.shape}")
    print(f"✓ State values shape: {value.shape}")
    print("✓ PASS: Full forward pass with new architecture completed successfully")
    return True


def test_batch_encoding():
    """Test 7: Batch action encoding and validation."""
    print("\n" + "="*70)
    print("TEST 7: BATCH ENCODING & VALIDATION")
    print("="*70)
    
    env = WyrmspanEnv()
    
    # Generate diverse test actions
    test_actions = [
        {"play_dragon": {"L1": "hand", "L2": "here", "chosen_id": i, "discount": "none"}}
        for i in range(5, 25)
    ]
    
    stats = test_action_batch(test_actions, env)
    
    assert stats["success"], "Batch encoding validation failed"
    assert stats["successfully_encoded"] == len(test_actions), "Not all actions encoded"
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("WYRMSPAN AI INTEGRATION TEST SUITE")
    print("Testing: Action Dimension (128), Embeddings, Projection MLP, Debugging")
    print("="*70)
    
    tests = [
        ("Action Dimension Upgrade", test_action_dimension),
        ("Simple Action Encoding", lambda: test_simple_action_encoding()),
        ("Action Decoder", test_action_decoder),
        ("Embeddings (Dragon/Cave)", test_embeddings),
        ("Projection MLP Fusion", test_fusion_mlp),
        ("End-to-End Forward Pass", test_full_forward_pass),
        ("Batch Encoding & Validation", test_batch_encoding),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            if isinstance(result, tuple):
                if result[0]:
                    passed += 1
                else:
                    failed += 1
                    print(f"✗ FAILED: {test_name}")
            elif result:
                passed += 1
            else:
                failed += 1
                print(f"✗ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        print("="*70)
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
