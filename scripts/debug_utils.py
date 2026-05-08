"""
Debugging utilities for Wyrmspan action encoding and embedding verification.
Provides tools to decode, visualize, and validate action vectors and encodings.
"""

import numpy as np
import torch
from game_env import WyrmspanEnv
from game_states import CAVE_NAMES, RESOURCES, DRAGON_PERSONALITIES


class ActionDecoder:
    """Decode action vectors back to human-readable representations."""
    
    def __init__(self, env: WyrmspanEnv):
        self.env = env
        self.action_type_keys = env.action_type_keys
        self.source_keys = env.source_keys
        self.target_keys = env.target_keys
        self.discount_keys = env.discount_keys
        self.random_source_keys = env.random_source_keys
    
    def decode_vector(self, vec: np.ndarray) -> dict:
        """
        Decode an action vector (128-dim) into a human-readable dictionary.
        Returns structured interpretation of the encoded action.
        """
        if len(vec) != self.env.action_dim:
            raise ValueError(f"Vector size {len(vec)} != expected {self.env.action_dim}")
        
        result = {
            "action_types": [],
            "wrappers": {},
            "costs": {},
            "location": {},
            "card_ids": {},
            "flags": {},
            "metadata": {}
        }
        
        # Extract action types
        action_type_vec = vec[self.env.ACTION_TYPE_OFFSET:self.env.ACTION_TYPE_OFFSET + self.env.ACTION_TYPE_SIZE]
        for idx, val in enumerate(action_type_vec):
            if val > 0.5:
                result["action_types"].append(self.action_type_keys[idx])
        
        # Extract wrappers
        wrapper_vec = vec[self.env.WRAP_OFFSET:self.env.WRAP_OFFSET + self.env.WRAP_SIZE]
        wrapper_names = ["make_payment", "adv_effects", "choice", "random", "sequence"]
        for i, name in enumerate(wrapper_names):
            if wrapper_vec[i] > 0.5:
                result["wrappers"][name] = True
        
        # Extract cave location
        cave_vec = vec[self.env.CAVE_OFFSET:self.env.CAVE_OFFSET + self.env.CAVE_SIZE]
        cave_idx = np.argmax(cave_vec)
        if cave_vec[cave_idx] > 0.5:
            if cave_idx < len(CAVE_NAMES):
                result["location"]["cave"] = CAVE_NAMES[cave_idx]
            elif cave_idx == 3:
                result["location"]["cave"] = "mat_slots"
            else:
                result["location"]["cave"] = "other"
        
        # Extract column
        col_vec = vec[self.env.COL_OFFSET:self.env.COL_OFFSET + self.env.COL_SIZE]
        col_idx = np.argmax(col_vec)
        if col_vec[col_idx] > 0.5:
            result["location"]["column"] = int(col_idx)
        
        result["location"]["cave_normalized"] = float(vec[self.env.CAVE_NORM_INDEX])
        result["location"]["col_normalized"] = float(vec[self.env.COL_NORM_INDEX])
        result["location"]["coord_flag"] = bool(vec[self.env.COORD_FLAG_INDEX] > 0.5)
        
        # Extract costs
        cost_vec = vec[self.env.COST_OFFSET:self.env.COST_OFFSET + self.env.COST_SIZE]
        for i, res in enumerate(RESOURCES):
            if i < len(cost_vec) and cost_vec[i] != 0:
                result["costs"][res] = float(cost_vec[i])
        if len(cost_vec) > len(RESOURCES):
            result["costs"]["coin"] = float(cost_vec[len(RESOURCES)])
            result["costs"]["egg"] = float(cost_vec[len(RESOURCES) + 1])
            result["costs"]["dragon_card"] = float(cost_vec[len(RESOURCES) + 2])
            result["costs"]["cave_card"] = float(cost_vec[len(RESOURCES) + 3])
            result["costs"]["any_resource"] = float(cost_vec[len(RESOURCES) + 4])
        
        # Extract source/target
        source_vec = vec[self.env.SOURCE_OFFSET:self.env.SOURCE_OFFSET + self.env.SOURCE_SIZE]
        source_idx = np.argmax(source_vec)
        if source_vec[source_idx] > 0.5 and source_idx < len(self.source_keys):
            result["location"]["source"] = self.source_keys[source_idx]
        
        target_vec = vec[self.env.TARGET_OFFSET:self.env.TARGET_OFFSET + self.env.TARGET_SIZE]
        target_idx = np.argmax(target_vec)
        if target_vec[target_idx] > 0.5 and target_idx < len(self.target_keys):
            result["location"]["target"] = self.target_keys[target_idx]
        
        # Extract discount
        discount_vec = vec[self.env.DISCOUNT_OFFSET:self.env.DISCOUNT_OFFSET + self.env.DISCOUNT_SIZE]
        discount_idx = np.argmax(discount_vec)
        if discount_vec[discount_idx] > 0.5 and discount_idx < len(self.discount_keys):
            result["metadata"]["discount"] = self.discount_keys[discount_idx]
        
        # Extract card IDs
        result["card_ids"]["dragon_id_norm"] = float(vec[self.env.DRAGON_ID_INDEX])
        result["card_ids"]["cave_id_norm"] = float(vec[self.env.CAVE_ID_INDEX])
        result["card_ids"]["display_index_norm"] = float(vec[self.env.DISPLAY_INDEX_INDEX])
        
        # Extract flags
        result["flags"]["aux"] = float(vec[self.env.AUX_INDEX])
        result["flags"]["rand_flag"] = bool(vec[self.env.RAND_FLAG_INDEX] > 0.5)
        result["flags"]["skip"] = bool(vec[self.env.SKIP_FLAG_INDEX] > 0.5)
        result["flags"]["pass"] = bool(vec[self.env.PASS_FLAG_INDEX] > 0.5)
        result["flags"]["has_cost"] = bool(vec[self.env.HAS_COST_FLAG_INDEX] > 0.5)
        
        # Extract sequence/choice lengths
        result["metadata"]["seq_length_norm"] = float(vec[self.env.SEQ_LEN_INDEX])
        result["metadata"]["choice_length_norm"] = float(vec[self.env.CHOICE_LEN_INDEX])
        
        # Extract random source
        rand_source_vec = vec[self.env.RAND_SOURCE_OFFSET:self.env.RAND_SOURCE_OFFSET + self.env.RAND_SOURCE_SIZE]
        rand_idx = np.argmax(rand_source_vec)
        if rand_source_vec[rand_idx] > 0.5 and rand_idx < len(self.random_source_keys):
            result["metadata"]["random_source"] = self.random_source_keys[rand_idx]
        
        # Extract resource type
        res_type_vec = vec[self.env.RESOURCE_TYPE_OFFSET:self.env.RESOURCE_TYPE_OFFSET + self.env.RESOURCE_TYPE_SIZE]
        res_idx = np.argmax(res_type_vec)
        if res_type_vec[res_idx] > 0.5 and res_idx < len(RESOURCES):
            result["metadata"]["resource_type"] = RESOURCES[res_idx]
        
        # Extract personality
        pers_vec = vec[self.env.PERSONALITY_OFFSET:self.env.PERSONALITY_OFFSET + self.env.PERSONALITY_SIZE]
        pers_idx = np.argmax(pers_vec)
        if pers_vec[pers_idx] > 0.5 and pers_idx < len(DRAGON_PERSONALITIES):
            result["metadata"]["personality"] = DRAGON_PERSONALITIES[pers_idx]
        
        # Padding dims (93-191 are padding/unused)
        padding_start = 93
        padding_dims = vec[padding_start:]
        non_zero_padding = np.count_nonzero(padding_dims)
        result["metadata"]["padding_non_zeros"] = int(non_zero_padding)
        
        return result
    
    def print_summary(self, vec: np.ndarray, label: str = "Action"):
        """Print a human-readable summary of the action vector."""
        decoded = self.decode_vector(vec)
        print(f"\n{'='*60}")
        print(f"DECODED {label} VECTOR")
        print(f"{'='*60}")
        
        print(f"\nACTION TYPES: {decoded['action_types']}")
        if decoded["wrappers"]:
            print(f"WRAPPERS: {[k for k, v in decoded['wrappers'].items() if v]}")
        
        print(f"\nLOCATION:")
        for key, val in decoded["location"].items():
            if val or val == 0:
                print(f"  {key}: {val}")
        
        if decoded["costs"]:
            print(f"\nCOSTS: {decoded['costs']}")
        
        print(f"\nCARD IDS:")
        for key, val in decoded["card_ids"].items():
            print(f"  {key}: {val:.4f}")
        
        if decoded["flags"]:
            active_flags = {k: v for k, v in decoded["flags"].items() if v is True}
            if active_flags:
                print(f"\nFLAGS: {active_flags}")
        
        if decoded["metadata"]:
            print(f"\nMETADATA:")
            for key, val in decoded["metadata"].items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")
        
        print(f"{'='*60}\n")


def visualize_action_encoding(json_action: dict, env: WyrmspanEnv, label: str = "Action"):
    """
    Encode a JSON action and print detailed visualization.
    Shows which dimensions are active and what they represent.
    """
    vec = env.featurize_json(json_action)
    print(f"\n{'='*60}")
    print(f"ACTION ENCODING VISUALIZATION: {label}")
    print(f"{'='*60}")
    
    print(f"\nInput JSON (truncated): {str(json_action)[:100]}...")
    print(f"Vector shape: {vec.shape}")
    print(f"Non-zero elements: {np.count_nonzero(vec)} / {len(vec)}")
    print(f"Non-zero ratio: {np.count_nonzero(vec) / len(vec) * 100:.1f}%")
    
    # Show active regions
    active_regions = []
    if np.count_nonzero(vec[0:32]) > 0:
        active_regions.append("ACTION_TYPE")
    if np.count_nonzero(vec[32:37]) > 0:
        active_regions.append("WRAPPERS")
    if np.count_nonzero(vec[37:46]) > 0:
        active_regions.append("LOCATION(cave/col)")
    if np.count_nonzero(vec[49:58]) > 0:
        active_regions.append("COSTS")
    if np.count_nonzero(vec[58:68]) > 0:
        active_regions.append("SOURCE/TARGET")
    if np.count_nonzero(vec[72:92]) > 0:
        active_regions.append("METADATA")
    
    print(f"Active regions: {', '.join(active_regions)}")
    
    # Detailed decode
    decoder = ActionDecoder(env)
    decoder.print_summary(vec, label)


def test_action_batch(actions: list, env: WyrmspanEnv, batch_size: int = 32) -> dict:
    """
    Encode a batch of JSON actions and check for NaN/Inf and statistics.
    Returns validation report.
    """
    print(f"\n{'='*60}")
    print(f"ACTION BATCH VALIDATION TEST ({len(actions)} actions)")
    print(f"{'='*60}")
    
    encoded_vectors = []
    errors = []
    
    for i, action in enumerate(actions):
        try:
            vec = env.featurize_json(action)
            encoded_vectors.append(vec)
            
            # Check for NaN/Inf
            if np.isnan(vec).any():
                errors.append(f"Action {i}: Contains NaN")
            if np.isinf(vec).any():
                errors.append(f"Action {i}: Contains Inf")
        except Exception as e:
            errors.append(f"Action {i}: Encoding failed - {str(e)}")
    
    if not encoded_vectors:
        print("ERROR: No actions encoded successfully!")
        return {"success": False, "errors": errors}
    
    # Stack into batch
    batch_array = np.array(encoded_vectors)
    
    # Statistics
    stats = {
        "total_actions": len(actions),
        "successfully_encoded": len(encoded_vectors),
        "encoding_errors": len(errors),
        "success": len(errors) == 0,
        "errors": errors,
        "statistics": {
            "mean_non_zeros": float(np.mean([np.count_nonzero(v) for v in encoded_vectors])),
            "min_non_zeros": int(np.min([np.count_nonzero(v) for v in encoded_vectors])),
            "max_non_zeros": int(np.max([np.count_nonzero(v) for v in encoded_vectors])),
            "mean_value": float(np.mean(batch_array)),
            "std_value": float(np.std(batch_array)),
            "min_value": float(np.min(batch_array)),
            "max_value": float(np.max(batch_array)),
        }
    }
    
    print(f"\nResults:")
    print(f"  Successfully encoded: {stats['successfully_encoded']}/{stats['total_actions']}")
    print(f"  Encoding errors: {stats['encoding_errors']}")
    
    print(f"\nVector Statistics (across all {len(encoded_vectors)} actions):")
    print(f"  Non-zero elements per vector:")
    print(f"    Mean: {stats['statistics']['mean_non_zeros']:.2f}")
    print(f"    Min:  {stats['statistics']['min_non_zeros']}")
    print(f"    Max:  {stats['statistics']['max_non_zeros']}")
    print(f"  Value range: [{stats['statistics']['min_value']:.4f}, {stats['statistics']['max_value']:.4f}]")
    print(f"  Mean value: {stats['statistics']['mean_value']:.4f}")
    print(f"  Std dev: {stats['statistics']['std_value']:.4f}")
    
    if errors:
        print(f"\nEncoding Errors:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    print(f"{'='*60}\n")
    return stats


def test_embedding_integration(agent: torch.nn.Module, action_batch: torch.Tensor, 
                               card_ids: torch.Tensor) -> dict:
    """
    Test that embeddings are properly integrated into the agent.
    Verifies embedding lookups and dimensions match.
    """
    print(f"\n{'='*60}")
    print(f"EMBEDDING INTEGRATION TEST")
    print(f"{'='*60}")
    
    try:
        # Extract embeddings
        batch_size, num_actions = action_batch.shape[:2]
        
        # Get card ID info
        dragon_ids = torch.clamp(card_ids[..., 0], min=0, max=agent.dragon_embed.num_embeddings - 1)
        cave_ids = torch.clamp(card_ids[..., 1], min=0, max=agent.cave_embed.num_embeddings - 1)
        
        # Look up embeddings
        dragon_vecs = agent.dragon_embed(dragon_ids)
        cave_vecs = agent.cave_embed(cave_ids)
        
        # Verify dimensions
        expected_dragon_dim = agent.dragon_embed.embedding_dim
        expected_cave_dim = agent.cave_embed.embedding_dim
        
        report = {
            "success": True,
            "dragon_embedding": {
                "shape": tuple(dragon_vecs.shape),
                "expected_dim": expected_dragon_dim,
                "actual_dim": dragon_vecs.shape[-1],
                "mean_norm": float(torch.norm(dragon_vecs.view(-1, expected_dragon_dim), dim=1).mean()),
                "matches": dragon_vecs.shape[-1] == expected_dragon_dim
            },
            "cave_embedding": {
                "shape": tuple(cave_vecs.shape),
                "expected_dim": expected_cave_dim,
                "actual_dim": cave_vecs.shape[-1],
                "mean_norm": float(torch.norm(cave_vecs.view(-1, expected_cave_dim), dim=1).mean()),
                "matches": cave_vecs.shape[-1] == expected_cave_dim
            },
            "concatenation": {
                "action_batch_dim": action_batch.shape[-1],
                "dragon_embed_dim": dragon_vecs.shape[-1],
                "cave_embed_dim": cave_vecs.shape[-1],
                "total_after_concat": action_batch.shape[-1] + dragon_vecs.shape[-1] + cave_vecs.shape[-1]
            }
        }
        
        # Check for NaN/Inf
        if torch.isnan(dragon_vecs).any() or torch.isinf(dragon_vecs).any():
            report["success"] = False
            report["dragon_embedding"]["error"] = "Contains NaN or Inf"
        if torch.isnan(cave_vecs).any() or torch.isinf(cave_vecs).any():
            report["success"] = False
            report["cave_embedding"]["error"] = "Contains NaN or Inf"
        
        print(f"\nDragon Embedding:")
        print(f"  Shape: {report['dragon_embedding']['shape']}")
        print(f"  Expected dim: {report['dragon_embedding']['expected_dim']}, Actual: {report['dragon_embedding']['actual_dim']}")
        print(f"  Mean norm: {report['dragon_embedding']['mean_norm']:.4f}")
        print(f"  ✓ Matches" if report['dragon_embedding']['matches'] else "  ✗ MISMATCH")
        
        print(f"\nCave Embedding:")
        print(f"  Shape: {report['cave_embedding']['shape']}")
        print(f"  Expected dim: {report['cave_embedding']['expected_dim']}, Actual: {report['cave_embedding']['actual_dim']}")
        print(f"  Mean norm: {report['cave_embedding']['mean_norm']:.4f}")
        print(f"  ✓ Matches" if report['cave_embedding']['matches'] else "  ✗ MISMATCH")
        
        print(f"\nConcatenation:")
        print(f"  Action batch: {report['concatenation']['action_batch_dim']} dims")
        print(f"  Dragon embed: {report['concatenation']['dragon_embed_dim']} dims")
        print(f"  Cave embed: {report['concatenation']['cave_embed_dim']} dims")
        print(f"  Total: {report['concatenation']['total_after_concat']} dims")
        
        print(f"\nResult: {'✓ PASS' if report['success'] else '✗ FAIL'}")
        print(f"{'='*60}\n")
        
        return report
        
    except Exception as e:
        print(f"ERROR during embedding test: {str(e)}")
        print(f"{'='*60}\n")
        return {"success": False, "error": str(e)}


def get_action_type_from_vector(vec: np.ndarray, env: WyrmspanEnv) -> list:
    """Extract action type(s) from a vector."""
    action_type_vec = vec[env.ACTION_TYPE_OFFSET:env.ACTION_TYPE_OFFSET + env.ACTION_TYPE_SIZE]
    action_types = []
    for idx, val in enumerate(action_type_vec):
        if val > 0.5:
            action_types.append(env.action_type_keys[idx])
    return action_types


def get_action_summary(vec: np.ndarray, env: WyrmspanEnv) -> str:
    """Get a one-line summary of an action."""
    decoder = ActionDecoder(env)
    decoded = decoder.decode_vector(vec)
    
    action_type = decoded["action_types"][0] if decoded["action_types"] else "unknown"
    location = decoded["location"].get("cave", "?")
    col = decoded["location"].get("column", "?")
    
    summary = f"{action_type}"
    if location != "?":
        summary += f" @ {location}[{col}]"
    
    if decoded["costs"]:
        cost_str = ", ".join([f"{k}:{v:.1f}" for k, v in list(decoded["costs"].items())[:2]])
        summary += f" (costs: {cost_str})"
    
    return summary
