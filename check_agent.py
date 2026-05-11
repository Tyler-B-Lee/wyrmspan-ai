import torch
import numpy as np
from game_env import WyrmspanEnv
from model_arch import WyrmspanAgent

def check_tensor(name, t):
    is_finite = torch.isfinite(t).all().item()
    print(f"{name} shape: {list(t.shape)}")
    print(f"{name} is finite: {is_finite}")
    if name == "action_scores":
        print(f"{name} min: {t.min().item()}")
        print(f"{name} max: {t.max().item()}")
    elif name == "state_value":
        print(f"{name}: {t.item()}")
    
    if not is_finite:
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        print(f"{name} contains NaN: {has_nan}, Inf: {has_inf}")

env = WyrmspanEnv()
obs, info = env.reset()

print("Observation keys:", obs.keys())
for k, v in obs.items():
    if isinstance(v, np.ndarray):
        print(f"Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}")
    else:
        print(f"Key: {k}, Value: {v}, Type: {type(v)}")

agent = WyrmspanAgent(action_vocab_size=env.action_token_vocab_size)
agent.eval()

obs_t = {}
for k, v in obs.items():
    if isinstance(v, np.ndarray):
        obs_t[k] = torch.from_numpy(v).unsqueeze(0)
    elif isinstance(v, (int, float)):
        obs_t[k] = torch.tensor([v]).unsqueeze(0)
    else:
        obs_t[k] = torch.tensor(v).unsqueeze(0)

# Check queue_masks content
if 'queue_masks' in obs_t:
    print(f"queue_masks sum: {obs_t['queue_masks'].sum().item()}")

try:
    with torch.no_grad():
        action_scores, state_value = agent.policy_value(obs_t)

    check_tensor("action_scores", action_scores)
    check_tensor("state_value", state_value)
    print(f"obs['action_mask'].sum(): {obs['action_mask'].sum()}")
except Exception as e:
    print(f"Error calling policy_value: {e}")
    import traceback
    traceback.print_exc()
