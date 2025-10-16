import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import Bounded
from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
from torchrl.objectives.td3 import TD3Loss

# --- Step 1: Define toy network dimensions
n_obs = 3
n_act = 2
batch_size = [5]

# --- Step 2: Define actor and critic networks
actor_module = nn.Sequential(nn.Linear(n_obs, 64), nn.ReLU(), nn.Linear(64, n_act))
critic_module = nn.Sequential(nn.Linear(n_obs + n_act, 64), nn.ReLU(), nn.Linear(64, 1))

# --- Step 3: Wrap in TensorDictModules
action_spec = Bounded(low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,))
actor = Actor(actor_module, spec=action_spec)
critic = ValueOperator(critic_module, in_keys=["observation", "action"])

# --- Step 4: Initialize TD3Loss
loss_module = TD3Loss(
    actor_network=actor,
    qvalue_network=critic,
    action_spec=action_spec,
    num_qvalue_nets=2,  # <-- TD3 uses two critics internally
)

# --- Step 5: Create dummy batch
data = TensorDict(
    {
        "observation": torch.randn(*batch_size, n_obs),
        "action": torch.randn(*batch_size, n_act).clamp(-1, 1),
        ("next", "observation"): torch.randn(*batch_size, n_obs),
        ("next", "reward"): torch.randn(*batch_size, 1),
        ("next", "done"): torch.zeros(*batch_size, 1, dtype=torch.bool),
        ("next", "terminated"): torch.zeros(*batch_size, 1, dtype=torch.bool),
    },
    batch_size=batch_size,
)

# --- Step 6: Compute loss
td_out = loss_module(data)

print("TD3Loss output keys:", list(td_out.keys()))
print("Loss actor:", td_out['loss_actor'].item())
print("Loss qvalue:", td_out['loss_qvalue'].item())

# --- Step 7: Check internal critics
print("\nNumber of critics inside loss:", len(loss_module.qvalue_network_params))
for i, params in enumerate(loss_module.qvalue_network_params.unbind(0)):
    mean_val = params.mean().item()
    print(f"  Critic {i+1} param mean: {mean_val:.4f}")
