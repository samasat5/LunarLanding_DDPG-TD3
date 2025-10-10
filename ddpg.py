import torch
import gymnasium as gym
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from Agent import env
from torchrl.modules import MLP

module = MLP(
    out_features=env.action_spec.shape[-1],
    num_cells=[32, 64],
    activation_class=torch.nn.Tanh,
)


module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])

policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],)

rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)