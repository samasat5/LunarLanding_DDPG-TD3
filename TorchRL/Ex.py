import torch
from torch import nn, optim
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import DDPGLoss
from tensordict.nn import TensorDictModule         # now lives in tensordict
# from torchrl.modules.value import ValueOperator 
import pdb


# 1. Environment
env = TransformedEnv(
    GymEnv("LunarLander-v3"),
    Compose(DoubleToFloat())
)

obs_dim = env.observation_spec["observation"].shape[0]
act_dim = env.action_spec.shape[-1]

# 2. Actor
actor_net = nn.Sequential(
    nn.Linear(obs_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, act_dim),
    nn.Tanh(),  # actions in [-1,1]
)








actor = TensorDictModule(
    actor_net,
    in_keys=["observation", "action"],
    out_keys=["state_action_value"],
)

# 3. Critic
critic_net = nn.Sequential(
    nn.Linear(obs_dim + act_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)

critic = TensorDictModule(
    critic_net,
    in_keys=["observation", "action"],
    out_keys=["state_action_value"],
)



loss_module = DDPGLoss(
    actor_network=actor,
    value_network=critic,)

# Replay buffer
storage = LazyTensorStorage(max_size=1000000)
sampler = SamplerWithoutReplacement()
replay_buffer = ReplayBuffer(storage=storage, sampler=sampler)

# 6. Collector
collector = SyncDataCollector(
    env,
    actor,device="cpu",)










# 7. Optimizers
optim_actor = optim.Adam(actor.parameters(), lr=1e-4)
optim_critic = optim.Adam(critic.parameters(), lr=1e-3)

# 8. Training loop
for i, data in enumerate(collector):
    replay_buffer.extend(data)
    batch = replay_buffer.sample(128)

    # compute DDPG losses
    loss_td = loss_module(batch)
    actor_loss = loss_td["loss_actor"]
    critic_loss = loss_td["loss_qvalue"]

    # critic step
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()

    # actor step
    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()

    if i % 50 == 0:
        print(f"Step {i}: actor_loss={actor_loss.item():.3f}, critic_loss={critic_loss.item():.3f}")
