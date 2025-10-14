import time
import matplotlib.pyplot as plt
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TensorDict, TensorDictSequential as Seq
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger

# Environment
env = TransformedEnv(
    GymEnv("LunarLanderContinuous-v3"),
    StepCounter(max_steps=1000))
env.set_seed(0)

obs_dim = env.observation_spec.shape[-1]
n_actions = env.action_spec.shape[-1]
max_action = env.action_spec.space.high[0]
min_action = env.action_spec.space.low[0]

#  Actor
MLP_SIZE = 64
actor_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[MLP_SIZE, MLP_SIZE])
actor_net = TensorDict(actor_mlp, in_keys=["observation"], out_keys=["action_value"])


#Critic
critic_mlp = MLP(out_features=1, num_cells=[MLP_SIZE, MLP_SIZE])
critic_net = TensorDict(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) 
qvalue = QValueModule(critic_net)

