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


# Critic
MLP_SIZE = 64
critic_mlp = MLP(out_features=1, num_cells=[MLP_SIZE, MLP_SIZE])
critic_net = TensorDict(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) 
qvalue = QValueModule(critic_net)

#  Actor
actor_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[MLP_SIZE, MLP_SIZE])
actor_net = TensorDict(actor_mlp, in_keys=["observation"], out_keys=["action_value"])

EPS_0 = 0.2
exploration_module = EGreedyModule(
    action_spec=env.action_spec,
    eps_init=EPS_0)
policy = Seq(actor_net, exploration_module)

# Collect the data from the agent’s interactions with the environment
collector = SyncDataCollector(
    env,
    policy,
    total_frames=10000,
    frames_per_batch=1000,
    max_frames_per_traj=1000,
    device="cpu",
)

# Replay buffer
BUFFER_LEN = 100000
rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_LEN))
