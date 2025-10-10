import torch
import gymnasium as gym
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from Agent import env
from torchrl.modules import MLP

import copy
import tempfile

import torch

from matplotlib import pyplot as plt
from tensordict import TensorDictBase

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

from torchrl.envs import (
    check_env_specs,
    ExplorationType,
    PettingZooEnv,
    RewardSum,
    set_exploration_type,
    TransformedEnv,
    VmasEnv,
)

from torchrl.modules import (
    AdditiveGaussianModule,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
)
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder
from tqdm import tqdm


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