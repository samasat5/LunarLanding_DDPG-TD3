import gymnasium as gym
import numpy as np
# import pygame
import torch
torch.manual_seed(0)
import time
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.envs.transforms import Compose



env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    # policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()


