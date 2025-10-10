import gymnasium
import numpy as np
import torch
torch.manual_seed(0)
import time
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.envs.transforms import Compose



env = (GymEnv("CartPole-v1"))
# env = TransformedEnv(env, Compose(StepCounter()))
# time.sleep(10)
episode = 0
env.set_seed(0)
obs, _ = env.reset()

while True:
    action = torch.tensor(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs)
    time.sleep(10)
    if done:
        episode += 1
        time.sleep(2)   
        obs, _ = env.reset()


