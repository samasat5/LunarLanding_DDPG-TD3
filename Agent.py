import gym
import numpy as np
import torch
torch.manual_seed(0)
import time
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from lunar_lander import LunarLander
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq



# env = TransformedEnv(GymEnv("CartPole-v1"), [StepCounter()])
env = TransformedEnv(GymEnv("LunarLander-v2"), [StepCounter()])
env.set_seed(0)
obs, _ = env.reset()

while True:
    action = torch.tensor([env.action_space.sample()])
    obs, reward, done, _, info = env.step(action)
    print(obs)
    time.sleep(0.1)
    if done:
        obs, _ = env.reset()
        break
        
env.close()


