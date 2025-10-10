import gymnasium
import numpy as np
import torch
torch.manual_seed(0)
import time
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.envs.transforms import Compose


# env = TransformedEnv(GymEnv("CartPole-v1"), Compose(StepCounter()))
env = TransformedEnv(GymEnv("LunarLander-v3", render_mode="human"))
env.set_seed(0)
obs, _ = env.reset()

while True:
    action = torch.tensor(env.action_space.sample())
    obs, reward, done, _, info = env.step(action)
    print(obs)
    time.sleep(1)
    if done:
        obs, _ = env.reset()
        # break

time.sleep(3)
env.close()


