import gymnasium as gym
from stable_baselines3 import DDPG

env = gym.make("LunarLanderContinuous-v3", render_mode="human")
env.reset()
print("sample action : ", env.action_space.sample())
print("observation space shape : ", env.observation_space.shape)
print("sample observation : ", env.observation_space.sample())

for step in range(200):
    obs, reward, done, _, _ = env.step(env.action_space.sample())
    print(reward)

env.close()