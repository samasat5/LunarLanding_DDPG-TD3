import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=50_000, log_interval=1000)
model.save("ddpg_lunarlander")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_lunarlander")
episodes = 10

for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        

env.close()