import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env  
from stable_baselines3.common.vec_env import VecNormalize
from utils_sb3 import QBiasLoggerDDPG, plot_qbias_ddpg

# TO ADD : ????????????????????????????????????????????????
# Optional: critic_target_std = q_next.std().item() — catches target explosion
# Optional: gradient norms every K steps ???
# eval_freq means: “Run an evaluation every eval_freq calls to env.step().”
# But here’s the subtlety:
# Each env.step() processes 8 frames in parallel, one from each environment.
# So after 1 call to env.step(), you’ve already advanced 8 timesteps total in the real world.
# add load and inference
# os.remove(save_csv) if exists

# parameters and hyperparameters
INIT_RAND_STEPS = 5_000 
TOTAL_FRAMES = 100_000 # 1_000_000
FRAMES_PER_BATCH = 100 # train freq
OPTIM_STEPS =  10 # gradient steps
BUFFER_LEN = 1_000_000
REPLAY_BUFFER_SAMPLE = 256 # 128
LOG_EVERY = 1_000
MLP_SIZE = 256
TAU = 0.005
GAMMA = 0.99
EVAL_EVERY = 10_000   # frames
EVAL_EPISODES = 10
DEVICE = "auto" 


env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=0)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
eval_env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=1) # use a separate environment for training and eval to avoid training bias + different seed
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False) # do not normalize rewards for eval env

eval_env.obs_rms = env.obs_rms # That copies the mean and variance of observations learned by the training env, so your evaluation uses the same normalization scale.

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)
)
# OU noise (often good for DDPG)
# action_noise = OrnsteinUhlenbeckActionNoise(
#     mean=np.zeros(n_actions),
#     sigma=0.2 * np.ones(n_actions),
#     theta=0.15
# )

logger = configure("./logs_ddpg/", ["stdout", "csv", "tensorboard"])
eval_callback = EvalCallback( # The callback runs episodes on eval_env every EVAL_EVERY steps and saves the best model.
    eval_env,
    # best_model_save_path="./ddpg_best",
    # log_path="./ddpg_eval",
    eval_freq=EVAL_EVERY,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=True,
)
qbias_cb = QBiasLoggerDDPG(gamma=GAMMA, sample_n=50_000, save_csv="./logs_ddpg/qbias/qbias_log.csv")
# trigger every EVAL_EVERY timesteps (works with n_envs>1 too, because it uses num_timesteps)
every_qbias = EveryNTimesteps(n_steps=EVAL_EVERY, callback=qbias_cb)

model = DDPG(
    policy="MlpPolicy", 
    env=env, 
    verbose=1, 
    seed=0, 
    learning_rate=1e-3,
    buffer_size=BUFFER_LEN,
    batch_size=REPLAY_BUFFER_SAMPLE,
    tau=TAU,
    gamma=GAMMA,
    device=DEVICE,
    action_noise=action_noise,
    train_freq=FRAMES_PER_BATCH,
    gradient_steps=OPTIM_STEPS,
    learning_starts=INIT_RAND_STEPS,
    policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300])), # Note that for DDPG/TD3, the default architecture is [400, 300]
)
model.set_logger(logger)
model.learn(
    total_timesteps=TOTAL_FRAMES, 
    log_interval=LOG_EVERY, 
    callback=[eval_callback, every_qbias],
    progress_bar=True,
) # train the agent, collects rollouts and optimizes the actor and critic networks
model.save("ddpg_lunarlander")
#model = DDPG.load("ddpg_lunarlander")

# episodes = 10  
# for ep in range(episodes):  
#     obs = eval_env.reset()  
#     done = False  
#     while not done:  
#         action, _states = model.predict(obs, deterministic=True) # no noise during evaluation  
#         obs, rewards, dones, info = eval_env.step(action)  
#         eval_env.render()  
#         done = dones[0]  # Extract boolean from array  
  
eval_env.close()
env.close()

if __name__ == "__main__":
    plot_qbias_ddpg()
    

    
