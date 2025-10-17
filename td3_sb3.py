import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env  
from stable_baselines3.common.vec_env import VecNormalize
from utils_sb3 import QBiasLoggerTD3, plot_stats_td3

# parameters and hyperparameters
INIT_RAND_STEPS = 5_000 
TOTAL_FRAMES = 1_00_000 # 1_000_000
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

eval_env.obs_rms = env.obs_rms
# That copies the mean and variance of observations learned by the training env, so your evaluation uses the same normalization scale.

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)
)

logger = configure("./logs_td3/", ["stdout", "csv", "tensorboard"])
qbias_cb = QBiasLoggerTD3(gamma=GAMMA, sample_n=10_000, save_csv="./logs_td3/stats/stats_log.csv")
eval_callback = EvalCallback( # The callback runs episodes on eval_env every EVAL_EVERY steps and saves the best model.
    eval_env,
    best_model_save_path="./td3_best",
    log_path="./td3_eval",
    eval_freq=EVAL_EVERY,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False,
)
every_qbias = EveryNTimesteps(n_steps=EVAL_EVERY, callback=qbias_cb)


model = TD3(
    policy="MlpPolicy", 
    env=env, 
    action_noise=action_noise, 
    verbose=1,
    seed=0, 
    learning_rate=1e-3,
    buffer_size=BUFFER_LEN,
    batch_size=REPLAY_BUFFER_SAMPLE,
    tau=TAU,
    gamma=GAMMA,
    device=DEVICE,
    train_freq=FRAMES_PER_BATCH,
    gradient_steps=OPTIM_STEPS,
    learning_starts=INIT_RAND_STEPS,
    #policy_delay=??
    policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300])),
)
model.set_logger(logger)
model.learn(
    total_timesteps=TOTAL_FRAMES, 
    log_interval=LOG_EVERY, 
    callback=[eval_callback, every_qbias], 
    progress_bar=True,
)# train the agent, collects rollouts and optimizes the actor and critic networks
model.save("td3_lunarlander")
#model = TD3.load("td3_lunarlander")

eval_env.close()
env.close()

if __name__ == "__main__":
    plot_stats_td3()
