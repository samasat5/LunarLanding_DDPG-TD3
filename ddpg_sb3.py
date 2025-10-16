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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env  
from stable_baselines3.common.vec_env import VecNormalize

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
EVAL_EPISODES = 3
DEVICE = "auto" 


env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=0)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
eval_env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=1) # use a separate environment for training and eval to avoid training bias + different seed
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False) # do not normalize rewards for eval env

eval_env.obs_rms = env.obs_rms
# That copies the mean and variance of observations learned by the training env, so your evaluation uses the same normalization scale.

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
    best_model_save_path="./ddpg_best",
    log_path="./ddpg_eval",
    eval_freq=EVAL_EVERY,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=True,
)


class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress", ncols=100, dynamic_ncols=True)

    def _on_step(self) -> bool:
        self.pbar.update(1)  # one step per env step
        return True

    def _on_training_end(self):
        self.pbar.close()
progress_bar = TqdmProgressCallback(TOTAL_FRAMES)


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
model.learn(total_timesteps=TOTAL_FRAMES, log_interval=LOG_EVERY, callback=[eval_callback, progress_bar]) # train the agent, collects rollouts and optimizes the actor and critic networks
model.save("ddpg_lunarlander")

#vec_env = model.get_env() # returns the correct environment

model = DDPG.load("ddpg_lunarlander")

episodes = 10  
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
    eval_path = "./ddpg_eval/evaluations.npz"

    if os.path.exists(eval_path):
        data = np.load(eval_path, allow_pickle=True)
        timesteps  = data["timesteps"]
        results    = data["results"]
        ep_lengths = data["ep_lengths"]

        # Compute mean and std over episodes at each evaluation
        mean_returns = results.mean(axis=1)
        std_returns  = results.std(axis=1)
        mean_lengths = ep_lengths.mean(axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, mean_returns, label="Mean Eval Reward")
        plt.fill_between(
            timesteps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.2,
            label="±1 std"
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Evaluation Reward")
        plt.title("DDPG Performance on LunarLanderContinuous-v3")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, mean_lengths, color="orange", label="Mean Episode Length")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Length")
        plt.title("Mean Evaluation Episode Length")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("No evaluation file found at:", eval_path)
