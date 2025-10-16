import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

# parameters and hyperparameters
INIT_RAND_STEPS = 5_000 
TOTAL_FRAMES = 100_000
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
BUFFER_LEN = 1_000_000
REPLAY_BUFFER_SAMPLE = 256
LOG_EVERY = 1_000
MLP_SIZE = 256
TAU = 0.005
GAMMA = 0.99
EVAL_EVERY = 10_000   # frames
EVAL_EPISODES = 3
DEVICE = "auto" 


env = Monitor(gym.make("LunarLanderContinuous-v3"))
eval_env = Monitor(gym.make("LunarLanderContinuous-v3")) # use a separate environment for training and eval to avoid training bias
env.reset(seed=0)
eval_env.reset(seed=0)

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
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ddpg_best",
    log_path="./ddpg_eval",
    eval_freq=EVAL_EVERY,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False,
)

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
    policy_kwargs=dict(net_arch=[MLP_SIZE, MLP_SIZE]),
)
model.set_logger(logger)
model.learn(total_timesteps=TOTAL_FRAMES, log_interval=LOG_EVERY, callback=eval_callback) # train the agent
model.save("ddpg_lunarlander")
vec_env = model.get_env() # returns the correct environment


model = DDPG.load("ddpg_lunarlander")
episodes = 10

for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        env.render()
        

env.close()