import time
from copy import deepcopy
import numpy as np
import pdb
import torchrl
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TDM, TensorDictSequential as Seq
from torchrl.modules import OrnsteinUhlenbeckProcessModule as OUNoise, MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, DDPGLoss,TD3Loss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger
import torch
from torch import nn, optim
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, InitTracker, ObservationNorm, StepCounter
from torchrl.envs.utils import check_env_specs
from utils2 import MultiCriticSoftUpdate, soft_update, QValueEnsembleModule
from tqdm import tqdm




"""
Comparing:
DDPG (single critic) with 
TD3 (twin critics + clipped double-Q + target smoothing + delayed actor updates) 
on LunarLanderContinuous-v3 environment.
"""


# parameters and hyperparameters
INIT_RAND_STEPS = 5000 
TOTAL_FRAMES = 200_000
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
BUFFER_LEN = 1_000_000
REPLAY_BUFFER_SAMPLE = 128
LOG_EVERY = 1000
MLP_SIZE = 256
TAU = 0.01
GAMMA = 0.99
EVAL_EVERY = 10_000   # frames
EVAL_EPISODES = 10
DEVICE = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"

# Seed the Python and RL environments to replicate similar results across training sessions. 

# 1. Environment
env = TransformedEnv(
    GymEnv("LunarLanderContinuous-v3"),
    Compose(
        DoubleToFloat(),
        InitTracker(),
        ObservationNorm(in_keys=["observation"]),
        StepCounter(),
    )
)

eval_env = TransformedEnv(
    GymEnv("LunarLanderContinuous-v3"),
    Compose(
        DoubleToFloat(),
        InitTracker(),
        ObservationNorm(in_keys=["observation"]),
        StepCounter(),
    )
)

env.transform[2].init_stats(1024) 
torch.manual_seed(0)
env.set_seed(0)
check_env_specs(env) 

eval_env.transform[2].init_stats(1024) 
eval_env.set_seed(0)
check_env_specs(eval_env) 

obs_dim = env.observation_spec["observation"].shape[-1] # observation_spec : the observation space
act_dim = env.action_spec.shape[-1] #action_spec : the action space



# 2. Actor (policy)
actor_mlp = MLP(out_features=act_dim, num_cells=[MLP_SIZE, MLP_SIZE], activation_class=nn.ReLU, activate_last_layer=False )
actor = TDM(actor_mlp, in_keys=["observation"], out_keys=["action_raw"])
tanh_on_action = TDM(nn.Tanh(), in_keys=["action_raw"], out_keys=["action"])
exploration_module = OUNoise(
    spec=env.action_spec,
    theta=0.15,
    sigma=0.05,
    dt=1e-2,)
policy = Seq(actor, tanh_on_action, selected_out_keys=["action"])   # deterministic policy
rollout_policy = Seq(actor, tanh_on_action, exploration_module)      # stochastic policy with exploration noise



# 3. Critic (action value function)
critic_mlp = MLP(
    out_features=1, 
    num_cells=[MLP_SIZE, MLP_SIZE],
    activation_class=nn.ReLU,
    activate_last_layer=False)
critic = TDM(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) # = QValue

# Target Networks
actor_target = deepcopy(actor) 
critic_target = deepcopy(critic)

# 4.  loss module
# --- 4) Warm-up forward to initialize lazy modules BEFORE loss/opt
with torch.no_grad():
    td0 = env.reset()          # has "observation"
    _ = policy(td0.clone())    # init actor
    td1 = td0.clone()
    td1["action"] = env.action_spec.rand(td1.batch_size)
    _ = critic(td1)            # init critic
    _ = actor_target(td0.clone())     # init target actor
    _ = critic_target(td1.clone())    # init target critic

loss_ddpg = DDPGLoss(
    actor_network=policy, # deterministic 
    value_network=critic,
    loss_function="l2",
    # delay_actor=True, # for more stability Default is False
    # delay_value=True, # for more stability Default is True
)
loss_td3 = TD3Loss(
    actor_network=policy, # deterministic 
    qvalue_network=critic,
    loss_function="l2",
    action_spec=env.action_spec,
    delay_actor=True, # for more stability, Default is False
    delay_qvalue=True, # for more stability, Default is True
)





# 5. Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=BUFFER_LEN), 
    #sampler=SamplerWithoutReplacement(),
    sampler=RandomSampler(),
)

# 6. Collector
collector = SyncDataCollector( # renvoie des batches de transitions prêts à mettre dans le rb
    env,
    rollout_policy,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=TOTAL_FRAMES, # how many timesteps to run the agent, If the total_frames is not divisible by frames_per_batch, an exception is raised.
    init_random_frames=INIT_RAND_STEPS, # number of initial random steps to populate the replay buffer before training begins, #if len(replay_buffer) > INIT_RAND_STEPS:
    device=DEVICE,
    # replay_buffer=replay_buffer,
    # extend_buffer=False, # =(env.step -> transition -> immediately added to replay_buffer)
)

# 7. Optimizers
optim_actor = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.0)
optim_critic = optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-2)


def train(
    method,
    loss,
    optim_critic,
    optim_actor,
    replay_buffer,
    collector,
    total_frames=TOTAL_FRAMES,
    eval_env=None,
    eval_episodes=EVAL_EPISODES,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    opt_steps=OPTIM_STEPS,
    batch_size=REPLAY_BUFFER_SAMPLE,
):
    
    updater = SoftUpdate(loss, tau=TAU)
    loss.make_value_estimator(gamma=GAMMA)
    total_count = 0
    total_episodes = 0
    t0 = time.time()
    success_steps, qvalues = [], []
    biases = []
    eval_rewards = []       
    eval_steps = []


    pbar = tqdm(total=TOTAL_FRAMES, desc="Training DDPG", dynamic_ncols=True) if method=="DDPG" else tqdm(total=TOTAL_FRAMES, desc="Training TD3", dynamic_ncols=True)
    for i, data in enumerate(collector): # runs through the data collected from the agent’s interactions with the environment
        replay_buffer.extend(data) # add data to the replay buffer
        max_length = replay_buffer[:]["next", "step_count"].max()
        # pdb.set_trace()
        if len(replay_buffer) <= INIT_RAND_STEPS: 
            pbar.update(data.numel())
            continue
        for _ in range(OPTIM_STEPS):
            td = replay_buffer.sample(REPLAY_BUFFER_SAMPLE)
            
            #New Ordering
            # single forward pass, reuse loss_out
            loss_out = loss(td)
            # === Actor update 
            for p in critic.parameters(): p.requires_grad = False
            optim_actor.zero_grad(set_to_none=True)
            loss_pi = loss_out["loss_actor"]        # from the same forward pass
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # or policy.parameters() / policy param accessor
            optim_actor.step()
            optim_actor.zero_grad(set_to_none=True)
            for p in critic.parameters():
                p.requires_grad = True
            # === Critic update ===
            optim_critic.zero_grad(set_to_none=True)
            loss_q = loss_out["loss_qvalue"] if method == "TD3" else loss_out["loss_value"]
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optim_critic.step()
            # === Update targets after both updates ===
            updater.step()
            
    

            
            
            
            # loss_out = loss(td)
            # loss_q = loss_out["loss_qvalue"] if method == "TD3" else loss_out["loss_value"]
            
            # # Critic update
            # optim_critic.zero_grad(set_to_none=True)
            # loss_q.backward()
            # optim_critic.step()
            # updater.step()

            # # Recompute loss_out so actor loss uses the updated critic params
            # loss_out = loss(td)
            

            # # Actor update
            # for p in critic.parameters(): p.requires_grad = False
            # optim_actor.zero_grad(set_to_none=True)
            # loss_pi = loss_out["loss_actor"]
            # loss_pi.backward()
            # optim_actor.step()
            # for p in critic.parameters(): p.requires_grad = True
        
            # Record TD bias
            if method == "DDPG":
                pred_q = loss_out["pred_value"]
                target_q = loss_out["target_value"]
                bias_batch = (pred_q - target_q).detach().mean().item()
            else:
                pred_q1, pred_q2 = loss_out["pred_value"] 
                target_q = loss_out["target_value"]
                bias_q1 = (pred_q1 - target_q).mean().item()
                bias_q2 = (pred_q2 - target_q).mean().item()
                bias_batch = (bias_q1 + bias_q2) / 2
            
            biases.append(bias_batch)

            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
            
            
            qvalues.append(loss_out["pred_value"].mean().item()) 


        success_steps.append(max_length)
        total_count += data.numel()
        total_episodes += data["next", "done"].sum().item()
        pbar.set_postfix({
            "Steps": total_count,
            "Episodes": total_episodes,
            "Mean Q": f"{torch.tensor(qvalues[-50:]).mean().item():.2f}",
            "Bias": f"{torch.tensor(biases[-50:]).mean().item():.2f}",
        })
        pbar.update(data.numel())

        if total_count % LOG_EVERY == 0:
            torchrl_logger.info(f"Successful steps in the last episode: {max_length}, Q: {torch.tensor(qvalues[-50:]).mean().item():.3f}, rb length {len(replay_buffer)}, Number of episodes: {total_episodes}")
            # torchrl_logger.info(f"Steps: {total_count}, Episodes: {total_episodes}, Max Ep Len: {max_length}, ReplayBuffer: {len(replay_buffer)}, Q: {torch.tensor(qvalues[-50:]).item():.3f} [END]")
        if total_count % EVAL_EVERY < FRAMES_PER_BATCH: # A vérifier
            policy.eval()
            with torch.no_grad():
                rewards, lens = [], []
                for _ in range(EVAL_EPISODES):
                    td = eval_env.reset()
                    done = False
                    episode_reward = 0.0
                    while not done:
                        td = policy(td)
                        td = eval_env.step(td)
                        episode_reward += td["next", "reward"].item()
                        done = td["next", "done"].item()
                        td = td.get("next")
                    lens.append(int(td.get("step_count",0)))    
                    rewards.append(episode_reward)
                mean_reward = sum(rewards) / EVAL_EPISODES
                eval_rewards.append(mean_reward)
                eval_steps.append(total_count)
                torchrl_logger.info(f"Evaluation over {EVAL_EPISODES} episodes: {mean_reward:.2f}")
            policy.train()

    pbar.close()
    t1 = time.time()
    print(f"Training took {t1-t0:.2f}s")
    torchrl_logger.info(
        f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
    )
    # window = 200  # adjust for smoothing strength
    # smooth_bias = np.convolve(biases, np.ones(window)/window, mode='valid')
    # plt.figure(figsize=(12,5))
    # plt.plot(biases, label="Raw Bias", color='tab:blue', alpha=0.3)  # transparent fluctuating curve
    # plt.plot(np.arange(window-1, len(biases)), smooth_bias, label="Smoothed Bias", color='tab:blue', linewidth=2)
    # plt.title(f"Training {method} - TD Bias")
    # plt.title(f"Training {method} - Bias")
    # plt.xlabel("Training Steps")
    # plt.show()

    # smooth_qvalue = np.convolve(qvalues, np.ones(window)/window, mode='valid')
    # plt.figure(figsize=(12,5))
    # plt.plot(qvalues, label="Raw q_values", color='tab:blue', alpha=0.3)  # transparent fluctuating curve
    # plt.plot(np.arange(window-1, len(qvalues)), smooth_qvalue, label="Smoothed q_values", color='tab:blue', linewidth=2)
    # # plt.title(f"Training {method} - smoothed Q Values")
    # plt.xlabel("Training Steps")
    # plt.show()
    
    # accumulated_rewards = np.cumsum(eval_rewards)

    # plt.figure(figsize=(10,5))
    # plt.plot(eval_steps, accumulated_rewards, label=f'Accumulated Evaluation Rewards - {method}', color='tab:green')
    # plt.xlabel('Training Steps')
    # plt.ylabel('Accumulated Reward')
    # plt.title(f'{method} - Accumulated Evaluation Reward')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    return eval_steps, eval_rewards



# train(
#     method="TD3",
#     loss=loss_td3,
#     optim_critic=optim_critic,
#     optim_actor=optim_actor,
#     replay_buffer=replay_buffer,
#     collector=collector,
#     total_frames=TOTAL_FRAMES,
#     eval_env=eval_env,
#     eval_episodes=EVAL_EPISODES,
#     log_every=LOG_EVERY,
#     eval_every=EVAL_EVERY,
#     opt_steps=OPTIM_STEPS,
#     batch_size=REPLAY_BUFFER_SAMPLE,
# )
# DDPG training
ddpg_steps, ddpg_rewards = train(
    method="DDPG",
    loss=loss_ddpg,
    optim_critic=optim_critic,
    optim_actor=optim_actor,
    replay_buffer=replay_buffer,
    collector=collector,
    total_frames=TOTAL_FRAMES,
    eval_env=eval_env,
    eval_episodes=EVAL_EPISODES,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    opt_steps=OPTIM_STEPS,
    batch_size=REPLAY_BUFFER_SAMPLE,
)

# TD3 training
td3_steps, td3_rewards = train(
    method="TD3",
    loss=loss_td3,
    optim_critic=optim_critic,
    optim_actor=optim_actor,
    replay_buffer=replay_buffer,
    collector=collector,
    total_frames=TOTAL_FRAMES,
    eval_env=eval_env,
    eval_episodes=EVAL_EPISODES,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    opt_steps=OPTIM_STEPS,
    batch_size=REPLAY_BUFFER_SAMPLE,
)


ddpg_accumulated = np.cumsum(ddpg_rewards)
td3_accumulated  = np.cumsum(td3_rewards)

plt.figure(figsize=(10,5))
plt.plot(ddpg_steps, ddpg_accumulated, label='DDPG Accumulated Reward', linewidth=2)
plt.plot(td3_steps, td3_accumulated, label='TD3 Accumulated Reward', linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Accumulated Reward')
plt.title('Accumulated Rewards - DDPG vs TD3')
plt.legend()
plt.grid(True)
plt.show()
