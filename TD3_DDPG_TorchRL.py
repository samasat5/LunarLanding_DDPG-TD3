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
from utils2 import MultiCriticSoftUpdate, soft_update, plot_mc_estimate
from tqdm import tqdm








# parameters and hyperparameters
INIT_RAND_STEPS = 5000 
TOTAL_FRAMES = 100_000
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
BUFFER_LEN = 1_000_000
REPLAY_BUFFER_SAMPLE = 128
LOG_EVERY = 1000
MLP_SIZE = 256
TAU = 0.01
GAMMA = 0.99
EVAL_EVERY = 1000   # frames
EVAL_EPISODES = 50
DEVICE = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
UPDATE_ACTOR_EVERY = 2
# Seed the Python and RL environments to replicate similar results across training sessions. 


def save_series(fname, raw, smooth, window):
    sm_padded = np.concatenate([np.full(window-1, np.nan), smooth])
    data = np.column_stack([np.arange(len(raw)), raw, sm_padded])
    np.savetxt(
        fname, data, delimiter=",",
        header="step,raw,smoothed", comments=""
    )
    
    
    
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
eval_env.set_seed(4)
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


# 4.  loss module
# --- 4) Warm-up forward to initialize lazy modules BEFORE loss/opt
with torch.no_grad():
    td0 = env.reset()          # has "observation"
    _ = policy(td0.clone())    # init actor
    td1 = td0.clone()
    td1["action"] = env.action_spec.rand(td1.batch_size)
    _ = critic(td1)            # init critic
    # _ = actor_target(td0.clone())     # init target actor
    # _ = critic_target(td1.clone())    # init target critic

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
    num_qvalue_nets=2 ,
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
# optim_actor = optim.Adam(policy.parameters(), lr=3e-4, weight_decay=0.0)
# optim_critic = optim.Adam(critic.parameters(), lr=3e-3, weight_decay=0)
optim_actor = Adam(loss_td3.actor_network_params.parameters(),  lr=3e-4)
optim_critic = Adam(loss_td3.qvalue_network_params.parameters(), lr=3e-4)


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
    update_step = 0
    rewards, rewards_eval = [], []
    qvalue1, qvalue2 = [], []
    episode_returns = []
    ep_return = 0.0
    mc_bias_means, mc_bias_steps = [], []
    frames_since_eval,total_frames_seen = 0, 0
    

    optim_actor = Adam(loss.actor_network_params.values(True, True),  lr=2e-4)
    
    if method == "TD3":
        optim_critic = Adam(loss.qvalue_network_params.values(True, True), lr=2e-4)
    else:
        optim_critic = Adam(loss.value_network_params.values(True, True), lr=2e-4)
        
    pbar = tqdm(total=TOTAL_FRAMES, desc="Training DDPG", dynamic_ncols=True) if method=="DDPG" else tqdm(total=TOTAL_FRAMES, desc="Training TD3", dynamic_ncols=True)
    for i, data in enumerate(collector): # runs through the data collected from the agent’s interactions with the environment

        replay_buffer.extend(data) # add data to the replay buffer
        # episodic return 
        rs = data["next", "reward"].cpu().numpy().reshape(-1)
        ds = data["next", "done"].cpu().numpy().reshape(-1)
        for r, d in zip(rs, ds):
            ep_return += float(r)
            if d:                      # episode ended
                episode_returns.append(ep_return)
                ep_return = 0.0


        max_length = replay_buffer[:]["next", "step_count"].max()
        
        
 
        if len(replay_buffer) <= INIT_RAND_STEPS: 
            pbar.update(data.numel())
            continue
        for _ in range(OPTIM_STEPS):
            td = replay_buffer.sample(REPLAY_BUFFER_SAMPLE)
            update_step += 1
            
            loss_out = loss(td)
            loss_q = loss_out["loss_qvalue"] if method == "TD3" else loss_out["loss_value"]
            
            # Critic update
            optim_critic.zero_grad(set_to_none=True)
            loss_q.backward()
            # torch.nn.utils.clip_grad_norm_(loss.value_network_params.values(True, True), 1.0) if method == "DDPG" else torch.nn.utils.clip_grad_norm_(loss.qvalue_network_params.values(True, True), 1.0)
            optim_critic.step()

            # Recompute loss_out so actor loss uses the updated critic params
            loss_out = loss(td)
            

            # Actor update
            if method == "DDPG":
                for p in critic.parameters(): p.requires_grad = False
                optim_actor.zero_grad(set_to_none=True)
                loss_pi = loss_out["loss_actor"]
                loss_pi.backward()
                optim_actor.step()
                updater.step()
                for p in critic.parameters(): p.requires_grad = True
            
            if method == "TD3":
                if update_step % UPDATE_ACTOR_EVERY == 0:
                    for p in critic.parameters(): p.requires_grad = False
                    optim_actor.zero_grad(set_to_none=True)
                    loss_pi = loss_out["loss_actor"]
                    loss_pi.backward()
                    # torch.nn.utils.clip_grad_norm_(loss.actor_network_params.values(True, True), 1.0)
                    optim_actor.step()
                    for p in critic.parameters(): p.requires_grad = True

                    # === Soft update targets only when actor is updated ===
                    updater.step()
            
        
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
                bias_batch = min(bias_q1, bias_q2)
            
            biases.append(bias_batch)

            batch_frames = data.numel()                
            total_frames_seen += batch_frames
            frames_since_eval += batch_frames
            total_episodes += data["next", "done"].sum()
            
            
            qvalues.append(loss_out["pred_value"].mean().item()) 
            

        rewards.append((i,td["next", "reward"].mean().item(),))
    
        
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

        # if total_count % LOG_EVERY == 0:
        #         torchrl_logger.info(f"Successful steps in the last episode: {max_length}, Q: {torch.tensor(qvalues[-50:]).mean().item():.3f}, rb length {len(replay_buffer)}, Number of episodes: {total_episodes}")
        
        
        # if (total_count // eval_every) != ((total_count - data.numel()) // eval_every):
        N = eval_env.batch_size #????\
        returns, successes = [], 0
        eval_max_steps = eval_env._max_episode_steps
        print(frames_since_eval, total_count , eval_every, data.numel() )
        while frames_since_eval >= EVAL_EVERY:
            frames_since_eval -= EVAL_EVERY
            actor = loss.actor_network 
            actor.eval()
            critic.eval()
            for i in range(EVAL_EPISODES): 
                td = eval_env.reset() 
                traj_q, traj_r, biases_all = [], [], []
                G, gpow = 0.0, 1.0 
                for _ in range(eval_max_steps): 
                    actor = loss.actor_network
                    s = td.select("observation")
                    a = actor(s)["action"]
                    td_in = td.empty_like().set("observation", obs)
                    q = critic(td_in.clone().set("action", a))["state_action_value"].item()
                    traj_q.append(q)
                    td = eval_env.step(td.clone().set("action", a))
                    traj_r.append(float(td["next","reward"]))
                    G += gpow * td["next","reward"].item()
                    gpow *= GAMMA
                    done = bool(td.get("done", False))
                    if "terminated" in td.keys(True) or "truncated" in td.keys(True):
                        done = done or bool(td.get("terminated", False)) or bool(td.get("truncated", False))
                    if done:
                        if "success" in td.keys(True) and bool(td.get("success")):
                            successes += 1
                        break
                
                # print(f"Eval episode nb {i} return: {G}")
                returns.append(G)
                G_t = []
                acc = 0.0
                for r in reversed(traj_r):
                    acc = r + GAMMA * acc
                    G_t.append(acc)
                G_t.reverse()
                biases_all.extend([q - g for q, g in zip(traj_q, G_t)])
            # print(G)
            plot_mc_estimate(returns, biases_all,  title="MC estimate of J(μ) with 95% CI")
                

            

        


    pbar.close()
    t1 = time.time()    
    print(f"Training took {t1-t0:.2f}s")
    torchrl_logger.info(
        f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
    )
    # window = 200  # adjust for smoothing strength
    # smooth_bias = np.convolve(biases, np.ones(window)/window, mode='valid')
    # smooth_qvalue = np.convolve(qvalues, np.ones(window)/window, mode='valid')
    
    # save_series("biases_newtd3.csv", biases, smooth_bias, window)
    # save_series("qvalues_newtd3.csv", qvalues, smooth_qvalue, window)
    
    # smooth_returns = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
    # plt.figure(figsize=(12,4))
    # plt.plot(episode_returns, label="Raw Bias", color='tab:blue', alpha=0.5)  # transparent fluctuating curve
    # plt.plot(np.arange(window-1, len(episode_returns)), smooth_returns, label="Smoothed Return", color='tab:blue', linewidth=2)
    # plt.xtitle(f"{method} – episodic return")
    # plt.title(f"{method} – episodic return")
    # plt.xlabel("Episode")
    # plt.ylabel("Return")
    # plt.tight_layout()
    # plt.show()
    
    
    # plt.figure(figsize=(12,5))
    # plt.xtitle(f"Training {method} - Bias")
    # plt.plot(biases, label="Raw Bias", color='tab:blue', alpha=0.5)  # transparent fluctuating curve
    # plt.plot(np.arange(window-1, len(biases)), smooth_bias, label="Smoothed Bias", color='tab:blue', linewidth=2)
    # plt.title(f"Training {method} - TD Bias")
    # plt.title(f"Training {method} - Bias")
    # plt.xlabel("Training Steps")
    # plt.show()

    # plt.figure(figsize=(12,5))
    # plt.xtitle(f"Training {method} - Q Values")
    # plt.plot(qvalues, label="Raw q_values", color='tab:blue', alpha=0.5)  # transparent fluctuating curve
    # plt.plot(np.arange(window-1, len(qvalues)), smooth_qvalue, label="Smoothed q_values", color='tab:blue', linewidth=2)
    # plt.title(f"Training {method} - smoothed Q Values")
    # plt.xlabel("Training Steps")
    # plt.show()





train(
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

