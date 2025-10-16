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



@torch.no_grad()
def soft_update(source, target, tau):
    src_dict = source.state_dict()
    tgt_dict = target.state_dict()
    for key in src_dict:
        tgt_dict[key] = tau * src_dict[key] + (1 - tau) * tgt_dict[key]
    target.load_state_dict(tgt_dict)
    
    
# configurations
INIT_RAND_STEPS = 5000 
TOTAL_FRAMES = 100_000
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
BUFFER_LEN = 1_000_000
REPLAY_BUFFER_SAMPLE = 128
LOG_EVERY = 1000
MLP_SIZE = 256
TAU = 0.005
GAMMA = 0.99
EVAL_EVERY = 10_000   # frames
EVAL_EPISODES = 3
DEVICE = "cpu"
EPS_0 = 0.2
BUFFER_LEN = 100000

# Environment
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


# Critic
critic_mlp_1 = MLP(
    out_features=1, 
    num_cells=[MLP_SIZE, MLP_SIZE],
    activation_class=nn.ReLU,
    activate_last_layer=False
)
critic_net_1 = TDM(critic_mlp_1, in_keys=["observation", "action"], out_keys=["state_action_value1"]) 
critic_mlp_2 = deepcopy(critic_mlp_1)
critic_net_2 = TDM(critic_mlp_2, in_keys=["observation", "action"], out_keys=["state_action_value2"])      


#  Actor
actor_mlp = MLP(
    out_features=act_dim, 
    num_cells=[MLP_SIZE, MLP_SIZE], 
    activation_class=nn.ReLU,
    activate_last_layer=False
)
actor_net = TDM(actor_mlp, in_keys=["observation"], out_keys=["action_raw"])
tanh_on_action = TDM(nn.Tanh(), in_keys=["action_raw"], out_keys=["action"])
policy = Seq(actor_net, tanh_on_action, selected_out_keys=["action"])   # final tanh ensures [-1, 1] range
exploration_module = OUNoise(
    spec=env.action_spec,
    theta=0.15,
    sigma=0.2,
    dt=1e-2,
)
rollout_policy = Seq(actor_net, tanh_on_action, exploration_module)


# TD3 Loss
# --- 4) Warm-up 
# initiate the TensorDict objects by passing some data through the modules
with torch.no_grad():
    td0 = env.reset()          # has "observation"
    _ = policy(td0.clone())    # init actor
    td1 = td0.clone()
    td1["action"] = env.action_spec.rand(td1.batch_size)
    _ = critic_net_1(td1)            # init critic
    _ = critic_net_2(td1)            # init critic
 # now we initiated td0 and td1 tensordicts
 # TensorDict with keys [\'action\', \'done\', \'is_init\', \'observation\',
 # \'state_action_value1\', \'state_action_value2\', \'step_count\', \'terminated\',
 # \'truncated\']'


# Targets
actor_target = deepcopy(actor_net) # no noise in target
critic_1_target = deepcopy(critic_net_1)
critic_2_target = deepcopy(critic_net_2)




qvalue_ensemble = QValueEnsembleModule(critic_net_1, critic_net_2)
loss = TD3Loss(
    actor_network=Seq(actor_net, tanh_on_action), 
    qvalue_network=qvalue_ensemble,
    action_spec=env.action_spec,
    loss_function="l2",
    delay_actor=True,
    delay_qvalue=True,
)



        
loss.make_value_estimator(gamma=GAMMA)

# Collect the data from the agent’s interactions with the environment
collector = SyncDataCollector(
    env,
    rollout_policy,
    total_frames=TOTAL_FRAMES,
    frames_per_batch=FRAMES_PER_BATCH,
    init_random_frames=INIT_RAND_STEPS,
    device=DEVICE,
)

# Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(BUFFER_LEN),
    sampler=RandomSampler(),
)

optim_actor = optim.Adam(policy.parameters(), lr=1e-4)
optim_critic_1 = optim.Adam(critic_net_1.parameters(), lr=1e-3)
optim_critic_2 = optim.Adam(critic_net_2.parameters(), lr=1e-3)


total_count = 0
total_episodes = 0
t0 = time.time()
success_steps = []
qvalues = []
biases = []

num_batches = TOTAL_FRAMES // FRAMES_PER_BATCH
pbar = tqdm(total=num_batches, desc="Training TD3", dynamic_ncols=True)

for i, data in enumerate(collector):  # Data from env rollouts
    replay_buffer.extend(data)
    max_length = replay_buffer[:]["next", "step_count"].max()

    if len(replay_buffer) <= INIT_RAND_STEPS:
        pbar.update(1)
        continue

    for _ in range(OPTIM_STEPS):
        td = replay_buffer.sample(REPLAY_BUFFER_SAMPLE)

        # --- Critic 1 update
        loss_out = loss(td)
        optim_critic_1.zero_grad(set_to_none=True)
        loss_q1 = loss(td)["loss_qvalue"]
        loss_q1.backward()
        optim_critic_1.step()

        # --- Critic 2 update
        optim_critic_2.zero_grad(set_to_none=True)
        loss_q2 = loss(td)["loss_qvalue"]
        loss_q2.backward()
        optim_critic_2.step()

        # --- Actor update (delayed)
        for p in critic_net_1.parameters():
            p.requires_grad = False
        for p in critic_net_2.parameters():
            p.requires_grad = False

        optim_actor.zero_grad(set_to_none=True)
        loss_pi = loss(td)["loss_actor"]
        loss_pi.backward()
        optim_actor.step()

        for p in critic_net_1.parameters():
            p.requires_grad = True
        for p in critic_net_2.parameters():
            p.requires_grad = True

        # --- Noise annealing
        exploration_module.step(data.numel())

        # --- Soft update targets
        with torch.no_grad():
            soft_update(actor_net, actor_target, TAU)
            soft_update(critic_net_1, critic_1_target, TAU)
            soft_update(critic_net_2, critic_2_target, TAU)



        # Record TD bias
        pred_q1, pred_q2 = loss_out["pred_value"] 
        target_q = loss_out["target_value"]
        bias_q1 = (pred_q1 - target_q).mean().item()
        bias_q2 = (pred_q2 - target_q).mean().item()
        bias_avg = (bias_q1 + bias_q2) / 2
        biases.append(bias_avg)
        
        
        # --- Stats
        total_count += data.numel()
        total_episodes += data["next", "done"].sum()
        qvalues.append((loss_q1.item() + loss_q2.item()) / 2) #TODO
        pdb.set_trace()

    success_steps.append(max_length)

    # --- Logging
    if total_count % LOG_EVERY == 0:
        torchrl_logger.info(
            f"Steps: {total_count}, Episodes: {total_episodes}, Max Ep Len: {max_length}, "
            f"ReplayBuffer: {len(replay_buffer)}, MeanQ: {np.mean(qvalues[-50:]):.3f}"
        )

    # --- Eval
    if total_count % EVAL_EVERY < FRAMES_PER_BATCH:
        policy.eval()
        with torch.no_grad():
            rewards = []
            lens = []
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
                lens.append(int(td.get("step_count", 0)))
                rewards.append(episode_reward)

            mean_reward = np.mean(rewards)
            torchrl_logger.info(f"[Eval] Mean reward: {mean_reward:.2f}")
        policy.train()

    # --- Update tqdm bar
    pbar.set_postfix({
        "Steps": total_count,
        "Episodes": total_episodes,
        "Mean Q": f"{torch.tensor(qvalues[-50:]).mean().item():.2f}",
        "Bias": f"{torch.tensor(biases[-50:]).mean().item():.2f}",
    })
    pbar.update(1)

# --- End training
pbar.close()
t1 = time.time()

print(f"\nTraining completed in {t1 - t0:.2f}s")
torchrl_logger.info(f"Solved after {total_count} steps, {total_episodes} episodes, time {t1 - t0:.2f}s")

# --- Plot results
plt.figure(figsize=(10, 5))
plt.title("Average Q-values over training")
plt.xlabel("Steps")
plt.ylabel("Q-value")
plt.plot(qvalues)
plt.grid(True)
plt.show()