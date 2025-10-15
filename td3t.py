import time
from copy import deepcopy
import pdb
import torchrl
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TDM, TensorDictSequential as Seq
from torchrl.modules import OrnsteinUhlenbeckProcessModule as OUNoise, MLP, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate, DDPGLoss,TD3Loss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger
import torch
from torch import nn, optim
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, InitTracker, ObservationNorm, StepCounter
from torchrl.envs.utils import check_env_specs


# configurations
INIT_RAND_STEPS = 5000 
TOTAL_FRAMES = 50_000
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

env.transform[2].init_stats(1024) 
torch.manual_seed(0)
env.set_seed(0)
check_env_specs(env) 

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
rollout_policy = Seq(actor_net, exploration_module)

# Targets
actor_target = deepcopy(actor_net) # no noise in target
critic_1_target = deepcopy(critic_net_1)
critic_2_target = deepcopy(critic_net_2)


# TD3 Loss
# 4. DDPG loss module
# --- 4) Warm-up forward to initialize lazy modules BEFORE loss/opt
with torch.no_grad():
    td0 = env.reset()          # has "observation"
    _ = policy(td0.clone())    # init actor
    td1 = td0.clone()
    td1["action"] = env.action_spec.rand(td1.batch_size)
    _ = critic_net_1(td1)            # init critic
    _ = critic_net_2(td1)            # init critic

loss = TD3Loss(
    actor_network=actor_net,    
    qvalue_network=[critic_net_1, critic_net_2],   
    action_spec=env.action_spec,       
)
loss.make_value_estimator(gamma=GAMMA)
updater = SoftUpdate(loss, tau=TAU) # for updating target networks

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
rb = ReplayBuffer(
    storage=LazyTensorStorage(BUFFER_LEN),
    sampler=RandomSampler(),
)

optim_actor = optim.Adam(policy.parameters(), lr=1e-4)
optim_critic_1 = optim.Adam(critic_net_1.parameters(), lr=1e-3)
optim_critic_2 = optim.Adam(critic_net_2.parameters(), lr=1e-3)


# total_count = 0
# total_episodes = 0
# t0 = time.time()
# success_steps = []


# OPTIM_STEPS = 10
# INIT_RAND_STEPS = 5000 # number of random steps at the beginning of training
# LOG_EVERY = 1000

# for i, data in enumerate(collector):
#     rb.extend(data)
#     if len(rb) > INIT_RAND_STEPS:
#         for _ in range(OPTIM_STEPS):
#             batch = rb.sample(128)
#             loss_dict = loss(batch)
#             optim.zero_grad()
#             loss_dict["loss"].backward()
#             optim.step()
#             # Update exploration factor
#             exploration_module.step(data.numel())
#             updater.step()
            
            
#             total_count += data.numel()
#             total_episodes += data["done"].sum().item()
#             if (i + 1) % 10 == 0:
#                 fps = total_count / (time.time() - t0)
#                 print(
#                     f"frames: {total_count}, episodes: {total_episodes}, fps: {fps:.2f}, "
#                     f"loss: {loss_dict['loss'].item():.3f}, "
#                     f"qvalue: {loss_dict['qvalue'].mean().item():.3f}, "
#                     f"policy_loss: {loss_dict['policy_loss'].item():.3f}"
#                 )
#                 t0 = time.time()
#                 total_count = 0
#                 total_episodes = 0
#                 success_steps.append(data["step_count"][data["done"]].cpu().numpy())

#                 # plot the q values over the episodes
#                 plt.plot([loss_dict['qvalue'].mean().item()] * len(success_steps), color='r')
#         success_steps.append(max_length)

#     if total_count > 0 and total_count % LOG_EVERY == 0:
#         torchrl_logger.info(f"Successful steps in the last episode: {max_length}, rb length {len(rb)}, Number of episodes: {total_episodes}")

#     if max_length > 475:
#         print("TRAINING COMPLETE")
#         break

# plt.figure(figsize=(10,5))
# plt.title("QValues per episode")
# plt.xlabel("QValues")
# plt.ylabel("Steps")
# plt.show()

