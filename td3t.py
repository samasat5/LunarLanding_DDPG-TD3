import time
import pdb
import torchrl
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TDM, TensorDictSequential as Seq
from torchrl.modules import OrnsteinUhlenbeckProcessModule as OUNoise, MLP, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate, DDPGLoss,TD3Loss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger
import torch
from torch import nn, optim
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, InitTracker, ObservationNorm, StepCounter
from torchrl.envs.utils import check_env_specs


# configurations
MLP_SIZE = 256

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
# # qvalue_1 = QValueModule(critic_net_1, spec=env.action_spec)  
# observation_example = torch.randn( obs_dim) 
# action_example = torch.randn( act_dim)  
# input_data = TensorDict({'observation': observation_example, 'action': action_example})


critic_mlp_2 = MLP(out_features=1, num_cells=[MLP_SIZE, MLP_SIZE])
critic_net_2 = TDM(critic_mlp_2, in_keys=["observation", "action"], out_keys=["state_action_value2"])      
# # qvalue_2 = QValueModule(critic_net_2, spec=env.action_spec)  
# qvalue_2 = critic_net_2


#  Actor
actor_mlp = MLP(
    out_features=act_dim, 
    num_cells=[MLP_SIZE, MLP_SIZE], )

actor_net = TDM(actor_mlp, in_keys=["observation"], out_keys=["action"])



EPS_0 = 0.2
exploration_module = OUNoise(
    spec=env.action_spec,
    theta=0.15,
    sigma=0.2,
    dt=1e-2,
)
policy = Seq(actor_net, exploration_module)



# Collect the data from the agent’s interactions with the environment
collector = SyncDataCollector(
    env,
    policy,
    total_frames=10000,
    frames_per_batch=1000,
    device="cpu",
)

# Replay buffer
BUFFER_LEN = 100000
rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_LEN))



# # test
# # Test forward with dummy batch
# dummy_obs = torch.randn(10, obs_dim)  # Batch of 10
# dummy_action = torch.randn(10, act_dim)

# td = TensorDict({
#     "observation": dummy_obs,
#     "action": dummy_action
# }, batch_size=[10])

# pdb.set_trace() 
# # Forward pass
# print("Actor output:", actor_net(td)["action"].shape)
# print("Critic 1 output:", critic_net_1(td)["state_action_value1"].shape)
# print("Critic 2 output:", critic_net_2(td)["state_action_value2"].shape)


print(torchrl.__version__)
# from torchrl.modules import TensorDictSequential as TDS, CatTensors

# critic_net_1 = TDS(
#     CatTensors(in_keys=["observation", "action"], out_key="obs_act"),
#     TDM(critic_mlp_1, in_keys=["obs_act"], out_keys=["state_action_value1"])
# )

# critic_net_2 = TDS(
#     CatTensors(in_keys=["observation", "action"], out_key="obs_act"),
#     TDM(critic_mlp_2, in_keys=["obs_act"], out_keys=["state_action_value2"])
# )



# # TD3 Loss
# loss = TD3Loss(
#     qvalue_network=(critic_net_1, critic_net_2),  
#     actor_network=actor_net,           
# )


# # optimizer
# ALPHA = 1e-3
# optim = Adam(loss.parameters(), lr=ALPHA)

# # Target network update
# TAU = 0.01
# updater = SoftUpdate(loss, tau=TAU) # for updating target networks

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

