import time
import matplotlib.pyplot as plt
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from tensordict.nn import TensorDictModule as TensorDict, TensorDictSequential as Seq
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, DDPGLoss,TD3Loss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger

# Environment
env = TransformedEnv(
    GymEnv("LunarLanderContinuous-v3"),
    StepCounter(max_steps=1000))
env.set_seed(0)

obs_dim = env.observation_spec.shape[-1]
n_actions = env.action_spec.shape[-1]
max_action = env.action_spec.space.high[0]
min_action = env.action_spec.space.low[0]


# Critic
MLP_SIZE = 64
critic_mlp = MLP(out_features=1, num_cells=[MLP_SIZE, MLP_SIZE])
critic_net = TensorDict(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) 
qvalue = QValueModule(critic_net)

#  Actor
actor_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[MLP_SIZE, MLP_SIZE])
actor_net = TensorDict(actor_mlp, in_keys=["observation"], out_keys=["action_value"])

EPS_0 = 0.2
exploration_module = EGreedyModule(
    action_spec=env.action_spec,
    eps_init=EPS_0)
policy = Seq(actor_net, exploration_module)

# Collect the data from the agent’s interactions with the environment
collector = SyncDataCollector(
    env,
    policy,
    total_frames=10000,
    frames_per_batch=1000,
    max_frames_per_traj=1000,
    device="cpu",
)

# Replay buffer
BUFFER_LEN = 100000
rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_LEN))


# DDPG Loss
loss = TD3Loss(value_network=policy, action_space=env.action_spec, delay_value=True)




# optimizer
ALPHA = 1e-3
optim = Adam(loss.parameters(), lr=ALPHA)

# Target network update
TAU = 0.01
updater = SoftUpdate(loss, tau=TAU) # for updating target networks

total_count = 0
total_episodes = 0
t0 = time.time()
success_steps = []


for i, data in enumerate(collector):
    rb.extend(data)
    batch = rb.sample(128)
    loss_dict = loss(batch)
    optim.zero_grad()
    loss_dict["loss"].backward()
    optim.step()
    updater.step()
    total_count += data.numel()
    total_episodes += data["done"].sum().item()
    if (i + 1) % 10 == 0:
        fps = total_count / (time.time() - t0)
        print(
            f"frames: {total_count}, episodes: {total_episodes}, fps: {fps:.2f}, "
            f"loss: {loss_dict['loss'].item():.3f}, "
            f"qvalue: {loss_dict['qvalue'].mean().item():.3f}, "
            f"policy_loss: {loss_dict['policy_loss'].item():.3f}"
        )
        t0 = time.time()
        total_count = 0
        total_episodes = 0
        success_steps.append(data["step_count"][data["done"]].cpu().numpy())
        plt.figure(figsize=(10,5))
        plt.title("Steps per episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.plot([step for sublist in success_steps for step in sublist])
        plt.show()


