import pdb
import time
from copy import deepcopy
import torch
from torch import nn, optim
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, InitTracker, ObservationNorm, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage, RandomSampler
from torchrl.objectives import DDPGLoss, SoftUpdate
from torchrl.modules import OrnsteinUhlenbeckProcessModule as OUNoise, MLP
from tensordict.nn import TensorDictModule as TDM, TensorDictSequential as Seq

# parameters and hyperparameters
INIT_RAND_STEPS = 5000 
FRAMES_PER_BATCH = 100
OPTIM_STEPS = 10
EPS_0 = 0.5
BUFFER_LEN = 1_000_000
TARGET_UPDATE_EPS = 0.95
REPLAY_BUFFER_SAMPLE = 128
LOG_EVERY = 1000
MLP_SIZE = 256
DISCOUNT_FACTOR = 0.99
TAU = 0.005
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Seed the Python and RL environments to replicate similar results across training sessions. 
#torch.manual_seed(0)

# 1. Environment
env = TransformedEnv(
    GymEnv("LunarLanderContinuous-v3"),
    Compose(
        DoubleToFloat(),
        InitTracker(),
        ObservationNorm(in_keys=["observation"]),
        StepCounter()
        #RewardNorm(),
    )
)
# env.set_seed(0)

obs_norm = env.transform[2] # assuming ObservationNorm is at index 2 in the Compose
obs_norm.init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
obs_dim = env.observation_spec["observation"].shape[-1] # observation_spec : the observation space
act_dim = env.action_spec.shape[-1] #action_spec : the action space

# 2. Actor (policy)
actor_mlp = MLP(
    out_features=act_dim, 
    num_cells=[MLP_SIZE, MLP_SIZE], 
    activation_class=nn.ReLU,     # hidden activations
    activate_last_layer=False 
)
actor = TDM(actor_mlp, in_keys=["observation"], out_keys=["action_raw"])
# Wrap Tanh so it applies to the "action" tensor 
tanh_on_action = TDM(nn.Tanh(), in_keys=["action_raw"], out_keys=["action"])
policy = Seq(actor, tanh_on_action)   # final tanh ensures [-1, 1] range
ou_noise = OUNoise(
    spec=env.action_spec,
    theta=0.15,
    sigma=0.2,
    dt=1e-2,
)
rollout_policy = Seq(policy, ou_noise)

# 3. Critic (action value function)
critic_mlp = MLP(
    out_features=1, 
    num_cells=[MLP_SIZE, MLP_SIZE],
    activation_class=nn.ReLU,
    activate_last_layer=False
)
critic = TDM(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) # = QValue

# Target Networks
actor_target = deepcopy(actor) # no noise in target
critic_target = deepcopy(critic)

# 4. DDPG loss module
# --- 4) Warm-up forward to initialize lazy modules BEFORE loss/opt
with torch.no_grad():
    td0 = env.reset()          # has "observation"
    _ = policy(td0.clone())    # init actor
    td1 = td0.clone()
    td1["action"] = env.action_spec.rand(td1.batch_size)
    _ = critic(td1)            # init critic

loss = DDPGLoss(
    actor_network=policy, # deterministic 
    value_network=critic,
    loss_function="l2",
    delay_actor=True,            
    delay_value=True, 
)

# 5. Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=BUFFER_LEN), 
    #sampler=SamplerWithoutReplacement(),
    sampler=RandomSampler(),
)

# 6. Collector
collector = SyncDataCollector(
    env,
    rollout_policy,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=50_000,
    device=DEVICE,
)

# 7. Optimizers
optim_actor = optim.Adam(policy.parameters(), lr=1e-4)
optim_critic = optim.Adam(critic.parameters(), lr=1e-3)

updater = SoftUpdate(loss, tau=TAU)

# 8. Training loop
total_count = 0
total_episodes = 0
t0 = time.time()
success_steps = []
for i, data in enumerate(collector):
    replay_buffer.extend(data)
    max_length = replay_buffer[:]["next", "step_count"].max()
    if len(replay_buffer) > INIT_RAND_STEPS:
        for _ in range(OPTIM_STEPS):
            if len(replay_buffer) < REPLAY_BUFFER_SAMPLE:
                break
            td = replay_buffer.sample(REPLAY_BUFFER_SAMPLE)

            # Critic update
            optim_critic.zero_grad(set_to_none=True)
            loss_vals = loss(td)
            loss_q = loss_vals["loss_value"]
            loss_q.backward()
            optim_critic.step()
            updater.step()

            # Actor update
            optim_actor.zero_grad(set_to_none=True)
            loss_vals = loss(td)
            loss_pi = loss_vals["loss_actor"]
            loss_pi.backward()
            optim_actor.step()
            updater.step()

            # Update target params
            updater.step()
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    success_steps.append(max_length)

print(f"Training took {time.time()-t0:.2f}s")
env.close()
