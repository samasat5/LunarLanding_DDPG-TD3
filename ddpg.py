import pdb
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import nn, optim
from torchrl.envs import GymEnv, TransformedEnv, Compose, DoubleToFloat, InitTracker, ObservationNorm, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage, RandomSampler
from torchrl.objectives import DDPGLoss, SoftUpdate
from torchrl.modules import OrnsteinUhlenbeckProcessModule as OUNoise, MLP
from tensordict.nn import TensorDictModule as TDM, TensorDictSequential as Seq
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import check_env_specs


"""
Comparing:
DDPG (single critic) with 
TD3 (twin critics + clipped double-Q + target smoothing + delayed actor updates) 
on LunarLanderContinuous-v3 environment.
"""


# parameters and hyperparameters
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
actor_mlp = MLP(
    out_features=act_dim, 
    num_cells=[MLP_SIZE, MLP_SIZE], 
    activation_class=nn.ReLU,     # hidden activations
    activate_last_layer=False 
)
actor = TDM(actor_mlp, in_keys=["observation"], out_keys=["action_raw"])
# Wrap Tanh so it applies to the "action" tensor 
tanh_on_action = TDM(nn.Tanh(), in_keys=["action_raw"], out_keys=["action"])
exploration_module = OUNoise(
    spec=env.action_spec,
    theta=0.15,
    sigma=0.2,
    dt=1e-2,
)
# mettre en place gaussian noise
# rollout_policy = Seq(policy, exploration_module) # à vérifier de ne pas dépasser les bornes de l'espace des actions?????
policy = Seq(actor, tanh_on_action, selected_out_keys=["action"])   # deterministic policy
rollout_policy = Seq(actor, tanh_on_action, exploration_module)      # stochastic policy with exploration noise

# 3. Critic (action value function)
critic_mlp = MLP(
    out_features=1, 
    num_cells=[MLP_SIZE, MLP_SIZE],
    activation_class=nn.ReLU,
    activate_last_layer=False
)
critic = TDM(critic_mlp, in_keys=["observation", "action"], out_keys=["state_action_value"]) # = QValue

# Target Networks
actor_target = deepcopy(policy) # no noise in target
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
loss.make_value_estimator(gamma=GAMMA)
updater = SoftUpdate(loss, tau=TAU)

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
optim_actor = optim.Adam(policy.parameters(), lr=1e-4)
optim_critic = optim.Adam(critic.parameters(), lr=1e-3)

# 8. Training loop
total_count = 0
total_episodes = 0
t0 = time.time()
success_steps, qvalues = [], []



pbar = tqdm(total=TOTAL_FRAMES, desc="Training DDPG", dynamic_ncols=True)
for i, data in enumerate(collector): # runs through the data collected from the agent’s interactions with the environment
    replay_buffer.extend(data) # add data to the replay buffer
    max_length = replay_buffer[:]["next", "step_count"].max()
    # pdb.set_trace()
    if len(replay_buffer) <= INIT_RAND_STEPS: 
        pbar.update(data.numel())
        continue
    for _ in range(OPTIM_STEPS):
        # if len(replay_buffer) < REPLAY_BUFFER_SAMPLE:
        #     break
        td = replay_buffer.sample(REPLAY_BUFFER_SAMPLE)

        # Critic update
        optim_critic.zero_grad(set_to_none=True)
        loss_q = loss(td)["loss_value"]
        loss_q.backward()
        optim_critic.step()
        updater.step()

        # Actor update (freeze critic params or detach inside loss)
        for p in critic.parameters(): p.requires_grad = False
        optim_actor.zero_grad(set_to_none=True)
        loss_pi = loss(td)["loss_actor"]
        loss_pi.backward()
        optim_actor.step()
        updater.step()
        for p in critic.parameters(): p.requires_grad = True

        # ou_noise.step(data.numel()) # make the noise decay over time

        total_count += data.numel()
        total_episodes += data["next", "done"].sum()
        
        qvalues.append(loss(td)["loss_value"].item())  #TODO
        # qvalues.append(loss(td)["pred_value"].mean().item())

    success_steps.append(max_length)
    total_count += data.numel()
    total_episodes += data["next", "done"].sum().item()
    pbar.set_postfix({
        "Steps": total_count,
        "Episodes": total_episodes,
        "Mean Q": f"{torch.tensor(qvalues[-50:]).mean().item():.2f}"
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
            torchrl_logger.info(f"Evaluation over {EVAL_EPISODES} episodes: {mean_reward:.2f}")
        policy.train()

pbar.close()
t1 = time.time()
print(f"Training took {t1-t0:.2f}s")
torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)

plt.figure(figsize=(10,5))
plt.title("QValues per episode")
plt.xlabel("Steps")
plt.ylabel("QValues")
plt.plot(qvalues)
plt.show()
