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
from utils2 import MultiCriticSoftUpdate, soft_update, plot_q_vs_mc, plot_bias_stats, plot_mc_estimate  
from tqdm import tqdm








# parameters and hyperparameters
INIT_RAND_STEPS = 5000 
TOTAL_FRAMES = 50_000
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
    def _ci95(x):
        if len(x) < 2:
            return float("nan")
        x = np.asarray(x, dtype=float)
        se = x.std(ddof=1) / np.sqrt(len(x))
        return 1.96 * se

    updater = SoftUpdate(loss, tau=TAU)
    loss.make_value_estimator(gamma=GAMMA)

    # --- bookkeeping
    total_count = 0
    total_episodes = 0
    t0 = time.time()
    success_steps, qvalues = [], []
    biases = []
    update_step = 0
    episode_returns = []
    ep_return = 0.0
    frames_since_eval, total_frames_seen = 0, 0

    # periodic eval history (returned to caller)
    eval_history = {
        "frames": [],
        "ret_mean": [],
        "ret_ci95": [],
        "bias_mean": [],
        "bias_ci95": [],
        "succ_rate": [],
    }

    # --- fresh optimizers bound to current loss params
    optim_actor = Adam(loss.actor_network_params.values(True, True), lr=2e-4)
    if method == "TD3":
        optim_critic = Adam(loss.qvalue_network_params.values(True, True), lr=2e-4)
    else:
        optim_critic = Adam(loss.value_network_params.values(True, True), lr=2e-4)

    desc = "Training TD3" if method == "TD3" else "Training DDPG"
    pbar = tqdm(total=total_frames, desc=desc, dynamic_ncols=True)

    for i, data in enumerate(collector):
        replay_buffer.extend(data)

        # episodic return tracking
        rs = data["next", "reward"].cpu().numpy().reshape(-1)
        ds = data["next", "done"].cpu().numpy().reshape(-1)
        for r, d in zip(rs, ds):
            ep_return += float(r)
            if d:
                episode_returns.append(ep_return)
                ep_return = 0.0

        max_length = replay_buffer[:]["next", "step_count"].max()

        # warm-up
        if len(replay_buffer) <= INIT_RAND_STEPS:
            batch_frames = data.numel()
            total_frames_seen += batch_frames
            frames_since_eval += batch_frames
            success_steps.append(max_length)
            total_count += batch_frames
            total_episodes += data["next", "done"].sum().item()
            pbar.update(batch_frames)
            continue

        # --- optimization steps
        for _ in range(opt_steps):
            td = replay_buffer.sample(batch_size)
            update_step += 1

            loss_out = loss(td)
            loss_q = loss_out["loss_qvalue"] if method == "TD3" else loss_out["loss_value"]

            # critic update
            optim_critic.zero_grad(set_to_none=True)
            loss_q.backward()
            optim_critic.step()

            # recompute for actor
            loss_out = loss(td)

            # actor update (+ soft update timing)
            if method == "DDPG":
                for p in critic.parameters(): p.requires_grad = False
                optim_actor.zero_grad(set_to_none=True)
                loss_pi = loss_out["loss_actor"]
                loss_pi.backward()
                optim_actor.step()
                updater.step()
                for p in critic.parameters(): p.requires_grad = True

            else:  # TD3
                if update_step % UPDATE_ACTOR_EVERY == 0:
                    for p in critic.parameters(): p.requires_grad = False
                    optim_actor.zero_grad(set_to_none=True)
                    loss_pi = loss_out["loss_actor"]
                    loss_pi.backward()
                    optim_actor.step()
                    for p in critic.parameters(): p.requires_grad = True
                    updater.step()  # only when actor updated

            # --- record TD bias
            if method == "DDPG":
                pred_q = loss_out["pred_value"]              # tensor
                target_q = loss_out["target_value"]          # tensor
                bias_batch = (pred_q - target_q).detach().mean().item()
            else:
                pred_q1, pred_q2 = loss_out["pred_value"]    # two tensors
                target_q = loss_out["target_value"]
                bias_q1 = (pred_q1 - target_q).mean().item()
                bias_q2 = (pred_q2 - target_q).mean().item()
                bias_batch = min(bias_q1, bias_q2)
            biases.append(bias_batch)

            # --- record qvalues (fix TD3: use min head)
            pv = loss_out["pred_value"]
            if isinstance(pv, (list, tuple)):
                qvalues.append(torch.minimum(pv[0], pv[1]).mean().item())
            else:
                qvalues.append(pv.mean().item())

        # progress counters
        batch_frames = data.numel()
        total_frames_seen += batch_frames
        frames_since_eval += batch_frames
        success_steps.append(max_length)
        total_count += batch_frames
        total_episodes += data["next", "done"].sum().item()

        # --- periodic evaluation ---
        if (eval_env is not None) and (frames_since_eval >= eval_every):
            # (optional) vary eval seed set across calls if you like:
            # eval_env.set_seed(1000 + len(eval_history["frames"]))
            rets, biases_eval, succ, _, _ = run_eval(
                method=method,
                loss=loss,
                eval_env=eval_env,
                eval_episodes=eval_episodes,
                gamma=GAMMA,
                eval_max_steps=getattr(eval_env, "_max_episode_steps", None),
            )
            ret_mean = float(np.mean(rets)) if len(rets) else float("nan")
            ret_ci = _ci95(rets) if len(rets) else float("nan")
            bias_mean = float(np.mean(biases_eval)) if len(biases_eval) else float("nan")
            bias_ci = _ci95(biases_eval) if len(biases_eval) else float("nan")
            succ_rate = succ / float(eval_episodes) if eval_episodes else 0.0

            eval_history["frames"].append(total_frames_seen)
            eval_history["ret_mean"].append(ret_mean)
            eval_history["ret_ci95"].append(ret_ci)
            eval_history["bias_mean"].append(bias_mean)
            eval_history["bias_ci95"].append(bias_ci)
            eval_history["succ_rate"].append(succ_rate)

            # show on the bar
            pbar.set_postfix({
                "Steps": total_count,
                "Episodes": int(total_episodes),
                "Mean Q": f"{torch.tensor(qvalues[-50:]).mean().item():.2f}" if qvalues else "n/a",
                "Bias": f"{torch.tensor(biases[-50:]).mean().item():.2f}" if biases else "n/a",
                "EvalR": f"{ret_mean:.1f}±{ret_ci:.1f}",
                "EvalB": f"{bias_mean:.2f}±{bias_ci:.2f}",
                "Succ": f"{100*succ_rate:.0f}%",
            })
            frames_since_eval = 0  # reset the eval counter

        pbar.update(batch_frames)

        # optional: break if we reached the requested total frames
        if total_frames_seen >= total_frames:
            break

    pbar.close()
    t1 = time.time()
    print(f"Training took {t1-t0:.2f}s")

    # return whatever you need; keeping eval history is the key new part
    return {
        "biases": biases,
        "qvalues": qvalues,
        "episode_returns": episode_returns,
        "eval_history": eval_history,
    }




@torch.no_grad()
def run_eval(method, loss, eval_env, eval_episodes, gamma, eval_max_steps):
    actor_eval  = loss.actor_network
    critic_eval = loss.qvalue_network if method == "TD3" else loss.value_network
    actor_eval.eval(); critic_eval.eval()

    returns, biases_all = [], []
    q_vals_all, g_t_all = [], []     # <--- add these
    successes = 0
    max_steps = eval_max_steps or getattr(eval_env, "_max_episode_steps", None) or 10_000

    for _ in range(eval_episodes):
        td = eval_env.reset()
        traj_q, traj_r = [], []
        G, gpow = 0.0, 1.0

        for t in range(max_steps):
            obs = td["observation"] if t == 0 else td["next", "observation"]
            s = TensorDict({"observation": obs}, batch_size=obs.shape[:-1])
            a = actor_eval(s)["action"]

            # Q(s,a)
            td_q = TensorDict({"observation": obs, "action": a}, batch_size=obs.shape[:-1])
            q_out = critic_eval(td_q)["state_action_value"]
            
            if isinstance(q_out, (list, tuple)):
                q = torch.minimum(q_out[0], q_out[1]).squeeze(-1).item()
            else:
                q = q_out.squeeze(-1).item()
            traj_q.append(q)

            # step
            td = eval_env.step(td.clone().set("action", a))
            r = float(td["next", "reward"])
            traj_r.append(r)
            G += gpow * r
            gpow *= gamma

            done = bool(td.get(("next","done"), False))
            if ("next","terminated") in td.keys(True): done |= bool(td.get(("next","terminated")))
            if ("next","truncated")  in td.keys(True): done |= bool(td.get(("next","truncated")))
            if done:
                if ("next","success") in td.keys(True) and bool(td.get(("next","success"))):
                    successes += 1
                break

        returns.append(G)

        # MC G_t and biases
        G_t, acc = [], 0.0
        for r in reversed(traj_r):
            acc = r + gamma * acc
            G_t.append(acc)
        G_t.reverse()

        biases_all.extend([q - g for q, g in zip(traj_q, G_t)])
        q_vals_all.extend(traj_q)     # <--- collect
        g_t_all.extend(G_t)           # <--- collect

    return returns, biases_all, successes, np.array(q_vals_all, dtype=float), np.array(g_t_all, dtype=float)




out_td3 = train(
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
    eval_every=EVAL_EVERY,   # <- periodic evaluation every 1000 frames
)

# After training, you still can run a final eval if you wish:
rets, biases, succ, q_vals, g_t  = run_eval(
    method="TD3",
    loss=loss_td3,
    eval_env=eval_env,
    eval_episodes=EVAL_EPISODES,
    gamma=GAMMA,
    eval_max_steps=getattr(eval_env, "_max_episode_steps", None),
)

# Plot learning curves from periodic evals
eh = out_td3["eval_history"]
plt.figure(figsize=(10,4))
plt.plot(eh["frames"], eh["ret_mean"], label="Return (mean)")
# error band
lo = np.array(eh["ret_mean"]) - np.array(eh["ret_ci95"])
hi = np.array(eh["ret_mean"]) + np.array(eh["ret_ci95"])
plt.fill_between(eh["frames"], lo, hi, alpha=0.2, label="95% CI")
plt.xlabel("Env frames"); plt.ylabel("Eval return"); plt.title("TD3 periodic evaluation"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(eh["frames"], eh["bias_mean"], label="MC bias mean (Q - G_t)")
lo = np.array(eh["bias_mean"]) - np.array(eh["bias_ci95"])
hi = np.array(eh["bias_mean"]) + np.array(eh["bias_ci95"])
plt.fill_between(eh["frames"], lo, hi, alpha=0.2, label="95% CI")
plt.axhline(0, ls="--", lw=1)
plt.xlabel("Env frames"); plt.ylabel("Bias"); plt.title("TD3 MC bias over training"); plt.legend(); plt.tight_layout(); plt.show()






