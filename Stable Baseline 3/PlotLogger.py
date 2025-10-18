import numpy as np, torch, os, csv
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn.functional as F


class PlotLoggerDDPG(BaseCallback):
    """
    
    """
    def __init__(self, gamma: float, sample_n: int = 50_000, reward_window: int = 200, save_csv: str = "./logs_ddpg/stats/stats_log.csv"):
        super().__init__()
        self.gamma = gamma
        self.sample_n = sample_n
        self.save_csv = save_csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timesteps",
                    "bias_mean", "bias_std",
                    "bias_abs_mean", "bias_abs_std",
                    "q_values_mean", "q_values_std",
                    "target_mean", "target_std",
                    "critic_loss",
                ])


    def _on_step(self) -> bool:
        rb = getattr(self.model, "replay_buffer", None)
        if rb is None or rb.size() == 0:
            return True
        
        n = min(self.sample_n, rb.size())
        batch = rb.sample(n)

        policy = self.model.policy
        device = policy.device
        # If it’s not already a tensor, convert it to a tensor and put it on the same device as the model.
        to = lambda x: x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=device)

        #print(batch)
        obs = to(batch.observations)
        act = to(batch.actions)
        nxt = to(batch.next_observations)
        rew = to(batch.rewards).squeeze(-1) #removes the last dimension if it’s size 1 (usually Nx1)
        done = to(batch.dones).squeeze(-1)

        timeouts = None
        if hasattr(batch, "timeouts") and batch.timeouts is not None: #Some environments end because the agent fails or succeeds (done=True),
            timeouts = to(batch.timeouts).squeeze(-1)                 #others just reach a time limit (max episode steps), which should not count as a terminal state for bootstrapping.

        with torch.no_grad():
            #print(policy.critic(obs, act)[0].shape)
            q = policy.critic(obs, act)[0].squeeze(-1)

            next_a = policy.actor_target(nxt)
            #print(policy.critic_target(nxt, next_a)[0].shape)
            next_q = policy.critic_target(nxt, next_a)[0].squeeze(-1)

            nonterminal = 1.0 - done # if timeout is None, done is enough
            if timeouts is not None:
                nonterminal = (1.0 - done) * (1.0 - timeouts)

            target = rew + self.gamma * nonterminal * next_q
            bias = (q - target).cpu().numpy() # should be in evaluation

        mean_bias = float(np.mean(bias))
        std_bias  = float(np.std(bias))

        mean_abs  = float(np.mean(np.abs(bias)))
        std_abs  = float(np.std(np.abs(bias)))

        q_values_mean = float(np.mean(q.cpu().numpy()))
        q_values_std = float(np.std(q.cpu().numpy()))

        target_mean = float(np.mean(target.cpu().numpy()))
        target_std = float(np.std(target.cpu().numpy())) 

        critic_loss = float(F.mse_loss(q, target).item())

        self.logger.record("stats/bias_mean", mean_bias), self.logger.record("stats/bias_std", std_bias)
        self.logger.record("stats/bias_abs_mean", mean_abs), self.logger.record("stats/bias_abs_std", std_abs)
        self.logger.record("stats/q_values_mean", q_values_mean), self.logger.record("stats/q_values_std", q_values_std)
        self.logger.record("stats/target_mean", target_mean), self.logger.record("stats/target_std", target_std)
        self.logger.record("stats/critic_loss", critic_loss)

        with open(self.save_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps, 
                mean_bias, std_bias, 
                mean_abs, std_abs, 
                q_values_mean, q_values_std,
                target_mean, target_std,
                critic_loss,
            ])

        return True
    
class PlotLoggerTD3(BaseCallback):
    """
    TD3 Q-bias/TD-error logger.
    Logs:
      - q_bias/q1_mean, q2_mean, min_mean
      - mean |bias| and std for each
      - overestimation_gap = mean(|q1 - q2|)
    Also appends a CSV for easy plotting.
    """
    def __init__(self, gamma: float, sample_n: int = 50_000, save_csv: str = "./logs_td3/stats/stats_log.csv"):
        super().__init__()
        self.gamma = gamma
        self.sample_n = sample_n
        self.save_csv = save_csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timesteps",
                    "q1_mean", "q1_std", "q1_mean_abs", "q1_std_abs",
                    "q2_mean", "q2_std", "q2_mean_abs", "q2_std_abs",
                    "min_mean", "min_std", "min_mean_abs", "min_std_abs",
                    "overestimation_gap_mean_abs", "overestimation_gap_std_abs",
                    "loss1", "loss2", "critic_loss",
                    "mean_q_1", "std_q_1",
                    "mean_q_2", "std_q_2"
                ])


    def _td3_target_actions(self, nxt: torch.Tensor) -> torch.Tensor:
        """
        TD3 target policy smoothing: add clipped Gaussian noise and clip to action bounds.
        Uses the algorithm's noise hyperparams & env action space.
        """
        next_a = self.model.policy.actor_target(nxt)

        # TD3 target smoothing noise
        noise_scale = float(getattr(self.model, "target_policy_noise", 0.2))
        noise_clip = float(getattr(self.model, "target_noise_clip", 0.5))
        noise = torch.normal(mean=0.0, std=noise_scale, size=next_a.shape, device=next_a.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_a = next_a + noise

        # Clip to env action bounds
        low  = torch.as_tensor(self.model.action_space.low,  device=next_a.device)
        high = torch.as_tensor(self.model.action_space.high, device=next_a.device)
        next_a = torch.max(torch.min(next_a, high), low)
        return next_a

    def _on_step(self) -> bool:
        rb = getattr(self.model, "replay_buffer", None)
        if rb is None or rb.size() == 0:
            return True

        n = min(self.sample_n, rb.size())
        batch = rb.sample(n)

        policy = self.model.policy
        device = policy.device
        to  = lambda x: x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=device)

        obs = to(batch.observations)
        act = to(batch.actions)
        nxt = to(batch.next_observations)
        rew = to(batch.rewards).squeeze(-1)
        done = to(batch.dones).squeeze(-1)

        timeouts = None
        if hasattr(batch, "timeouts") and batch.timeouts is not None:
            timeouts = to(batch.timeouts).squeeze(-1)

        with torch.no_grad():
            q1, q2 = policy.critic(obs, act)
            q1 = q1.squeeze(-1)
            q2 = q2.squeeze(-1)

            next_a = self._td3_target_actions(nxt)
            qt1, qt2 = policy.critic_target(nxt, next_a)
            qt1 = qt1.squeeze(-1)
            qt2 = qt2.squeeze(-1)
            next_q_min = torch.minimum(qt1, qt2)

            nonterminal = 1.0 - done
            if timeouts is not None:
                nonterminal = (1.0 - done) * (1.0 - timeouts)

            target = rew + self.gamma * nonterminal * next_q_min
            bias1 = (q1 - target).cpu().numpy()
            bias2 = (q2 - target).cpu().numpy()

            q_min_curr = torch.minimum(q1, q2)
            bias_min = (q_min_curr - target).cpu().numpy()

            # overestimation gap between critics 
            over_gap = torch.abs(q1 - q2).cpu().numpy()


        q1_mean, q1_std = float(np.mean(bias1)), float(np.std(bias1))
        q1_mean_abs, q1_std_abs = float(np.mean(np.abs(bias1))), float(np.std(np.abs(bias1)))

        q2_mean, q2_std = float(np.mean(bias2)), float(np.std(bias2))
        q2_mean_abs, q2_std_abs = float(np.mean(np.abs(bias2))), float(np.std(np.abs(bias2))) 

        mn_mean, mn_std= float(np.mean(bias_min)), float(np.std(bias_min))
        mn_abs_mean, mn_abs_std = float(np.mean(np.abs(bias_min))), float(np.std(np.abs(bias_min)))

        gap_abs_mean, gap_abs_std = float(np.mean(over_gap)), float(np.std(over_gap))

        loss1 = float(F.mse_loss(q1, target).item())
        loss2 = float(F.mse_loss(q2, target).item())
        critic_loss = 0.5 * (loss1 + loss2)

        mean_q_1, std_q_1 = float(torch.mean(q1).item()), float(torch.std(q1,unbiased=False).item())
        mean_q_2, std_q_2 = float(torch.mean(q2).item()), float(torch.std(q2,unbiased=False).item())

        # log to SB3 (TensorBoard/CSV if configured)
        self.logger.record("stats/q1_mean", q1_mean), self.logger.record("stats/q1_std", q1_std)
        self.logger.record("stats/q1_mean_abs", q1_mean_abs), self.logger.record("stats/q1_std_abs", q1_std_abs)
        
        self.logger.record("stats/q2_mean", q2_mean), self.logger.record("stats/q2_std", q2_std)
        self.logger.record("stats/q2_mean_abs", q2_mean_abs), self.logger.record("stats/q2_std_abs", q2_std_abs)

        self.logger.record("stats/min_mean", mn_mean), self.logger.record("stats/min_std", mn_std)
        self.logger.record("stats/min_mean_abs", mn_abs_mean), self.logger.record("stats/min_std_abs", mn_abs_std)
        
        self.logger.record("stats/overestimation_gap_abs_mean", gap_abs_mean), self.logger.record("stats/overestimation_gap_abs_std", gap_abs_std)
        self.logger.record("stats/loss1", loss1), self.logger.record("stats/loss2", loss2), self.logger.record("stats/critic_loss", critic_loss)
        self.logger.record("stats/mean_q_1", mean_q_1), self.logger.record("stats/std_q_1", std_q_1)
        self.logger.record("stats/mean_q_2", mean_q_2), self.logger.record("stats/std_q_2", std_q_2)

        # append to our CSV
        with open(self.save_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps,
                q1_mean, q1_std, q1_mean_abs, q1_std_abs,
                q2_mean, q2_std, q2_mean_abs, q2_std_abs,
                mn_mean, mn_std, mn_abs_mean, mn_abs_std,
                gap_abs_mean, gap_abs_std,
                loss1, loss2, critic_loss,
                mean_q_1, std_q_1,
                mean_q_2, std_q_2
            ])

        return True
    

class EvalMCBiasChild(BaseCallback):
    """
    Child of EvalCallback: runs immediately AFTER each evaluation.
    Computes Monte-Carlo bias on the eval episodes (not from replay buffer).
    """
    def __init__(self, gamma=0.99, bootstrap_time_limit=True,
                 save_csv="./logs_ddpg/stats/mc_eval_stats.csv", verbose=1):
        super().__init__(verbose=verbose)
        self.gamma = gamma
        self.bootstrap_time_limit = bootstrap_time_limit
        self.save_csv = save_csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timesteps",
                    "mc_bias_mean","mc_bias_std",
                    "mc_bias_abs_mean","mc_bias_abs_std",
                    "q_values_mean","q_values_std",
                    "mc_return_mean","mc_return_std",
                    "critic_mse_q_vs_mc",
                    "eval_mean_reward","eval_std_reward",
                    "eval_mean_ep_length","eval_std_ep_length",
                ])

    def _rollout_one_eval_episode(self, eval_env, deterministic=True):
        obs = eval_env.reset()
        done = False
        traj = dict(obs=[], act=[], rew=[], done=[], timeout=[], next_obs=[])
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            next_obs, reward, dones, infos = eval_env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            truncated = bool(
                ("TimeLimit.truncated" in info and info["TimeLimit.truncated"])
                or ("truncated" in info and info["truncated"])
            )
            traj["obs"].append(obs[0].copy())
            traj["act"].append(action[0].copy())
            traj["rew"].append(float(reward[0]))
            traj["done"].append(bool(dones[0]))
            traj["timeout"].append(truncated)
            traj["next_obs"].append(next_obs[0].copy())
            obs, done = next_obs, bool(dones[0])
        return traj

    def _on_step(self) -> bool:
        # 2) Roll out the SAME number of episodes as the parent
        n_episodes = getattr(self.parent, "n_eval_episodes", 5)
        deterministic = getattr(self.parent, "deterministic", True)

        ep_rewards, ep_lengths, trajs = [], [], []
        for _ in range(n_episodes):
            tr = self._rollout_one_eval_episode(self.parent.eval_env, deterministic=deterministic)
            ep_rewards.append(sum(tr["rew"]))
            ep_lengths.append(len(tr["rew"]))
            trajs.append(tr)

        # 3) Compute MC returns (with time-limit bootstrap)
        policy = self.model.policy
        dev = policy.device
        gamma = self.gamma

        Q_all, G_all = [], []
        for tr in trajs:
            obs = torch.as_tensor(np.asarray(tr["obs"]), device=dev, dtype=torch.float32)
            act = torch.as_tensor(np.asarray(tr["act"]), device=dev, dtype=torch.float32)
            rew = torch.as_tensor(np.asarray(tr["rew"]), device=dev, dtype=torch.float32)
            done= torch.as_tensor(np.asarray(tr["done"]),device=dev, dtype=torch.bool)
            tout= torch.as_tensor(np.asarray(tr["timeout"]),device=dev, dtype=torch.bool)
            nxt = torch.as_tensor(np.asarray(tr["next_obs"]),device=dev, dtype=torch.float32)

            V_boot = None
            if self.bootstrap_time_limit and tout.any():
                with torch.no_grad():
                    na = policy.actor_target(nxt)
                    nq = policy.critic_target(nxt, na)[0].squeeze(-1)
                V_boot = nq  # [T]

            T = rew.shape[0]
            G = torch.zeros(T, device=dev)
            G_next = torch.zeros((), device=dev)
            for t in range(T - 1, -1, -1):
                true_terminal = done[t] & (~tout[t])
                G_next = torch.where(true_terminal, torch.zeros_like(G_next), G_next)
                if self.bootstrap_time_limit and tout[t]:
                    g_t = rew[t] + gamma * (V_boot[t] if V_boot is not None else 0.0)
                else:
                    g_t = rew[t] + gamma * G_next
                G[t] = g_t
                G_next = g_t

            with torch.no_grad():
                Q = policy.critic(obs, act)[0].squeeze(-1)

            Q_all.append(Q); G_all.append(G)

        Q_all = torch.cat(Q_all) if Q_all else torch.empty(0, device=dev)
        G_all = torch.cat(G_all) if G_all else torch.empty(0, device=dev)

        if Q_all.numel():
            mc_bias = (Q_all - G_all).detach().cpu().numpy()
            mc_mean, mc_std = float(mc_bias.mean()), float(mc_bias.std())
            mc_abs_mean, mc_abs_std = float(np.abs(mc_bias).mean()), float(np.abs(mc_bias).std())
            q_mean, q_std = float(Q_all.mean()), float(Q_all.std(unbiased=False))
            g_mean, g_std = float(G_all.mean()), float(G_all.std(unbiased=False))
            critic_mse = float(F.mse_loss(Q_all, G_all).item())
        else:
            mc_mean=mc_std=mc_abs_mean=mc_abs_std=q_mean=q_std=g_mean=g_std=critic_mse=float("nan")

        # 4) Log to TB
        self.logger.record("mc_eval/bias_mean", mc_mean)
        self.logger.record("mc_eval/bias_std", mc_std)
        self.logger.record("mc_eval/bias_abs_mean", mc_abs_mean)
        self.logger.record("mc_eval/bias_abs_std", mc_abs_std)
        self.logger.record("mc_eval/q_values_mean", q_mean)
        self.logger.record("mc_eval/q_values_std", q_std)
        self.logger.record("mc_eval/return_mean", g_mean)
        self.logger.record("mc_eval/return_std", g_std)
        self.logger.record("mc_eval/critic_mse_q_vs_mc", critic_mse)

        # 5) Also save CSV (with eval aggregates for convenience)
        eval_mean_reward = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        eval_std_reward = float(np.std(ep_rewards))  if ep_rewards else float("nan")
        eval_mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
        eval_std_len = float(np.std(ep_lengths))  if ep_lengths else float("nan")

        with open(self.save_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps,
                mc_mean, mc_std, mc_abs_mean, mc_abs_std,
                q_mean, q_std, g_mean, g_std, critic_mse,
                eval_mean_reward, eval_std_reward, eval_mean_len, eval_std_len,
            ])

        if self.verbose:
            print(f"[MC-after-eval] t={self.num_timesteps}  "
                  f"R={eval_mean_reward:.2f}  MCbias={mc_mean:.4f}  MSE={critic_mse:.4f}")
        return True