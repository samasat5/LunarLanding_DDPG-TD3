# utils_sb3.py
import numpy as np, torch, os, csv, matplotlib.pyplot as plt, pdb
from stable_baselines3.common.callbacks import BaseCallback

class QBiasLoggerDDPG(BaseCallback):
    def __init__(self, gamma: float, sample_n: int = 50_000, save_csv: str = "./logs_ddpg/qbias/qbias_log.csv"):
        super().__init__()
        self.gamma = gamma
        self.sample_n = sample_n
        self.save_csv = save_csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timesteps",
                    "mean_bias",
                    "mean_abs_bias",
                    "std_bias"
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
            bias = (q - target).cpu().numpy()

        mean_bias = float(np.mean(bias))
        mean_abs  = float(np.mean(np.abs(bias)))
        std_bias  = float(np.std(bias))

        self.logger.record("q_bias/mean", mean_bias)
        self.logger.record("q_bias/mean_abs", mean_abs)
        self.logger.record("q_bias/std", std_bias)

        with open(self.save_csv, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, mean_bias, mean_abs, std_bias])

        return True
    

class QBiasLoggerTD3(BaseCallback):
    """
    TD3 Q-bias/TD-error logger.
    Logs:
      - q_bias/q1_mean, q2_mean, min_mean
      - mean |bias| and std for each
      - overestimation_gap = mean(|q1 - q2|)
    Also appends a CSV for easy plotting.
    """
    def __init__(self, gamma: float, sample_n: int = 50_000, save_csv: str = "./logs_td3/qbias/qbias_log.csv"):
        super().__init__()
        self.gamma = gamma
        self.sample_n = sample_n
        self.save_csv = save_csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timesteps",
                    "q1_mean","q1_mean_abs","q1_std",
                    "q2_mean","q2_mean_abs","q2_std",
                    "min_mean","min_mean_abs","min_std",
                    "overestimation_gap_mean_abs"
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

        q1_mean, q1_abs, q1_std = float(np.mean(bias1)), float(np.mean(np.abs(bias1))), float(np.std(bias1))
        q2_mean, q2_abs, q2_std = float(np.mean(bias2)), float(np.mean(np.abs(bias2))), float(np.std(bias2))
        mn_mean, mn_abs, mn_std = float(np.mean(bias_min)), float(np.mean(np.abs(bias_min))), float(np.std(bias_min))
        gap_abs_mean = float(np.mean(over_gap))

        # log to SB3 (TensorBoard/CSV if configured)
        self.logger.record("q_bias/q1_mean", q1_mean)
        self.logger.record("q_bias/q1_mean_abs", q1_abs)
        self.logger.record("q_bias/q1_std", q1_std)

        self.logger.record("q_bias/q2_mean", q2_mean)
        self.logger.record("q_bias/q2_mean_abs", q2_abs)
        self.logger.record("q_bias/q2_std", q2_std)

        self.logger.record("q_bias/min_mean", mn_mean)
        self.logger.record("q_bias/min_mean_abs", mn_abs)
        self.logger.record("q_bias/min_std", mn_std)

        self.logger.record("q_bias/overestimation_gap_abs_mean", gap_abs_mean)

        # append to our CSV
        with open(self.save_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.num_timesteps,
                q1_mean, q1_abs, q1_std,
                q2_mean, q2_abs, q2_std,
                mn_mean, mn_abs, mn_std,
                gap_abs_mean
            ])

        return True
    

def plot_qbias_ddpg(csv_path="./logs_ddpg/qbias/qbias_log.csv", save_path=None):
    """
    Read the q-bias CSV saved by QBiasLoggerDDPG and plot the mean, |mean| and std.

    :param csv_path: Path to the CSV created by QBiasLogger.
    :param save_path: Optional path to save the figure. If None, saves next to CSV.
    :return: Path to the saved PNG.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No file found at {csv_path}")

    # Read CSV with column names
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
 
    timesteps = data["timesteps"]
    mean_bias = data["mean_bias"]
    mean_abs = data["mean_abs_bias"]
    std_bias = data["std_bias"]

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, mean_bias, label="mean(Q - target)")
    plt.plot(timesteps, mean_abs, label="mean |Q - target|")
    plt.plot(timesteps, std_bias, label="std(Q - target)")
    plt.xlabel("Timesteps")
    plt.ylabel("Q-bias value")
    plt.title("DDPG Q-bias vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(csv_path), "qbias_plot.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Q-bias plot saved at: {save_path}")
    return save_path

def plot_qbias_td3(csv_path="./logs_td3/qbias/qbias_log.csv", save_path=None):
    """
    Read the TD3 q-bias CSV saved by QBiasLoggerTD3 and plot the mean, |mean|, std,
    and overestimation gap for both critics.

    :param csv_path: Path to the CSV created by QBiasLoggerTD3.
    :param save_path: Optional path to save the figure. If None, saves next to CSV.
    :return: Path to the saved PNG.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No file found at {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["timesteps"]

    plt.figure(figsize=(9, 5))

    # Plot both critics’ mean biases
    plt.plot(t, data["q1_mean"], label="Q1 mean bias")
    plt.plot(t, data["q2_mean"], label="Q2 mean bias")
    plt.plot(t, data["min_mean"], label="min(Q1, Q2) mean bias")

    # Plot abs bias as dashed lines
    plt.plot(t, data["q1_mean_abs"], "--", label="Q1 mean |bias|")
    plt.plot(t, data["q2_mean_abs"], "--", label="Q2 mean |bias|")
    plt.plot(t, data["min_mean_abs"], "--", label="min(Q1,Q2) mean |bias|")

    # Plot the overestimation gap
    if "overestimation_gap_mean_abs" in data.dtype.names:
        plt.plot(t, c("overestimation_gap_mean_abs"), ":", color="black",
                 label="|Q1 - Q2| (mean abs gap)")

    plt.xlabel("Timesteps")
    plt.ylabel("Bias value")
    plt.title("TD3 Q-bias & Overestimation Gap vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(csv_path), "qbias_td3_plot.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"TD3 Q-bias plot saved at: {save_path}")
    return save_path


if __name__ == "__main__":
    #plot_qbias_td3()
    plot_qbias_ddpg()