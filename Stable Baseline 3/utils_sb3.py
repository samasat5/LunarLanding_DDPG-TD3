# utils_sb3.py
import numpy as np, torch, os, csv, matplotlib.pyplot as plt, pdb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch.nn.functional as F
from collections import deque
import pandas as pd

def plot_stats_ddpg(csv_path="./logs_ddpg/stats/stats_log.csv"):
    """
    Read the q-bias CSV saved by PlotLoggerDDPG and plot the mean, |mean| and std.

    :param csv_path: Path to the CSV created by PlotLogger.
    :param save_path: Optional path to save the figure. If None, saves next to CSV.
    :return: Path to the saved PNG.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No file found at {csv_path}")
    
    # Read CSV with column names
    data = np.genfromtxt(csv_path, delimiter=",", names=True) 
    timesteps = data["timesteps"]
    mean_bias, std_bias = data["bias_mean"], data["bias_std"]
    mean_abs, std_abs = data["bias_abs_mean"], data["bias_abs_std"]
    mean_q, std_q = data["q_values_mean"], data["q_values_std"]
    mean_target, std_target = data["target_mean"], data["target_std"]
    critic_loss = data["critic_loss"]

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, mean_bias, label="mean(Q - target)")
    plt.fill_between(timesteps, mean_bias - std_bias, mean_bias + std_bias, alpha=0.2, label="std")
    # plt.plot(timesteps, mean_abs, label="mean |Q - target|")
    # plt.fill_between(timesteps, mean_abs - std_abs, mean_abs + std_abs, alpha=0.2, label="std")
    plt.xlabel("Timesteps")
    plt.ylabel("Q-bias value")
    plt.title("DDPG Q-bias vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "qbias_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, mean_q, label="mean Q-values")
    plt.fill_between(timesteps, mean_q - std_q, mean_q + std_q, alpha=0.2, label="std")
    plt.xlabel("Timesteps")
    plt.ylabel("Q-value")
    plt.title("DDPG Q-value vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "qvalue_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, mean_target, label="mean Target")
    plt.fill_between(timesteps, mean_target - std_target, mean_target + std_target, alpha=0.2, label="std")
    plt.xlabel("Timesteps")
    plt.ylabel("Target Value")
    plt.title("DDPG Target Value vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "target_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, critic_loss, label="mean critic loss values")
    plt.xlabel("Timesteps")
    plt.ylabel("Critic Loss")
    plt.title("DDPG Critic Loss vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "critic_loss_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    data = np.load("./ddpg_eval/evaluations.npz", allow_pickle=True)
    # arrays of shape (num_evals, variable-length lists)
    timesteps = data["timesteps"]
    results = data["results"]        # list of lists of rewards
    reward_std = np.array([np.std(r) for r in results])
    reward_mean = np.array([np.mean(r) for r in results])
    # ep_lengths = data["ep_lengths"]  # list of lists of lengths
    # mean_lengths = np.array([np.mean(l) for l in ep_lengths])
    # std_lengths = np.array([np.std(l) for l in ep_lengths])
    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, reward_mean, label="Reward")
    plt.fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, label="std")
    # plt.plot(timesteps, mean_lengths, label="Reward Episode Length")
    # plt.fill_between(timesteps, mean_lengths- std_lengths, mean_lengths+ std_lengths, alpha=0.2, label="±1 std (seeds)")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("DDPG Reward vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "rewards_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    data = np.genfromtxt("./logs_ddpg/stats/mc_stats_log.csv", delimiter=",", names=True)
    timesteps = data["timesteps"]
    bias_mean = data["mc_mean"]
    bias_std = data["mc_std"]
    g_mean = data["G_mean"]
    g_std = data["G_std"]
    plt.plot(timesteps, bias_mean, label="Mean MC Bias")
    plt.fill_between(
        timesteps,
        bias_mean - bias_std,
        bias_mean + bias_std,
        alpha=0.3,
        label="std"
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Monte Carlo Bias Evolution")
    plt.ylabel("Q(s,a) - G(s,a)")
    plt.xlabel("Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "MCBias_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    plt.plot(timesteps, bias_mean, label="Mean MC Bias")
    plt.fill_between(
        timesteps,
        g_mean - g_std,
        g_mean + g_std,
        alpha=0.3,
        label="std"
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Monte Carlo Bias Evolution")
    plt.ylabel("G(s,a)")
    plt.xlabel("Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "GBias_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"DDPG plots saved at: {save_path}")
    return save_path
    

def plot_stats_td3(csv_path="./runs/td3_eval", save_path=None):
    """
    Read the TD3 q-bias CSV saved by PlotLoggerTD3 and plot the mean, |mean|, std,
    and overestimation gap for both critics.

    :param csv_path: Path to the CSV createdPlotLoggerTD3.
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
    plt.fill_between(t, data["q1_mean"] - data["q1_std"], data["q1_mean"] + data["q1_std"], alpha=0.2, label="std (batch)")
    plt.plot(t, data["q2_mean"], label="Q2 mean bias")
    plt.fill_between(t, data["q2_mean"] - data["q2_std"], data["q2_mean"] + data["q2_std"], alpha=0.2, label="std (batch)")
    plt.plot(t, data["min_mean"], label="min(Q1, Q2) mean bias")
    plt.fill_between(t, data["min_mean"] - data["min_std"], data["min_mean"] + data["min_std"], alpha=0.2, label="std (batch)")

    # Plot abs bias as dashed lines
    plt.plot(t, data["q1_mean_abs"], "--", label="Q1 mean |bias|")
    plt.fill_between(t, data["q1_mean_abs"] - data["q1_std_abs"], data["q1_mean_abs"] + data["q1_std_abs"], alpha=0.2, label="std (batch)")
    plt.plot(t, data["q2_mean_abs"], "--", label="Q2 mean |bias|")
    plt.fill_between(t, data["q2_mean_abs"] - data["q2_std_abs"], data["q2_mean_abs"] + data["q2_std_abs"], alpha=0.2, label="std (batch)")
    plt.plot(t, data["min_mean_abs"], "--", label="min(Q1,Q2) mean |bias|")
    plt.fill_between(t, data["min_mean_abs"] - data["min_std_abs"], data["min_mean_abs"] + data["min_std_abs"], alpha=0.2, label="std (batch)")


    # Plot the overestimation gap
    if "overestimation_gap_mean_abs" in data.dtype.names:
        plt.plot(t, data["overestimation_gap_mean_abs"], ":", color="black",
                 label="|Q1 - Q2| (mean abs gap)")
        plt.fill_between(t, data["overestimation_gap_mean_abs"] - data["overestimation_gap_std_abs"], data["overestimation_gap_mean_abs"] + data["overestimation_gap_std_abs"], alpha=0.2, label="std (batch)")

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

   # --- Critic losses ---
    plt.figure(figsize=(9, 5))
    plt.plot(t, data["loss1"], label="loss1 (MSE(Q1,target))")
    plt.plot(t, data["loss2"], label="loss2 (MSE(Q2,target))")
    plt.plot(t, data["critic_loss"], label="critic_loss (avg)")
    plt.xlabel("Timesteps"); plt.ylabel("Loss")
    plt.title("TD3 Critic Loss vs Timesteps")
    plt.grid(True, alpha=0.3); plt.legend()
    p2 = os.path.join(os.path.dirname(csv_path), "td3_critic_loss.png")
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    # --- Mean Q values ---
    plt.figure(figsize=(9, 5))
    plt.plot(t, data["mean_q_1"], label="mean Q1")
    plt.fill_between(t, data["mean_q_1"] - data["std_q_1"], data["mean_q_1"] + data["std_q_1"], alpha=0.2, label="±1 std (natch)")
    plt.plot(t, data["mean_q_2"], label="mean Q2")
    plt.fill_between(t, data["mean_q_2"] - data["std_q_2"], data["mean_q_2"] + data["std_q_2"], alpha=0.2, label="±1 std (batch)")
    plt.xlabel("Timesteps"); plt.ylabel("Q-value")
    plt.title("TD3 Mean Q-values vs Timesteps")
    plt.grid(True, alpha=0.3); plt.legend()
    p3 = os.path.join(os.path.dirname(csv_path), "td3_mean_q.png")
    plt.tight_layout(); plt.savefig(p3, dpi=200); plt.close()

    data = np.load("./runs/td3_eval/evaluations.npz", allow_pickle=True)
    # arrays of shape (num_evals, variable-length lists)
    timesteps = data["timesteps"]
    results = data["results"]        # list of lists of rewards
    reward_std = np.array([np.std(r) for r in results])
    reward_mean = np.array([np.mean(r) for r in results])
    ep_lengths = data["ep_lengths"]  # list of lists of lengths
    mean_lengths = np.array([np.mean(l) for l in ep_lengths])
    std_lengths = np.array([np.std(l) for l in ep_lengths])
    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, reward_mean, label="Reward")
    plt.fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, label="std (batch)")
    # plt.plot(timesteps, mean_lengths, label="Reward Episode Length")
    # plt.fill_between(timesteps, mean_lengths- std_lengths, mean_lengths+ std_lengths, alpha=0.2, label="±1 std (seeds)")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("TD3 Reward vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(os.path.dirname(csv_path), "rewards_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"TD3 plots saved at: {save_path}")

    return save_path

def tmp_plot(csv_path="./logs_td3/stats/stats_log.csv"):
    """
    Read the q-bias CSV saved by PlotLoggerDDPG and plot the mean, |mean| and std.

    :param csv_path: Path to the CSV created by PlotLogger.
    :param save_path: Optional path to save the figure. If None, saves next to CSV.
    :return: Path to the saved PNG.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No file found at {csv_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Read CSV with column names
    data = np.genfromtxt(csv_path, delimiter=",", names=True) 
    timesteps = data["timesteps"]
    mean_bias, std_bias = data["bias_mean"], data["bias_std"]
    mean_abs, std_abs = data["bias_abs_mean"], data["bias_abs_std"]
    mean_q, std_q = data["q_values_mean"], data["q_values_std"]
    mean_target, std_target = data["target_mean"], data["target_std"]
    critic_loss = data["critic_loss"]

    axes[0,0].plot(timesteps, mean_bias, label="mean(Q - target)")
    axes[0,0].fill_between(timesteps, mean_bias - std_bias, mean_bias + std_bias, alpha=0.2, label="std")
    # plt.plot(timesteps, mean_abs, label="mean |Q - target|")
    # plt.fill_between(timesteps, mean_abs - std_abs, mean_abs + std_abs, alpha=0.2, label="std")
    axes[0,0].set_xlabel("Timesteps")
    axes[0,0].set_ylabel("Q-bias value")
    axes[0,0].set_title("DDPG Q-bias vs Timesteps")
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()

    axes[0,1].plot(timesteps, mean_q, label="mean Q-values")
    axes[0,1].fill_between(timesteps, mean_q - std_q, mean_q + std_q, alpha=0.2, label="std")
    axes[0,1].set_xlabel("Timesteps")
    axes[0,1].set_ylabel("Q-value")
    axes[0,1].set_title("DDPG Q-value vs Timesteps")
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()

    # axes[1,0].plot(timesteps, mean_target, label="mean Target")
    # axes[1,0].fill_between(timesteps, mean_target - std_target, mean_target + std_target, alpha=0.2, label="std")
    # axes[1,0].set_xlabel("Timesteps")
    # axes[1,0].set_ylabel("Target Value")
    # axes[1,0].set_title("DDPG Target Value vs Timesteps")
    # axes[1,0].grid(True, alpha=0.3)
    # axes[1,0].legend()

    axes[0,1].plot(timesteps, critic_loss, label="mean critic loss values")
    axes[0,1].set_xlabel("Timesteps")
    axes[0,1].set_ylabel("Critic Loss")
    axes[0,1].set_title("DDPG Critic Loss vs Timesteps")
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()

    data = np.load("./td3_eval/evaluations.npz", allow_pickle=True)
    # arrays of shape (num_evals, variable-length lists)
    timesteps = data["timesteps"]
    results = data["results"]        # list of lists of rewards
    reward_std = np.array([np.std(r) for r in results])
    reward_mean = np.array([np.mean(r) for r in results])
    ep_lengths = data["ep_lengths"]  # list of lists of lengths
    mean_lengths = np.array([np.mean(l) for l in ep_lengths])
    std_lengths = np.array([np.std(l) for l in ep_lengths])
    axes[1,1].plot(timesteps, reward_mean, label="Reward")
    axes[1,1].fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, label="std")
    # plt.plot(timesteps, mean_lengths, label="Reward Episode Length")
    # plt.fill_between(timesteps, mean_lengths- std_lengths, mean_lengths+ std_lengths, alpha=0.2, label="±1 std (seeds)")
    axes[1,1].set_xlabel("Timesteps")
    axes[1,1].set_ylabel("Reward")
    axes[1,1].set_title("DDPG Reward vs Timesteps")
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    plt.show()
    return

if __name__ == "__main__":
    #plot_stats_td3()
    plot_stats_ddpg()
    #tmp_plot()