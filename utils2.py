import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate
from tensordict import TensorDict
import numpy as np
import pdb  

import math
import numpy as np
import matplotlib.pyplot as plt




def plot_bias_stats(biases, title="On-policy bias Q - MC G_t"):
    if biases.size == 0: return
    k = np.arange(1, len(biases)+1)
    run_mean = np.cumsum(biases)/k
    diffsq = np.cumsum((biases - run_mean)**2)
    var = np.zeros_like(run_mean); 
    if len(biases) >= 2: var[1:] = diffsq[1:] / (k[1:]-1)
    ci = 1.96*np.sqrt(var)/np.sqrt(k)

    plt.figure(figsize=(7,4))
    plt.plot(k, run_mean, label="Running mean bias")
    plt.fill_between(k, run_mean-ci, run_mean+ci, alpha=0.2, label="95% CI")
    plt.axhline(0, ls="--", lw=1, label="Zero bias")
    plt.title(title); plt.xlabel("On-policy steps (k)"); plt.ylabel("Q - G_t")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_q_vs_mc(q_vals, g_t_all, title="Calibration: Q(s,μ) vs MC G_t"):
    if q_vals.size == 0: return
    plt.figure(figsize=(5,5))
    plt.scatter(g_t_all, q_vals, s=6, alpha=0.3)
    m = np.nanmean(np.concatenate([q_vals, g_t_all]))
    lo, hi = np.percentile(np.concatenate([q_vals, g_t_all]), [2, 98])
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1)  # ideal y=x
    plt.xlabel("MC return-to-go G_t"); plt.ylabel("Critic Q(s,μ)")
    plt.title(title); plt.tight_layout(); plt.show()
    
    
    
def plot_mc_estimate(returns, title="MC estimate of J(μ) with 95% CI"):
    """returns: list/array of episodic (discounted) returns G_1..G_N"""
    r = np.asarray(returns, dtype=float)
    N = len(r)
    # running mean & CI
    k = np.arange(1, N+1)
    running_mean = np.cumsum(r) / k
    # unbiased running std (nan for k=1 -> replace with 0)
    diffsq = np.cumsum((r - running_mean)**2)
    running_var = np.zeros(N)
    running_var[1:] = diffsq[1:] / (k[1:] - 1)
    running_std = np.sqrt(running_var)
    ci95 = 1.96 * running_std / np.sqrt(k)
    lo = running_mean - ci95
    hi = running_mean + ci95

    plt.figure(figsize=(7,4))
    plt.plot(k, running_mean, label="Running mean Ȳₖ")
    plt.fill_between(k, lo, hi, alpha=0.2, label="95% CI")
    plt.axhline(running_mean[-1], linestyle="--", linewidth=1, label=f"Final mean = {running_mean[-1]:.2f}")
    plt.xlabel("Episodes (k)")
    plt.ylabel("Estimated return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def soft_update(target_network, source_network, tau):
    """Soft update the target network using Polyak averaging."""
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

class MultiCriticSoftUpdate(SoftUpdate):
    def __init__(self, loss_module, tau: float):
        super().__init__(loss_module, tau=tau)

    def _step(self, critic_sources: list, critic_targets: list) -> None:
        """Update both target critic networks using Polyak averaging."""
        for p_source, p_target in zip(critic_sources, critic_targets):
            p_target.data.lerp_(p_source.data, 1 - self.tau)

    def update(self, td1):
        # Apply the update using the TD3Loss for both Q networks (Q1, Q2)
        critic_sources = [self.loss_module.qvalue_network[0], self.loss_module.qvalue_network[1]]  # Q1, Q2
        critic_targets = [self.loss_module.qvalue_network[0].target, self.loss_module.qvalue_network[1].target]  # Q1_target, Q2_target
        self._step(critic_sources, critic_targets)



