import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate
from tensordict import TensorDict
import numpy as np
import pdb  

@torch.no_grad()
def evaluate_mc_bias(loss, 
                     eval_env,
                     gamma=0.99,
                     episodes=5,
                     ):

    actor = loss.actor_network  
    qnet = getattr(loss, "qvalue_network", None) or loss.value_network

    all_bias = []
    all_predq = []
    all_returns = []

    for _ in range(episodes):
        td = eval_env.reset()
        ep_obs = []
        ep_act = []
        ep_rew = []

        # rollout with deterministic policy
        while True:
           
            td_in = td.select("observation")
            td_pi = actor(td_in.clone()) # gives action and observation
            td_env = td.clone()
            td_env.set("action", td_pi["action"])
            td = eval_env.step(td_env)

    
            ep_obs.append(td_in["observation"])
            ep_act.append(td_pi["action"])

            pdb.set_trace()
            # Step env
            td = eval_env.step(td_pi)
            ep_rew.append(td["next", "reward"].item())

            if td["next", "done"].item():
                break

        # Compute Monte-Carlo returns G_t backward
        G = []
        g = 0.0
        for r in reversed(ep_rew):
            g = r + gamma * g
            G.append(g)
        G = list(reversed(G))  # align with timesteps
        obs_batch = torch.stack(ep_obs, 0)
        act_batch = torch.stack(ep_act, 0)
        td_batch = TensorDict(
            {"observation": obs_batch, "action": act_batch},
            batch_size=[obs_batch.shape[0]],
        )
        td_q = qnet(td_batch.clone())

        # TorchRL q-nets usually write "state_action_value"
        q_pred = td_q.get("state_action_value")
        # If TD3 with two critics, q_pred may be a tuple/list; take min
        if isinstance(q_pred, (tuple, list)):
            q1, q2 = q_pred
            q_pred = torch.minimum(q1, q2)

        q_pred = q_pred.squeeze(-1)  # [T]
        G_t = torch.tensor(G, dtype=q_pred.dtype)

        bias = (q_pred - G_t).cpu().numpy()
        all_bias.extend(bias.tolist())
        all_predq.extend(q_pred.cpu().numpy().tolist())
        all_returns.extend(G_t.cpu().numpy().tolist())

    mean_bias = float(np.mean(all_bias))
    details = {
        "bias_per_step": np.array(all_bias),
        "q_pred_per_step": np.array(all_predq),
        "return_per_step": np.array(all_returns),
    }
    return mean_bias, details



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



