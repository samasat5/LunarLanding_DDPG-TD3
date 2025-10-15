import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate
from tensordict import TensorDict



class QValueEnsembleModule(nn.Module):
    """
    Wraps two TensorDictModules (critic_net_1, critic_net_2) into a single module
    that returns a TensorDict containing both outputs and exposes in_keys.
    """
    def __init__(self, q1, q2):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.in_keys = getattr(q1, "in_keys", getattr(q1, "_in_keys", None))
        if getattr(self, "_in_keys", None) is None:
            self._in_keys = getattr(self, "in_keys", None)

    def forward(self, tensordict):
        # Ensure we don't mutate the incoming tensordict unexpectedly
        td1 = self.q1(tensordict.clone())
        td2 = self.q2(tensordict.clone())
        q1_val = td1.get("state_action_value1")
        q2_val = td2.get("state_action_value2")

        # Sanity check
        if q1_val is None or q2_val is None:
            raise KeyError(
                f"Critic outputs missing 'state_action_value': got {list(td1.keys())}, {list(td2.keys())}"
            )
        # Merge
        td = TensorDict({}, batch_size=tensordict.batch_size)
        td["state_action_value"] = torch.min(q1_val, q2_val)
        return td

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



