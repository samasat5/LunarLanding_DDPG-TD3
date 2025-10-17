import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate
from tensordict import TensorDict


class QValueEnsembleModule(nn.Module):
    """
    Wrap two TensorDictModules (q1, q2) that each read ('observation','action')
    and produce two separate outputs:
      - 'state_action_value1' : [..., 1]
      - 'state_action_value2' : [..., 1]
    TD3Loss will take the min for target bootstrapping and compute both critic losses.
    """
    def __init__(self, q1: nn.Module, q2: nn.Module):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        # propagate in_keys so TorchRL can inspect them
        self.in_keys = getattr(q1, "in_keys", getattr(q1, "_in_keys", None))
        if getattr(self, "_in_keys", None) is None:
            self._in_keys = self.in_keys
        self.out_keys = ["state_action_value1", "state_action_value2"]

    def forward(self, tensordict: TensorDict):
        td1 = self.q1(tensordict.clone())
        td2 = self.q2(tensordict.clone())

        v1 = td1.get("state_action_value")
        v2 = td2.get("state_action_value")
        if v1 is None or v2 is None:
            raise KeyError(
                f"Critics must write 'state_action_value'. Got {list(td1.keys())} and {list(td2.keys())}."
            )

        out = tensordict.clone()
        out.set("state_action_value1", v1)  # [..., 1]
        out.set("state_action_value2", v2)  # [..., 1]
        return out



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



