import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate
from tensordict import TensorDict


class QValueEnsembleModule(nn.Module):
    """
    Wrap two TensorDictModules (q1, q2) that each write 'state_action_value' (shape [..., 1])
    and return a single TensorDict with 'state_action_value' of shape [..., 2].
    TD3Loss(num_qvalue_nets=2) can then split those heads internally.
    """
    def __init__(self, q1: nn.Module, q2: nn.Module, out_key: str = "state_action_value"):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.out_key = out_key
        # propagate in_keys so TorchRL can inspect them
        self.in_keys = getattr(q1, "in_keys", getattr(q1, "_in_keys", None))
        if getattr(self, "_in_keys", None) is None:
            self._in_keys = self.in_keys

    def forward(self, tensordict):
        td1 = self.q1(tensordict.clone())
        td2 = self.q2(tensordict.clone())

        q1 = td1.get(self.out_key)  # expected shape [..., 1]
        q2 = td2.get(self.out_key)  # expected shape [..., 1]
        if q1 is None or q2 is None:
            raise KeyError(
                f"Critics must write '{self.out_key}'. Got keys {list(td1.keys())} and {list(td2.keys())}."
            )

        # Concatenate heads on the last dimension => shape [..., 2]
        q = torch.cat([q1, q2], dim=-1)

        out = TensorDict({self.out_key: q}, batch_size=tensordict.batch_size)
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



