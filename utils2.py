import torch
import torch.nn as nn
from torchrl.objectives import SoftUpdate

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










class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.obs_fc = nn.Linear(obs_dim, hidden_size)
        self.act_fc = nn.Linear(act_dim, hidden_size)
        self.output_fc = nn.Linear(hidden_size, 1)

    def forward(self, observation, action):
        obs_out = torch.relu(self.obs_fc(observation))
        act_out = torch.relu(self.act_fc(action))
        combined = obs_out + act_out 
        state_action_value = self.output_fc(combined)
        return state_action_value


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc = nn.Linear(obs_dim, hidden_size)
        self.mu_fc = nn.Linear(hidden_size, act_dim)
        self.log_std_fc = nn.Linear(hidden_size, act_dim)
    def forward(self, observation):
        x = torch.relu(self.fc(observation))
        mu = self.mu_fc(x)
        log_std = self.log_std_fc(x)
        std = torch.exp(log_std)
        return mu, std
