import torch
import torch.nn as nn
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
