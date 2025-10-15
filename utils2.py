import torch
import torch.nn as nn

# Critic class
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.obs_fc = nn.Linear(obs_dim, hidden_size)
        self.act_fc = nn.Linear(act_dim, hidden_size)
        self.output_fc = nn.Linear(hidden_size, 1)

    def forward(self, observation, action):
        # Process observation and action separately
        obs_out = torch.relu(self.obs_fc(observation))
        act_out = torch.relu(self.act_fc(action))

        # Combine the outputs
        combined = obs_out + act_out  # You can also concatenate, etc.
        
        # Output the state-action value
        state_action_value = self.output_fc(combined)
        return state_action_value

