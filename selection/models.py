import torch.nn as nn


class AgentSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # score for each agent
        )

    def forward(self, state_feats):
        # state_feats: (N, D)
        scores = self.net(state_feats).squeeze(-1)  # -> (N,)
        return scores
