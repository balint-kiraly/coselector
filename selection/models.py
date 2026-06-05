import torch.nn as nn


class AgentSelectorMLP(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise final layer bias to +0.5 so sigmoid(0.5) ≈ 0.62 at cold
        # start → the policy selects ~62% of agents by default, guaranteeing a
        # non-trivial detection signal before any gradient updates.
        nn.init.constant_(self.net[-1].bias, 0.5)

    def forward(self, state_feats):
        # state_feats: (N, D) → returns (N,) logits
        return self.net(state_feats).squeeze(-1)
