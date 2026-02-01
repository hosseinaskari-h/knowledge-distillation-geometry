"""
Value network for RL (critic)
"""
import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Estimates value of states"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [hidden_dim] embedding
        Returns:
            value: scalar value estimate
        """
        return self.net(state).squeeze(-1)
