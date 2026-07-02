import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, context_size, embedding_dim)

        returns:
            (batch_size, context_size, embedding_dim)
        """
        return self.net(x)