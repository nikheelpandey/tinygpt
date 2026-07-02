import torch
import torch.nn as nn

from model.attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_size: int,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                SelfAttention(
                    embedding_dim=embedding_dim,
                    head_size=head_size,
                    context_size=context_size,
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, context_size, embedding_dim)

        Returns:
            (batch_size, context_size, embedding_dim)
        """

        # Run all attention heads
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )

        # Project back to embedding dimension
        out = self.proj(out)

        return out