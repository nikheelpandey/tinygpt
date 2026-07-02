import torch
import torch.nn as nn

from model.multi_head import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_size: int,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_size=context_size,
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
        )

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

if __name__ == "__main__":
    block = TransformerBlock(
    embedding_dim=128,
    num_heads=8,
    context_size=32,
    )

    x = torch.randn(4, 32, 128)

    out = block(x)

    print(out.shape)