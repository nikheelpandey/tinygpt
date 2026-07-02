import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        context_size: int,
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            head_size,
            bias=False,
        )

        self.key = nn.Linear(
            embedding_dim,
            head_size,
            bias=False,
        )

        self.value = nn.Linear(
            embedding_dim,
            head_size,
            bias=False,
        )

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_size, context_size))
        )


    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, context_size, embedding_dim)
        """

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2,-1)
        scores = scores / math.sqrt(k.size(-1))


        # Apply causal mask
        seq_len = x.size(1)

        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0,
            float("-inf"),
        )

        # Normalize
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        out = weights @ v

        return out

if __name__ == "__main__":
    attention = SelfAttention(
        embedding_dim=128,
        head_size=64,
        context_size=128
    )

    x = torch.randn(8, 32, 128)

    out = attention(x)

    print(out.shape)