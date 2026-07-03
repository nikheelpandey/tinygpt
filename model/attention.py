import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kv_cache import HeadCache


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


    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_len: int,
    ) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.size(-1))

        k_len = k.size(1)

        if q_len == k_len:
            scores = scores.masked_fill(
                self.mask[:q_len, :k_len] == 0,
                float("-inf"),
            )

        weights = F.softmax(scores, dim=-1)
        return weights @ v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: HeadCache | None = None,
        *,
        use_cache: bool = False,
    ):
        """
        x: (batch_size, seq_len, embedding_dim)

        When use_cache is True, returns (out, new_kv_cache).
        Otherwise returns out only.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if use_cache:
            if kv_cache is not None:
                k_prev, v_prev = kv_cache
                k = torch.cat([k_prev, k], dim=1)
                v = torch.cat([v_prev, v], dim=1)

            out = self._attend(q, k, v, x.size(1))
            return out, (k, v)

        out = self._attend(q, k, v, x.size(1))
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