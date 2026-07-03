import torch
import torch.nn as nn

from model.attention import SelfAttention
from model.kv_cache import LayerCache


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

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: LayerCache | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, context_size, embedding_dim)

        Returns:
            (batch_size, context_size, embedding_dim)

        When layer_cache is provided, returns (out, new_layer_cache).
        """
        use_cache = layer_cache is not None
        head_outs = []
        new_layer_cache = [] if use_cache else None

        for i, head in enumerate(self.heads):
            head_cache = (
                layer_cache[i]
                if layer_cache and i < len(layer_cache)
                else None
            )

            if use_cache:
                out, new_head_cache = head(x, head_cache, use_cache=True)
                new_layer_cache.append(new_head_cache)
            else:
                out = head(x)

            head_outs.append(out)

        out = torch.cat(head_outs, dim=-1)
        out = self.proj(out)

        if use_cache:
            return out, new_layer_cache

        return out