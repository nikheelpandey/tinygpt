import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import GPTEmbedding
from model.block import TransformerBlock
from model.kv_cache import KVCache


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.context_size = context_size

        # Token + Position Embeddings
        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_size=context_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Language Modeling Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        pos_offset: int = 0,
    ):
        """
        Args:
            x: (batch_size, context_size)
            targets: (batch_size, context_size)
            kv_cache: per-layer cached keys/values for incremental inference
            pos_offset: absolute position of the first token in x

        Returns:
            logits:
                (batch_size, context_size, vocab_size)

            loss:
                scalar tensor (or None during inference)

            kv_cache:
                updated cache when kv_cache was provided, else omitted
        """

        x = self.embedding(x, pos_offset=pos_offset)

        new_kv_cache = None

        if kv_cache is not None:
            new_kv_cache = []

            for i, block in enumerate(self.blocks):
                layer_cache = (
                    kv_cache[i]
                    if i < len(kv_cache)
                    else []
                )
                x, layer_new_cache = block(x, layer_cache)
                new_kv_cache.append(layer_new_cache)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, V = logits.shape

            logits = logits.view(B * T, V)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets,
            )

        if kv_cache is not None:
            return logits, loss, new_kv_cache

        return logits, loss
        
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_kv_cache: bool = True,
    ):
        self.eval()

        kv_cache = None

        for _ in range(max_new_tokens):
            if use_kv_cache and kv_cache is not None:
                if idx.size(1) > self.context_size:
                    # Positions are rebased when the window slides — re-prefill.
                    idx_cond = idx[:, -self.context_size:]
                    logits, _, kv_cache = self(
                        idx_cond,
                        kv_cache=[],
                        pos_offset=0,
                    )
                else:
                    idx_cond = idx[:, [-1]]
                    pos_offset = idx.size(1) - 1
                    logits, _, kv_cache = self(
                        idx_cond,
                        kv_cache=kv_cache,
                        pos_offset=pos_offset,
                    )
            elif use_kv_cache:
                idx_cond = idx[:, -self.context_size:]
                logits, _, kv_cache = self(
                    idx_cond,
                    kv_cache=[],
                    pos_offset=0,
                )
            else:
                idx_cond = idx[:, -self.context_size:]
                logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    k=top_k,
                )

                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(
                logits,
                dim=-1,
            )

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs,
                    descending=True,
                    dim=-1,
                )
                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1,
                )
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(
                    -1,
                    sorted_indices,
                    sorted_mask,
                )
                probs = probs.masked_fill(mask, 0.0)
                probs = probs / probs.sum(
                    dim=-1,
                    keepdim=True,
                )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )

        return idx