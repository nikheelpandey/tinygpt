import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import GPTEmbedding
from model.block import TransformerBlock


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
        self.blocks = nn.Sequential(
            *[
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
    ):
        """
        Args:
            x: (batch_size, context_size)
            targets: (batch_size, context_size)

        Returns:
            logits:
                (batch_size, context_size, vocab_size)

            loss:
                scalar tensor (or None during inference)
        """

        x = self.embedding(x)

        x = self.blocks(x)

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

        return logits, loss
        
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.context_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            # -----------------------------
            # Top-k sampling
            # -----------------------------
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

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )

        return idx