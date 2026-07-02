import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, context_size)

        returns:
            (batch_size, context_size, embedding_dim)
        """

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device,
        )

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings



if __name__=="__main__":
    import torch

    embedding = GPTEmbedding(
        vocab_size=5000,
        embedding_dim=128,
        context_size=32,
    )

    x = torch.randint(0, 5000, (8, 32))

    y = embedding(x)

    print(y.shape)