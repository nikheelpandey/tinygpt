import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """
    Dataset for next-token prediction.

    Example:
        tokens = [1, 2, 3, 4, 5, 6]
        context_size = 3

        Sample 0:
            x = [1, 2, 3]
            y = [2, 3, 4]

        Sample 1:
            x = [2, 3, 4]
            y = [3, 4, 5]
    """

    def __init__(self, tokens: list[int], context_size: int):
        self.tokens = tokens
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.tokens) - self.context_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.context_size]
        y = self.tokens[idx + 1 : idx + self.context_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )