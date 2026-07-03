import torch

# Per-head (key, value) tensors shaped (batch, seq_len, head_size)
HeadCache = tuple[torch.Tensor, torch.Tensor]

# One entry per attention head in a layer
LayerCache = list[HeadCache]

# One entry per transformer layer
KVCache = list[LayerCache]
