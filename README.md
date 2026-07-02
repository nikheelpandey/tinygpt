# TinyGPT

Train a small GPT-style language model from scratch in pure PyTorch — no Hugging Face, no pretrained weights, no magic.

TinyGPT is a minimal, readable implementation of a decoder-only transformer. It includes a byte-level BPE tokenizer, a full GPT model, a training loop with validation and early stopping, and a simple inference CLI. The default corpus is Shakespeare (~1.1 MB), so you can go from zero to generated text in minutes on a laptop.

**Good for:** learning how LLMs work, experimenting with hyperparameters, swapping in your own text corpus, and understanding tokenization + attention end to end.

## Features

- **Byte-level BPE tokenizer** trained on your corpus (vocab size configurable)
- **GPT architecture** — token + position embeddings, multi-head self-attention, feed-forward blocks, causal masking
- **Training loop** with train/val split, perplexity, accuracy, AMP on CUDA, and early stopping
- **Checkpointing** — saves best and final model weights under `checkpoints/`
- **Inference CLI** with temperature sampling and optional top-k filtering

## Quick start

```bash
git clone https://github.com/nikheelpandey/tinygpt.git
cd tinygpt

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python train.py
```

Training prints loss, perplexity, and a short text sample each epoch. When it finishes, generate text:

```bash
python inference.py "The king" --max-new-tokens 150 --temperature 0.8
```

Use the best checkpoint if you prefer:

```bash
python inference.py "To be or not" --checkpoint checkpoints/tiny_gpt_best.pt
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- ~2 GB RAM for training (CPU/MPS/CUDA all supported)

| Device | Notes |
|--------|-------|
| **CUDA** | Fastest; uses automatic mixed precision |
| **Apple MPS** | Supported during training |
| **CPU** | Works; slower but fine for experimentation |

## Project structure

```
tinygpt/
├── train.py              # End-to-end training script
├── inference.py          # Text generation CLI
├── data/
│   └── corpus.txt        # Default training text (Shakespeare)
├── tokenizer/
│   ├── bpe.py            # BPE tokenizer
│   ├── trainer.py        # BPE merge training
│   └── utils.py          # Byte tokenizer + corpus helpers
├── model/
│   ├── gpt.py            # GPT model + generate()
│   ├── attention.py      # Scaled dot-product attention
│   ├── multi_head.py     # Multi-head attention
│   ├── block.py          # Transformer block
│   ├── feed_forward.py   # MLP block
│   └── embeddings.py     # Token + position embeddings
├── dataset/
│   └── dataset.py        # Sliding-window language modeling dataset
└── checkpoints/          # Saved models (created during training, gitignored)
```

## Training your own corpus

Replace `data/corpus.txt` with any UTF-8 text file, then run `python train.py`. Hyperparameters live at the top of `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB_SIZE` | 5000 | BPE vocabulary size |
| `CONTEXT_SIZE` | 128 | Max sequence length |
| `EMBEDDING_DIM` | 128 | Model width |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_LAYERS` | 4 | Transformer blocks |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 5 | Max training epochs |
| `LR` | 3e-4 | AdamW learning rate |
| `PATIENCE` | 2 | Early stopping patience |

Checkpoints include the model weights, architecture config, and BPE merges — everything needed for inference.

## How it works

1. **Tokenize** — Text is split into bytes, then BPE merges are learned until the vocabulary reaches `VOCAB_SIZE`.
2. **Dataset** — Sliding windows of `CONTEXT_SIZE` tokens produce `(input, target)` pairs for next-token prediction.
3. **Model** — A stack of causal transformer blocks predicts the next token at each position.
4. **Train** — Cross-entropy loss, AdamW optimizer, validation perplexity, and greedy samples each epoch.
5. **Generate** — At inference time, the model autoregressively samples tokens with temperature scaling.

## Default model size

With the default hyperparameters on the Shakespeare corpus, the model has roughly **2.1M parameters** — small enough to train quickly, large enough to produce recognizable (if imperfect) Shakespearean prose.

## Inference options

```
python inference.py PROMPT [options]

  --max-new-tokens N   Tokens to generate (default: 100)
  --temperature T      Sampling temperature (default: 1.0; lower = more deterministic)
  --checkpoint PATH    Checkpoint file (default: checkpoints/tiny_gpt.pt)
```

## License

MIT — use it, fork it, learn from it.

## Contributing

Issues and pull requests welcome. Ideas that fit the project's scope:

- Better sampling (top-p / repetition penalty)
- Saving and loading tokenizer artifacts separately
- Training on custom corpora via CLI flags
- Small benchmark or eval script

---

If this helped you understand transformers, consider starring the repo so others can find it.
