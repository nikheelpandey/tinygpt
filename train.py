from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer.trainer import train_bpe
from tokenizer.bpe import BPETokenizer
from tokenizer.utils import ByteTokenizer, build_corpus

from dataset.dataset import GPTDataset
from model.gpt import GPT


# =====================================================
# Hyperparameters
# =====================================================

BATCH_SIZE = 64
CONTEXT_SIZE = 128

VOCAB_SIZE = 5000

EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4

LR = 3e-4
EPOCHS = 5

VAL_SPLIT = 0.2          # fraction of tokens held out for validation
GEN_TOKENS = 100         # how many tokens to sample for the qualitative check
GEN_PROMPT = "The "      # seed text for generation preview

PATIENCE = 2             # stop early if val loss doesn't improve for this many epochs

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "tiny_gpt.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "tiny_gpt_best.pt"


# =====================================================
# Read Corpus
# =====================================================

with open("data/corpus.txt", encoding="utf-8") as f:
    text = f.read()


# =====================================================
# Train Tokenizer
# =====================================================

corpus = build_corpus(text, ByteTokenizer())

print("Training BPE tokenizer...")

merges, corpus = train_bpe(
    corpus=corpus,
    vocab_size=VOCAB_SIZE,
)

tokenizer = BPETokenizer(merges)

# `corpus` was mutated in-place by train_bpe with every learned merge already
# applied — no need to re-encode `text` from scratch. Just flatten it.
tokens = [tok for sequence in corpus for tok in sequence]

print(f"Corpus tokens : {len(tokens)}")
print(f"Vocabulary    : {VOCAB_SIZE}")
print(f"Merges        : {len(merges)}")


# =====================================================
# Train / Validation Split
# =====================================================

# Split by position (not randomly) so both halves stay made of contiguous,
# coherent text rather than shuffled fragments.
split_idx = int(len(tokens) * (1 - VAL_SPLIT))
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

print(f"Train tokens  : {len(train_tokens)}")
print(f"Val tokens    : {len(val_tokens)}")


# =====================================================
# Device
# =====================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =====================================================
# Datasets
# =====================================================

train_dataset = GPTDataset(
    tokens=train_tokens,
    context_size=CONTEXT_SIZE,
)

val_dataset = GPTDataset(
    tokens=val_tokens,
    context_size=CONTEXT_SIZE,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    num_workers=4 if device.type == "cuda" else 0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,   # no need to shuffle for evaluation
    pin_memory=(device.type == "cuda"),
    num_workers=4 if device.type == "cuda" else 0,
)


# =====================================================
# Model
# =====================================================

model = GPT(
    vocab_size=256 + len(merges),
    embedding_dim=EMBEDDING_DIM,
    context_size=CONTEXT_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# =====================================================
# Optimizer
# =====================================================

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1)

use_amp = device.type == "cuda"  # AMP/GradScaler kept CUDA-only; MPS support is inconsistent
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =====================================================
# Metrics helpers
# =====================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp) -> dict:
    """
    Runs one full pass over `loader` and returns validation metrics:
      - loss: average cross-entropy loss
      - perplexity: exp(loss), a standard LM quality metric.
        ~vocab_size perplexity = model is basically guessing uniformly (gibberish).
        Lower is better; a coherent tiny model should land well below vocab_size.
      - accuracy: fraction of positions where the model's top-1 prediction
        matches the actual next token. Rough but intuitive "is it learning" signal.
    """

    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_tokens = 0
    num_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, loss = model(x, y)

        total_loss += loss.detach()
        num_batches += 1

        # Flatten both sides before comparing, regardless of whether logits
        # come back as (B, T, V) or already-flattened (B*T, V).
        vocab_size = logits.size(-1)
        preds = logits.reshape(-1, vocab_size).argmax(dim=-1)
        targets = y.reshape(-1)

        total_correct += (preds == targets).sum()
        total_tokens += targets.numel()

    avg_loss = (total_loss / max(num_batches, 1)).item()
    accuracy = (total_correct / max(total_tokens, 1)).item()
    perplexity = float(torch.exp(torch.tensor(avg_loss)))

    model.train()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
    }

@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int,
                     context_size: int, device) -> str:
    """
    Greedy-decodes a short continuation from `prompt` so you can eyeball
    whether the model produces plausible text or gibberish. Not a rigorous
    metric, but often the fastest way to catch a broken/undertrained model
    that still shows a "reasonable-looking" loss number.
    """
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # keep only the last `context_size` tokens, matching training context
        input_crop = input_ids[:, -context_size:]

        logits, _ = model(input_crop, None)
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    model.train()

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


# =====================================================
# Training
# =====================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in tqdm(range(EPOCHS), desc="Epochs"):

    # ---------------- Train ----------------
    model.train()
    running_loss = torch.zeros((), device=device)

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False):

        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, enabled=True):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.detach()

    train_loss = (running_loss / len(train_loader)).item()

    # ---------------- Validate ----------------
    val_metrics = evaluate(model, val_loader, device, use_amp)

    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val PPL: {val_metrics['perplexity']:.2f} | "
        f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%"
    )

    # ---------------- Qualitative check ----------------
    sample = generate_sample(
        model, tokenizer, GEN_PROMPT, GEN_TOKENS, CONTEXT_SIZE, device
    )
    print(f"Sample: {sample!r}\n")

    # ---------------- Track best checkpoint / early stopping ----------------
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0

        CHECKPOINT_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": 256 + len(merges),
                    "embedding_dim": EMBEDDING_DIM,
                    "context_size": CONTEXT_SIZE,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                },
                "merges": merges,
                "val_loss": best_val_loss,
                "epoch": epoch + 1,
            },
            BEST_CHECKPOINT_PATH,
        )
    else:
        epochs_without_improvement += 1
        print(
            f"No val improvement for {epochs_without_improvement} epoch(s) "
            f"(best so far: {best_val_loss:.4f})\n"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val improvement for {PATIENCE} epochs)"
            )
            break


# =====================================================
# Save Final Checkpoint
# =====================================================

CHECKPOINT_DIR.mkdir(exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": 256 + len(merges),
            "embedding_dim": EMBEDDING_DIM,
            "context_size": CONTEXT_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
        },
        "merges": merges,
    },
    CHECKPOINT_PATH,
)

print(f"\nFinal checkpoint saved to {CHECKPOINT_PATH}")
print(f"Best checkpoint (val_loss={best_val_loss:.4f}) saved to {BEST_CHECKPOINT_PATH}")