import argparse

import torch

from model.gpt import GPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_PATH = "checkpoints/tiny_gpt.pt"


def load_model(checkpoint_path: str):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    tokenizer = BPETokenizer(
        checkpoint["merges"]
    )

    model = GPT(**config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(
        description="TinyGPT Inference"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt to generate from",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="Checkpoint path",
    )

    args = parser.parse_args()

    model, tokenizer, device = load_model(
        args.checkpoint
    )

    tokens = tokenizer.encode(args.prompt)

    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        idx=x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    output = tokenizer.decode(
        generated[0].tolist()
    )

    print("\n" + "=" * 80)
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()
