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
        help="Maximum number of new tokens",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )

    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable KV cache during generation",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated text",
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

    prompt_tokens = tokenizer.encode(
        args.prompt
    )

    idx = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    print("\n" + "*" * 80)
    # print(args.prompt, end="", flush=True)

    if args.stream:

        for next_token in model.generate_stream(
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.disable_kv_cache,
        ):
            token = next_token.item()

            # Decode only the newly generated token.
            print(
                tokenizer.decode([token]),
                end="",
                flush=True,
            )

        print()

    else:

        generated = model.generate(
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.disable_kv_cache,
        )

        print(
            tokenizer.decode(
                generated[0].tolist()
            )
        )

    print("*" * 80)


if __name__ == "__main__":
    main()