from pathlib import Path
from collections import defaultdict


def read_txt(path: str | Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


def build_corpus(text: str, tokenizer: ByteTokenizer) -> list[list[int]]:
    corpus = []

    for line in text.splitlines():
        if line.strip():
            corpus.append(tokenizer.encode(line))

    return corpus


def get_pair_counts(corpus: list[list[int]]) -> dict[tuple[int, int], int]:
    counts = defaultdict(int)

    for sequence in corpus:
        for pair in zip(sequence, sequence[1:]):
            counts[pair] += 1

    return counts


def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_symbol: int,
) -> list[int]:
    """
    Replace every non-overlapping occurrence of pair
    with new_symbol.
    """

    merged = []

    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and (sequence[i], sequence[i + 1]) == pair
        ):
            merged.append(new_symbol)
            i += 2

        else:
            merged.append(sequence[i])
            i += 1

    return merged



# if __name__ == "__main__":

#     tokenizer = ByteTokenizer()
#     text = read_txt("data/corpus.txt")
#     corpus = build_corpus(text, tokenizer)
#     merges = train_bpe(
#         corpus,
#         vocab_size=300,
#     )
#     print("\nLearned merges:\n")
#     for pair, token in merges.items():
#         print(pair, "->", token)
#     print("\nFinal corpus:\n")
#     for sequence in corpus:
#         print(sequence)