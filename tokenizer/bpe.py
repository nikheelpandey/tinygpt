from tokenizer.utils import ByteTokenizer, merge_pair


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int]):
        self.tokenizer = ByteTokenizer()
        self.merges = merges  # pair -> new_symbol; insertion order == learned rank

        # Reverse lookup: token_id -> pair
        self.reverse_merges = {
            new_symbol: pair
            for pair, new_symbol in merges.items()
        }

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text)

        if len(tokens) < 2:
            return tokens

        while True:
            # Only consider pairs that are actually present in the sequence
            pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

            # Among those, find the one that was learned earliest (lowest rank)
            candidate = min(
                pairs,
                key=lambda p: self.merges.get(p, float("inf")),
                default=None,
            )

            if candidate is None or candidate not in self.merges:
                break  # no mergeable pair left

            new_symbol = self.merges[candidate]
            tokens = merge_pair(tokens, candidate, new_symbol)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        bytes_ = []
        cache: dict[int, list[int]] = {}

        def expand(token: int) -> list[int]:
            if token < 256:
                return [token]
            if token in cache:
                return cache[token]
            left, right = self.reverse_merges[token]
            result = expand(left) + expand(right)
            cache[token] = result
            return result

        for token in tokens:
            bytes_.extend(expand(token))

        return self.tokenizer.decode(bytes_)