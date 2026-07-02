import heapq
from collections import defaultdict

from tokenizer.utils import merge_pair


def _pairs(seq):
    return zip(seq, seq[1:])


def train_bpe(
    corpus: list[list[int]],
    vocab_size: int = 5000,
    min_frequency: int = 1,
) -> dict[tuple[int, int], int]:
    """
    Train a byte-level BPE tokenizer.
    """
    merges = {}
    corpus = [list(seq) for seq in corpus]  # mutable working copies

    # pair -> total count across corpus
    pair_counts = defaultdict(int)
    # pair -> set of sequence indices where it currently (or recently) occurs
    pair_seqs = defaultdict(set)

    for idx, seq in enumerate(corpus):
        for pair in _pairs(seq):
            pair_counts[pair] += 1
            pair_seqs[pair].add(idx)

    # Max-heap via negation, with lazy deletion of stale entries
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    while 256 + len(merges) < vocab_size:

        # Pop stale heap entries until the top actually matches current count
        best_pair = None
        best_count = 0
        while heap:
            neg_count, pair = heap[0]
            current = pair_counts.get(pair, 0)
            if current <= 0 or -neg_count != current:
                heapq.heappop(heap)
                continue
            best_pair, best_count = pair, current
            break

        if best_pair is None or best_count < min_frequency:
            print(f"Stopping: highest pair frequency = {best_count}")
            break

        heapq.heappop(heap)
        next_symbol = 256 + len(merges)

        # Only touch sequences that actually contain best_pair
        for idx in pair_seqs.get(best_pair, ()):
            seq = corpus[idx]

            # Remove this sequence's old pair contributions
            for pair in _pairs(seq):
                pair_counts[pair] -= 1

            new_seq = merge_pair(seq, best_pair, next_symbol)
            corpus[idx] = new_seq

            # Add back the new pair contributions
            for pair in _pairs(new_seq):
                pair_counts[pair] += 1
                pair_seqs[pair].add(idx)
                heapq.heappush(heap, (-pair_counts[pair], pair))

        pair_counts[best_pair] = 0
        merges[best_pair] = next_symbol

        # print(f"Learned token {next_symbol}: {best_pair} (count={best_count})")

    print(f"\nLearned {len(merges)} merges.")
    print(f"Final vocabulary size: {256 + len(merges)}")

    return merges, corpus