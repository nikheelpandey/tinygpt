import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import GPTEmbedding
from model.block import TransformerBlock
from model.kv_cache import KVCache


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.context_size = context_size

        # Token + Position Embeddings
        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_size=context_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Language Modeling Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        pos_offset: int = 0,
    ):
        """
        Args:
            x:
                (batch_size, sequence_length)

            targets:
                (batch_size, sequence_length)

            kv_cache:
                Cached keys/values for incremental decoding.

            pos_offset:
                Absolute position of the first token in x.

        Returns:
            logits:
                (batch_size, sequence_length, vocab_size)

            loss:
                Cross-entropy loss if targets are provided.

            kv_cache:
                Updated cache during generation.
        """

        x = self.embedding(
            x,
            pos_offset=pos_offset,
        )

        new_kv_cache = None

        if kv_cache is not None:
            new_kv_cache = []

            for i, block in enumerate(self.blocks):
                layer_cache = (
                    kv_cache[i]
                    if i < len(kv_cache)
                    else []
                )

                x, layer_new_cache = block(
                    x,
                    layer_cache,
                )

                new_kv_cache.append(layer_new_cache)

        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )

        if kv_cache is not None:
            return logits, loss, new_kv_cache

        return logits, loss

    # ---------------------------------------------------------------
    # Shared sampling primitive.
    #
    # Pure "given the context so far, produce the next token" logic --
    # no model-mode side effects (no self.eval()) and no knowledge of
    # tokenizers. Both generate() and generate_stream() are thin loops
    # built on top of this.
    #
    # Kept private for now since only generate()/generate_stream() call
    # it. If a chat loop, inference server, or beam-search variant is
    # added later that needs it directly, this is a natural candidate
    # to make public (drop the leading underscore).
    # ---------------------------------------------------------------
    @torch.no_grad()
    def _sample_next_token(
        self,
        idx: torch.Tensor,
        kv_cache: KVCache | None,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
        use_kv_cache: bool,
        seen_tokens: list[set] | None = None,
    ):
        """
        Runs one decoding step: forward pass + sampling for a single
        new token.

        Args:
            idx:
                (batch_size, sequence_length_so_far)

            kv_cache:
                Current cache, or None (no cache yet / caching disabled).

            seen_tokens:
                Optional list of length batch_size, where seen_tokens[b]
                is the set of token ids already generated for sequence b.
                Used for repetition_penalty instead of recomputing
                idx[b].unique() every step -- the caller is responsible
                for building this set from the prompt and updating it
                with each newly sampled token (see generate() /
                generate_stream()). If None, falls back to
                idx[b].unique().

        Returns:
            next_token:
                (batch_size, 1)

            kv_cache:
                Updated cache (unchanged / None if use_kv_cache=False).
        """

        # ---------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------
        if use_kv_cache and kv_cache is not None:

            if idx.size(1) > self.context_size:
                # Context window has slid.
                # Rebuild the KV cache from scratch.
                idx_cond = idx[:, -self.context_size:]

                logits, _, kv_cache = self(
                    idx_cond,
                    kv_cache=[],
                    pos_offset=0,
                )

            else:
                # Decode only the newest token.
                idx_cond = idx[:, [-1]]

                logits, _, kv_cache = self(
                    idx_cond,
                    kv_cache=kv_cache,
                    pos_offset=idx.size(1) - 1,
                )

        elif use_kv_cache:

            # First forward pass (prefill).
            idx_cond = idx[:, -self.context_size:]

            logits, _, kv_cache = self(
                idx_cond,
                kv_cache=[],
                pos_offset=0,
            )

        else:

            # Full forward pass every iteration.
            idx_cond = idx[:, -self.context_size:]

            logits, _ = self(idx_cond)

        # Only keep logits for the next token prediction.
        logits = logits[:, -1, :]

        # ---------------------------------------------------------
        # Temperature
        # ---------------------------------------------------------
        logits = logits / temperature

        # ---------------------------------------------------------
        # Repetition Penalty
        #
        # Penalize every token that has already appeared in the
        # generated sequence.
        #
        # Positive logits:
        #     logit /= penalty
        #
        # Negative logits:
        #     logit *= penalty
        #
        # penalty = 1.0 disables this feature.
        # ---------------------------------------------------------
        if repetition_penalty != 1.0:

            for batch_idx in range(idx.size(0)):

                if seen_tokens is not None:
                    # O(distinct tokens) instead of O(sequence length).
                    #
                    # TODO: this still allocates a fresh tensor from a
                    # Python set every step (Python-object overhead +
                    # a host->device copy). For long-context / hot-path
                    # use, replace seen_tokens with a persistent
                    # boolean mask of shape (vocab_size,) per batch
                    # element, set via `seen[next_token] = True` after
                    # each step, with previous_tokens obtained from
                    # `seen.nonzero(as_tuple=True)[0]` (or the penalty
                    # applied directly through the mask). GPU-friendly
                    # and allocation-free. Not worth it yet.
                    previous_tokens = torch.tensor(
                        list(seen_tokens[batch_idx]),
                        device=logits.device,
                        dtype=torch.long,
                    )
                else:
                    previous_tokens = idx[batch_idx].unique()

                if previous_tokens.numel() == 0:
                    continue

                token_logits = logits[
                    batch_idx,
                    previous_tokens,
                ]

                logits[
                    batch_idx,
                    previous_tokens,
                ] = torch.where(
                    token_logits < 0,
                    token_logits * repetition_penalty,
                    token_logits / repetition_penalty,
                )

        # ---------------------------------------------------------
        # Top-k
        # ---------------------------------------------------------
        if top_k is not None:

            values, _ = torch.topk(
                logits,
                k=top_k,
            )

            logits[
                logits < values[:, [-1]]
            ] = float("-inf")

        # Convert logits into probabilities.
        probs = F.softmax(
            logits,
            dim=-1,
        )

        # ---------------------------------------------------------
        # Top-p (Nucleus Sampling)
        # ---------------------------------------------------------
        if top_p is not None:

            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending=True,
                dim=-1,
            )

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            # Remove everything after cumulative probability > top_p.
            sorted_mask = cumulative_probs > top_p

            # Keep the first token that crosses the threshold.
            sorted_mask[..., 1:] = (
                sorted_mask[..., :-1].clone()
            )
            sorted_mask[..., 0] = False

            # Scatter mask back into original vocabulary order.
            mask = torch.zeros_like(
                sorted_mask,
                dtype=torch.bool,
            )

            mask.scatter_(
                dim=-1,
                index=sorted_indices,
                src=sorted_mask,
            )

            probs = probs.masked_fill(
                mask,
                0.0,
            )

            # Renormalize probabilities.
            probs = probs / probs.sum(
                dim=-1,
                keepdim=True,
            )

        # ---------------------------------------------------------
        # Sample next token
        # ---------------------------------------------------------
        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        return next_token, kv_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
    ):
        """
        Autoregressive text generation (blocking -- returns only once
        all max_new_tokens have been generated).

        Decoding pipeline (per step, see _sample_next_token):

            Forward pass
                ↓
            Temperature
                ↓
            Repetition penalty
                ↓
            Top-k
                ↓
            Softmax
                ↓
            Top-p
                ↓
            Sample next token

        For token-by-token streaming, use generate_stream() instead.
        """

        self.eval()

        kv_cache = None

        # Maintain seen-token sets incrementally instead of scanning
        # the whole sequence with .unique() every step.
        seen_tokens = None
        if repetition_penalty != 1.0:
            seen_tokens = [
                set(idx[b].tolist())
                for b in range(idx.size(0))
            ]

        for _ in range(max_new_tokens):

            next_token, kv_cache = self._sample_next_token(
                idx,
                kv_cache,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                use_kv_cache,
                seen_tokens,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )

            if seen_tokens is not None:
                for b in range(idx.size(0)):
                    seen_tokens[b].add(next_token[b].item())

        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        eos_token_id: int | None = None,
    ):
        """
        Streaming autoregressive text generation. A generator that
        yields as soon as each new token is sampled, instead of
        waiting for the full max_new_tokens to complete.

        Uses the exact same per-step logic as generate() via
        _sample_next_token, so outputs are distributionally identical
        -- this only changes when you get to see the tokens.

        The model stays tokenizer-agnostic: it only ever deals in
        tensors. Decode outside, e.g.:

            for next_token in model.generate_stream(idx, 200):
                print(
                    tokenizer.decode([next_token.item()]),
                    end="",
                    flush=True,
                )

        Only the newest token is yielded each step (not the whole
        growing context) -- yielding the full sequence every step
        would mean transferring O(1 + 2 + ... + N) = O(N^2) token ids
        over N steps, which gets expensive for long generations. If
        the caller needs the full context (e.g. for stop-sequence
        checks), maintain it on the caller side the same way
        generate() does:

            context = idx
            for next_token in model.generate_stream(idx, 200):
                context = torch.cat((context, next_token), dim=1)

        Args:
            idx:
                (1, sequence_length) prompt tokens. Streaming only
                supports batch_size == 1 -- a single interleaved
                stream doesn't make sense for multiple sequences
                generated in lockstep.

            eos_token_id:
                Optional. If the sampled token equals this id,
                generation stops after yielding it.

        Yields:
            next_token:
                (1, 1) -- the newest sampled token id, one per step.
        """

        self.eval()

        if idx.size(0) != 1:
            raise ValueError(
                "generate_stream only supports batch_size=1 "
                f"(got batch_size={idx.size(0)}). Use generate() for "
                "batched generation."
            )

        kv_cache = None

        seen_tokens = None
        if repetition_penalty != 1.0:
            seen_tokens = [set(idx[0].tolist())]

        for _ in range(max_new_tokens):

            next_token, kv_cache = self._sample_next_token(
                idx,
                kv_cache,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                use_kv_cache,
                seen_tokens,
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1,
            )

            if seen_tokens is not None:
                seen_tokens[0].add(next_token[0].item())

            yield next_token

            if eos_token_id is not None and next_token.item() == eos_token_id:
                return