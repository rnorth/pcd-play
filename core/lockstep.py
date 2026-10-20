"""
Lockstep multi-token decoding across many fields that share one prefilled prompt.

Each field is a row in an mlx_lm BatchKVCache. Every step asks each unfinished row
for its next token and feeds them all in one (M, 1) forward pass, so decoding costs
about as many passes as the longest field rather than the sum of all fields.
"""

from typing import List

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

# Grammars finish well inside this (datetime is 20 symbols); hitting it means a bug.
MAX_STEPS = 32


def broadcast_batch_cache(prefix_cache, suffix_lengths: List[int]) -> List[BatchKVCache]:
    """
    Copies a single-row prefix cache into M rows, ready for a right-padded suffix batch.

    After the suffix pass, call finalize() on each layer to roll each row's padding to
    the left, where it is masked and RoPE offsets are per row, before decoding further.
    """
    m = len(suffix_lengths)
    width = max(suffix_lengths)
    out = []
    for c in prefix_cache:
        bc = BatchKVCache([0] * m)
        bc.keys = mx.repeat(c.keys[..., :c.offset, :], m, axis=0)
        bc.values = mx.repeat(c.values[..., :c.offset, :], m, axis=0)
        bc._idx = c.offset
        bc.offset = mx.array([c.offset] * m)
        bc.prepare(right_padding=[width - l for l in suffix_lengths])
        out.append(bc)
    return out


def lockstep_decode(model, cache: List[BatchKVCache], first_logits: mx.array, rows: list) -> int:
    """
    Advances every row until each one's step(logits) returns None. Returns the number
    of forward passes used. The cache must already be finalized and hold exactly these
    rows, in order; finished rows are filtered out of it as they complete.
    """
    active, logits, passes = list(range(len(rows))), first_logits, 0
    for _ in range(MAX_STEPS):
        keep, nxt = [], []
        for j, r in enumerate(active):
            tok = rows[r].step(logits[j])
            if tok is not None:
                keep.append(j)
                nxt.append(tok)
        if not keep:
            return passes
        if len(keep) < len(active):
            for c in cache:
                c.filter(mx.array(keep))
        active = [active[j] for j in keep]
        logits = model(mx.array(nxt)[:, None], cache=cache)[:, -1, :]
        passes += 1
    raise RuntimeError(f"lockstep decode did not finish within {MAX_STEPS} steps")
