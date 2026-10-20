import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from conftest import load_preset
from core.prompt_builder import build_parallel_prompt
from core.schema import StructuredSchema

SUFFIXES = ['  "a": "', '  "requires_executive_escalation": "', '  "x":']


class GreedyRow:
    def __init__(self, n):
        self.n = n
        self.tokens = []

    def step(self, logits):
        self.tokens.append(int(mx.argmax(logits)))
        return self.tokens[-1] if len(self.tokens) < self.n else None


def base_prompt():
    preset = load_preset("support_triage")
    return build_parallel_prompt(preset["context"], StructuredSchema(preset["schema"]))


def reference_decode(model, prompt_toks, n):
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_toks)[None], cache=cache)[0, -1]
    out = []
    while True:
        out.append(int(mx.argmax(logits)))
        if len(out) == n:
            return out
        logits = model(mx.array([[out[-1]]]), cache=cache)[0, -1]


def run_lockstep(model, tokenizer, rows):
    from core.lockstep import broadcast_batch_cache, lockstep_decode

    base = tokenizer.encode(base_prompt())
    sufs = [tokenizer.encode(s, add_special_tokens=False) for s in SUFFIXES]
    lens = [len(s) for s in sufs]
    width = max(lens)
    padded = mx.array([s + [0] * (width - len(s)) for s in sufs], dtype=mx.int32)

    cache = make_prompt_cache(model)
    model(mx.array(base)[None], cache=cache)
    b_cache = broadcast_batch_cache(cache, lens)
    out = model(padded, cache=b_cache)
    first = mx.stack([out[i, lens[i] - 1] for i in range(len(sufs))])
    for c in b_cache:
        c.finalize()
    passes = lockstep_decode(model, b_cache, first, rows)
    return base, sufs, passes


def test_lockstep_matches_sequential(engine):
    model, tokenizer = engine
    rows = [GreedyRow(24) for _ in SUFFIXES]
    base, sufs, _ = run_lockstep(model, tokenizer, rows)
    for row, suf in zip(rows, sufs):
        assert row.tokens == reference_decode(model, base + suf, 24)


def test_rows_finishing_at_different_steps(engine):
    model, tokenizer = engine
    ns = [3, 20, 10]
    rows = [GreedyRow(n) for n in ns]
    base, sufs, passes = run_lockstep(model, tokenizer, rows)
    for row, suf, n in zip(rows, sufs, ns):
        assert row.tokens == reference_decode(model, base + suf, n)
    assert passes == 19
