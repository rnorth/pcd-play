# Typed number/date fields via lockstep decoding — Design

**Status:** draft · **Date:** 2026-10-20 · **Backend:** MLX only

## Problem

The parallel engine only supports `boolean` and `enum` fields: each is decided by one
next-token distribution read from a single batched forward pass. Numbers and dates need
several tokens per field, which the current right-padded broadcast cache cannot decode
in a batch (later tokens would attend to padding and get wrong RoPE positions).

## Goals

1. Decode multi-token values for many fields **in parallel** ("lockstep"): one `(M, 1)`
   forward pass per step, cost ≈ longest field, not sum of fields.
2. Add field types `number`, `date`, `datetime`, each **grammar-constrained** (output is
   always a valid value) and **nullable** with a calibrated `p_null`.
3. Leave `boolean`/`enum` results unchanged (bit-for-bit values, probs within 1e-3).

## Non-goals (for now)

- Free-text fields (follow-up PR; reuses the lockstep loop).
- PyTorch backend (new types raise `NotImplementedError` there).
- Non-Qwen tokenizers (number/date grammars rely on one-token-per-digit).
- Time zones, units, min/max bounds, computed values, folding forced separators into
  the previous pass.

## Design

### Cache: `BatchKVCache` with right-padding roll (proven in spike)

- Broadcast the single prefilled prefix into an `mlx_lm` `BatchKVCache` per layer,
  then `prepare(right_padding=…)` and run the existing right-padded suffix batch.
  First-step logits are identical to today's engine.
- Before multi-token decoding, `finalize()` rolls each row's padding to the left, where
  `left_padding` masks it and per-row `offset` fixes RoPE positions.
- `filter(rows)` drops finished rows so later steps get cheaper.
- Spike result (M1, Qwen2.5-1.5B-4bit, 650-token prefix): lockstep tokens identical to
  per-row sequential decode over 64 steps; 19 ms/step at M=1, 95 ms/step at M=28.

### Lockstep loop

Each multi-token field is a *row decoder* with `step(logits) -> next token | None`.
The loop asks every active row for its next token, drops rows that returned `None`,
feeds the rest in one forward pass, repeats.

### Prompt and suffix

- Generative fields use the suffix `'  "{f}":'` (**no trailing space**). Spike: with a
  trailing space the model skips the opening quote and writes dates unquoted; without it
  the canonical next token is `' "'` (date) or `' '` (number, since Qwen splits digits).
- When a schema has any generative field, the system prompt gains
  `Use null for any attribute the text does not state.` (enum/boolean-only schemas keep
  today's prompt byte-for-byte).

### Grammars (string-level, model-free)

A grammar answers "given the text so far, which symbols may come next?" Each symbol is a
literal string that must encode to exactly one token (one id per symbol, no variants):
`0`–`9`, `-`, `.`, `T`, `:`, `' "'`, `' '`, `' null'` (NULL), `',\n'` (END).
Softmax over the allowed symbols' logits gives the step probability.

| Type | Symbols emitted | Rules |
|---|---|---|
| `number` (`decimals` 0–6, default 0) | `' '` then `12` / `-3.5`, or NULL | optional `-`; 1–15 integer digits, no leading zeros; if `decimals>0`, optional `.` + 1..`decimals` digits; END terminates |
| `date` | `' "'` then `2026-10-20`, or NULL | `YYYY-MM-DD`; month 01–12; day 01–(days in that month/year); complete after 10 chars (closing quote never generated) |
| `datetime` | `' "'` then `2026-10-20T13:45:00`, or NULL | `date` + `THH:MM:SS`, hour 00–23, min/sec 00–59; no zone |

- Step 1 always offers NULL; its probability is `p_null`. If NULL wins, value is `null`.
- `confidence` = product of the chosen symbols' step probabilities (forced symbols = 1).
- Thousands separators need no support: constrained to digits, the model decodes
  `£2,350.50` as `2350.50` (spike).

### p_null evaluation (spike, with the null rule)

| Text | Field absent? | p_null |
|---|---|---|
| Unrelated email / invoice | all claim fields | 0.89–1.00 ✅ |
| Claim text | `police_report_date` (absent) | 0.018 ❌ hallucinates the incident date |
| Claim text | present fields | 0.00–0.04 ✅ |

**p_null catches gross absence only.** Do not use it for routing subtle absences without
further work (e.g. per-field description hints, a bigger model). Documented in README.

### Output

`parsed_json[f] = {"value": <int|float|str|None>, "prob": confidence}`;
`field_telemetry[f]` adds `type`, `confidence`, `p_null`. `sequential_forward_passes`
counts lockstep passes. Naive-baseline validation stops treating new types as enums;
its schema prompt shows `number | null`, `"YYYY-MM-DD" | null`, etc.

## Risks

- Latency: each lockstep step ≈ one batched decode pass (~95 ms at M=28 on M1).
  Datetime = 20 steps. Acceptable for a demo; documented.
- Confidence is a product of step probabilities, not a calibrated softmax over values.
- `mlx-lm` floor rises to the tested `0.32.0` (`BatchKVCache.prepare/finalize/filter`).
