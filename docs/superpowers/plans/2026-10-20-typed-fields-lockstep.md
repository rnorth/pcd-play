# Typed Number/Date Fields via Lockstep Decoding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add grammar-constrained, nullable `number`, `date` and `datetime` field types to the MLX parallel engine, decoded for all fields in lockstep.

**Architecture:** One PR, five commits: golden snapshots → `BatchKVCache` swap + lockstep loop → model-free string grammars → schema types → engine wiring + demo preset. Grammars choose among single-token symbols; the lockstep loop advances every unfinished field one token per batched forward pass.

**Tech Stack:** Python 3.12 (mise), `mlx` / `mlx-lm 0.32.x`, `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-typed-fields-lockstep-design.md`

## Global Constraints

- Branch `feat/number-date-fields`; one PR. MLX only — `core/engine_torch.py` raises `NotImplementedError` for new types.
- `mlx-lm>=0.32.0` in `requirements.txt` and `requirements-mlx.txt`.
- Enum/boolean-only schemas: prompt byte-for-byte unchanged; outputs identical `value`s, `prob`s within `1e-3`.
- Generative-field suffix: `'  "{fname}":'` (no trailing space).
- Null rule appended to the system prompt **only** when the schema has a generative field: `\nUse null for any attribute the text does not state.`
- Symbols (each must encode to exactly one token, else `ValueError`): `0`–`9`, `-`, `.`, `T`, `:`, `' "'`, `' '`, `NULL = ' null'`, `END = ',\n'`.
- `number.decimals`: int 0–6, default 0. Integer part ≤ 15 digits.
- Lockstep hard cap `MAX_STEPS = 32` (module constant); exceeding it → `RuntimeError` (grammar bug).
- No `Protocol` classes — duck typing (`.step()`, `.allowed()`, `.parse()`).

## Review Focus

1. **Text unrelated to the schema** → every new-type field `null`, `p_null > 0.5`. Test: Task 5 `test_unrelated_text_is_null`.
2. **Schema with only one new-type field** (no single-token fields; M=1) → works. Test: Task 5 `test_single_generative_field`.
3. **Collision enum + new type in one schema** (resolver slices the cache *before* `finalize()` rolls it) → enum values equal golden. Test: Task 5 `test_collision_enum_alongside_number`.
4. **Thousands separators in the text** (`£2,350.50`) → `2350.5`. Test: Task 5 `test_insurance_claim_values`.
5. **Non-Qwen tokenizer** → `ValueError` at vocab build. Test: Task 3 `test_symbol_vocab_rejects_multi_token_symbol`.

Known limitation (documented, not tested): subtle absence — e.g. `police_report_date` in a claim that never mentions police — gets `p_null ≈ 0.02` and a hallucinated date.

---

### Task 1: Test harness + golden snapshots of today's behaviour

**Files:**
- Create: `requirements-dev.txt` (`-r requirements-mlx.txt`, `pytest`), `tests/conftest.py`, `tests/test_presets_golden.py`, `tests/golden/<preset_id>.json` (×5)

**Interfaces:**
- Produces: session fixture `engine -> (model, tokenizer)` via `core.engine_mlx.get_engine()`; helper `load_preset(preset_id) -> dict`.

- [ ] **Step 1:** `git switch -c feat/number-date-fields`; `.venv/bin/pip install -r requirements-dev.txt`.
- [ ] **Step 2:** Write `test_presets_match_golden`, parametrised over presets that have a golden file: run `run_parallel_generation(preset["context"], StructuredSchema(preset["schema"]))`; assert each field's `value` equals golden and `abs(prob - golden) < 1e-3`. With `UPDATE_GOLDEN=1`, write goldens for every preset instead and skip.
- [ ] **Step 3:** On **unchanged** code: `UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_presets_golden.py`, then `.venv/bin/pytest tests/test_presets_golden.py -v` → 5 PASS.
- [ ] **Step 4:** Commit: `test: add golden snapshots for parallel engine presets`.

### Task 2: `BatchKVCache` broadcast + lockstep loop

**Files:**
- Create: `core/lockstep.py`, `tests/test_lockstep.py`
- Modify: `core/engine_mlx.py` (warmup in `get_engine`, broadcast in `run_parallel_generation`, cache construction in `_resolve_collision_mlx`), `requirements.txt`, `requirements-mlx.txt`

**Interfaces:**
- Produces:
  - `broadcast_batch_cache(prefix_cache: list[KVCache], suffix_lengths: list[int]) -> list[BatchKVCache]` — per layer: `BatchKVCache([0]*M)`; keys/values = `mx.repeat(prefix[..., :offset, :], M, 0)`; `_idx = offset`; `offset = mx.array([offset]*M)`; then `prepare(right_padding=[max(suffix_lengths) - l for l in suffix_lengths])`.
  - `lockstep_decode(model, cache: list[BatchKVCache], first_logits: mx.array, rows: list) -> int` — each row has `step(logits: mx.array) -> Optional[int]`: record a choice from this row's `[V]` logits, return the token to feed next, or `None` when complete (the final choice is **not** fed). Returns forward passes used. Precondition: caller already ran `finalize()` and filtered the cache to exactly these rows, in order.

- [ ] **Step 1: Write failing tests** in `tests/test_lockstep.py`. Helpers: `GreedyRow(n)` — `step` records the argmax; returns it while fewer than `n` tokens are recorded, `None` once the `n`th is recorded. `reference_decode(model, prompt_toks, n) -> list[int]` — plain `make_prompt_cache`, greedy, one row.
  - `test_lockstep_matches_sequential`: base prompt from the `support_triage` preset + suffixes `'  "a": "'`, `'  "requires_executive_escalation": "'`, `'  "x":'`; `GreedyRow(24)` each; each row's tokens `== reference_decode(...)`.
  - `test_rows_finishing_at_different_steps`: same suffixes, `GreedyRow(3)`, `GreedyRow(20)`, `GreedyRow(10)`; each equals its reference's first n tokens; returned passes `== 19`.
- [ ] **Step 2:** `.venv/bin/pytest tests/test_lockstep.py -v` → FAIL (`ModuleNotFoundError: core.lockstep`).
- [ ] **Step 3: Implement `core/lockstep.py`:**

```python
active, logits, passes = list(range(len(rows))), first_logits, 0
for _ in range(MAX_STEPS):
    keep, nxt = [], []
    for j, r in enumerate(active):
        tok = rows[r].step(logits[j])
        if tok is not None:
            keep.append(j); nxt.append(tok)
    if not keep:
        return passes
    if len(keep) < len(active):
        for c in cache: c.filter(mx.array(keep))
    active = [active[j] for j in keep]
    logits = model(mx.array(nxt)[:, None], cache=cache)[:, -1, :]
    passes += 1
raise RuntimeError(...)
```

- [ ] **Step 4:** `.venv/bin/pytest tests/test_lockstep.py -v` → PASS.
- [ ] **Step 5: Engine uses the new cache.** `run_parallel_generation`: replace the `mx.repeat` broadcast block with `broadcast_batch_cache(cache, suffix_lengths)`. `get_engine` warmup: same (28 rows, suffix length 6). `_resolve_collision_mlx`: build each layer's per-row cache as a fresh `mlx_lm.models.cache.KVCache()` with sliced `keys`/`values` (`[row:row+1, :, :true_len, :]`, valid because `finalize()` has not run) and `offset = true_len`, instead of `copy.copy(c)`. Bump `mlx-lm>=0.32.0` in both requirements files.
- [ ] **Step 6:** `.venv/bin/pytest -v` → all PASS (golden proves no behaviour change, incl. `smart_home_voice` collisions).
- [ ] **Step 7:** Commit: `feat: batch cache with per-row padding and lockstep decode loop`.

### Task 3: Symbol vocab + grammars (model-free)

**Files:**
- Create: `core/grammars.py`, `tests/test_grammars.py`

**Interfaces:**
- Produces:
  - `NULL = " null"`, `END = ",\n"`; `SYMBOLS` = the Global Constraints list.
  - `build_symbol_vocab(tokenizer) -> dict[str, int]` — `{s: tokenizer.encode(s, add_special_tokens=False)[0]}`; `ValueError` naming the symbol if any encodes to ≠ 1 token.
  - `NumberGrammar(decimals: int)`, `DateGrammar(with_time: bool)` — `allowed(text: str) -> list[str]` (symbols allowed after `text`; `[]` = complete) and `parse(text: str) -> int | float | str`. `text` = concatenation of chosen symbols excluding `END`; if `NULL` is chosen, `parse` is not called.
  - `make_grammar(fdef) -> NumberGrammar | DateGrammar` (by `fdef.field_type`, `fdef.decimals`).

- [ ] **Step 1: Write failing tests** (compare `allowed()` as sets; `D` = digits `0`–`9`):
  - Number, `decimals=0`: `""` → `{NULL, " "}`; `" "` → `D ∪ {-}`; `" -"` → `D`; `" 0"` → `{END}`; `" 12"` → `D ∪ {END}`; `" "` + 15 digits → `{END}`; `parse(" -42") == -42` (int).
  - Number, `decimals=2`: `" 12"` → `D ∪ {., END}`; `" 1."` → `D`; `" 1.2"` → `D ∪ {END}`; `" 1.23"` → `{END}`; `parse(" 1.5") == 1.5`; `parse(" 3") == 3.0` (float).
  - Date (`Q = ' "'`): `""` → `{NULL, Q}`; `Q` → `{1, 2}`; `Q+"2024"` → `{-}`; `Q+"2024-"` → `{0, 1}`; `Q+"2024-0"` → `D − {0}`; `Q+"2024-1"` → `{0, 1, 2}`; `Q+"2024-02-"` → `{0, 1, 2}`; `Q+"2024-02-0"` → `D − {0}`; `Q+"2024-02-2"` → `D` (leap); `Q+"2023-02-2"` → `D − {9}`; `Q+"2024-04-3"` → `{0}`; `Q+"2024-01-3"` → `{0, 1}`; `Q+"2024-10-20"` → `[]`; `parse(Q+"2024-10-20") == "2024-10-20"`.
  - Datetime: `Q+"2024-10-20"` → `{T}`; `…T` → `{0, 1, 2}`; `…T2` → `{0, 1, 2, 3}`; `…T23:` → `{0..5}`; `Q+"2024-10-20T23:59:59"` → `[]`; `parse` returns `"2024-10-20T23:59:59"`.
  - `test_symbol_vocab_qwen`: real Qwen tokenizer (`transformers.AutoTokenizer.from_pretrained(MODEL_ID)`, no model load) → every symbol present.
  - `test_symbol_vocab_rejects_multi_token_symbol`: fake tokenizer whose `encode("5")` returns two ids → `ValueError` mentioning `'5'`.
- [ ] **Step 2:** `.venv/bin/pytest tests/test_grammars.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** Date/datetime: positional template after `Q`: `YYYY-MM-DD` (+ `THH:MM:SS`); literal positions allow only their char. Year `[1-2]DDD`. Day digits use `max_day = calendar.monthrange(y, m)[1]`:

```python
# first day digit:            0 .. max_day // 10
# second day digit, given d1: d1 == 0 → 1..9
#                             d1 == max_day // 10 → 0 .. max_day % 10
#                             else → 0..9
```
Hour: first `0–2`; second `0–3` if first is `2`, else `0–9`. Min/sec: first `0–5`. `NULL` only when `text == ""`.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_grammars.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: add number/date/datetime token grammars`.

### Task 4: Schema accepts the new types

**Files:**
- Modify: `core/schema.py`, `core/engine_torch.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Produces: `FieldDefinition(name, field_type, description, choices=None, decimals=0)`; property `is_generative -> bool` (True for `number`/`date`/`datetime`; their `choices is None`); `StructuredSchema` passes `spec.get("decimals", 0)`; property `StructuredSchema.has_generative -> bool`. In `compile_parallel_metadata`, generative fields get suffix `'  "{fname}":'`, `cands=[]`, `seqs=[]`, collision `False`.

- [ ] **Step 1: Write failing tests:** `test_number_field_defaults` (`decimals == 0`, `is_generative`, `choices is None`); `test_decimals_out_of_range` (`7` and `-1` → `ValueError`); `test_json_schema_prompt_new_types` — `to_json_schema_prompt_str()` contains `'"amount": number | null'`, `'"when": "YYYY-MM-DD" | null'`, `'"at": "YYYY-MM-DDTHH:MM:SS" | null'`; `test_generative_suffix_has_no_trailing_space`; `test_unknown_type_message_lists_all_types`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `to_dict` emits `choices: None`, `cardinality: None` for generative fields. `engine_torch.run_parallel_generation_torch`: `if schema.has_generative: raise NotImplementedError("number/date/datetime fields need the MLX backend")`.
- [ ] **Step 4:** `.venv/bin/pytest tests/test_schema.py tests/test_presets_golden.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: accept number/date/datetime field definitions`.

### Task 5: Engine decodes the new types + demo preset

**Files:**
- Modify: `core/grammars.py` (add `GrammarRow`), `core/engine_mlx.py` (`get_engine`, `run_parallel_generation`, naive validation in `run_naive_generation` and `stream_naive_generation`), `README.md`
- Create: `presets/insurance_claim.json`, `tests/test_engine_typed_fields.py`

**Interfaces:**
- Consumes: `broadcast_batch_cache`, `lockstep_decode` (Task 2); `build_symbol_vocab`, `make_grammar`, `NULL`, `END` (Task 3); `is_generative`, `has_generative` (Task 4).
- Produces: `GrammarRow(grammar, vocab: dict[str, int], temperature: float)` with `step()` per Task 2, and attributes `value`, `confidence: float`, `p_null: float` once finished. `step`: `syms = grammar.allowed(text)`; softmax over `logits[vocab[s]] / temperature`; take argmax `s` with prob `p`; `confidence *= p`; on the first step `p_null = prob(NULL)`; if `s` is `NULL` → `value = None`, return `None`; if `s` is `END` → `value = parse(text)`, return `None`; else `text += s`; if `grammar.allowed(text) == []` → `value = parse(text)`, return `None`; else return `vocab[s]`. Module global `_symbol_vocab` built once in `get_engine()`.

- [ ] **Step 1: Create `presets/insurance_claim.json`** (`id: "insurance_claim"`, title "Insurance Claim Intake (6 Fields)"). Context:

  > Claim CLM-88213. Policyholder Dana Okafor reports a rear-end collision on 14 March 2026 at about 5:40pm on the A40. Repair estimate from Westway Motors: £2,350.50. Excess on the policy is £250. Two passengers, no injuries reported. Claim submitted online 2026-03-16 at 09:12:30.

  Schema: `incident_date` (date), `submitted_at` (datetime), `repair_estimate_gbp` (number, `decimals: 2`), `passenger_count` (number), `claim_type` (enum: `COLLISION`, `THEFT`, `FIRE`, `WEATHER`, `VANDALISM`), `injuries_reported` (boolean).
- [ ] **Step 2: Write failing tests:**
  - `test_insurance_claim_values`: `incident_date == "2026-03-14"`, `submitted_at == "2026-03-16T09:12:30"`, `repair_estimate_gbp == 2350.5`, `passenger_count == 2`, `claim_type == "COLLISION"`, `injuries_reported is False`; each new-type field's telemetry has `type`, `confidence`, `p_null`.
  - `test_unrelated_text_is_null`: insurance schema on `"Hi team, the offsite is moving to next quarter, details to follow. Cheers, Sam"` → every new-type value `None`, `p_null > 0.5`.
  - `test_single_generative_field`: schema `{"incident_date": date}` on the claim context → `"2026-03-14"`.
  - `test_collision_enum_alongside_number`: `smart_home_voice` schema + `{"brightness_pct": number}` on its context → enum/boolean **values** (not probs — the prompt changed) equal `tests/golden/smart_home_voice.json`; `brightness_pct` is `None` or an `int`.
  - `test_naive_does_not_flag_typed_fields`: `run_naive_generation` on `insurance_claim` → `invalid_enums` has no new-type field names.
  - No golden file for `insurance_claim` (Task 1's test skips presets without one).
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4: Implement in `run_parallel_generation`:** if `schema.has_generative`, append the null rule to the system prompt. Collect generative row indices `gen` (schema order). Single-token and collision handling skip `gen` rows and run first (collisions need the un-rolled cache). Then if `gen`: `for c in b_cache: c.finalize(); c.filter(mx.array(gen))`; `first = mx.stack([suffix_out[i, suffix_lengths[i] - 1] for i in gen])`; one `GrammarRow(make_grammar(fdef), _symbol_vocab, temperature)` per field; `passes = lockstep_decode(model, b_cache, first, rows)`; fill `parsed_json` (`value`, `prob = confidence`) and `field_telemetry` (`value`, `type`, `confidence`, `p_null`); `sequential_forward_passes = 1 + collision_passes + passes`; restore declared field order (existing block, now unconditional). Naive validation: skip `fdef.is_generative` fields in the `invalid_enums` check (both functions).
- [ ] **Step 5:** `.venv/bin/pytest -v` → all PASS.
- [ ] **Step 6: README:** document `number` (`decimals`), `date`, `datetime`; the `value: null` / `p_null` contract and its limitation (catches unrelated text, not subtle absence — spec's p_null table); latency (one batched decode pass per step; datetime ≈ 20 steps). Run `bash run.sh`, pick *Insurance Claim Intake*, confirm the parallel pane shows the values.
- [ ] **Step 7:** Commit: `feat: decode number/date/datetime fields in lockstep`.

**PR description note:** `test_collision_enum_alongside_number` checks enum values, not probs, because adding a generative field changes the prompt (null rule + catalogue).
