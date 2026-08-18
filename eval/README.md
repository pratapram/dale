# Evaluation harness

Not part of the `dale` package — this project's own research-validation tooling (the same role
`examples/` plays for demos), built on `dale.agent` directly (no shortcut around real
`dispatch`/tool-calling). It implements the evaluation design in `DESIGN.md` Section 5.

## What's here

- `harness.py` — `run_trial`/`run_trials`: runs an agent N times against a use case, checks each
  run against ground truth by inspecting registry state after the run (never by parsing the
  model's free-text answer), and reports success rate, failure-mode categorization (by
  `DaleError` code), the wasted-turn-rate metric (fraction of logged operation calls that were
  rejected — `cost_gate_exceeded` doesn't count, that's a correct safety response, not a mistake),
  and per-trial token usage.

  **Token accounting** (`TrialResult.usage`, `TrialSummary.usage_stats()`, and a line in
  `render()`) is the instrument for `paper.md` Section 4.2 part (C). It reports two different
  input-token figures — cumulative spend and *peak* single-request context — because only the
  second tests a bounded-context claim; see `dale.agent.TokenUsage` for the full reasoning. Means
  are taken over measured trials only, and `TestModel` runs are excluded outright (its usage is
  synthetic), so a free structural batch can't contribute a fake zero. Peak context is reported as
  a **max**, not a mean: the number that tests a bound is the largest one observed. Failed trials
  still report cumulative spend, via a caller-owned accumulator passed to `run_sync(usage=...)` —
  reading `result.usage` would lose exactly the runs (a repetition loop, an exhausted budget) whose
  cost is most worth knowing.

  **Per-call timing** rides along in the same entries (`elapsed_ms`, plus `auto_inspect_ms` for
  `peek_at_every_step`'s share), summed by `ActionLog.total_host_ms` and reported by
  `TrialSummary.timing_stats()` as a host-compute vs. model-and-network split. Unlike token stats
  this covers every trial including `TestModel` ones — elapsed time is real whatever produced it.
  A measured 10,005-row trial: `26.7s wall — 17.9 ms host compute (0.07%), 26.7s model + network`.
  The value of tracking it is not the current number but that it *stays* a rounding error as
  datasets grow, since operation cost scales with rows and a round trip doesn't.

  **Checkers grade the answer, not the route.** A correct result counts however the model shaped
  it — a list of records carrying `account_id`, or `dict_diff` rows keyed `key` with the record
  nested under `previous_value`. Route quality is a separate axis, already measured by
  wasted-turn rate and call count; folding it into correctness would report "wrong" for a run
  that answered correctly by a different valid pipeline. `_identifier_sets` in `use_cases.py` is
  where that tolerance lives, and it compares *exact* sets so tolerance never becomes a rubber
  stamp. This was a real false negative: it failed 2 of 3 live `uc3_large` runs whose answers
  were correct.

  **Reading the failure-mode tally:** most codes describe the model's call (`HANDLE_NOT_FOUND`,
  `TYPE_MISMATCH` — it asked for something wrong, and the error told it so). `INTERNAL_ERROR` is
  categorically different: it means an operation dereferenced something it never validated, so the
  model received `{"message": "operation execution failed", "details": {}}` and had nothing to act
  on. **Treat every `INTERNAL_ERROR` in a trial as a DALE defect to file, never as a model
  failure** — an `INTERNAL_ERROR` always means a missing precondition. One such bug
  (`join_lookup` on a value-valued `reduce_by` index) was found exactly this way, from a single
  `failure modes: INTERNAL_ERROR=1` line in a `uc3_large` run.

  Prints progress live by default (`verbose=True`, mapped internally to `dale.agent`'s
  `verbosity="debug"` — see below): a `[i/n] running...`/`[i/n] ok in Xs` marker per trial, and every
  individual tool call as it happens inside a trial. A live model run can legitimately take anywhere
  from ~30 seconds to a few minutes depending on retries and exploratory steps — without this, a
  slow-but-working run looks identical to a hung one. Pass `verbose=False` for quiet mode (summary
  only). `TrialSummary.render(show_action_logs=True)` (also the default) prints each trial's full
  action log at `debug=True` — full RETURN payloads, useful for exactly the kind of checker-bug
  diagnosis this kind of evaluation needs.

  Every call renders as `handle.operation(args)` (a receiver method call, not a raw params dict);
  handle-creating operations (`OperationSpec.creates_handle`) require the model to supply a `name`
  and `description` — `name` is not just a display label, it's the handle's real
  `DataRegistry`-level identifier now (rejected outright on collision with an already-alive handle,
  see `DESIGN.md` Section 2's Pointer-Based State Management), so a call renders as e.g.
  `in_stock_widgets = products.filter_where(...)` with no separate opaque id to reconcile — the name
  in the CALL line, in `REGISTRY STATE`, and in raw `DaleError`/`dispatch.py` output are all the same
  string. Full API and rendering-behavior detail lives in `src/dale/agent/`'s own docstrings
  (`ActionLog._render_registry_state`/`_render_entry`, `dale.agent.Verbosity`) rather than duplicated
  here — this file tracks eval-harness-specific behavior, not `dale.agent`'s API.

  `dale.agent.Verbosity` (`"quiet"`/`"normal"`/`"debug"`, passed to `build_agent`/`build_tools` and
  `ActionLog.render`) controls how much of each call prints: `"normal"` shows the `CALL` line, a
  one-line `RETURN` status (with a concise error summary on failure), and `REGISTRY STATE`; `"debug"`
  adds the full JSON-indented `RETURN` payload. `REGISTRY STATE` itself is a column-aligned table —
  `HANDLE`, `TYPE` (list/dict/set), `SIZE`, `MEMORY` (a humanized `avg_record_bytes * size`
  estimate, `cost.py`'s own approximation formatted for reading), `DESCRIPTION` (left ragged, not
  wrapped, matching `kubectl`/`docker ps`'s convention for a trailing free-text column):
  ```
  REGISTRY STATE:
    HANDLE                     TYPE  SIZE  MEMORY  DESCRIPTION
    -------------------------  ----  ----  ------  -----------
    people                     list  5     270 B   List of people with their department.
    engineering_people *       list  3     171 B   People in the Engineering department.
  ```
  A trailing `*` marks the most recently *created* handle (the only kind of "update" a handle can
  undergo — DALE handles are immutable once created). Shows every handle still alive across the
  whole run, so you can see the model's working set grow and shrink without re-deriving it from the
  call history. Handles pre-loaded by a use case's `setup()` (bypassing the agent's own tool calls)
  are seeded into this listing via `ActionLog.seed_from_registry`, using the real `name`/`description`
  the invoker supplied to `registry.create()`/`load_csv` at setup time — not blank, since every handle
  has always had both from creation now, whether it came from the model or the invoker.
- `use_cases.py` — setup/task/checker triples for Use Cases 1-4, reusing `examples/data/` and the
  same ground truth already pinned down in `tests/test_use_case_pipelines.py`. Checkers search all
  handles still alive at the end of a run for one matching the expected shape and content —
  structural checking, not NLP grading of the answer text.
- `run_trials.py` — CLI entry point. Prints `harness.describe_setup`'s "story" of the use case
  before any trial runs — the task text, which virtual files are registered, the resulting starting
  handles, and which operations are available — so you know what the model is working with before
  watching it work.

## Usage

```bash
# structural-only, free, no API key (verifies the harness itself, not real correctness):
uv run --extra agent python -m eval.run_trials uc1 test 2

# a real trial run (costs API credit):
uv run --env-file .env --extra agent python -m eval.run_trials uc1 anthropic:claude-sonnet-5 10

# cap spend per trial (pydantic-ai's own default is request_limit=50):
uv run --env-file .env --extra agent python -m eval.run_trials uc1 anthropic:claude-sonnet-5 5 \
  --max-requests 25

# Section 4.2 part (F): same use case batched vs. one operation per round trip
# -- compare the reported mean request count between the two runs. There is only
# one tool now (run_plan), so the unbatched condition is a one-step ceiling on it
# rather than a missing tool.
uv run --env-file .env --extra agent python -m eval.run_trials uc1 anthropic:claude-sonnet-5 5
uv run --env-file .env --extra agent python -m eval.run_trials uc1 anthropic:claude-sonnet-5 5 \
  --steps-per-call 1
```

Use cases: `uc1` (inventory reconciliation), `uc2` (log sessionization / `window_flag`), `uc3`
(churn / feature usage, alternative pipeline), `uc4` (org permission inheritance /
`graph_walk_resolve`). Gemini models are expected to fail entirely — `filter_where`'s recursive
predicate schema breaks Gemini's function-calling format (tracked in issue #1) — this is a known,
already-diagnosed limitation, not a harness bug,
if you point a `google:*` model at any use case here.

## Validated so far

Structural correctness confirmed two ways, at zero API cost:
1. All four use cases run cleanly through the harness against `pydantic_ai`'s built-in `TestModel`
   (garbage tool arguments — correctly produces 0% success, 100% wasted-turn rate, and sensible
   `HANDLE_NOT_FOUND`/`FILE_NOT_REGISTERED` failure-mode counts).
2. Positive control: running each use case's real deterministic pipeline (the same one
   `tests/test_use_case_pipelines.py` asserts) and calling its checker directly confirms all four
   checkers correctly return `True` on genuinely correct data — the harness can actually detect
   success, not just always report failure.

Real trial batches have been run against `anthropic:claude-sonnet-5`. The first batch gave a
usable N=5 for two of the four use cases — **UC3 80% (4/5)** and **UC4 100% (5/5)** — while UC1 and
UC2 were cut short before reaching a trustworthy N. That run also surfaced two genuine checker bugs
rather than model failures, both since fixed: UC1's left-join-vs-inner-join task ambiguity, and
UC2's hardcoded assumption about `window_flag`'s flag-field name.

These are small-N results and are reported as such; the full trial plan (N=10-20 per pattern, at
multiple dataset sizes) has not been run. Issues #14-#19 track the remaining measurement work.
