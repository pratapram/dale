# DALE — Declarative Algorithmic Logic Engine

**Describe the task. Point to the data. DALE agent finishes the task. No code generated.**

Data stays in host memory — a `DataRegistry` holding native Python `list`/`dict`/`set` — and the
LLM addresses it only through opaque handles, never as rows in its own context. It selects from a
small, closed set of declarative primitives and supplies structured parameters (predicates,
computed fields, priority orders), never expressions or logic.

This is what gives DALE two properties code-generation-based alternatives (agents that write and
execute Python against your data) can't offer:

- **No code-execution attack surface.** There's nothing to sandbox because there's nothing being
  executed — the LLM's entire output is validated data, not instructions. That removes the sandbox
  problem, not every knob: `max_tool_calls` defaults to unlimited, there's no OS-level backstop, and
  process isolation is the invoker's job — see [`docs/environment.md`](docs/environment.md).
- **Pre-execution cost estimation.** Because the primitive set is closed and known in advance, the
  cost of an operation — a join's output size, for example — can be computed *before* it runs, not
  discovered by running it and hoping it doesn't blow up.

**How large is "large"?** The model's context does not grow with your data. It sees handle metadata
(`kind`, `size`, `description`) plus a hard-capped sample, so its context is identical at 39 rows
and at 10,000; only host memory scales, since the registry holds native Python collections. An
operation whose estimated output exceeds the default ceiling — **1,000,000 rows or 500 MB** — is
refused before it runs unless you pass `confirm=True` on that call or raise the ceiling with
`DataRegistry(limits=RegistryLimits(...))`. The largest run measured so far is **10,005 rows**,
where all host compute totalled 17.9 ms — 0.07% of wall clock (the breakdown is in
[`docs/agent.md`](docs/agent.md#where-the-time-goes)). **No upper bound has been established.** The
planned multi-size evaluation (100 / 1,000 / 10,000+ rows, N=10–20 trials) has not been run.

### Why not just let the agent write pandas?

You can, and for open-ended exploration you probably should. DALE takes the other side of a trade
text-to-SQL already made at scale: rather than let a model write arbitrary code against your data,
constrain it to a bounded query language and let a separate, trusted engine execute it. DALE
generalizes that from a relational database to in-memory Python collections, with a grammar aimed
at the graph-walk, sliding-window, and priority-conflict operations SQL doesn't express naturally.

**The cost is real, and there is no escape hatch.** If your task doesn't decompose into the
primitives below, you cannot drop into Python to finish it — the catalog grows only when a
developer writes and reviews a new primitive at build time. Don't use DALE for **long-horizon,
turn-by-turn state mutation** (a ledger accumulating transactions, a controller tracking relative
commands — stateful code-execution runtimes are right for that; see `DESIGN.md`'s "Non-Goal / Scope
Boundary"), for **open-ended analysis needing pandas' full surface**, or against **live data
sources** — there are no database or API connectors, deliberately, so something upstream has to
materialize the data first. And note that evaluation so far is a **preliminary pilot** (N=5, four
of eight patterns, one dataset size), not a validated success rate.

**Status:** alpha (`0.1.0`). Core engine and a foundational primitive set, with tests, plus a
PydanticAI-based agent-integration module (`src/dale/agent/`) verified end-to-end against a live
Anthropic account. Interfaces may still change. Issues: https://github.com/pratapram/dale/issues

## Install

Requires Python 3.10+. Core DALE depends only on `pydantic>=2.6` — no network access, no
credentials. The `agent` extra adds `pydantic-ai` for the LLM-orchestration layer.

```bash
git clone https://github.com/pratapram/dale
cd dale
uv sync --extra dev      # or: pip install -e ".[dev]"
```

For the LLM examples, add the agent extra: `uv sync --extra agent` (or `pip install -e ".[agent]"`).
[`uv`](https://docs.astral.sh/uv/) is the project's package manager; pip works everywhere uv is
shown.

*Not published to PyPI yet — install from source. `pip install dale` will work once it is.*

## Quick start

You have an employee roster in memory, and you want to know how many people are in Engineering. You
say that in plain English. The agent picks the primitives, runs them against the roster, and hands
back a handle to the answer — it never sees the rows themselves.

```python
import dale
from dale.agent import ActionLog, build_agent, pick_model, run_agent

# Your data, as ordinary Python.
people = [
    {"name": "Alice", "department": "Engineering", "age": 34},
    {"name": "Bob", "department": "Sales", "age": 29},
    {"name": "Carol", "department": "Engineering", "age": 41},
]

# Hand it to DALE. From here on the model sees only the handle's name, kind,
# size, and description — never the rows.
registry = dale.DataRegistry()
registry.create(
    "list",
    people,
    name="people",
    description="employee roster with name, department, and age",
    created_by="quickstart",
)

# Ask for what you want, in plain English.
action_log = ActionLog()
action_log.seed_from_registry(registry)
agent = build_agent(registry, action_log, model=pick_model())

outcome = run_agent(
    agent,
    "How many people are in Engineering, and what are their names?",
    deps=registry,
    action_log=action_log,
)
if not outcome.success:
    raise SystemExit(f"run failed: {outcome.error}")

# The answer comes back as real Python, read out of the registry by handle.
engineering = registry.materialize(outcome.result.output.handle)

print(engineering)
print(outcome.result.output.note)
```

```
[{'name': 'Alice', 'department': 'Engineering', 'age': 34}, {'name': 'Carol', 'department': 'Engineering', 'age': 41}]
There are 2 people in Engineering: Alice and Carol.
```

**What DALE actually did.** The model never wrote code. It chose one primitive, filled in its
parameters, and DALE ran it. That trace is what `action_log` holds — for a task this small it's a
single step:

```
[1] Intent: keep only Engineering staff
    Action: engineering_people = people.filter_where(predicate="department == 'Engineering'")
    Result: ok (<0.1 ms)
```

The `Intent` line is the model's own reason for the call, required on every step. The `Action` line
is the entire extent of what it was permitted to emit: a primitive name plus structured parameters —
`{"field": "department", "op": "==", "value": "Engineering"}` — never an expression, never a line of
Python. A different model might reach the same answer another way (`group_by` on `department`, then
read the bucket), but whatever it picks comes from the seventeen primitives below and nothing else.

Needs `uv sync --extra agent` and one of `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` /
`MOONSHOTAI_API_KEY` / `DEEPSEEK_API_KEY` / `ALIBABA_API_KEY`; `pick_model()` takes the first it
finds, or `DALE_MODEL` if you set one.

**`engineering` is the answer; the note is not.** The model returns a `DaleResult` — a handle plus a
one-sentence summary. The rows in `engineering` came out of DALE's own primitives; the note is LLM
prose about them, and its wording varies from run to run.

Primitives can also be called directly, with no model involved — `dale.call_primitive(registry,
name, params)`, same validation and same cost gates. See `examples/01`–`03`.

## Primitives

Every primitive is called through `dale.call_primitive(registry, name, params)`, which validates
`params` against the primitive's schema, runs its cost estimator if it has one, and executes it —
the same path the agent loop uses, not a shortcut test-only path. Every primitive that **creates a
handle** takes two extra required params, `name` and `description` (as does `registry.create()`).
`name` *becomes* the handle: it must be a valid Python identifier, and colliding with a live handle
is an error, not a silent rename. `description` is mandatory too — and an honest "unknown,
uninspected data" is a complete, encouraged answer when true.

`dale.call_primitive()` returns a `PrimitiveOutput` (`.status`, `.handle`, `.result`, `.estimate`)
whose `.handle` is a `HandleMeta` (`.handle`, `.kind`, `.size`, `.description`) — so the string id of
a newly created handle is `out.handle.handle`. Primitives that create a handle set `.handle` and
leave `.result` empty; inspection primitives (`peek`, `describe`) do the reverse; a refused operation
returns `status="cost_gate_exceeded"` with its `.estimate` and no handle at all.

| Primitive | Purpose |
|---|---|
| `load_csv` | Load a local CSV into a `list` handle, with deterministic type inference. Takes a `file` — a virtual name registered on a `FileRegistry` ahead of time, never a raw path (see [`docs/environment.md`](docs/environment.md)) |
| `load_json` | Load a local JSON file into a `list` handle (top-level array) or `dict` handle (top-level object) — JSON is a loading path, not a new handle kind. Same `file`-via-`FileRegistry` contract as `load_csv`. Optional `remove_envelope=True` unwraps a single-list-key envelope (e.g. Salesforce's `{"records": [...]}`) straight to a `list` handle — only when the model explicitly asserts it recognizes the shape; DALE never guesses this on its own. Nested structure is otherwise preserved as-is |
| `filter_where` | Keep records matching a predicate (comparisons, `and`/`or`/`not`) |
| `compute_field` | Add a derived field (`add`/`subtract`/`multiply`/`divide` over fields or constants) |
| `sort_by` | Stable multi-key sort, nulls sorted last |
| `index_by` | Build a unique-keyed `dict` (composite key → single record); errors on duplicates |
| `group_by` | Build a bucketed `dict` (composite key → list of records) |
| `priority_reduce` | `index_by`, but for duplicate keys instead of erroring: resolves each group's `value_field` to a single winning value via a priority order (e.g. license tiers, "gold" beats "silver" beats "bronze") — the "Hash map + priority-ordered reduction" pattern |
| `dict_diff` | Compare two `dict` handles keyed the same way; returns one row per key across their union, each tagged `new`/`removed`/`changed`/`unchanged` |
| `join_lookup` | Merge a `list` against an `index_by`/`group_by`-built `dict`; real fan-out risk, and a real cost estimator |
| `window_flag` | Sliding-window occurrence counting/flagging over a group key + orderable field (numeric or ISO 8601) — the "Sliding Window / Two-Pointer" pattern (log-stream sessionization) |
| `graph_walk_resolve` | Walk each node's single-parent ancestor chain in an `index_by`-built `dict`, collect applicable rules per group field, resolve conflicts via `resolve_priority` — the "Adjacency Graph Traversal" pattern (org-chart permission inheritance) |
| `flatten_json` | Explode a nested array field into one row per element, carrying selected parent fields down onto each row (e.g. a GitHub issue's `labels` array → one row per label, with the issue number/title attached). A record whose target field is absent, `null`, or an empty array contributes zero rows — not an error. `path` supports one level of nesting today, not arbitrary depth |
| `peek` | A small sample of a handle — a sanity check, not a data-dump channel. Hard-capped at 50 items **and 4 KB serialized**, whatever the handle holds and however deeply it nests, so a `peek` of a 10-million-row handle costs the same as one of a 10-row handle. Anything cut to fit says so in place: a shortened list ends with `"+N more items"`, a shortened dict gains a `"..."` entry counting the keys not shown, a shortened string ends with `"...(+N more chars)"`, and `"truncated": true` sits alongside the sample — a sample never understates what is there |
| `describe` | Aggregate statistics for a field (numeric min/max/mean/null-rate, or categorical distinct-count/top-k) — never individual values dumped in bulk. `top_k`'s *values* are subject to the same byte cap and the same in-place markers as `peek`; its counts never are |
| `release_handle` | Explicit cleanup of a handle no longer needed |
| `export_handle` | Write a handle's real content straight to a registered output destination (CSV or JSON) — the LLM gets back only a confirmation (row/byte count), never the content itself |

**The table gives purpose; the library gives parameters.** There is no hand-written parameter table
anywhere — it would drift. Ask the catalog instead; this is the same schema the LLM is handed, so
it can't be out of date:

```python
import inspect
import dale

print(dale.list_primitives())

spec = dale.get_primitive("window_flag")
print(inspect.getdoc(spec.fn))

schema = spec.param_schema.model_json_schema()
required = set(schema["required"])
for field in schema["properties"]:
    print(f"  {field:<13} {'required' if field in required else 'optional'}")
```

```
['compute_field', 'describe', 'dict_diff', 'export_handle', 'filter_where', 'flatten_json', 'graph_walk_resolve', 'group_by', 'index_by', 'join_lookup', 'load_csv', 'load_json', 'peek', 'priority_reduce', 'release_handle', 'sort_by', 'window_flag']
Flag records with >= threshold qualifying occurrences (matching an
optional predicate) within a trailing window over an orderable field,
grouped by one or more key fields. Returns a new list handle with two
fields added per record: `as` (bool) and `{as}_count` (int). Output order
is grouped, not guaranteed to match input order — sort_by afterward if a
specific order matters.
  handle        required
  group_by      required
  window_field  required
  window_size   required
  threshold     required
  predicate     optional
  as            optional
  name          required
  description   required
```

Read `model_json_schema()` itself (not just its keys) for types, enums, defaults, and the nested
predicate grammar. It reports the **wire names** you actually pass — `window_flag`'s field is `as_`
on the Python class but `as` in a call, and reading the schema rather than `model_fields` is what
gets that right. On the spec: `cost_estimator` is `None` for primitives with no pre-execution
estimate, and `bounded_by_input` marks the ones whose output provably can't exceed their input.

Composite keys: `index_by`/`group_by` accept `key_fields: list[str]`. A single field produces a
scalar key; more than one produces a tuple — not a fourth handle type.

Deliberately not built yet: general graph traversal
(`graph_bfs`/`graph_dfs`/`graph_topological_sort`/`graph_connected_components` —
`graph_walk_resolve`'s bounded single-parent walk covered its use case without needing these),
entity resolution, `top_k`, `dict_frequency`/`set_difference`, shape metadata and nested `peek()`
for JSON, `flatten_json`'s multi-level `path` support, `load_jsonl`/`load_parquet`.

## Examples

Scripts **01, 02, 03 and 08 need no API key** — they call primitives directly, the same way an LLM
would. The rest drive a real model and need the `agent` extra plus a key. (Gemini can't run tasks
that need `filter_where`'s recursive and/or/not predicates — its function-calling format can't
represent recursive schemas; see `examples/04`'s docstring.)

| Script | Demonstrates |
|---|---|
| `01_filter_sort_compute.py` | No key. `filter_where`, `compute_field`, `sort_by`, `peek`, `describe`, `release_handle` |
| `02_composite_key_join.py` | No key. `index_by`/`group_by` with multi-field (composite/tuple) keys, `join_lookup` |
| `03_cost_estimation_guardrail.py` | No key. **Start here for the safety story.** An oversized join is refused *before* it runs, with an exact row-count estimate (`estimated_rows: 100 (threshold: 20)`) and no handle created — then succeeds under an explicit `confirm=True` |
| `04_llm_orchestrated.py` | **The actual pitch.** An LLM solves `01`'s task from a plain-language description, never seeing the data and never writing code |
| `05_hello_world_memory.py` | Shortest possible `dale.agent` example — in-memory data, a one-line task. Start here before `04` |
| `06_hello_world_file.py` | File-to-file: the model calls `load_csv` and `export_handle` against virtual names, never constructing a real path on either end. Prints each step live |
| `07_export_to_file.py` | The write half of `06` on its own: results go to disk via `export_handle`, never back through the model's context |
| `08_json_flatten.py` | No key. `load_json` + `flatten_json` over a real GitHub issues snapshot — one row per label, issues with no labels contributing zero rows |
| `09_license_reconciliation.py` | DALE's original motivating problem, solved by a real LLM: per-tier eligibility lists resolved via `priority_reduce`, then `dict_diff`'d against the previous hour |
| `10_named_handle_referencing.py` | Three named datasets referred to by name in the task prompt; the model chains `index_by` + `join_lookup` twice |

```bash
uv run python examples/01_filter_sort_compute.py
uv run python examples/02_composite_key_join.py
uv run python examples/03_cost_estimation_guardrail.py
uv run python examples/08_json_flatten.py

# the rest need the agent extra + a real API key:
uv sync --extra dev --extra agent
cp .env.example .env   # then fill in your real key(s) — .env is gitignored, .env.example isn't
uv run --env-file .env --extra agent python examples/05_hello_world_memory.py
uv run --env-file .env --extra agent python examples/04_llm_orchestrated.py

# DALE_MODEL forces a provider/model; DALE_VERBOSITY is quiet | normal | debug | raw
DALE_MODEL="openai:gpt-5.6" DALE_VERBOSITY=debug \
  uv run --env-file .env --extra agent python examples/06_hello_world_file.py
```

## Extending the primitive catalog

New primitives are registered **at build time by a developer**, never chosen or imported by the LLM
at runtime — that boundary is the whole safety argument, so it is deliberate, not a gap.
Built-in primitives use the exact same mechanism:

```python
from pydantic import BaseModel
from dale.catalog import primitive
from dale.catalog import PrimitiveOutput

class MyParams(BaseModel):
    handle: str

@primitive("my_primitive", MyParams)
def my_primitive(registry, params: MyParams) -> PrimitiveOutput:
    ...
```

If your primitive creates a handle, declare `creates_handle=True` and put `name: str` and
`description: str` on its param schema — that is how the agent layer knows to ask the model for
them, without hardcoding a primitive-name list (`src/dale/catalog.py`). Pass a `cost_estimator=` to
keep pre-execution cost estimation intact, or `bounded_by_input=True` if the output is provably no
larger than the input.

## Development

```bash
uv sync --extra dev --extra agent && uv run pytest -q   # 366 passed, 1 deselected
uv sync --extra dev && uv run pytest -q                 # core only — 3 failed, 209 passed, 4 skipped
```

Measured on Python 3.12 at commit `b33b198`. Both lines start with `uv sync`, deliberately: `uv
sync` *replaces* the environment's extras, so it is the only way to reach a genuinely
`pydantic_ai`-free env — `uv run --extra dev` alone will not remove a `pydantic_ai` an earlier sync
installed, and reports the green 366 instead. That also means the second line uninstalls the agent
extra; re-run the first to get it back. Name both extras when you want both: `uv sync --extra agent`
on its own drops `pytest`.

**Known issue:** the core-only run is red. `tests/test_use_case_pipelines.py` imports `dale.agent`
without the `pytest.importorskip` guard `tests/test_agent.py` has, so its three tests fail on
`ModuleNotFoundError: No module named 'pydantic_ai'` instead of skipping. Install the `agent` extra
for a green run.

Live tests that hit a real provider are deselected two ways — `addopts = "-m 'not live'"` in
`pyproject.toml`, plus a `DALE_LIVE_TESTS` gate inside the test file — so a bare `pytest` never
spends money. `DALE_LIVE_TESTS=1 uv run --extra dev --extra agent pytest -m live` is the one
deliberate way to run them; it needs `ANTHROPIC_API_KEY`.

Tests are per module (`test_registry`, `test_grammar`, `test_errors`, `test_primitives_core`,
`test_primitives_phase2`, `test_cost_estimation`, `test_baseline`, `test_dict_diff`, `test_files`,
`test_flatten_json`, `test_priority_reduce`, `test_harness_usage`, `test_agent`, `test_agent_live`)
with small isolated fixtures. `test_use_case_pipelines.py` is the deliberate exception: it runs the
`DESIGN.md` §3 use cases end to end against `examples/data/`, asserting independently-computed
ground truth. `test_agent.py` skips itself without the `agent` extra, and drives the full tool-call
loop against PydanticAI's offline models — no key needed for structural verification.

## Docs

- [`DESIGN.md`](DESIGN.md) — architecture, core principles, scope boundary
- [`docs/agent.md`](docs/agent.md) — `dale.agent` reference: batching, action log, privacy mode, loop termination, run cost
- [`docs/environment.md`](docs/environment.md) — what the invoker must provide: runtime, files, credentials, limits, lifecycle

A paper covering the related work, mechanism, evaluation, and limitations in full is in
preparation; it will be added to `paper/` when it's ready.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
