# `dale.agent` — the LLM orchestration layer

[`src/dale/agent/`](../src/dale/agent/) (requires `uv sync --extra agent`, or
`pip install -e ".[agent]"`) is the real integration surface, not example-scoped code. It is built
on [PydanticAI](https://ai.pydantic.dev/). See the README's "Use it with an LLM" for the
five-line version; this page is the reference.

Entry points: `build_agent`, `run_agent`, `build_tools`, `pick_model`, `ActionLog`, `DaleResult`,
`AgentRunOutcome`, `TokenUsage`, `AgentLoopTerminated`, `registry_state_summary`,
`render_raw_messages`, `DEFAULT_SYSTEM_PROMPT`.

## `run_plan` — the model's only tool

`build_tools(action_log, ...)` returns **exactly one** PydanticAI `Tool`, `run_plan`. (It returns a
*list* of one because that is what `Agent(tools=...)` takes, and so a second agent-layer tool later
needs no caller changes.)

`run_plan` submits a list of `steps`, which DALE runs in order; a single operation call is a list of
one, not a special case. Each step keeps the exact same Pydantic-validated schema a standalone call
to that operation would have — a real discriminated union built from the live catalog, each variant
carrying its operation's own docstring, so it can never drift from what `dale.call_operation`
actually accepts. A plan stops at the first step that doesn't succeed and returns everything
completed so far, and each step is logged as its own real `ActionLog` entry — so a batched call is
indistinguishable in the resulting trace from the same steps issued one turn at a time.

One surface rather than two means the catalog is published once per request instead of twice, and
the model never chooses between two encodings of the same call.

- `max_steps_per_call=1` caps a plan at one step, reproducing the unbatched, one-call-per-turn
  condition (this is the ablation the paper's evaluation plan needs).
- `operations=` selects the model's action space, and is **the largest token lever a deployment
  has**. The `run_plan` schema is ~78% of a request body and is sent on every request, so what is
  in it dominates cost regardless of dataset size — a 3-row question pays the same catalog cost as
  a 10,000-row one. Three modes:

  | value | offered | schema |
  |---|---|---|
  | omitted (default) | `CORE_OPERATIONS`, 9 ops | 10,591 chars |
  | `ALL_OPERATIONS` | the whole catalog, 17 ops | 20,752 chars |
  | `[...]` | exactly what you name | scales with the list |

  The core is `filter_where`, `sort_by`, `index_by`, `group_by`, `join_lookup`, `compute_field`,
  `peek`, `describe`, `release_handle` — chosen from measured use across 245 trial runs, not
  intuition. Loaders (`load_csv`/`load_json`), `export_handle`, and the specialist algorithms
  (`window_flag`, `graph_walk_resolve`, `dict_diff`, `reduce_by`, `flatten_json`) are opt-in.

  **Don't guess the list — measure it.** Develop against `ALL_OPERATIONS`, then read
  `ActionLog.operations_used()`, which returns a paste-ready allowlist of what the run actually
  called (failed calls included: a rejected `join_lookup` still proves the pipeline needed it).
  `run_agent` prints it at any verbosity above `"quiet"`:

  ```
  operations used: ['compute_field', 'filter_where', 'index_by', 'join_lookup', 'sort_by']
  ```

  `dale.call_operation` stays unrestricted throughout — the allowlist constrains the model's action
  space, not the engine's, so a host loading its own fixtures is unaffected.

**Per-step fields.** Every step injects one extra required field, `intent` — a short
natural-language note on why the model is making that specific call. It is stripped before the call
reaches `dale.call_operation` (operations don't know about it) and recorded, alongside the call and
its result, in an `ActionLog` entry. Separately, handle-creating operations declare `name` and
`description` on their *own* param schema (not injected here): `name` becomes the handle's real
identifier in `DataRegistry`, so it must reach dispatch, not just the log.

## `ActionLog`

The append-only `(intent → tool call → result)` trace, used in place of a resume/checkpoint
feature: on failure, a human reads the log instead of needing full registry-state serialization.
It is also the artifact the evaluation harness compares against an expected call sequence, for
per-step rather than only end-to-end correctness.

Seed it from the registry before the run — `action_log.seed_from_registry(registry)` — so the
starting handles appear in the trace too.

## `build_agent` and `pick_model`

`build_agent(registry, action_log, model=...)` wires the two together into a ready-to-run `Agent`.
Its default `system_prompt` is built from `registry_state_summary(registry)`, listing whatever
handles already exist.

`pick_model()` returns `DALE_MODEL` verbatim if it is set. Otherwise it walks the following list in
order and returns the pinned default for the **first** environment variable it finds — so with two
keys in your shell, the higher row wins. If none is set it raises `RuntimeError` naming all of them.

| Checked in this order | Model it returns |
|---|---|
| `ANTHROPIC_API_KEY` | `anthropic:claude-haiku-4-5` |
| `OPENAI_API_KEY` | `openai:gpt-5.6` |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `google:gemini-3.6-flash` |
| `MOONSHOTAI_API_KEY` | `moonshotai:kimi-k2.6` |
| `DEEPSEEK_API_KEY` | `deepseek:deepseek-v4-flash` |
| `ALIBABA_API_KEY` or `DASHSCOPE_API_KEY` | `alibaba:qwen-plus` |

These are cheap, fast defaults chosen for the examples, not a recommendation for your workload — set
`DALE_MODEL` (e.g. `DALE_MODEL="anthropic:claude-sonnet-5"`, which is what the paper's pilot used)
to pin anything else. `pick_model()` never reads a key's *value*; presence is the whole test, and
the provider classes do the authenticating.

## Strict privacy mode

`DataRegistry(privacy_mode=True)` turns on the `strict_privacy` mode (`DESIGN.md`, "Optional
Strict-Privacy Mode"):

- `peek` is dropped from the operations offered to the model entirely — it is not among `run_plan`'s
  step variants at all, so there is nothing to redact. `describe` is the only inspection operation
  offered. The default system prompt gets an explicit note saying so, rather than making the model
  spend a turn discovering it.
- `describe`'s categorical `top_k` *values* are redacted; its numeric aggregate stats and
  `distinct_count`/`null_rate` stay real.
- Validation-error messages stop echoing the caller's raw input.

Called directly through dispatch (host-side, not via the model), `peek` still redacts everything a
redaction has to cover: leaf values become type placeholders, a dict handle's own keys become
positional `<key 1>` placeholders (they are data — `index_by` on a patient id keys the handle by
real identifiers), and truncation markers drop their counts, since `shown + "+N more"` is a bucket
size and a set of bucket sizes is exactly the `top_k` that `describe` withholds under this flag.

## `peek_at_every_step`

`build_agent(..., peek_at_every_step=True)` (default on) splices a small automatic `peek`/`describe`
into the system prompt for every starting handle, and into every handle-creating call's own result
afterward (under an `auto_inspect` key), so the model doesn't have to spend a separate turn
sanity-checking what it just built. Free — no extra `max_tool_calls` slot, no separate `ActionLog`
entry. Ignored entirely, regardless of its own value, whenever `privacy_mode` is on.

## Repetition nudge and repetition limit

`build_agent(..., repetition_nudge=True)` (default on) — after a call fails (or hits
`cost_gate_exceeded`) identically — same operation, same params — three times, its result gets a
`repetition_warning` field quoting DALE's own already-known error back at the model explicitly,
instead of silently letting it resend the same rejected call indefinitely. This is mechanical, not
diagnostic: DALE already has the error, the model just needed to be told it's repeating itself.
Counts across the whole session, not just consecutive calls, and works identically whether the
calls arrive one at a time or batched in a single `run_plan`.

`build_agent(..., repetition_limit=5)` (default 5, `None` to disable) — the same signal escalated
from a warning to a stop. Once one identically-failing call reaches that many attempts, the run ends
with `dale.agent.AgentLoopTerminated` instead of the model being warned yet again. It reuses the
nudge's own count, so it needed no new detection mechanism: attempts 1–2 return a plain error, 3–4
add the `repetition_warning`, the 5th terminates — the model is always warned before it's stopped,
and a limit low enough to break that promise (anything not greater than 3) is rejected with a
`ValueError` when the tools/agent are built rather than quietly honored.

Motivated by a real pilot failure: 46 byte-for-byte identical calls that never recovered
(see the pilot write-up). Note the loop was never *unbounded* — pydantic-ai applies
`UsageLimits(request_limit=50)` by default — this makes it stop early and with a stated reason
(`REPETITION_LIMIT_EXCEEDED`, naming the operation and quoting the real error) instead of late and
anonymously.

`AgentLoopTerminated` deliberately does **not** subclass `DaleError`: every `DaleError` describes
one *call* and is handed back to the model as a payload it can ignore, which is precisely why this
must not be one. It only catches byte-identical resends — a model that varies its params escapes it
by design; that case stays the job of `max_tool_calls` and `UsageLimits`.

How the stop *reaches you* depends on how you run the agent. `run_agent` catches it like any other
run failure and hands back an `AgentRunOutcome(success=False, error=...)`, still printing the
`Result: Failure` block with its usage and timing lines at non-quiet verbosity. Calling
`agent.run_sync` directly — as `examples/02`, `03`, `04`, `09` and `10` do — lets it propagate to
the caller as a raised exception, so a script taking that route should catch it itself.

## Verbosity, and `verbosity="raw"`

`build_agent(..., verbosity=...)` / `run_agent(..., verbosity=...)` accept `"quiet"`, `"normal"`,
`"debug"`, and `"raw"`. The examples read this from `DALE_VERBOSITY`.

`"raw"` is for investigating the literal exchange with the model, not just DALE's own
dispatch-level view of it: it interleaves the raw protocol messages (system/user prompt, the
model's exact tool-call args before `intent` is stripped, tool returns, any free text) right before
each call's usual human-readable block.

Use `run_agent` rather than calling `agent.run_sync(...)` directly to get two things a bare
`run_sync` structurally can't provide: the trailing raw messages after the run ends (the last tool
return plus the model's closing output — nothing else triggers after them, so nothing else would
print them), and a final `Result: Success`/`Result: Failure` section at any `verbosity != "quiet"`.
`render_raw_messages(messages)` is the underlying pure-rendering function, reusable standalone
against `result.all_messages()` if you want the raw exchange without any of DALE's own trace.

## What a run cost

`AgentRunOutcome.usage` / `dale.agent.TokenUsage` report what a run cost, on the success *and*
failure paths, and print in `run_agent`'s `Result:` block at any `verbosity != "quiet"`:

```
Result: Success
  handle='engineering_people_list'
  note: There are 3 people in Engineering: Alice, Carol, and Erin
  tokens: 2 requests, 25,848 in / 222 out
          peak context 13,091
```

**Two input-token numbers, because they answer different questions.** `input_tokens` is *cumulative*
across every request — what the run cost — and grows with the number of turns even when the context
window is perfectly bounded. `peak_context_tokens` is the largest single request's prompt (input +
cache-read, since a cached prefix still occupies context while being excluded from `input_tokens`),
which is what a bounded-context claim is actually about. The run above shows why reporting only the
first would mislead: 25,848 cumulative is nearly double the 13,091 actually resident at any one
moment, and that gap widens with every turn.

Usage comes from a caller-owned accumulator handed to `run_sync`'s `usage=` parameter rather than
read off the result afterward, so a run that *raises* still reports its spend — a repetition loop or
an exhausted request budget are exactly the runs whose cost matters most. Peak context needs
`result.all_messages()`, which an exception discards, so it carries a separate `peak_available` flag
instead of a silently-zero value. Figures from pydantic-ai's offline `TestModel`/`FunctionModel` are
marked `available=False` rather than reported, since their token counts are synthetic.

## Where the time goes

Per-call timing lands in the same trace: every `ActionLogEntry` carries `elapsed_ms` (the
`dale.call_operation` call itself) and `auto_inspect_ms` (`peek_at_every_step`'s extra
peek/describe, kept separate so a convenience feature's cost can't hide inside the operation it
wraps), rendered inline as `Result: ok (2.0 ms + 0.1 ms inspect)`. `ActionLog.total_host_ms` sums
them, and `run_agent`'s `Result:` block subtracts that from wall clock to show the split:

```
time: 26.7s wall — 17.9 ms host compute (0.07%), 26.7s model + network
```

That ratio is the most counter-intuitive fact about operating DALE, which is why it prints on every
run rather than only under a profiler. In a measured 10,005-row run, `group_by` over all 10,005
events took 2.0 ms and the whole pipeline 17.9 ms — **0.07%** of wall clock. The dataset size that
dominates a user's mental model contributes almost nothing to the time; round trips are the only
thing worth optimizing, which is also the concrete case for batching steps into one `run_plan`.

## Module layout

`dale.agent` was one 1,740-line module until it was split by concern; the split was a pure move, no
behaviour changed. Everything importable from `dale.agent` before still is — the submodules are an
internal arrangement, not a new API surface.

| Module | Concern |
|---|---|
| `log.py` | the `ActionLog` and how a run renders |
| `prompt.py` | the system prompt, and `peek_at_every_step`'s auto-inspection |
| `execution.py` | the one choke point every call goes through, and loop control |
| `tools.py` | building `run_plan` — what actually goes on the wire |
| `usage.py` | what a run cost, in tokens and in wall clock |

`examples/04_llm_orchestrated.py` is a thin script on top of all this — the fixture data, task text,
and printing, nothing else.
