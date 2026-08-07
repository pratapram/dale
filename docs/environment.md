# The DALE environment — what the invoker must provide

DALE doesn't discover, fetch, or persist anything on its own. Everything it operates on has to be
handed to it explicitly by whatever application embeds it (the "invoker") — this is deliberate (no
live connectors, no checkpoint/resume — both deliberate). This
page is the checklist of preconditions, not a tutorial — see the README's Quick start and
[`examples/`](../examples/) for usage.

**Runtime.** Python 3.10+. `pip install "dale-engine[agent]"` is the normal install (the distribution is
`dale-engine`; the import is `dale`).

**`pip install dale-engine` — core only, for a different agent loop.** If you are driving DALE from the
OpenAI SDK, LangChain, or your own loop rather than [PydanticAI](https://ai.pydantic.dev/)'s, the
core install gives you the whole engine — all 17 primitives, the registry, the declarative grammar,
cost estimation, `register_primitive` — without the provider SDKs. That is **6 packages and 8 MB**,
against **103 packages and 103 MB** for the `agent` extra, which pulls the Anthropic, OpenAI and
Google SDKs plus a full OpenTelemetry stack. Every primitive's parameters are a pydantic model, so
`dale.get_primitive("filter_where").param_schema.model_json_schema()` gives you a tool schema for
any framework. Core `dale` needs only `pydantic>=2.6`, and has **zero network dependency and needs
no credentials at all** — a real property, not just an omission ([`DESIGN.md`](../DESIGN.md)'s
Build-Time Extensibility section states this explicitly: no live connectors means no network access
is needed in the common case).

**Filesystem.** Local files only — `load_csv` and `load_json` are the built loaders today
(`load_jsonl`/`load_parquet` are unbuilt; whenever they land, the same local-only constraint
applies). The invoker registers each file the LLM should be able to load under a virtual name on a
`FileRegistry` — `files = dale.FileRegistry(); files.register("sales_data",
"/secure/data/sales.csv")` — and passes it to `DataRegistry(files=files)`. The LLM only ever picks
among names the invoker explicitly registered; it never sees or constructs a raw filesystem path.
This isn't just tidiness — `load_csv`'s `path` parameter used to accept an LLM-constructed string
directly, which is an unrestricted local file *read* primitive with no scoping (bounded only by the
running process's own OS file permissions). `FileRegistry` closes that the same way `DataRegistry`
already closes the equivalent problem for in-memory data: opaque references the host controls,
never a raw address/path the LLM supplies itself. There's no schema file to prepare upfront —
types are inferred from the CSV itself at load time. DALE never reads a file it wasn't explicitly
pointed at, and never reaches out to a database, API, or any other live source — that's a hard
scope boundary, not a current gap.

The same registry has a writable counterpart for output: `files.register_output("results",
"/secure/output/results.csv")` registers a destination `export_handle` can write to — same
never-an-LLM-constructed-path property, just for writes instead of reads. Unlike `register`, the
target need not exist yet (that's the point), but its parent directory must.

**Credentials (agent layer only).** One of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `MOONSHOTAI_API_KEY`, `DEEPSEEK_API_KEY`,
`ALIBABA_API_KEY`/`DASHSCOPE_API_KEY`, read directly from the process environment by
`pick_model()`/pydantic-ai's provider classes. With no `DALE_MODEL` set, the first key found decides
both provider and model — see [the default-model table](agent.md#build_agent-and-pick_model), which
also gives the order they're checked in. `DALE_MODEL` overrides all of it with a literal model
string, e.g. `DALE_MODEL="moonshotai:kimi-k2.6"`. Core DALE (direct `dale.call_primitive` calls, no
LLM) needs none of this.

**Initial data.** The invoker constructs a fresh `dale.DataRegistry()` (with a `FileRegistry` if
any `load_csv` calls are expected) and populates it — either via `load_csv` primitive calls (the
LLM's own call, or the invoker's, against pre-registered virtual names), or directly via
`registry.create(kind, value, name=..., description=..., created_by=...)` for data already in
memory — *before* the agent runs. `name` becomes the handle itself and must be a valid Python
identifier that no live handle already uses; `description` is mandatory, and an honest "unknown,
uninspected data" is a complete answer when true. DALE never has data it wasn't given; there's no
ambient "current dataset."

**The task.** A natural-language string passed to `run_agent(agent, task, deps=registry,
action_log=...)` (or `agent.run_sync(task, deps=registry)`). Optionally a custom `system_prompt`
(`build_agent`'s default is built from `registry_state_summary(registry)`, listing whatever handles
already exist).

**Resource limits — mostly opt-in, worth setting deliberately for a real deployment.**
`DataRegistry(limits=RegistryLimits(...))` accepts `max_handles`, `max_result_rows`,
`max_result_bytes`, and `max_tool_calls` (the in-process runaway-loop backstop — on reaching it the
run ends with `dale.agent.AgentLoopTerminated`, rather than the call merely being rejected and left
for the model to retry past) — all have defaults, but `max_tool_calls` defaults to `None`
(unlimited), so a production deployment should set it explicitly. It's the blunt counterpart to
[`repetition_limit`](agent.md#repetition-nudge-and-repetition-limit): that one stops a
byte-identical resend loop early and says exactly why, but sees nothing if the model varies its
params; this one catches that case, reporting only that the budget ran out. Separately, at the
agent-loop level: pydantic-ai applies its own default `UsageLimits(request_limit=50)` automatically
whenever `run_sync` isn't given an explicit `usage_limits` argument (verified directly against the
installed version's source — `usage_limits = usage_limits or UsageLimits()`) — a real existing cap
the invoker inherits for free, separate from DALE's own counter, and tightenable (or loosenable) by
passing `usage_limits=UsageLimits(...)` to `run_sync` directly; `dale.agent.build_agent` doesn't
currently set or expose this itself. The **OS-level backstop** (ulimit/cgroup around the whole
process) is explicitly not built into DALE — today that's entirely the invoker's responsibility.

**Process lifecycle.** One `DataRegistry` per invocation, never reused across sessions/users
— no cross-session state, no persistence, nothing survives past the process.
Tool calls within a session are assumed strictly sequential — no concurrent primitive calls. The
single-tenant, one-dedicated-process-per-invocation deployment model is an
assumption DALE's resource-governance design depends on, not something DALE creates or enforces —
providing that isolation (a container, a subprocess, a sandboxed worker) is the invoker's job.

**What DALE explicitly does not need:** a database, a message queue, persistent storage of any
kind, or special hardware. The registry is fully in-memory and ephemeral by design.
