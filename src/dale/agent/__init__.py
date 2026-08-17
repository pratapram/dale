"""The real agent-integration surface — not example-scoped.

Two things live here that examples/04_llm_orchestrated.py used to hand-roll
locally: the PydanticAI tool-building code (`build_tools`/`build_agent`), and
the intent/action log (`ActionLog`) that stands in for
a resume/checkpoint feature. The log gives a human enough context to see what
an agent run did after a failure, and gives the evaluation harness a per-step
correctness signal (intent -> tool call -> result), not just a final-answer
pass/fail.

The model is offered exactly one tool, `run_plan`, whose `steps` are a
discriminated union over the operation catalog. A single operation call is a
`steps` list of length one — not a special case, the trivial case — so the
model never chooses between two encodings of the same call and the catalog is
published once per request rather than twice.

Every step injects one extra required field, `intent`, onto the operation's
own param schema — a short natural-language note on *why* the model is making
this specific call. `intent` is stripped before the call reaches
`dale.call_operation` (operations themselves don't know about it) and is
recorded, alongside the call and its result, as an ActionLogEntry.
Handle-creating operations also declare `name`/`description` directly on
their own param schema (not injected here) — `name` becomes the handle's
real identifier in DataRegistry itself, so it must reach dispatch, not just
the log.

Requires the `agent` extra (`uv sync --extra agent`) — not a core DALE
dependency. Importing this module without pydantic-ai installed raises
ImportError; callers that want to degrade gracefully should catch it, the
same way examples/04 does.

**Layout.** This was one 1,740-line module until it was split by concern; the
split was a pure move, no behaviour changed. Everything a caller used to
import from `dale.agent` is still importable from `dale.agent`, including the
private names the tests reach for — the submodules are an internal
arrangement, not a new API surface.

  - `log.py`       the ActionLog and how a run renders
  - `prompt.py`    the system prompt, and peek_at_every_step's auto-inspection
  - `execution.py` the one choke point every call goes through, and loop control
  - `tools.py`     building `run_plan` — what actually goes on the wire
  - `usage.py`     what a run cost, in tokens and in wall clock

The entry points themselves — `build_agent`, `run_agent`, `pick_model` and the
`DaleResult` output type — stay here, since they are what a caller reaches for
first and are the only things that depend on all five.
"""

import os
import time
from typing import Any, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.usage import RunUsage

import dale

from dale.agent.execution import (
    _REPETITION_LIMIT_DEFAULT,
    _REPETITION_NUDGE_THRESHOLD,
    _TOOL_MAX_RETRIES,
    _count_prior_identical_failures,
    _execute_and_log_step,
    _repetition_nudge_text,
    _repetition_stop_message,
    _validate_repetition_limit,
    AgentLoopTerminated,
)
from dale.agent.log import (
    ActionLog,
    ActionLogEntry,
    HandleLabel,
    Verbosity,
    _format_bytes,
    _render_raw_part,
    render_raw_messages,
)
from dale.agent.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    _AUTO_PEEK_N,
    _PRIVACY_MODE_NOTE,
    _auto_inspect,
    _initial_inspect_summary,
    default_system_prompt,
    registry_state_summary,
)
from dale.agent.tools import (
    _build_run_plan_tool,
    _call_params,
    _params_for_plan_step,
    _selected_operations,
    _UntitledToolJsonSchema,
    build_tools,
)
from dale.agent.usage import (
    AgentRunOutcome,
    DaleResult,
    TokenUsage,
    _is_synthetic_model,
    _render_run_timing,
)

__all__ = [
    "ActionLog",
    "ActionLogEntry",
    "AgentLoopTerminated",
    "AgentRunOutcome",
    "DaleResult",
    "DEFAULT_SYSTEM_PROMPT",
    "HandleLabel",
    "TokenUsage",
    "Verbosity",
    "build_agent",
    "build_tools",
    "default_system_prompt",
    "pick_model",
    "registry_state_summary",
    "render_raw_messages",
    "run_agent",
]


def pick_model() -> str:
    override = os.environ.get("DALE_MODEL")
    if override:
        return override
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic:claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-5.6"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "google:gemini-3.6-flash"
    if os.environ.get("MOONSHOTAI_API_KEY"):
        return "moonshotai:kimi-k2.6"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek:deepseek-v4-flash"
    if os.environ.get("ALIBABA_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
        return "alibaba:qwen-plus"
    raise RuntimeError(
        "No API key found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "GEMINI_API_KEY, GOOGLE_API_KEY, MOONSHOTAI_API_KEY, "
        "DEEPSEEK_API_KEY, ALIBABA_API_KEY/DASHSCOPE_API_KEY "
        "(or set DALE_MODEL and the matching key)."
    )


def build_agent(
    registry: dale.DataRegistry,
    action_log: ActionLog,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    verbosity: Verbosity = "quiet",
    peek_at_every_step: bool = True,
    repetition_nudge: bool = True,
    repetition_limit: int | None = _REPETITION_LIMIT_DEFAULT,
    max_steps_per_call: int | None = None,
    operations: Sequence[str] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> Agent:
    """`peek_at_every_step` (default True) is ignored entirely — no initial
    or post-step auto-inspection, regardless of its value — whenever
    `registry.privacy_mode` is True; that flag always wins. Only the
    default `system_prompt` gets the initial peek/describe block and the
    privacy_mode constraints note appended; a caller-supplied `system_prompt`
    is used verbatim.

    `repetition_nudge`, `repetition_limit`, `max_steps_per_call` and
    `operations` are passed straight to build_tools — see there, and
    _execute_and_log_step for what the first two actually do at the choke point
    they act on. `repetition_limit` is validated here as well as there, both
    because it reads as a parameter of *this* function to anyone calling it and
    so a bad value surfaces as a configuration error rather than from behind an
    unrelated missing-API-key failure out of pick_model().

    `model_settings` is merged *over* DALE's defaults, not instead of them: a
    caller reaching for this usually wants one unrelated knob
    (`{"temperature": 0}`), and having that silently drop
    `parallel_tool_calls=False` — the setting this design's "one tool call per
    response, all multiplicity inside `steps`" premise rests on — would be a
    trap. Passing `parallel_tool_calls=True` explicitly does override it, which
    is deliberate: they said the words. The two defaults:
      - `parallel_tool_calls=False`, always. Mapped by pydantic-ai's Anthropic
        adapter to `disable_parallel_tool_use` and by the OpenAI one (which
        Kimi/DeepSeek/Qwen ride too); not mapped for Gemini, which is moot —
        Gemini can't run DALE at all, it rejects the recursive `Predicate`.
      - `anthropic_cache_tool_definitions=True`, `anthropic:*` only. Gated on
        the model string rather than sent everywhere because a provider that
        doesn't recognize a setting may reject the request outright, so a
        wrongly-applied one fails every run against it rather than degrading;
        a caller constructing an AnthropicModel object themselves opts in via
        `model_settings`. Cost, not speed: measured cached TTFT was marginally
        *slower* than uncached, but it cuts input-token spend ~90% after a
        run's first request, which is what an eval sweep is made of.
        TokenUsage.from_run folds `cache_read_tokens` back into
        `peak_context_tokens`, so paper.md Section 4.2 part (C)'s
        bounded-context claim stays honest under it."""
    # Every configuration check runs before pick_model(), never after: a bad
    # `operations` name or `max_steps_per_call` is the invoker's mistake, and
    # surfacing it as "No API key found" — which is what happens if build_tools
    # is left to run inside the Agent(...) call below — sends whoever hit it
    # looking for credentials they don't need. Same reason
    # _validate_repetition_limit is called here at all rather than left to
    # build_tools; the two newer parameters just have to be given the same
    # treatment, which means building the tools first.
    _validate_repetition_limit(repetition_limit)
    tools = build_tools(
        action_log,
        verbosity=verbosity,
        peek_at_every_step=peek_at_every_step,
        repetition_nudge=repetition_nudge,
        repetition_limit=repetition_limit,
        privacy_mode=registry.privacy_mode,
        max_steps_per_call=max_steps_per_call,
        operations=operations,
    )
    resolved_model = model or pick_model()

    prompt = (
        system_prompt
        if system_prompt is not None
        else default_system_prompt(registry, peek_at_every_step=peek_at_every_step)
    )

    defaults: dict[str, Any] = {"parallel_tool_calls": False}
    if isinstance(resolved_model, str) and resolved_model.startswith("anthropic:"):
        defaults["anthropic_cache_tool_definitions"] = True

    return Agent(
        resolved_model,
        deps_type=dale.DataRegistry,
        output_type=DaleResult,
        tools=tools,
        system_prompt=prompt,
        model_settings={**defaults, **(model_settings or {})},
    )


def run_agent(
    agent: Agent,
    task: str,
    *,
    deps: dale.DataRegistry,
    action_log: ActionLog,
    verbosity: Verbosity = "quiet",
) -> AgentRunOutcome:
    """Thin wrapper around `agent.run_sync(task, deps=deps)` — exists for two
    things a bare `run_sync` call structurally can't provide on its own:

    1. Under `verbosity="raw"`, flushes the trailing raw messages (the last
       tool return, plus the model's closing structured output) after the
       run completes. Nothing inside `run_plan_fn`'s per-call hook can ever
       show these — no further tool call follows them, so nothing triggers
       the live printing. Without this, "raw" mode would silently miss the
       end of every run — the exact gap flagged when this was designed.
    2. Always prints a final `Result:` section when `verbosity != "quiet"` —
       `Result: Success` with the output's `handle`/`exported_to` and `note`,
       or `Result: Failure` with the exception — so a run's outcome is
       visible the same way every step already was, not left for the caller
       to check and print separately.

    Catches the same broad Exception eval/harness.py's run_trial already
    does, for the same reason: a full run failure (a usage-limit hit, a
    model/provider error) is a failed run to report, not something that
    should propagate and crash the caller.

    Token usage is reported on both paths. That needs a caller-owned RunUsage
    handed to run_sync via its `usage=` parameter rather than reading
    `result.usage` afterward: on the failure path there is no result to read,
    and a failed run is exactly the one whose token spend is most worth
    knowing (a repetition loop, an exhausted request budget). The accumulator
    keeps whatever was spent before the exception.
    """
    accumulator = RunUsage()
    # Snapshotted, not read absolutely, for the same reason the RunUsage
    # accumulator is caller-owned rather than read off the result: the log
    # outlives the run. `wall_clock_s` below covers exactly this call, so the
    # host-compute figure it's split against has to as well, or a caller
    # reusing one ActionLog across two run_agent calls gets a second line
    # charging it for the first run's work. See _render_run_timing.
    host_ms_before = action_log.total_host_ms
    auto_inspect_ms_before = action_log.total_auto_inspect_ms
    started = time.perf_counter()
    try:
        result = agent.run_sync(task, deps=deps, usage=accumulator)
    except Exception as exc:
        wall_clock_s = time.perf_counter() - started
        usage = TokenUsage.from_run(accumulator, model=agent.model)
        if verbosity != "quiet":
            block = f"Result: Failure\n  {type(exc).__name__}: {exc}"
            if rendered := usage.render():
                block += f"\n{rendered}"
            block += "\n" + _render_run_timing(
                wall_clock_s,
                action_log,
                host_ms_before=host_ms_before,
                auto_inspect_ms_before=auto_inspect_ms_before,
            )
            print(block, flush=True)
        return AgentRunOutcome(
            success=False, error=repr(exc), usage=usage, wall_clock_s=wall_clock_s
        )

    if verbosity == "raw":
        all_messages = result.all_messages()
        tail = all_messages[action_log.raw_messages_seen :]
        if tail:
            print(render_raw_messages(tail), flush=True)
        action_log.raw_messages_seen = len(all_messages)

    wall_clock_s = time.perf_counter() - started
    usage = TokenUsage.from_run(accumulator, result.all_messages(), model=agent.model)

    if verbosity != "quiet":
        out = result.output
        if out.handle:
            where = f"handle={out.handle!r}"
        elif out.exported_to:
            where = f"exported_to={out.exported_to!r}"
        else:
            where = "(no handle or exported_to set)"
        block = f"Result: Success\n  {where}\n  note: {out.note}"
        if rendered := usage.render():
            block += f"\n{rendered}"
        block += "\n" + _render_run_timing(
            wall_clock_s,
            action_log,
            host_ms_before=host_ms_before,
            auto_inspect_ms_before=auto_inspect_ms_before,
        )
        print(block, flush=True)

    return AgentRunOutcome(
        success=True, result=result, usage=usage, wall_clock_s=wall_clock_s
    )
