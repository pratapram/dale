"""The intent/action log — DALE's audit trail, and how a run renders.

`ActionLog` stands in for a resume/checkpoint
feature: on failure a human reads it to see what a run did, without needing
full registry-state serialization. It is also the artifact the evaluation
compares against an expected call sequence, for per-step rather than only
end-to-end correctness.

Everything here is presentation and record-keeping. It knows nothing about how
a call was dispatched or how many arrived together — deliberately, since
paper.md Section 3.12's claim that a batched call is indistinguishable in the
trace from the same steps issued one at a time is exactly the property this
module not knowing the difference is what guarantees.
"""

import json
import textwrap
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, TypeAdapter

import dale
from dale.grammar import Predicate, render_predicate

_predicate_adapter: TypeAdapter[Any] = TypeAdapter(Predicate)


Verbosity = Literal["quiet", "normal", "debug", "raw"]
"""Controls both whether a call prints live as it happens and how much detail
it shows: "quiet" prints nothing during the run; "normal" prints the Action
line and Registry State but not the raw Result payload; "debug" adds the full
JSON-indented Result body. "raw" adds one more layer on top of "debug": the
literal wire-level exchange with the model (system/user prompt, the model's
own tool-call args before intent-stripping, tool returns, free text,
thinking) interleaved right before each call's own human-readable block --
see render_raw_messages and run_agent (the latter is required to see the
final leg of a run under "raw", since nothing else triggers after the last
tool call). The quiet/normal/debug three-way split applies to
ActionLog.render()'s post-hoc output via its own `debug` parameter; "raw" has
no post-hoc equivalent there since it's about the live model exchange, not
the operation-call trace ActionLog itself records."""


def _render_raw_part(part: Any) -> str:
    """One raw pydantic-ai message part, rendered as a labeled block — the
    literal wire-level content (the model's own tool-call args before
    intent-stripping/validation, the system/user prompt text, free text,
    thinking), not DALE's own dispatch-level view of the same events.
    Recognizes the common part types by name (avoids a hard pydantic-ai
    version-specific import list); anything else falls back to a generic
    labeled repr rather than being silently dropped."""
    kind = type(part).__name__
    if kind in ("SystemPromptPart", "InstructionPart"):
        return f"[SYSTEM PROMPT]\n{part.content}"
    if kind == "UserPromptPart":
        return f"[USER PROMPT]\n{part.content}"
    if kind in ("ToolCallPart", "NativeToolCallPart"):
        args = part.args
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        return f"[MODEL -> TOOL CALL] {part.tool_name}({json.dumps(args, default=str)})"
    if kind in ("ToolReturnPart", "NativeToolReturnPart"):
        return f"[TOOL -> MODEL RETURN] {part.tool_name}: {json.dumps(part.content, default=str)}"
    if kind == "TextPart":
        return f"[MODEL TEXT] {part.content}"
    if kind == "ThinkingPart":
        return f"[MODEL THINKING] {part.content}"
    if kind == "RetryPromptPart":
        content = part.content if isinstance(part.content, str) else part.model_response()
        return f"[RETRY PROMPT] {content}"
    return f"[{kind}] {part!r}"


def render_raw_messages(messages: Sequence[Any]) -> str:
    """Human-readable rendering of raw pydantic-ai ModelMessage objects —
    the literal to-and-fro with the model (system/user prompt, the model's
    own tool-call args, tool returns, free text, thinking), not DALE's own
    Action/Result/Registry State view of the same events. Used two ways:
    live, per new message batch, inside each tool call under
    verbosity="raw" (see run_plan_fn); and once more by run_agent after a
    run completes, to flush the trailing messages (the last tool return plus
    the model's closing output) that never trigger another tool call, so
    nothing else would ever print them."""
    return "\n\n".join(
        _render_raw_part(part) for msg in messages for part in msg.parts
    )


def _format_bytes(n: float) -> str:
    """Human-readable byte count for the Registry State table's Memory
    column — same avg_record_bytes * size approximation cost.py's own
    estimator already uses (documented there as a deliberate approximation,
    not a precise accounting), just formatted for a human instead of
    compared against a threshold."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


class HandleLabel(BaseModel):
    """Agent-layer log/display record for a handle's provenance — `handle`
    itself IS the real DataRegistry identifier now (see DataHandle), so
    there's no separate name to track here; this just carries what a step
    of the trace/registry-state table wants to show alongside it."""

    handle: str
    step: int
    operation: str
    type: str | None = None
    size: int | None = None
    avg_record_bytes: int | None = None
    description: str | None = None


class ActionLogEntry(BaseModel):
    step: int
    intent: str
    operation: str
    params: dict[str, Any]
    status: str
    result: dict[str, Any]
    # Host-side wall time for this step, split so DALE's own overhead can't
    # hide inside the operation's number. `elapsed_ms` is the
    # dale.call_operation call itself — the actual data processing;
    # `auto_inspect_ms` is peek_at_every_step's extra peek/describe splice,
    # 0.0 when it didn't run. That feature is documented as costing no extra
    # turn and no extra log entry, which is true, but it isn't free in host
    # time, and a number that says so is better than a claim that implies
    # otherwise. Neither includes any model or network time: summing every
    # entry's total and subtracting from a run's wall clock is what separates
    # host compute from waiting on the LLM.
    elapsed_ms: float = 0.0
    auto_inspect_ms: float = 0.0
    # Snapshots of ActionLog.alive taken immediately before/after this entry
    # was recorded — captured once, at record() time, rather than
    # reconstructed later by replaying the whole log. `alive_before` is what
    # annotates this call's own handle references (so e.g. a release_handle
    # call can still show what it released, even though that handle is gone
    # from `alive_after`); `alive_after` is what the registry-state listing
    # below this entry shows.
    alive_before: dict[str, HandleLabel] = Field(default_factory=dict)
    alive_after: dict[str, HandleLabel] = Field(default_factory=dict)


class ActionLog(BaseModel):
    """Append-only (intent -> tool call -> result) trace of one agent run.

    In place of a resume/checkpoint feature: on failure, a
    human reads this to see what happened without needing full registry-state
    serialization. Also the artifact the evaluation compares against an
    expected call sequence for per-step (not just end-to-end) correctness.
    """

    entries: list[ActionLogEntry] = Field(default_factory=list)
    alive: dict[str, HandleLabel] = Field(default_factory=dict)
    """Working state, incrementally updated by record() — grows on a
    handle-creating call, shrinks on a successful release_handle. Each
    entry's alive_before/alive_after are frozen copies of this at the time,
    so rendering never needs to replay the log to reconstruct history."""

    raw_messages_seen: int = 0
    """Only meaningful under verbosity="raw" — how many of this run's raw
    pydantic-ai messages (RunContext.messages / result.all_messages()) have
    already been printed live, so run_plan_fn's per-call hook shows only
    what's new since the last call, and run_agent's post-run tail flush
    shows only what wasn't already shown live. Untouched (stays 0) at any
    other verbosity."""

    def record(
        self,
        *,
        intent: str,
        operation: str,
        params: dict[str, Any],
        result: dict[str, Any],
        elapsed_ms: float = 0.0,
        auto_inspect_ms: float = 0.0,
    ) -> ActionLogEntry:
        step = len(self.entries) + 1
        entry = ActionLogEntry(
            step=step,
            intent=intent,
            operation=operation,
            params=params,
            status=result.get("status", "unknown"),
            result=result,
            elapsed_ms=elapsed_ms,
            auto_inspect_ms=auto_inspect_ms,
            alive_before=dict(self.alive),
        )
        self.entries.append(entry)

        handle_meta = result.get("handle") if isinstance(result, dict) else None
        if isinstance(handle_meta, dict) and "name" in handle_meta:
            handle_id = handle_meta["name"]
            self.alive[handle_id] = HandleLabel(
                handle=handle_id,
                step=step,
                operation=operation,
                type=handle_meta.get("type"),
                size=handle_meta.get("size"),
                avg_record_bytes=handle_meta.get("avg_record_bytes"),
                description=handle_meta.get("description"),
            )
        if operation == "release_handle" and entry.status == "ok":
            released = params.get("handle")
            if released is not None:
                self.alive.pop(released, None)

        entry.alive_after = dict(self.alive)
        return entry

    @property
    def total_host_ms(self) -> float:
        """Host-side wall time across every logged call — operation execution
        plus peek_at_every_step's inspection overhead.

        Named for what it totals rather than for the field it starts from: the
        per-entry `elapsed_ms` is operation-only, so a `total_elapsed_ms` would
        invite the obvious-looking reimplementation `sum(e.elapsed_ms ...)`,
        which silently drops auto-inspect from the very split this exists to
        produce.

        Subtracting this from a run's total wall clock is what separates data
        processing from waiting on the model. The gap is normally enormous:
        one measured 10,005-row run spent ~4 ms here against ~18.6 s of wall
        clock, i.e. host compute was ~0.02% of it. That ratio is the
        resource-cost hierarchy in paper.md Section 3.13 as a measurement
        rather than an argument."""
        return sum(e.elapsed_ms + e.auto_inspect_ms for e in self.entries)

    @property
    def total_auto_inspect_ms(self) -> float:
        """The peek_at_every_step share of total_host_ms, on its own — so
        the cost of a convenience feature is attributable rather than folded
        into the operations it wraps."""
        return sum(e.auto_inspect_ms for e in self.entries)

    def seed_from_registry(self, registry: dale.DataRegistry) -> None:
        """Register handles that exist before the agent's first tool call —
        e.g. CSV files or in-memory data an eval harness or invoker loaded
        directly via dale.call_operation, bypassing build_tools() entirely.
        Without this, such handles are real and usable but invisible to
        render()'s registry-state listing, since they never went through
        record(). Call once, right after construction, before the agent
        runs."""
        for meta in registry.list_handles():
            if meta.name in self.alive:
                continue
            self.alive[meta.name] = HandleLabel(
                handle=meta.name,
                step=0,
                operation=meta.created_by,
                type=meta.type,
                size=meta.size,
                avg_record_bytes=meta.avg_record_bytes,
                description=meta.description,
            )

    def render(self, *, debug: bool = False) -> str:
        """`debug=False` (default) omits the raw Result JSON body — just the
        Action line, a one-line Result status (with a concise error summary if
        the call failed), and Registry State. Pass `debug=True` for the full
        JSON-indented payload on every call, useful when you need to see
        exactly what an operation returned, not just that it succeeded."""
        blocks = [
            f"{self._render_entry(e, debug=debug)}\n{self._render_registry_state(e.alive_after)}"
            for e in self.entries
        ]
        return "\n\n".join(blocks)

    @classmethod
    def _render_entry(cls, e: ActionLogEntry, *, debug: bool = False) -> str:
        """Call rendered as a single-line handle.operation(args) signature —
        most operations are effectively a method call on a handle — with the
        result as a separate, JSON-indented block. Two visually distinct
        pieces instead of one wrapped line of Python dict repr, which was
        hard to parse, especially for anything returning a multi-record
        sample (e.g. peek). The Result JSON body itself is opt-in
        (`debug=True`): most of it (peek/describe samples, full DataHandle
        dumps) is exactly the kind of noise Registry State/the Action line
        already summarize; an error's code+message is still worth a
        one-liner even without debug, since "something failed" without why
        isn't useful."""
        call = cls._render_call(e.operation, cls._render_params(e.params))
        call = cls._render_assignment(e, call)
        timing = cls._render_timing(e)
        if debug:
            body = textwrap.indent(json.dumps(e.result, indent=2, default=str), "      ")
            return_line = f"    Result: {e.status}{timing}\n{body}"
        elif e.status == "ok":
            return_line = f"    Result: {e.status}{timing}"
        else:
            result = e.result if isinstance(e.result, dict) else {}
            code = result.get("code", "UNKNOWN")
            message = result.get("message", "")
            return_line = f"    Result: {e.status}{timing}: {code} — {message}"
        return f"[{e.step}] Intent: {e.intent}\n    Action: {call}\n{return_line}"

    @staticmethod
    def _render_timing(e: ActionLogEntry) -> str:
        """` (1.3 ms)`, or ` (1.3 ms + 0.4 ms inspect)` when
        peek_at_every_step added work. Empty for an unmeasured entry (e.g. one
        constructed directly in a test), so nothing renders a misleading
        0.0 ms. Sub-millisecond calls — most of them, at these data sizes —
        still print rather than rounding to zero, since "too fast to measure"
        is itself the finding."""
        if e.elapsed_ms <= 0 and e.auto_inspect_ms <= 0:
            return ""
        out = f" ({e.elapsed_ms:.1f} ms" if e.elapsed_ms >= 0.05 else " (<0.1 ms"
        if e.auto_inspect_ms >= 0.05:
            out += f" + {e.auto_inspect_ms:.1f} ms inspect"
        return out + ")"

    @staticmethod
    def _render_assignment(e: ActionLogEntry, call: str) -> str:
        """Prefix a handle-creating call with `name = `, so it's clear at a
        glance *when* a handle was created, not just that it exists once you
        reach the Result block or Registry State further down — e.g.
        `people_list = load_csv(file='people.csv')` instead of a bare
        `load_csv(file='people.csv')` that never says what it produced.
        `name` here is the handle's own real identifier (DataHandle.name),
        not a separate label — so unlike the pre-unification version of this
        method, there's no disambiguation to do: collisions are rejected
        outright at registry.create() time, so two alive handles can never
        render to the same name."""
        if not isinstance(e.result, dict):
            return call
        handle_meta = e.result.get("handle")
        if not isinstance(handle_meta, dict) or "name" not in handle_meta:
            return call
        return f"{handle_meta['name']} = {call}"

    @classmethod
    def _render_call(cls, operation: str, params: dict[str, Any]) -> str:
        """`handle.operation(arg=val, ...)` when params has a natural
        receiver, else `operation(arg=val, ...)`. The receiver is the
        `handle` field if present; otherwise the first `*_handle` field in
        declared order (join_lookup/graph_walk_resolve have two — the first
        one wins, the rest render as regular keyword args); operations with
        no handle field at all (load_csv, which creates one rather than
        operating on one) get no receiver. A handle reference needs no
        annotation/lookup to be readable — it already IS the semantic name
        (e.g. `people_list.peek(n=5)`) — since DataHandle.name and the
        LLM-supplied name are the same field. `name`/`description` are
        dropped from the argument list: `name` already appears via the
        `= ` assignment prefix, and `description` is shown in REGISTRY
        STATE below — repeating either here is noise."""
        receiver_key = None
        if "handle" in params:
            receiver_key = "handle"
        else:
            for key in params:
                if key.endswith("_handle"):
                    receiver_key = key
                    break

        remaining = {
            k: v for k, v in params.items() if k not in (receiver_key, "name", "description")
        }
        args = ", ".join(f"{k}={v!r}" for k, v in remaining.items())
        if receiver_key is not None:
            return f"{params[receiver_key]}.{operation}({args})"
        return f"{operation}({args})"

    @classmethod
    def _render_registry_state(cls, alive: dict[str, HandleLabel]) -> str:
        """A column-aligned table (HANDLE, TYPE, SIZE, MEMORY, DESCRIPTION)
        instead of one densely punctuated line per handle. HANDLE is the
        real DataRegistry identifier — also what raw DaleError/dispatch
        output uses — and doubles as the semantic name now that the two are
        unified; no separate NAME column. TYPE is the handle's type
        (list/dict/set). MEMORY is `avg_record_bytes * size` (the same
        approximation cost.py's own estimator uses, see _format_bytes), "-"
        if unknown (e.g. an empty handle has no sample to estimate from).
        DESCRIPTION is left ragged (not wrapped) since it's free text of
        varying length — matches how most CLI table output (kubectl,
        docker ps) handles a trailing free-text column. The single handle
        with the highest `step` (the most recently *created* handle — the
        only kind of "update" a handle can undergo, since DALE handles are
        immutable once created) gets a trailing `*` on its name."""
        if not alive:
            return "    Registry State: (empty)"
        newest_step = max(info.step for info in alive.values())
        rows = [
            (
                info.handle + (" *" if info.step == newest_step else ""),
                info.type or "-",
                str(info.size) if info.size is not None else "-",
                _format_bytes(info.avg_record_bytes * info.size)
                if info.avg_record_bytes is not None and info.size is not None
                else "-",
                info.description or "",
            )
            for info in alive.values()
        ]
        headers = ("Handle", "Type", "Size", "Memory", "Description")
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers) - 1)
        ]

        def fmt(cols: tuple[str, ...]) -> str:
            padded = [f"{cols[i]:<{widths[i]}}" for i in range(len(widths))]
            line = "      " + "  ".join(padded) + ("  " + cols[-1] if cols[-1] else "")
            return line.rstrip()

        lines = [
            "    Registry State:",
            fmt(headers),
            fmt(tuple("-" * w for w in widths) + ("-" * len(headers[-1]),)),
        ]
        lines.extend(fmt(row) for row in rows)
        return "\n".join(lines)

    @staticmethod
    def _render_params(params: dict[str, Any]) -> dict[str, Any]:
        """Render any `predicate` value as boolean-logic text (see
        grammar.render_predicate) instead of the raw nested dict/JSON the
        model actually produced — for human review, not evaluation. Falls
        back to the raw value if it doesn't parse as a Predicate."""
        if "predicate" not in params:
            return params
        display = dict(params)
        try:
            display["predicate"] = render_predicate(
                _predicate_adapter.validate_python(params["predicate"])
            )
        except Exception:
            pass
        return display
