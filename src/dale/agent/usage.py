"""What a run cost: tokens, and the host-compute vs. waiting-on-the-model split.

`TokenUsage` deliberately carries two different input-token figures because
they answer different questions — cumulative spend across every request, and
the largest single request's prompt, which is what a bounded-context claim is
actually about. Reporting only one is misleading, and the distinction was
confirmed by measurement rather than assumed.

`_render_run_timing` is the other half: host compute against wall clock. The
ratio is the single most counter-intuitive fact about operating DALE — one
measured 10,005-row run spent ~4 ms on host compute against ~18.6 s of wall
clock — which is why it prints on every run rather than only when profiling.

`DaleResult` and `AgentRunOutcome` are here for the same reason: one is what
the model produced, the other is whether producing it succeeded and what it
cost. Splitting them from the numbers that describe the same run would put a
circular import between this module and the entry points that use all of it.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResult

from dale.agent.log import ActionLog


class DaleResult(BaseModel):
    """The agent's structured final output — the same "never trust
    LLM-authored content as the actual data" principle
    applied to the last step of a run, not just the ones in between.
    Exactly one of `handle`/`exported_to` is expected to be set: `handle`
    when the result was left as a handle in the registry (the invoker calls
    registry.materialize(handle) themselves for the real data — never this
    model's own fields); `exported_to` when the run ended with
    export_handle writing straight to a file, bypassing the LLM's context
    entirely."""

    handle: str | None = Field(
        None, description="The handle holding this task's final result, if left in memory."
    )
    exported_to: str | None = Field(
        None,
        description="The destination name results were written to via export_handle, if any.",
    )
    note: str = Field(
        ...,
        description="One short sentence summarizing the result, for a human skimming the "
        "log. Not the data itself — the invoker reads `handle` or `exported_to` for that.",
    )


def _is_synthetic_model(model: Any) -> bool:
    """True for pydantic-ai's own offline stand-ins (TestModel/FunctionModel).

    They report plausible-looking but meaningless usage — a bare TestModel run
    returns input_tokens=51 — so token figures from them must be marked
    unavailable rather than averaged in. Without this guard a free structural
    batch produces a confident "0 tokens" mean that actually means "no data",
    which is exactly the kind of number that reaches a paper by accident."""
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.test import TestModel

    if isinstance(model, (TestModel, FunctionModel)):
        return True
    return isinstance(model, str) and model == "test"


@dataclass
class TokenUsage:
    """One run's token spend, flattened out of pydantic-ai's usage objects.

    Deliberately carries two different input-token figures, because they
    answer different questions and reporting only one is misleading:

    - `input_tokens` is *cumulative* across every request in the run — what
      the run cost. It grows with the number of turns even when the context
      window is perfectly bounded, so it is not a context measurement.
    - `peak_context_tokens` is the largest single request's prompt, which is
      what a bounded-context claim (paper.md Section 4.2 part C) is actually
      about. Cache-read tokens are added in because a cached prefix still
      occupies the context window while being excluded from input_tokens.

    Measured directly to confirm the distinction is real, not theoretical: a
    2-request run reporting RunUsage(input_tokens=103) was two requests of 51
    and 52 — the peak is 52, not 103.

    `peak_available` is separate from `available` because the two come from
    different places. Cumulative figures come from a caller-owned RunUsage
    accumulator, which survives a run that raises; the peak needs
    result.all_messages(), which an exception discards. A failed run therefore
    reports real cumulative spend and no peak — and those are precisely the
    runs whose cost is most worth knowing (a repetition loop, a usage-limit
    hit), so a silently-zero peak would be worse than an explicit flag.
    """

    available: bool = False
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    peak_context_tokens: int = 0
    peak_available: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_run(
        cls,
        accumulator: Any,
        messages: Sequence[Any] | None = None,
        *,
        model: Any = None,
    ) -> "TokenUsage":
        """Build from a caller-owned RunUsage (passed to run_sync via its
        `usage=` parameter) plus, when the run succeeded, its messages.

        The broad `except` below covers exactly one thing: *reading the
        reported numbers*. A provider that reports usage oddly, or not at all,
        must not turn a real run into a failed one, so anything unexpected
        while pulling fields off the accumulator or the messages degrades to an
        unavailable instance.

        It deliberately does not cover synthetic-model detection, which is why
        that call is hoisted out above it. _is_synthetic_model does runtime
        imports of pydantic_ai's internal model modules; inside the try, the
        day either module path moves, *every* call would quietly return an
        all-zero unavailable TokenUsage and the paper's whole measuring
        instrument would evaporate with nothing failing to say so. A broken
        import there is a broken instrument, not a quirky provider, and it
        should be loud."""
        available = not _is_synthetic_model(model)
        try:
            usage = cls(
                available=available,
                requests=getattr(accumulator, "requests", 0) or 0,
                tool_calls=getattr(accumulator, "tool_calls", 0) or 0,
                input_tokens=getattr(accumulator, "input_tokens", 0) or 0,
                cache_read_tokens=getattr(accumulator, "cache_read_tokens", 0) or 0,
                cache_write_tokens=getattr(accumulator, "cache_write_tokens", 0) or 0,
                output_tokens=getattr(accumulator, "output_tokens", 0) or 0,
            )
            if usage.requests == 0:
                usage.available = False
            if messages:
                # Only ModelResponse carries .usage; ModelRequest doesn't.
                peaks = [
                    (getattr(m.usage, "input_tokens", 0) or 0)
                    + (getattr(m.usage, "cache_read_tokens", 0) or 0)
                    for m in messages
                    if getattr(m, "usage", None) is not None
                ]
                if peaks:
                    usage.peak_context_tokens = max(peaks)
                    usage.peak_available = usage.available
            return usage
        except Exception:
            return cls()

    def render(self) -> str:
        """One or two indented lines for run_agent's Result block. Empty when
        nothing was measured, so callers can skip it without a second check."""
        if not self.available:
            return ""
        line = (
            f"  tokens: {self.requests} requests, "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out"
        )
        if self.peak_available:
            line += f"\n          peak context {self.peak_context_tokens:,}"
        return line


@dataclass
class AgentRunOutcome:
    """What run_agent returns — success or failure, always enough to know
    what happened without the caller needing their own try/except around
    run_sync. `result` (the real pydantic-ai AgentRunResult — .output,
    .all_messages(), .usage, etc. all still work; note `usage` is a property
    in pydantic-ai 2.21, not a method) is set only on success; `error` only on
    failure. `usage` is populated either way — see TokenUsage on why a failed
    run still reports real cumulative spend. Mirrors eval/harness.py's own
    TrialResult, which established this same success/error split for the same
    reason: a full run failure is a failed run, not a crash."""

    success: bool
    result: AgentRunResult[DaleResult] | None = None
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    wall_clock_s: float = 0.0
    """Total run wall time. The host-compute half of the split lives on the
    ActionLog (`total_host_ms`), which is where per-call timing is recorded —
    subtract it from this to get model + network time. Note that arithmetic is
    only valid for a log used by exactly one run; run_agent itself does the
    subtraction against a snapshot taken at its own start, so its printed split
    stays per-run even when a caller reuses one ActionLog across several."""


def _render_run_timing(
    wall_clock_s: float,
    action_log: ActionLog,
    *,
    host_ms_before: float = 0.0,
    auto_inspect_ms_before: float = 0.0,
) -> str:
    """The host-compute vs. waiting-on-the-model split, as one line.

    Worth printing on every run, not just when profiling: the ratio is the
    single most counter-intuitive fact about operating DALE. A measured
    10,005-row run spent ~4 ms on host compute against ~18.6 s of wall clock,
    so the data size that dominates the user's mental model contributes almost
    nothing to the time, and round trips are the only thing worth optimizing.

    The two `*_before` snapshots exist because the two halves of the split have
    different lifetimes: `wall_clock_s` measures one run, while an ActionLog's
    totals accumulate over its whole life. Nothing stops a caller from handing
    the same log to run_agent twice (it's the session's trace, and a
    multi-turn session is a perfectly reasonable thing to want), and if that
    happens the second run's line would charge it for the first run's compute —
    reporting a host share that could exceed 100% and a negative model time
    clamped to zero. Taking the totals at run start and diffing keeps the line
    honest per run without the log having to know anything about runs."""
    host_ms = action_log.total_host_ms - host_ms_before
    model_s = max(0.0, wall_clock_s - host_ms / 1000)
    share = (host_ms / 1000 / wall_clock_s * 100) if wall_clock_s > 0 else 0.0
    line = (
        f"  time: {wall_clock_s:.1f}s wall — "
        f"{host_ms:.1f} ms host compute ({share:.2f}%), "
        f"{model_s:.1f}s model + network"
    )
    if (inspect_ms := action_log.total_auto_inspect_ms - auto_inspect_ms_before) >= 0.05:
        line += f"\n        (host compute includes {inspect_ms:.1f} ms of auto-inspect)"
    return line
