"""Trial-runner harness: runs an LLM agent N
times against a use case, checks each run's result against independently-
computed ground truth by inspecting registry state after the run (never by
parsing the model's free-text answer — structured-data checking, consistent
with DALE's own philosophy), and reports success rate, failure-mode
categorization, and the wasted-turn-rate metric.

Routes every run through dale.agent.build_agent/ActionLog directly — the same
path examples/04 and a real deployment would use, not a shortcut.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_ai.usage import RunUsage

import dale
from dale.agent import (
    ALL_OPERATIONS,
    ActionLog,
    TokenUsage,
    build_agent,
    registry_state_summary,
)

UseCaseSetup = Callable[[dale.DataRegistry], None]
Checker = Callable[[dale.DataRegistry, ActionLog, str], bool]


def describe_setup(use_case_name: str, setup: UseCaseSetup, task: str) -> str:
    """The "story" of a use case, printed once before any trial runs: the
    task text, which virtual files are registered, what handles setup()
    produces from them, and which operations the model has to work with.
    Runs setup() on a throwaway registry purely to describe the resulting
    state — every trial gets its own fresh registry built the same way."""
    registry = dale.DataRegistry(files=dale.FileRegistry())
    setup(registry)

    files = registry.files.list_names() if registry.files else []
    file_lines = [f"  - {name}" for name in files] if files else ["  (none)"]

    lines = [
        f"=== {use_case_name} ===",
        "",
        "TASK:",
        f"  {task}",
        "",
        "FILES REGISTERED:",
        *file_lines,
        "",
        "STARTING STATE (handles the model will see):",
        *(f"  {line}" for line in registry_state_summary(registry).splitlines()),
        "",
        "OPERATIONS AVAILABLE:",
        # Names alone said what the model could reach for but nothing about
        # what any of it does, which is the part worth having in a trial log
        # you read weeks later.
        *(f"  {line}" for line in dale.render_catalog(format="compact").splitlines()),
    ]
    return "\n".join(lines)


@dataclass
class TrialResult:
    trial: int
    model: str
    use_case: str
    success: bool
    final_answer: str
    action_log: ActionLog
    error: str | None = None
    blocked_code: str | None = None
    """Set when the model reported it could not do the task at all
    (`DaleResult.status == "blocked"`). Distinct from `error`, which means the
    run broke, and from a plain False `success`, which means it answered
    wrongly. Without this a declined run is indistinguishable from a wrong one
    in batch output -- which cost a real wrong conclusion during the change
    that introduced blocked results."""
    usage: TokenUsage = field(default_factory=TokenUsage)
    wall_clock_s: float = 0.0

    @property
    def host_ms(self) -> float:
        """Host-side compute for this trial, from the action log's per-call
        timing. `wall_clock_s - host_ms/1000` is time spent waiting on the
        model."""
        return self.action_log.total_host_ms


@dataclass
class TrialSummary:
    use_case: str
    model: str
    n: int
    successes: int
    results: list[TrialResult]

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def wasted_turn_rate(self) -> float:
        """Fraction of logged operation calls that were rejected (status ==
        "error") — the metric the evaluation pilot added to test whether a
        closed, schema-validated action space produces fewer wasted turns
        from bad emissions than free-form code generation. `cost_gate_exceeded`
        is not counted as wasted — it's a correct safety response to an
        oversized op, not a mistake."""
        total = sum(len(r.action_log.entries) for r in self.results)
        if not total:
            return 0.0
        wasted = sum(
            1 for r in self.results for e in r.action_log.entries if e.status == "error"
        )
        return wasted / total

    @property
    def measured(self) -> list[TrialResult]:
        """Trials with real token data. Excludes TestModel runs, whose usage
        figures are synthetic — see dale.agent.TokenUsage."""
        return [r for r in self.results if r.usage.available]

    def usage_stats(self) -> dict[str, float]:
        """Mean token spend per trial, plus the worst peak context seen.

        Means are over *measured* trials only, so a batch mixing real and
        synthetic runs doesn't dilute the average with zeros. Peak context is
        reported as a max, not a mean: paper.md Section 4.2(C)'s claim is that
        DALE's context stays bounded, and the number that tests a bound is the
        largest one observed, not the typical one."""
        measured = self.measured
        if not measured:
            return {}
        n = len(measured)
        with_peak = [r for r in measured if r.usage.peak_available]
        stats = {
            "n_measured": float(n),
            "mean_requests": sum(r.usage.requests for r in measured) / n,
            "mean_tool_calls": sum(r.usage.tool_calls for r in measured) / n,
            "mean_input_tokens": sum(r.usage.input_tokens for r in measured) / n,
            "mean_output_tokens": sum(r.usage.output_tokens for r in measured) / n,
        }
        if with_peak:
            stats["max_peak_context_tokens"] = float(
                max(r.usage.peak_context_tokens for r in with_peak)
            )
        return stats

    def timing_stats(self) -> dict[str, float]:
        """Mean wall clock per trial, split into host compute and time spent
        waiting on the model.

        Unlike usage_stats this covers *every* trial, including TestModel
        ones: elapsed time is real regardless of which model produced it.
        The interesting output is the host share, which is normally a rounding
        error — the point of measuring it is that it stays one as datasets
        grow, since an operation's cost scales with rows while a round trip
        doesn't."""
        timed = [r for r in self.results if r.wall_clock_s > 0]
        if not timed:
            return {}
        n = len(timed)
        wall = sum(r.wall_clock_s for r in timed) / n
        host = sum(r.host_ms for r in timed) / n
        return {
            "n_timed": float(n),
            "mean_wall_clock_s": wall,
            "mean_host_ms": host,
            "mean_model_s": max(0.0, wall - host / 1000),
            "host_share_pct": (host / 1000 / wall * 100) if wall > 0 else 0.0,
        }

    def failure_mode_counts(self) -> Counter:
        """Error codes seen across all trials, most common first — the
        categorization DESIGN.md's evaluation design calls for
        (wrong operation, malformed predicate, wrong threshold, ...)."""
        counts: Counter = Counter()
        for r in self.results:
            for e in r.action_log.entries:
                if e.status == "error":
                    counts[e.result.get("code", "UNKNOWN")] += 1
            # A declined run has no failed *call* to count, so it would
            # otherwise vanish from this tally entirely -- the one failure mode
            # that leaves no trace in the action log.
            if r.blocked_code:
                counts[f"BLOCKED:{r.blocked_code}"] += 1
        return counts

    def render(self, show_action_logs: bool = True) -> str:
        """`show_action_logs=True` (default) prints each trial's full action
        log — predicates included, rendered as human-readable boolean logic
        via ActionLog.render() / grammar.render_predicate, not raw JSON. This
        is what would have made the UC1/UC2 checker bugs found in this
        project's own first real trial batch visible immediately, instead of
        needing a separate diagnostic script — worth the extra output length."""
        lines = [
            f"{self.use_case} / {self.model}: {self.successes}/{self.n} "
            f"({self.success_rate:.0%}) success, "
            f"{self.wasted_turn_rate:.1%} wasted-turn rate "
            f"({sum(len(r.action_log.entries) for r in self.results)} total calls)",
        ]
        stats = self.usage_stats()
        if stats:
            n_measured = int(stats["n_measured"])
            # Against the trials that actually produced a result, not against
            # `self.n` (how many were *requested*): a batch interrupted after 3
            # of 10 would otherwise report "7 unmeasured", folding "never ran"
            # into a word that means "ran, but its token figures aren't real".
            # Those are different facts, and only the second one is this
            # note's job to report.
            unmeasured = len(self.results) - n_measured
            note = f" ({unmeasured} unmeasured)" if unmeasured else ""
            line = (
                f"  tokens (mean of {n_measured}{note}): "
                f"{stats['mean_requests']:.1f} requests, "
                f"{stats['mean_tool_calls']:.1f} tool calls, "
                f"{stats['mean_input_tokens']:,.0f} in / "
                f"{stats['mean_output_tokens']:,.0f} out"
            )
            if "max_peak_context_tokens" in stats:
                line += f"; max peak context {stats['max_peak_context_tokens']:,.0f}"
            lines.append(line)
        timing = self.timing_stats()
        if timing:
            lines.append(
                f"  time (mean): {timing['mean_wall_clock_s']:.1f}s wall — "
                f"{timing['mean_host_ms']:.1f} ms host compute "
                f"({timing['host_share_pct']:.2f}%), "
                f"{timing['mean_model_s']:.1f}s model + network"
            )
        modes = self.failure_mode_counts()
        if modes:
            lines.append(
                "  failure modes: " + ", ".join(f"{k}={v}" for k, v in modes.most_common())
            )
        for r in self.results:
            if not r.success:
                if r.blocked_code:
                    detail = f"BLOCKED {r.blocked_code} — {r.final_answer[:160]}"
                else:
                    detail = r.error or "result did not match expected ground truth"
                lines.append(f"  trial {r.trial} FAILED: {detail}")
        if show_action_logs:
            for r in self.results:
                status = "ok" if r.success else "FAILED"
                lines.append(f"\n--- trial {r.trial} ({status}) action log ---")
                lines.append(r.action_log.render(debug=True) or "  (no calls logged)")
        return "\n".join(lines)


def run_trial(
    trial: int,
    use_case_name: str,
    setup: UseCaseSetup,
    task: str,
    checker: Checker,
    model: str,
    *,
    verbose: bool = True,
    max_steps_per_call: int | None = None,
    usage_limits: Any | None = None,
    operations: Any = ALL_OPERATIONS,
    privacy_mode: bool = False,
) -> TrialResult:
    """`max_steps_per_call=1` runs the same use case one operation per round
    trip — paired with TokenUsage.requests, that's paper.md Section 4.2 part
    (F)'s request-count comparison against the unrestricted default, needing no
    other mechanism. It replaces the old `enable_run_plan=False`: there is no
    second tool surface to withhold any more, so the unbatched condition is now
    expressed as a ceiling on batch size rather than as a missing tool.

    `usage_limits` is forwarded to run_sync (a pydantic_ai.usage.UsageLimits).
    pydantic-ai already applies request_limit=50 when none is given; passing
    one explicitly is how a live batch caps its own spend."""
    registry = dale.DataRegistry(files=dale.FileRegistry(), privacy_mode=privacy_mode)
    setup(registry)
    action_log = ActionLog()
    action_log.seed_from_registry(registry)
    agent = build_agent(
        registry,
        action_log,
        model=model,
        verbosity="debug" if verbose else "quiet",
        max_steps_per_call=max_steps_per_call,
        # Explicitly the whole catalog, not the default. build_agent's default
        # is CORE_OPERATIONS, which excludes window_flag and graph_walk_resolve
        # -- uc2 and uc4 would silently become unsolvable, and every other
        # number would describe a different system than the ones already
        # recorded in the project's measurement notes. A trial arm that wants a
        # narrowed catalog passes it deliberately; the baseline never inherits
        # one from a default that moved.
        operations=operations,
    )

    # Owned here, not read off the result afterward: run_sync accumulates into
    # this object, so a trial that raises still reports what it spent. Failed
    # trials are the ones whose token cost matters most (a repetition loop, an
    # exhausted request budget) — reading result.usage would lose exactly them.
    accumulator = RunUsage()
    started = time.perf_counter()

    try:
        result = agent.run_sync(task, deps=registry, usage=accumulator, usage_limits=usage_limits)
        final_answer = str(result.output)
        success = checker(registry, action_log, final_answer)
        blocked_code = getattr(result.output, "code", None) if (
            getattr(result.output, "status", "ok") == "blocked"
        ) else None
        return TrialResult(
            trial=trial,
            model=model,
            use_case=use_case_name,
            success=success,
            final_answer=final_answer,
            action_log=action_log,
            blocked_code=blocked_code,
            usage=TokenUsage.from_run(accumulator, result.all_messages(), model=agent.model),
            wall_clock_s=time.perf_counter() - started,
        )
    except Exception as exc:  # a full run failure is a failed trial, not a crashed harness
        return TrialResult(
            trial=trial,
            model=model,
            use_case=use_case_name,
            success=False,
            final_answer="",
            action_log=action_log,
            error=repr(exc),
            usage=TokenUsage.from_run(accumulator, model=agent.model),
            wall_clock_s=time.perf_counter() - started,
        )


def run_trials(
    use_case_name: str,
    setup: UseCaseSetup,
    task: str,
    checker: Checker,
    model: str,
    n: int,
    *,
    verbose: bool = True,
    max_steps_per_call: int | None = None,
    usage_limits: Any | None = None,
    operations: Any = ALL_OPERATIONS,
    privacy_mode: bool = False,
) -> TrialSummary:
    """Prints progress at two levels, both flushed immediately rather than
    buffered until the whole batch finishes: per-trial start/elapsed-time
    markers here, and (when verbose=True, the default) each individual tool
    call as it happens inside a trial, via dale.agent.build_tools' verbose
    mode — a live model run can legitimately take tens of seconds to a few
    minutes (retries, exploratory tool calls), and with no output in between
    at either level, a slow-but-working run is indistinguishable from a hang.

    `max_steps_per_call` and `usage_limits` are passed straight through to every
    trial unchanged — see run_trial for what each one is for. They belong to a
    whole batch rather than to individual trials by construction: an arm of the
    part-(F) comparison is only meaningful if every trial in it ran under the
    same conditions, so there is deliberately no way to vary them per trial."""
    results = []
    for i in range(n):
        print(f"  [{i + 1}/{n}] running...", flush=True)
        start = time.monotonic()
        result = run_trial(
            i + 1,
            use_case_name,
            setup,
            task,
            checker,
            model,
            verbose=verbose,
            max_steps_per_call=max_steps_per_call,
            usage_limits=usage_limits,
            operations=operations,
            privacy_mode=privacy_mode,
        )
        elapsed = time.monotonic() - start
        outcome = "ok" if result.success else "FAILED"
        print(
            f"  [{i + 1}/{n}] {outcome} in {elapsed:.1f}s "
            f"({len(result.action_log.entries)} calls)",
            flush=True,
        )
        results.append(result)
    return TrialSummary(
        use_case=use_case_name,
        model=model,
        n=n,
        successes=sum(1 for r in results if r.success),
        results=results,
    )
