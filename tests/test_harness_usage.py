"""Token accounting (dale.agent.TokenUsage + eval.harness reporting).

The instrument for paper.md Section 4.2 part (C). These tests exist mainly to
pin two things that would be easy to get quietly wrong and hard to notice in a
published number: that peak context is a max rather than a sum, and that
synthetic-model runs are excluded rather than averaged in as zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.usage import RunUsage

import dale
from dale.agent import ActionLog, TokenUsage, build_agent, build_tools, run_agent
from eval.harness import TrialResult, TrialSummary


def _run_plan(action_log, **kwargs):
    """`(tool, params_model)` for the one tool build_tools returns.

    Reads the params model off `inspect.signature` — the same accessor
    tests/test_agent.py uses. This file used to reach for
    `tool.function.__annotations__["params"]` instead: two accessors for one
    thing, and now both would point at the same single tool."""
    import inspect

    tools = build_tools(action_log, **kwargs)
    assert [t.name for t in tools] == ["run_plan"]
    tool = tools[0]
    return tool, inspect.signature(tool.function).parameters["params"].annotation


@dataclass
class _FakeRequestUsage:
    input_tokens: int = 0
    cache_read_tokens: int = 0


class _FakeResponse:
    """Stands in for a pydantic_ai ModelResponse — only `.usage` is read."""

    def __init__(self, input_tokens: int, cache_read_tokens: int = 0) -> None:
        self.usage = _FakeRequestUsage(input_tokens, cache_read_tokens)


class _FakeRequest:
    """ModelRequest has no usage; TokenUsage must skip it, not crash."""

    usage = None


def test_peak_context_is_the_max_request_not_the_sum():
    """The distinction the whole measurement rests on. RunUsage.input_tokens
    is cumulative (a real 2-request run reported 103 from requests of 51 and
    52); the bounded-context claim is about the peak, which is 52."""
    accumulator = RunUsage(input_tokens=103, output_tokens=12, requests=2)
    messages = [_FakeRequest(), _FakeResponse(51), _FakeRequest(), _FakeResponse(52)]

    usage = TokenUsage.from_run(accumulator, messages, model="anthropic:claude-haiku-4-5")

    assert usage.input_tokens == 103  # cumulative — the cost number
    assert usage.peak_context_tokens == 52  # max — the context number
    assert usage.peak_available is True


def test_peak_context_includes_cache_read_tokens():
    """A cached prefix still occupies the context window while being excluded
    from input_tokens, so it has to be added back in."""
    usage = TokenUsage.from_run(
        RunUsage(input_tokens=10, requests=1),
        [_FakeResponse(10, cache_read_tokens=4_000)],
        model="anthropic:claude-haiku-4-5",
    )
    assert usage.peak_context_tokens == 4_010


def test_usage_unavailable_for_synthetic_models():
    """TestModel reports a plausible input_tokens=51 that means nothing. Left
    unguarded it averages into a confident zero-ish figure that reads as data.

    A FunctionModel driving a real one-step batch, and `assert outcome.success`
    first, because the path being measured has to be pinned. With
    `model="test"` this run now *fails* — TestModel cannot synthesize a
    discriminated-union list item, so it exhausts run_plan's retry budget —
    run_agent catches that, and both assertions below still pass, via the
    failure path. The test would have gone on reading green while no longer
    testing "a *successful* synthetic run reports unavailable usage"."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    registry = dale.DataRegistry()
    registry.create(
        "list", [{"name": "W"}], name="items", description="d", created_by="fixture"
    )
    sent = {"n": 0}

    def emit(messages, info) -> ModelResponse:
        sent["n"] += 1
        if sent["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_plan",
                        {"steps": [{"operation": "peek", "handle": "items", "intent": "look"}]},
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("final_result", {"note": "done"})])

    log = ActionLog()
    agent = build_agent(registry, log, model=FunctionModel(emit))
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="quiet")

    assert outcome.success
    assert len(log.entries) == 1  # the step really ran, through dale.call_operation
    assert outcome.usage.available is False
    assert outcome.usage.render() == ""


def test_failed_run_still_reports_cumulative_spend_but_no_peak():
    """The failure path is the point, not an edge case: a run that raises has
    no result to read usage off, and those are exactly the runs (repetition
    loop, exhausted budget) whose spend is most worth knowing."""
    accumulator = RunUsage(input_tokens=4_200, output_tokens=90, requests=6)

    usage = TokenUsage.from_run(accumulator, None, model="anthropic:claude-haiku-4-5")

    assert usage.available is True
    assert usage.input_tokens == 4_200
    assert usage.requests == 6
    assert usage.peak_available is False  # not silently zero — explicitly absent
    assert "peak context" not in usage.render()


def test_render_shows_both_numbers_with_thousands_separators():
    usage = TokenUsage(
        available=True, requests=4, input_tokens=12_480, output_tokens=310,
        peak_context_tokens=3_905, peak_available=True,
    )
    assert usage.render() == (
        "  tokens: 4 requests, 12,480 in / 310 out\n          peak context 3,905"
    )


def test_from_run_never_raises_on_a_broken_accumulator():
    """A provider reporting usage oddly must not turn a real run into a failed
    one."""

    class Hostile:
        @property
        def requests(self):
            raise RuntimeError("nope")

    assert TokenUsage.from_run(Hostile()).available is False


def test_a_broken_synthetic_model_check_is_loud_not_silently_unavailable(monkeypatch):
    """The other side of that broad `except`: it covers reading the reported
    numbers, not deciding whether they mean anything. _is_synthetic_model does
    runtime imports of pydantic-ai internals, and swallowing a failure there
    would turn every run into an all-zero unavailable TokenUsage — the whole
    measuring instrument gone, with nothing failing to say so."""
    # Patched on dale.agent.usage, not dale.agent: from_run resolves the name in
    # the module that defines it, so patching the re-export in dale.agent's
    # __init__ rebinds an alias nothing reads. Everything else in these tests
    # imports from dale.agent unchanged — this one reaches inside because it is
    # substituting an implementation detail rather than calling the public API.
    import dale.agent.usage

    def exploded(model):
        raise ImportError("pydantic_ai.models.test moved")

    monkeypatch.setattr(dale.agent.usage, "_is_synthetic_model", exploded)
    with pytest.raises(ImportError):
        TokenUsage.from_run(RunUsage(input_tokens=10, requests=1))


# --- per-call timing -------------------------------------------------------


def test_per_call_timing_is_recorded_and_summed(registry):
    """Every logged call carries its own host-side cost, and the log sums
    them — that sum minus wall clock is what separates data processing from
    waiting on the model."""
    registry.create(
        "list",
        [{"name": f"r{i}", "keep": i % 2 == 0} for i in range(200)],
        name="rows",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    class Ctx:
        deps = registry

    tool.function(
        Ctx(),
        params_model(
            steps=[
                {
                    "operation": "filter_where",
                    "intent": "keep evens",
                    "handle": "rows",
                    "predicate": {"field": "keep", "op": "==", "value": True},
                    "name": "kept",
                    "description": "d",
                }
            ]
        ),
    )

    entry = log.entries[-1]
    assert entry.elapsed_ms > 0
    # peek_at_every_step defaults on, so the splice is attributed separately
    # rather than inflating the operation's own number.
    assert entry.auto_inspect_ms > 0
    assert log.total_host_ms == pytest.approx(entry.elapsed_ms + entry.auto_inspect_ms)
    assert log.total_auto_inspect_ms == pytest.approx(entry.auto_inspect_ms)


def test_auto_inspect_time_is_zero_when_the_feature_is_off(registry):
    registry.create("list", [{"a": 1}], name="rows", description="d", created_by="fixture")
    log = ActionLog()
    tool, params_model = _run_plan(log, peek_at_every_step=False)

    class Ctx:
        deps = registry

    tool.function(
        Ctx(),
        params_model(
            steps=[
                {
                    "operation": "sort_by",
                    "intent": "sort",
                    "handle": "rows",
                    "keys": [{"field": "a"}],
                    "name": "s",
                    "description": "d",
                }
            ]
        ),
    )

    assert log.entries[-1].elapsed_ms > 0
    assert log.entries[-1].auto_inspect_ms == 0.0
    assert log.total_auto_inspect_ms == 0.0


def test_timing_renders_on_the_result_line():
    from dale.agent import ActionLogEntry

    e = ActionLogEntry(
        step=1, intent="i", operation="filter_where", params={"handle": "h"},
        status="ok", result={"status": "ok"}, elapsed_ms=1.34, auto_inspect_ms=0.42,
    )
    assert "(1.3 ms + 0.4 ms inspect)" in ActionLog._render_entry(e)

    # An unmeasured entry renders nothing rather than a misleading "0.0 ms".
    bare = ActionLogEntry(
        step=1, intent="i", operation="peek", params={"handle": "h"},
        status="ok", result={"status": "ok"},
    )
    assert "ms" not in ActionLog._render_entry(bare)


def test_summary_timing_split():
    log = ActionLog()
    log.record(
        intent="i", operation="filter_where", params={}, result={"status": "ok"},
        elapsed_ms=3.0, auto_inspect_ms=1.0,
    )
    r = TrialResult(
        trial=1, model="m", use_case="uc", success=True, final_answer="",
        action_log=log, wall_clock_s=10.0,
    )
    summary = TrialSummary(use_case="uc", model="m", n=1, successes=1, results=[r])

    stats = summary.timing_stats()
    assert stats["mean_host_ms"] == pytest.approx(4.0)
    assert stats["mean_model_s"] == pytest.approx(9.996)
    assert stats["host_share_pct"] == pytest.approx(0.04)
    assert "host compute (0.04%)" in summary.render(show_action_logs=False)


def test_summary_omits_timing_block_when_no_trial_was_timed():
    """Symmetric with the token block: an untimed batch reports nothing rather
    than a confident 0.0s wall / 0.00% host share, which would read as a
    measurement instead of the absence of one."""
    summary = TrialSummary(
        use_case="uc", model="m", n=1, successes=0, results=[_trial(TokenUsage())]
    )
    assert summary.timing_stats() == {}
    assert "time (mean)" not in summary.render(show_action_logs=False)


def _trial(usage: TokenUsage) -> TrialResult:
    return TrialResult(
        trial=1,
        model="m",
        use_case="uc",
        success=True,
        final_answer="",
        action_log=ActionLog(),
        usage=usage,
    )


def test_summary_means_exclude_unmeasured_trials():
    """A batch mixing real and synthetic runs must not dilute the mean with
    zeros — the reported n says how many actually contributed."""
    measured = TokenUsage(
        available=True, requests=4, input_tokens=1_000, output_tokens=100,
        peak_context_tokens=800, peak_available=True,
    )
    summary = TrialSummary(
        use_case="uc",
        model="m",
        n=2,
        successes=2,
        results=[_trial(measured), _trial(TokenUsage())],
    )

    stats = summary.usage_stats()
    assert stats["n_measured"] == 1
    assert stats["mean_input_tokens"] == 1_000  # not 500
    assert stats["max_peak_context_tokens"] == 800

    rendered = summary.render(show_action_logs=False)
    assert "tokens (mean of 1 (1 unmeasured))" in rendered


def test_summary_omits_token_block_when_nothing_measured():
    summary = TrialSummary(
        use_case="uc", model="m", n=1, successes=0, results=[_trial(TokenUsage())]
    )
    assert summary.usage_stats() == {}
    assert "tokens" not in summary.render(show_action_logs=False)


# --- the part-(F) ablation mechanism ----------------------------------------


def test_run_trial_hands_the_step_ceiling_to_the_agent(monkeypatch):
    """`max_steps_per_call` is the only mechanism paper.md Section 4.2 part
    (F) has left: `--no-run-plan` used to express the unbatched arm by
    withholding a second tool, and there is no second tool now, so the arm is
    "cap the plan at one step" instead. That comparison is only meaningful if
    the flag actually reaches `build_agent` — and nothing asserted the old one
    did either. A trial silently run with the ceiling dropped reports the
    batched request count under the unbatched arm's name, which is a wrong
    published number rather than a failure anyone would see.

    Captured at the seam and aborted there: running a real trial needs a model,
    and the claim under test is about argument threading, not about a run."""
    import eval.harness as harness

    class _Captured(Exception):
        pass

    seen: dict = {}

    def capture(registry, action_log, **kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(harness, "build_agent", capture)

    with pytest.raises(_Captured):
        harness.run_trial(
            1,
            "uc",
            lambda registry: None,
            "task",
            lambda registry, log, answer: True,
            "test",
            verbose=False,
            max_steps_per_call=1,
        )
    assert seen["max_steps_per_call"] == 1

    # ...and the default arm is genuinely unbounded, not some other number that
    # happens to be large: `None` is what "no ceiling" means to build_tools.
    seen.clear()
    with pytest.raises(_Captured):
        harness.run_trial(
            1,
            "uc",
            lambda registry: None,
            "task",
            lambda registry, log, answer: True,
            "test",
            verbose=False,
        )
    assert seen["max_steps_per_call"] is None


def test_run_trials_passes_the_step_ceiling_to_every_trial(monkeypatch):
    """The batch-level half. An arm of the part-(F) comparison is only an arm
    if *every* trial in it ran under the same condition — the reason this
    belongs to the batch rather than to individual trials in the first place."""
    import eval.harness as harness

    seen: list = []

    def fake_run_trial(trial, use_case_name, setup, task, checker, model, **kwargs):
        seen.append(kwargs.get("max_steps_per_call"))
        return TrialResult(
            trial=trial,
            model=model,
            use_case=use_case_name,
            success=True,
            final_answer="",
            action_log=ActionLog(),
        )

    monkeypatch.setattr(harness, "run_trial", fake_run_trial)

    harness.run_trials(
        "uc",
        lambda registry: None,
        "task",
        lambda registry, log, answer: True,
        "test",
        3,
        verbose=False,
        max_steps_per_call=1,
    )

    assert seen == [1, 1, 1]
