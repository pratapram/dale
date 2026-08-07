"""Running one primitive call, and deciding when a run should stop.

`_execute_and_log_step` is the single choke point every primitive call goes
through — the real `dale.call_primitive` path, plus `peek_at_every_step`'s
auto_inspect splice, the repetition nudge, the action-log entry and the live
print. It was extracted so a call arriving as part of a batch is
indistinguishable in the resulting trace from one issued on its own; now that
batching is the only call shape there is only one caller, but the property it
was extracted to guarantee is the one tests/data's frozen baseline asserts.

The rest is loop control: `AgentLoopTerminated` and the repetition machinery
that decides a run has stopped making progress and says so, rather than
burning the request budget rediscovering the same rejection.
"""

import time
from typing import Any

import dale

from dale.agent.log import ActionLog, Verbosity
from dale.agent.prompt import _auto_inspect




class AgentLoopTerminated(Exception):
    """The session is over — deliberately *not* a DaleError.

    Every DaleError describes one *call*, and both dispatch and
    _execute_and_log_step hand it back to the model as a result payload it can
    read and react to. This describes the *session*: DALE has concluded there
    is no next call worth making, so returning it as a payload the model is
    free to ignore would be a category error.

    That distinction is not theoretical — it was a real bug. ToolCallLimitError
    is a DaleError, so _execute_and_log_step's broad `except dale.DaleError`
    used to convert it into exactly such an ignorable payload, which is why
    RegistryLimits.max_tool_calls refused *work* without ever stopping the
    *loop*. It's now caught specifically and re-raised as this instead. Not
    inheriting DaleError is what keeps that class of mistake from recurring:
    no existing handler anywhere in DALE catches this by accident.

    Two conditions raise it, distinguishable by `code`:
      - TOOL_CALL_LIMIT_EXCEEDED — the session's total call ceiling
        (RegistryLimits.max_tool_calls). Blunt, but catches a loop whose
        params vary; `attempts` is None, since no single call was repeated.
      - REPETITION_LIMIT_EXCEEDED — one byte-identical call failing
        `repetition_limit` times. Precise and diagnostic, but only catches
        verbatim resends.
    Neither subsumes the other, which is why both exist.

    Raised out of the tool function, so it propagates straight out of
    Agent.run_sync (verified: a non-ModelRetry exception raised inside a
    pydantic-ai tool terminates the run after exactly one tool invocation —
    ModelRetry is the opposite path, retrying to the budget, and this must
    never be one). Callers that treat a failed run as data rather than a crash
    — eval/harness.py's run_trial, dale.agent.run_agent — already catch broad
    Exception and record it, so they need no change to report this cleanly.
    """

    def __init__(
        self, message: str, *, code: str, primitive: str, attempts: int | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.primitive = primitive
        self.attempts = attempts


_REPETITION_NUDGE_THRESHOLD = 2
"""How many prior, identical, failed attempts (same primitive, same params,
status != "ok") trigger a repetition nudge on the next one — i.e. the nudge
first appears on the 3rd attempt. Matches Tier-1 stuck-detection
design ("nudge after 2-3 repeats"). Not exposed as a public parameter, same
as _AUTO_PEEK_N — a threshold worth tuning later from real data, not a
per-deployment knob yet."""

_REPETITION_LIMIT_DEFAULT = 5
"""Default `repetition_limit`: how many identical failed attempts (same
primitive, same params) end the run outright, via AgentLoopTerminated.

Must stay strictly greater than _REPETITION_NUDGE_THRESHOLD + 1, so the model
always gets at least one explicit warning quoting the real error before it's
stopped — being killed without having been told is both unfair and useless
diagnostically. At these two values: attempts 1-2 return a plain error, 3-4
add a repetition_warning, and the 5th terminates.

Why act at all rather than warn indefinitely: the real UC1 pilot failure
(the evaluation pilot) was 46 byte-for-byte identical calls that never recovered
and burned the entire request budget. paper.md Section 3.13 ranks an LLM
inference call as the most expensive resource tier in the system; spending 41
more of them re-deriving an answer DALE already has is the sharpest available
violation of that principle. Note the loop was never *unbounded* — pydantic-ai
applies UsageLimits(request_limit=50) by default — this makes it stop early and
with a stated reason instead of late and anonymously.

Both constants live here, above build_tools, because this one is used as a
default parameter value (evaluated at def time), not just inside a body."""

_TOOL_MAX_RETRIES = 3
"""How many schema-invalid `run_plan` calls in a row the model may send before
pydantic-ai gives up on the tool and ends the run.

pydantic-ai's default is 1, survivable only while there were 18 tools: its
retry budget is keyed per tool *name* and cleared on success
(tool_manager.py:120-136, verified), so 17 named tools meant 17 independent
budgets. With `run_plan` the only tool they collapse into one
consecutive-failure counter, and inheriting 1 would mean a single malformed
batch ends the whole run — a much harsher regime arrived at by accident. Hence
a number DALE chose.

Must exceed 1, for that reason. Kept below the *default* `repetition_limit` of
5 — deliberately not enforced against a caller-supplied one, unlike
_validate_repetition_limit's invariant, since a caller who raises or disables
that limit isn't creating an inconsistency, just a different policy for a
different mechanism. The two count different things and must not be
confusable: this counts
*schema-invalid* batches that never reached DALE at all (pydantic-ai rejected
the arguments; nothing ran, nothing was logged), while `repetition_limit`
counts *semantically* repeated failures — well-formed calls dispatch ran and
rejected identically, each a real ActionLog entry. A model stuck on the first
can't produce valid JSON for the schema; one stuck on the second understood the
schema perfectly and is wrong about the data."""


def _validate_repetition_limit(repetition_limit: int | None) -> None:
    """Turns the invariant stated in _REPETITION_LIMIT_DEFAULT's docstring into
    an enforced one.

    A limit that isn't strictly greater than _REPETITION_NUDGE_THRESHOLD + 1
    stops the run on — or before — the very attempt that first carries a
    repetition_warning, so the model gets killed without ever having been told
    it was repeating itself. That's the one thing the whole nudge-then-stop
    escalation exists to avoid, and leaving it as prose is precisely how
    `repetition_limit=0` became a way to end a run on its first *successful*
    call: nothing rejected the value, and the terminal comparison happily fired
    against an attempt count of zero.

    Checked when the tools are built rather than at the per-call choke point
    because a bad limit is the invoker's configuration mistake, not the model's
    behavior — the useful moment to hear about it is at construction, not
    several model turns into a live run. `None` stays valid: that's the
    documented way to disable the stop entirely."""
    if repetition_limit is None:
        return
    if repetition_limit <= _REPETITION_NUDGE_THRESHOLD + 1:
        raise ValueError(
            f"repetition_limit must be greater than {_REPETITION_NUDGE_THRESHOLD + 1} "
            f"(or None to disable it), got {repetition_limit!r}. The model is always "
            f"warned before it is stopped: the repetition_warning first appears on "
            f"attempt {_REPETITION_NUDGE_THRESHOLD + 1}, so any lower limit terminates "
            f"the run without ever having warned it."
        )


def _count_prior_identical_failures(
    action_log: ActionLog, primitive: str, call_params: dict[str, Any]
) -> int:
    """How many times this exact (primitive, params) pair has already failed
    (or hit cost_gate_exceeded) anywhere earlier in this run's log — not just
    consecutively, so a `peek` or other call interleaved between two
    identical failing attempts still counts them as repetition. This is the
    real, verbatim mechanism behind UC1's 46x-identical-retry pilot failure
    (the evaluation pilot) — a model that keeps resending the same rejected
    call, never varying it."""
    return sum(
        1
        for e in action_log.entries
        if e.primitive == primitive and e.params == call_params and e.status != "ok"
    )


def _repetition_nudge_text(primitive: str, payload: dict[str, Any], attempt_number: int) -> str:
    """Quotes DALE's own already-known error back at the model explicitly —
    Tier 1 of the stuck-detection design: DALE already has the
    theoretical reason (the error itself), so no model reasoning or new
    mechanism is needed, only surfacing what's already known instead of
    letting the model rediscover it by repeating the identical call.

    Two wordings, because the repetition counter deliberately counts
    `cost_gate_exceeded` alongside real errors (a model that keeps hitting the
    cost-estimate gate without ever setting `confirm=True` is genuinely stuck,
    and loop protection should apply to it) — but a cost_gate_exceeded payload
    carries an `estimate`, not a `code`/`message`, so the error wording renders
    the literal "UNKNOWN — " and calls a correct safety response a failure.
    That's not a cosmetic slip: it contradicts DALE's own semantics elsewhere
    (eval/harness.py's wasted_turn_rate excludes cost_gate_exceeded precisely
    because it is "a correct safety response, not a mistake"; paper.md Section
    3.5 frames the gate as the thing that *enables* self-correction), and it
    withholds from the model the one fact that would unstick it — that the
    remedy is `confirm=True`, not a different approach. Kept adjacent to
    _repetition_stop_message below on purpose: the warning and the stop have to
    describe the same situation the same way, and splitting them apart is how
    they'd drift."""
    if payload.get("status") == "cost_gate_exceeded":
        return (
            f"You have now made this exact {primitive} call {attempt_number} times, and the "
            f"cost-estimate gate refused it every time because you never set confirm=True. "
            f"Sending it again unchanged will be refused again. Either resend it with "
            f"confirm=True to accept the estimated cost, or make the operation smaller."
        )
    code = payload.get("code", "UNKNOWN")
    message = payload.get("message", "")
    return (
        f"You have now made this exact {primitive} call {attempt_number} times, and it failed "
        f"identically every time: {code} — {message}. Repeating the same call again will not "
        f"produce a different result. Try a genuinely different approach or different parameters."
    )


def _repetition_stop_message(primitive: str, payload: dict[str, Any], attempt_number: int) -> str:
    """The same distinction _repetition_nudge_text draws, for the message that
    ends the run — the post-mortem's one-line account of why it stopped, and
    the only text a caller catching AgentLoopTerminated ever sees. It's worth
    as much care as the nudge: "stopped after 5 identical failed join_lookup
    calls: UNKNOWN — " tells whoever reads the log that something went wrong
    inside DALE, when what actually happened is that the model kept asking for
    an expensive operation and never confirmed it."""
    if payload.get("status") == "cost_gate_exceeded":
        return (
            f"stopped after {attempt_number} identical {primitive} calls that the "
            f"cost-estimate gate refused: confirm=True was never set"
        )
    return (
        f"stopped after {attempt_number} identical failed {primitive} calls: "
        f"{payload.get('code', 'UNKNOWN')} — {payload.get('message', '')}"
    )


def _execute_and_log_step(
    registry: dale.DataRegistry,
    action_log: ActionLog,
    *,
    primitive: str,
    intent: str,
    call_params: dict[str, Any],
    peek_at_every_step: bool,
    repetition_nudge: bool,
    repetition_limit: int | None,
    verbosity: Verbosity,
) -> dict:
    """Runs one primitive call through the real dale.call_primitive path,
    splices in peek_at_every_step's auto_inspect and a repetition_nudge
    (the Tier-1 stuck-detection design) where applicable, records it
    in the action log, and prints it live at verbosity != "quiet". The single
    place both a normal one-call tool (_make_tool_fn) and run_plan's per-step
    loop go through, so a call arriving as part of a batch is
    indistinguishable in the resulting trace from one issued on its own
    turn — deliberately, since nothing about DALE's auditability should
    change just because calls arrived together.

    `repetition_limit` escalates repetition_nudge's warning into a stop: once
    the same (primitive, params) pair has failed that many times, the run ends
    with AgentLoopTerminated instead of the model being warned again. Both
    read the same count from _count_prior_identical_failures — the kill needed
    no new detection mechanism, only permission to act on what the nudge was
    already measuring. Because this is the single shared choke point, the
    limit applies identically to batched run_plan steps and standalone calls,
    and counts across the two together."""
    terminal: dale.DaleError | None = None
    started = time.perf_counter()
    try:
        output = dale.call_primitive(registry, primitive, call_params)
        payload = output.model_dump(exclude_none=True)
    except dale.ToolCallLimitError as exc:
        # The one DaleError that describes the session rather than the call:
        # handing it back as a payload lets the model shrug it off and keep
        # calling forever, which is what made max_tool_calls refuse work
        # without ever stopping the loop. Caught ahead of the broad handler
        # below (except clauses match in order) and re-raised after logging.
        payload = exc.to_payload()
        terminal = exc
    except dale.DaleError as exc:
        payload = exc.to_payload()
    # Measured before the auto-inspect splice below, so a primitive's own cost
    # is never inflated by DALE's optional inspection of its result.
    elapsed_ms = (time.perf_counter() - started) * 1000

    auto_inspect_ms = 0.0
    if peek_at_every_step and not registry.privacy_mode and payload.get("status") == "ok":
        handle_meta = payload.get("handle")
        if isinstance(handle_meta, dict) and "handle" in handle_meta:
            inspect_started = time.perf_counter()
            inspected = _auto_inspect(registry, handle_meta["handle"])
            auto_inspect_ms = (time.perf_counter() - inspect_started) * 1000
            if inspected is not None:
                payload["auto_inspect"] = inspected

    # Counted once, read twice: the nudge warns on it, the limit stops on it.
    # `attempt_number` includes this call (which isn't in the log yet), so it's
    # prior failures + 1; the nudge's original `prior_failures >= THRESHOLD`
    # is exactly `attempt_number > THRESHOLD`, unchanged in behavior.
    attempt_number = 0
    if payload.get("status") != "ok" and (repetition_nudge or repetition_limit is not None):
        attempt_number = _count_prior_identical_failures(action_log, primitive, call_params) + 1
        if repetition_nudge and attempt_number > _REPETITION_NUDGE_THRESHOLD:
            payload["repetition_warning"] = _repetition_nudge_text(
                primitive, payload, attempt_number
            )

    entry = action_log.record(
        intent=intent,
        primitive=primitive,
        params=call_params,
        result=payload,
        elapsed_ms=elapsed_ms,
        auto_inspect_ms=auto_inspect_ms,
    )
    if verbosity != "quiet":
        print(
            f"{ActionLog._render_entry(entry, debug=verbosity in ('debug', 'raw'))}\n"
            f"{ActionLog._render_registry_state(entry.alive_after)}",
            flush=True,
        )

    # Both terminal checks are deliberately after record() and the live print,
    # never before: the ActionLog is what stands in for a resume/checkpoint
    # feature, so the call that ended the run has to be *in*
    # it — carrying whatever error or repetition_warning explains why — or the
    # post-mortem loses the single most important entry. Terminating first
    # would destroy exactly the evidence this exists to surface.
    if terminal is not None:
        raise AgentLoopTerminated(
            f"tool call limit reached: {terminal.message}",
            code=terminal.code,
            primitive=primitive,
        ) from terminal

    # `attempt_number >= 1` is not redundant with the limit comparison: it is
    # what keeps a *non-attempt* from ever ending the run. The counter is only
    # assigned for a call that didn't succeed, so a successful call leaves it at
    # 0 — and with a limit of 0 (or a negative one) the bare `>=` fired on that
    # zero, terminating a perfectly good first call with the self-contradictory
    # "stopped after 0 identical failed calls: UNKNOWN — ".
    # _validate_repetition_limit now rejects such a limit at construction, so
    # this guard is the second half of a belt-and-braces pair: this function is
    # the choke point every call goes through, and it should not depend on its
    # caller having been well-behaved to avoid killing a run that is working.
    if repetition_limit is not None and attempt_number >= 1 and attempt_number >= repetition_limit:
        raise AgentLoopTerminated(
            _repetition_stop_message(primitive, payload, attempt_number),
            code="REPETITION_LIMIT_EXCEEDED",
            primitive=primitive,
            attempts=attempt_number,
        )
    return payload
