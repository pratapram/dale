"""Building the one tool the model is offered.

`run_plan` is the only tool DALE publishes; a single operation call is a
`steps` list of length one. `steps` is a real pydantic discriminated union
over every selected operation's own parameter schema, so each step keeps full
validation against the operation it names — the model chooses from the same
closed, structured-parameter grammar described in paper.md Section 3.2, just
submitting several choices in one message instead of one per message.

This module owns what goes on the wire: which operations are offered
(`_selected_operations`), what each step's schema looks like
(`_params_for_plan_step`), what is trimmed from it (`_UntitledToolJsonSchema`),
and how a validated step becomes the plain dict dispatch takes (`_call_params`).
"""

from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import BaseModel, Field, create_model
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import GenerateToolJsonSchema

import dale

from dale.agent.execution import (
    _REPETITION_LIMIT_DEFAULT,
    _TOOL_MAX_RETRIES,
    _execute_and_log_step,
    _validate_repetition_limit,
)
from dale.agent.log import ActionLog, Verbosity, render_raw_messages


def _selected_operations(
    operations: Sequence[str] | None, *, privacy_mode: bool
) -> list[str]:
    """Which operations the model is offered as `run_plan` steps — the single
    place either filter below is applied, so there is no second copy to drift.

    `operations` (default `None` = the whole catalog, today's behaviour) is the
    caller's allowlist, for a deployment that knows its task needs nine of the
    seventeen. An unknown name is a ValueError, not a silent drop: a typo would
    otherwise shrink the catalog and surface many model turns later as an
    inexplicable "it never tried filter_where". Read as a set against
    `dale.list_operations()` order, so a caller's ordering or a duplicate can't
    reorder the published union. This governs what the *model* is offered only;
    `dale.call_operation` stays unrestricted, since a host loading its own
    fixtures is not the party being constrained.

    `privacy_mode` then drops `peek` unconditionally — not merely redacted at
    call time (inspect.py's redaction stays as a dispatch-level backstop for
    direct callers) but never offered as a step, so the model can't spend a call
    on something that can only hand back type placeholders. `describe` stays
    fully functional: its aggregate statistics were never "individual real
    values" under this design's own definition, only categorical top_k is
    redacted."""
    catalog = dale.list_operations()
    if operations is not None:
        unknown = sorted(set(operations) - set(catalog))
        if unknown:
            raise ValueError(
                f"unknown operation(s) in the `operations` allowlist: {unknown}. "
                f"Known operations: {sorted(catalog)}"
            )
        allowed = set(operations)
    else:
        allowed = set(catalog)
    selected = [n for n in catalog if n in allowed and not (privacy_mode and n == "peek")]
    if not selected:
        # Caught here only so the message names a cause: falling through reaches
        # `Union[tuple([])]` and surfaces as "TypeError: Cannot take a Union of
        # no types" from inside `typing`, naming nothing the caller wrote. The
        # privacy_mode clause keys on peek having actually been in the allowlist
        # rather than on the flag being on, so an empty `operations=[]` under
        # privacy_mode doesn't send the reader off to fix the wrong thing.
        dropped_peek = privacy_mode and operations is not None and "peek" in operations
        raise ValueError(
            "the `operations` allowlist selects no operations at all"
            + (" once privacy_mode drops peek" if dropped_peek else "")
            + f" (got {list(operations) if operations is not None else None!r}), so "
            f"run_plan would have no step type the model could ever submit."
        )
    return selected


def build_tools(
    action_log: ActionLog,
    *,
    verbosity: Verbosity = "quiet",
    peek_at_every_step: bool = True,
    repetition_nudge: bool = True,
    repetition_limit: int | None = _REPETITION_LIMIT_DEFAULT,
    privacy_mode: bool = False,
    max_steps_per_call: int | None = None,
    operations: Sequence[str] | None = None,
) -> list[Tool]:
    """The model's entire tool surface: exactly one PydanticAI Tool, `run_plan`,
    whose `steps` are a discriminated union generated from the live operation
    catalog — so it always matches what dale.call_operation accepts, being built
    from the same OperationSpec rather than hand-maintained.

    A single operation call is a `steps` list of length one. That is the whole
    design: no second, per-operation tool surface, so the catalog is published
    once per request instead of twice and the model never chooses between two
    encodings of the same call. Returns a list of one because that is what
    `Agent(tools=...)` takes, and so a second agent-layer tool later needs no
    caller changes.

    This function is now a delegation, and each parameter is documented where it
    is actually used rather than restated here: `operations`/`privacy_mode` in
    _selected_operations; `verbosity` on the `Verbosity` alias;
    `peek_at_every_step`, `repetition_nudge` and `repetition_limit` in
    _execute_and_log_step, which is the choke point all three act at (and
    _REPETITION_LIMIT_DEFAULT / _validate_repetition_limit for what values that
    last one may take, checked here so a bad one fails at construction);
    `max_steps_per_call` in _build_run_plan_tool. Two things that live nowhere
    else:

    `max_steps_per_call` (default `None` = unbounded) replaces the old
    `enable_run_plan` flag, and names what it actually does. `1` reproduces the
    unbatched condition — one operation per round trip — which is what paper.md
    Section 4.2 part (F)'s batched-vs-unbatched ablation needs now that there is
    no second tool to withhold.

    Handle-creating operations declare `name`/`description` directly on their
    own param schema (not injected here) — `name` is the LLM-supplied identifier
    registry.create() uses as the handle itself (rejected outright on collision
    with an alive handle), and `description` is mandatory on every handle for the
    same reason. Both flow all the way to dispatch, not just the log: the same
    "explicit LLM-supplied structure, not automatic inference" principle `intent`
    already uses, applied to handle identity rather than display.

    Deliberately no `from __future__ import annotations` in this module:
    `run_plan_fn`'s parameter annotation is `RunPlanParams`, built inside
    _build_run_plan_tool from the selected catalog and never bound at module
    level. Postponed evaluation (PEP 563) would turn it into a string
    PydanticAI's introspection tries to eval() against module globals, where no
    such name exists — found the hard way in examples/04.
    """
    _validate_repetition_limit(repetition_limit)
    return [
        _build_run_plan_tool(
            action_log,
            verbosity=verbosity,
            peek_at_every_step=peek_at_every_step,
            repetition_nudge=repetition_nudge,
            repetition_limit=repetition_limit,
            privacy_mode=privacy_mode,
            max_steps_per_call=max_steps_per_call,
            operations=operations,
        )
    ]


def _call_params(params: BaseModel, *, exclude: set[str]) -> dict[str, Any]:
    """The model's tool-call arguments as the plain dict `dale.call_operation`
    takes, minus the agent-layer-only fields operations themselves don't know
    about (`intent` always; `operation` too, for a run_plan step).

    `exclude_unset=True`, deliberately, and the distinction from the
    `exclude_none=True` this used to pass is not cosmetic — it was a live bug,
    and a nasty one, because DALE was the party at fault while the model got
    the blame. `exclude_none` is *recursive*: a schema-valid predicate
    `{"field": "x", "op": "==", "value": None}` — a perfectly reasonable way to
    ask "which records have x set to null?" — arrived here whole and left as
    `{"field": "x", "op": "=="}`, which dispatch then rejected with
    `INVALID_PARAMS: predicate.Comparison.value Field required`. The model was
    told to supply a field it had already supplied, so its only available
    repair was to send the identical call again, and DALE ate the value again.
    That is byte-for-byte the malformed `filter_where` behind the 46-identical-
    retry pilot failure in the evaluation pilot — the failure NullComparison, the
    repetition nudge and `repetition_limit` were all built in response to. Those
    three defenses all sit downstream of this line: with the cause live, the
    kill switch fires on a loop DALE's own serialization created. Worse, the
    behavior was unpredictable from the schema the model was given, since
    `exclude_none` doesn't touch list elements — `{"op": "in", "value": [null, 1]}`
    sailed through untouched while `{"op": "==", "value": null}` did not.

    `exclude_unset` keeps the one thing `exclude_none` was actually there for.
    Optional fields the model never mentioned (`describe.field`,
    `join_lookup.fields`, `window_flag.predicate`, `export_handle.format`) are
    still stripped rather than being forwarded as explicit nulls, and every
    defaulted field (`confirm`, `how`, `as`, `carry_fields`, `n`) is simply
    absent from the dict, so `spec.param_schema.model_validate` re-applies
    exactly the same default it would have applied anyway — including the
    `confirm` that `dispatch.call_operation`'s cost gate reads. What it stops
    doing is second-guessing a null the model chose to send: an explicitly
    supplied `None` now reaches validation, where the schema — the same schema
    the model was shown — decides whether it is legal.

    Nothing regresses in exchange, and that is worth stating precisely rather
    than hedging, because the obvious worry — "won't a model that sends
    `carry_fields: null` now get an error where it used to be quietly repaired?"
    — turns out not to be reachable. By the time this function runs, `params`
    is an already-validated instance of the operation's own param model: a null
    for any field whose schema forbids one (`carry_fields`, `remove_envelope`,
    `n`, `how`, `confirm`, `as` — checked individually, that is all of them)
    was rejected one layer earlier, by pydantic-ai, identically before and
    after this change. The only nulls that ever arrive here are ones the schema
    already permits, which is exactly the set that should be forwarded rather
    than eaten.

    Kept as a function with one caller rather than inlined. Its original
    reason — two call sites had to serialize identically or a batched step would
    stop being indistinguishable from a standalone one — is now historical:
    there is one surface left, so there is nothing to disagree with. What is not
    historical is everything above: the account of a live bug that took three
    defenses to notice and one line to cause. Inlining would put that inside a
    loop body, where it would be under pressure to shorten."""
    return params.model_dump(exclude=exclude, exclude_unset=True, by_alias=True)


class _UntitledToolJsonSchema(GenerateToolJsonSchema):
    """pydantic-ai's own tool schema generator, minus the `title` keys.

    `title` is pydantic bookkeeping — the model class's own name, restating what
    the surrounding schema already says — and 823 tokens of it per request,
    across the 25 `$defs` `run_plan` publishes. The base class already strips
    *property* titles; overriding `model_schema` removes the model-level one on
    every `$def`, which is where the bulk of them are.

    Hooking pydantic's own per-model emitter rather than post-processing the
    finished schema is deliberate: a recursive walk that deletes every `title`
    key it meets cannot tell a schema from arbitrary data, so it also reaches
    into `default`, `const`, `enum` and `examples` values — an operation with
    `default={"title": "Mr"}` would have its default silently rewritten. This
    can only ever see a model's own schema node.

    Passed as `Tool(schema_generator=...)`, the only seam that affects
    `tool_def.parameters_json_schema` — i.e. what goes on the wire.
    `params_model.model_json_schema()` does *not* go through it, so anything
    asserting there is asserting about a schema nobody sends."""

    def model_schema(self, schema: Any) -> Any:
        emitted = super().model_schema(schema)
        emitted.pop("title", None)
        return emitted


def _params_for_plan_step(name: str, spec: dale.OperationSpec) -> type[BaseModel]:
    """One `steps` variant: the operation's own param model, plus an
    `operation: Literal[name]` discriminator field — what lets run_plan's
    `steps: list[PlanStep]` be a real pydantic discriminated union (each
    step keeps full validation against its own operation's actual schema,
    not a hand-parsed `dict`) — plus its own per-step `intent`: a batched step
    is a full ActionLog entry in its own right, not a sub-item of one
    "batch intent", consistent with _execute_and_log_step logging it exactly
    like a standalone call.

    `__doc__` carries the operation's own docstring, in full, because pydantic
    promotes it into the variant's schema `description` and JSON Schema has no
    other per-variant equivalent of a tool's `description`. Not a nicety: while
    each operation also had a named tool, that tool carried the docstring and
    these variants all shipped `description: None`, unnoticed because the
    information was still on the wire somewhere. Under batch-only nothing else
    carries it, so leaving them null would have the model selecting among 17
    operations on name and parameter shape alone — a 7,474-token *information
    deficit*, not a saving, and the easiest way to make this change look like a
    win while degrading what it exists to serve.

    In full, not truncated to a first line: the named tools published
    `(spec.fn.__doc__ or ...).strip()` whole, so this is information parity with
    what the model had before (pydantic dedents the continuation lines on the
    way in, which is the only difference), and it is what the measured token
    figure assumed. Truncating is a different change needing its own
    measurement.

    `intent` carries no per-field description — DEFAULT_SYSTEM_PROMPT states
    what it is for once, rather than 17 copies of one sentence. Its
    *requiredness* is not documentation and stays here."""
    fields: dict[str, Any] = {
        "operation": (Literal[name], Field(..., description=f"Must be {name!r}.")),
        "intent": (str, Field(...)),
    }
    return create_model(
        f"{spec.param_schema.__name__}PlanStep",
        __base__=spec.param_schema,
        __doc__=(spec.fn.__doc__ or f"DALE operation: {name}").strip(),
        **fields,
    )


def _build_run_plan_tool(
    action_log: ActionLog,
    *,
    verbosity: Verbosity,
    peek_at_every_step: bool,
    repetition_nudge: bool,
    repetition_limit: int | None,
    privacy_mode: bool,
    max_steps_per_call: int | None = None,
    operations: Sequence[str] | None = None,
) -> Tool:
    """`run_plan` — agent-layer orchestration, not an `@operation`/catalog
    entry (it doesn't operate on data itself, so it isn't in
    dale.list_operations() and can't recursively appear as one of its own
    steps). The model's only tool: it submits a sequence of steps, which DALE
    runs in order in a plain Python loop — see the design notes
    "Plan/batch tool" entry for the full design rationale this implements.
    Every step goes through `_execute_and_log_step` — the exact same
    dale.call_operation path, its own real ActionLog entry,
    peek_at_every_step's auto_inspect — so a step arriving as part of a batch
    is indistinguishable in the resulting trace from one that arrived alone,
    and a `steps` list of length one is not a special case but the trivial
    case.

    `run_plan_fn` takes exactly one non-context parameter, and must keep taking
    exactly one. PydanticAI flattens a function's single BaseModel parameter
    onto the tool's top-level schema only when it is the *only* non-context
    argument; add a second (even a defaulted one) and everything gets wrapped
    under a `"params"` key, with the second parameter shown to the model as a
    real, spurious field. Most models infer the intended flat shape anyway,
    which is what makes it unpleasant: a cheaper one sometimes doesn't, sends
    the flat call, fails validation, and the run dies as "Tool 'run_plan'
    exceeded max retries" — a live Haiku failure, back when this applied to 17
    tool functions built in a loop. Everything `run_plan_fn` needs is closed
    over lexically, which is what keeps it true here.

    Stops at the first step whose result isn't status="ok" (a real error, or
    cost_gate_exceeded) and returns everything completed so far, including
    the failure — consistent with "never waste a turn": a
    failed step 3 of 10 shouldn't discard steps 1-2's real results; the
    model sees exactly what happened and decides its next move itself, the
    same as it does today one call at a time.

    A real, known tradeoff, not silently glossed over: because all N steps
    arrive as one pydantic model (`steps: list[PlanStep]`, a discriminated
    union per step) validated in one pass before this function ever runs, a
    single step with a *malformed* (schema-invalid) argument fails the
    entire tool call up front — pydantic-ai's normal per-tool retry budget
    applies to the whole plan, not a partial one, so nothing from any step
    executes. Only semantic/runtime errors a syntactically valid step can
    still hit (HANDLE_NOT_FOUND, a real DaleError from dispatch) get the
    partial-results treatment described above. Left as a documented
    tradeoff (see paper.md) rather than "fixed", since fixing it would mean
    giving up per-step Pydantic validation — the whole reason a
    discriminated union was chosen over a raw `list[dict]` an operation
    would have to hand-validate itself.

    That tradeoff is what `_TOOL_MAX_RETRIES` exists to blunt: it is now the
    only tool, so its retry budget is the only one, and pydantic-ai's default
    of 1 would have made a single malformed batch end every run.
    """
    if max_steps_per_call is not None and max_steps_per_call < 1:
        raise ValueError(
            f"max_steps_per_call must be at least 1 (or None for no ceiling), got "
            f"{max_steps_per_call!r} — a plan of zero steps can do nothing, so this "
            f"would publish a tool the model is unable to call at all."
        )
    plan_step_models = [
        _params_for_plan_step(name, dale.get_operation(name))
        for name in _selected_operations(operations, privacy_mode=privacy_mode)
    ]
    PlanStep = Annotated[Union[tuple(plan_step_models)], Field(discriminator="operation")]

    class RunPlanParams(BaseModel):
        steps: list[PlanStep] = Field(
            ...,
            min_length=1,
            max_length=max_steps_per_call,
            description=(
                "A sequence of operation calls to run in order, in this one turn, "
                "when you already know the shape of what you need -- e.g. a filter "
                "followed by a sort, or several steps building toward one result. "
                "Each step takes exactly the same params a normal one-off call to "
                "that operation would, plus its own `operation` name and `intent`. "
                "Stops at the first step that doesn't succeed and returns everything "
                "completed so far, including the failure, so a bad step 3 doesn't "
                "lose steps 1-2's real results."
            ),
        )

    def run_plan_fn(ctx: RunContext[dale.DataRegistry], params: RunPlanParams) -> dict:
        if verbosity == "raw":
            new_messages = ctx.messages[action_log.raw_messages_seen :]
            if new_messages:
                print(render_raw_messages(new_messages), flush=True)
            action_log.raw_messages_seen = len(ctx.messages)

        results: list[dict] = []
        for step in params.steps:
            call_params = _call_params(step, exclude={"operation", "intent"})
            payload = _execute_and_log_step(
                ctx.deps,
                action_log,
                operation=step.operation,
                intent=step.intent,
                call_params=call_params,
                peek_at_every_step=peek_at_every_step,
                repetition_nudge=repetition_nudge,
                repetition_limit=repetition_limit,
                verbosity=verbosity,
            )
            results.append(payload)
            if payload.get("status") != "ok":
                break

        all_ok = len(results) == len(params.steps) and all(
            r.get("status") == "ok" for r in results
        )
        return {
            "status": "ok" if all_ok else "partial",
            "steps_completed": len(results),
            "steps_requested": len(params.steps),
            "results": results,
        }

    return Tool(
        run_plan_fn,
        takes_ctx=True,
        name="run_plan",
        max_retries=_TOOL_MAX_RETRIES,
        schema_generator=_UntitledToolJsonSchema,
        description=(
            "Run a sequence of operation calls in order, in one turn. Every "
            "operation is reached through this tool: one step is a single call, "
            "several steps are a plan you already know the shape of -- e.g. a filter "
            "then a sort, or several steps building toward one result. Each step is "
            "logged and executed exactly as if you'd called it on its own; stops at "
            "the first step that doesn't succeed and returns everything completed so "
            "far, including the failure, so you can see what happened and decide your "
            "next move. Batching costs nothing when it's wrong -- a plan that dies at "
            "step 3 still banks steps 1-2 -- so reach for it whenever you can see the "
            "path, and send one step when you can't."
        ),
    )
