from __future__ import annotations

import inspect
import json

import pytest

pytest.importorskip("pydantic_ai")

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import dale
from dale.agent import (
    ActionLog,
    AgentLoopTerminated,
    build_agent,
    build_tools,
    default_system_prompt,
    pick_model,
    registry_state_summary,
    render_raw_messages,
    run_agent,
)


class FakeCtx:
    """Stands in for pydantic_ai.RunContext — run_plan_fn reads ctx.deps, and
    ctx.messages under verbosity="raw". The latter is a class attribute rather
    than absent because run_plan_fn's raw block is now the *only* copy of that
    code (the per-operation tool's duplicate is gone), so it has to be
    reachable from here."""

    messages: list = []

    def __init__(self, deps):
        self.deps = deps


def _run_plan(action_log=None, **kwargs):
    """`(tool, params_model)` for the one tool build_tools returns.

    Asserts the "exactly one tool" invariant on every single use rather than
    once in a dedicated test: if a second tool surface ever comes back, it
    fails here, everywhere, immediately.

    The params model is read off the tool function's own annotation, which is
    also how `scripts/capture_trace_baseline.py` did it — one accessor for this
    across the whole suite, rather than test_agent.py's `inspect.signature` and
    test_harness_usage.py's `__annotations__` for the same thing."""
    tools = build_tools(action_log if action_log is not None else ActionLog(), **kwargs)
    assert [t.name for t in tools] == ["run_plan"]
    tool = tools[0]
    return tool, inspect.signature(tool.function).parameters["params"].annotation


def _batch(registry, tool, params_model, *steps):
    """Submit one run_plan call carrying `steps` and return its payload. A
    single operation call is `_batch(..., one_step)` — the trivial case, not a
    special one."""
    return tool.function(FakeCtx(registry), params_model(steps=list(steps)))


def _published(tool):
    """The schema a real request actually carries.

    `tool.tool_def.parameters_json_schema`, never
    `params_model.model_json_schema()`: the latter does not go through
    `Tool(schema_generator=...)`, so a hygiene assertion made against it would
    be an assertion about a schema nobody sends."""
    return tool.tool_def.parameters_json_schema


def _step_defs(tool) -> dict:
    """The published `<X>ParamsPlanStep` `$defs`, keyed by operation name."""
    defs = _published(tool).get("$defs", {})
    return {
        v["properties"]["operation"]["const"]: v
        for k, v in defs.items()
        if k.endswith("PlanStep")
    }


def _step_mapping(tool) -> dict:
    """The discriminator's own operation -> `$def` mapping — the other half of
    "is this operation offered", and the half a model actually dispatches on."""
    return _published(tool)["properties"]["steps"]["items"]["discriminator"]["mapping"]


def _plan_model(*batches: list, note: str = "done", handle: str | None = None):
    """A FunctionModel emitting one `run_plan` call per batch, in order, then
    the final structured output.

    The replacement for `TestModel(call_tools=[...])` throughout this file.
    TestModel synthesizes arguments field by field and cannot produce a valid
    discriminated-union list item at all, so against a run_plan-only agent it
    doesn't exercise the tool — it fails the run with "exceeded max retries".
    Hand-written step lists are the only way to drive a real agent loop
    offline now."""
    sent = {"n": 0}

    def emit(messages, info) -> ModelResponse:
        i = sent["n"]
        sent["n"] += 1
        if i < len(batches):
            return ModelResponse(
                parts=[ToolCallPart("run_plan", {"steps": list(batches[i])})]
            )
        return ModelResponse(
            parts=[ToolCallPart("final_result", {"handle": handle, "note": note})]
        )

    return FunctionModel(emit)


def _widgets(registry, **kwargs):
    return registry.create(
        "list",
        kwargs.pop("rows", [{"name": "Widget"}]),
        name=kwargs.pop("name", "widgets"),
        description="d",
        created_by="fixture",
    )


def test_build_tools_returns_exactly_one_tool():
    """The whole change in one assertion. The catalog is published once, as
    `run_plan`'s `steps` union, and the model never chooses between two
    encodings of the same call."""
    tools = build_tools(ActionLog())
    assert len(tools) == 1
    assert tools[0].name == "run_plan"
    assert tools[0].description


def test_every_operation_is_reachable_as_a_step():
    """The catalog didn't shrink when the tools did — every operation is still
    offered, just through one surface. Asserted on the *published* discriminator
    mapping, which is what the model dispatches on."""
    tool, _ = _run_plan()
    assert set(_step_mapping(tool)) == set(dale.list_operations())
    assert set(_step_defs(tool)) == set(dale.list_operations())


def test_intent_is_required_on_every_step_variant():
    """`intent` survives the move from 17 named tools to 17 union variants. Its
    per-field *description* deliberately does not (it is stated once in
    DEFAULT_SYSTEM_PROMPT instead) — requiredness is not documentation, and
    must not travel with it."""
    tool, _ = _run_plan()
    variants = _step_defs(tool)
    assert len(variants) == len(dale.list_operations())
    for name, schema in variants.items():
        assert "intent" in schema["properties"], name
        assert "intent" in schema["required"], name
        assert "description" not in schema["properties"]["intent"], name


def test_the_step_discriminator_is_the_wire_field_operation():
    """`operation` is not an internal attribute name — it is the JSON key the
    model writes on every single step it ever sends, and the key pydantic
    dispatches the union on. It travels in three places that have to agree:
    the discriminator declaration, each variant's `const`, and each variant's
    `required` list. Asserted on the *published* schema (`_published`), since
    that is the only one that reaches the model.

    A rename that updated the Python field but left any one of these three
    saying `primitive` would fail every model call with a validation error the
    model cannot repair, because the schema it was shown would disagree with
    the schema it is validated against."""
    tool, _ = _run_plan()
    steps = _published(tool)["properties"]["steps"]

    assert steps["items"]["discriminator"]["propertyName"] == "operation"
    assert set(steps["items"]["discriminator"]["mapping"]) == set(dale.list_operations())

    for name, schema in _step_defs(tool).items():
        assert schema["properties"]["operation"]["const"] == name, name
        assert "operation" in schema["required"], name
        assert "primitive" not in schema["properties"], name


def test_a_step_sent_with_the_pre_rename_primitive_field_is_rejected():
    """A clean break has to be enforced at the wire, not just in the schema
    text. `operation` was `primitive`, and the step models forbid extras — so
    a caller (or a cached tool definition, or a replayed transcript) still
    sending `primitive` must fail loudly at validation rather than have the
    key ignored and the step silently dropped or misrouted.

    The negative case is the one that matters: with `primitive` merely absent
    from the schema, a permissive model would have accepted the step, found no
    discriminator, and either raised somewhere far downstream or run the wrong
    operation. The positive control below is what proves this test would still
    notice if step validation stopped happening at all."""
    _, params_model = _run_plan()

    old_style = {
        "primitive": "peek",
        "handle": "widgets",
        "intent": "inspect",
    }
    with pytest.raises(ValidationError) as exc_info:
        params_model(steps=[old_style])
    # The failure is specifically "no discriminator", not some incidental
    # complaint about the extra key.
    assert "operation" in str(exc_info.value)

    # Positive control: the identical step with the new field name validates.
    new_style = dict(old_style)
    new_style["operation"] = new_style.pop("primitive")
    assert params_model(steps=[new_style]).steps[0].operation == "peek"


def test_a_created_handle_renders_with_its_name_as_an_assignment_prefix(registry):
    """`in_stock_widgets = widgets.filter_where(...)` — the prefix that says
    *when* a handle came into being, rather than leaving a reader to infer it
    from Registry State further down.

    `_render_assignment` reads the handle out of the *serialized* result dict
    by string key, and returns the call unchanged when the key is missing. So
    when the DataHandle's serialized `handle` key became `name`, the prefix
    just stopped being emitted — no exception, no failing type check, and
    every other render assertion (intent, status, Registry State) still
    passed. Only the byte-for-byte golden fixture caught it. This is the same
    property, stated directly, so the next reader gets a one-line diagnosis
    instead of a 400-line JSON diff.

    Also pinned: the non-creating call gets *no* prefix — that branch is what
    makes the prefix mean something."""
    handle = registry.create(
        "list",
        [{"name": "Widget", "in_stock": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "intent": "keep in-stock",
            "name": "in_stock_widgets",
            "description": "in-stock widgets",
        },
        {"operation": "peek", "handle": "in_stock_widgets", "intent": "check", "n": 1},
    )

    creating, inspecting = log.entries
    assert ActionLog._render_entry(creating).splitlines()[1] == (
        "    Action: in_stock_widgets = widgets.filter_where(predicate='in_stock == True')"
    )
    # peek creates nothing, so there is nothing to assign.
    assert ActionLog._render_entry(inspecting).splitlines()[1] == (
        "    Action: in_stock_widgets.peek(n=1)"
    )

    rendered = log.render()
    assert "in_stock_widgets = widgets.filter_where(" in rendered

    # And the reason it works: the name the prefix prints comes from the
    # serialized DataHandle's `name` key, which is also the real registry id.
    assert creating.result["handle"]["name"] == "in_stock_widgets"
    assert "handle" not in creating.result["handle"]


def test_tool_call_records_action_log_entry(registry):
    handle = registry.create(
        "list",
        [{"name": "Widget", "in_stock": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    outer = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "intent": "keep in-stock items",
            "name": "in_stock_widgets",
            "description": "widgets currently in stock",
        },
    )
    assert outer["status"] == "ok"
    assert outer["steps_completed"] == 1 and outer["steps_requested"] == 1
    payload = outer["results"][0]

    assert payload["status"] == "ok"
    assert len(log.entries) == 1
    entry = log.entries[0]
    assert entry.step == 1
    assert entry.intent == "keep in-stock items"
    assert entry.operation == "filter_where"
    assert entry.status == "ok"
    assert "intent" not in entry.params  # stripped before reaching dale.call_operation
    # ...and so is `operation`, the discriminator field this change introduced.
    # A leak would be loud at dispatch (every param schema forbids extras), but
    # entry.params *is* the audit record, so it is pinned here rather than left
    # to be inferred from some other test going red.
    assert "operation" not in entry.params
    assert entry.params["name"] == "in_stock_widgets"  # forwarded, unlike intent
    assert payload["handle"]["name"] == "in_stock_widgets"  # name IS the real handle now


def test_tool_call_error_path_is_logged(registry):
    log = ActionLog()
    tool, params_model = _run_plan(log)

    outer = _batch(
        registry,
        tool,
        params_model,
        {"operation": "peek", "handle": "does_not_exist", "intent": "inspect a bad handle"},
    )
    assert outer["status"] == "partial"
    payload = outer["results"][0]

    assert payload["status"] == "error"
    assert payload["code"] == "HANDLE_NOT_FOUND"
    assert len(log.entries) == 1
    assert log.entries[0].status == "error"
    assert log.entries[0].intent == "inspect a bad handle"


def test_action_log_step_numbers_increment_across_separate_batches(registry):
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)
    step = {"operation": "peek", "handle": handle.name, "intent": "check"}

    for _ in range(3):
        _batch(registry, tool, params_model, step)

    assert [e.step for e in log.entries] == [1, 2, 3]


def test_action_log_step_numbers_increment_within_one_batch(registry):
    """The same invariant on the other axis. Step numbering across a batch was
    only ever covered incidentally, and it is now the common case: three steps
    arriving together must number 1, 2, 3, exactly as three separate calls
    would."""
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)
    step = {"operation": "peek", "handle": handle.name, "intent": "check"}

    payload = _batch(registry, tool, params_model, step, dict(step), dict(step))

    assert payload["steps_completed"] == 3
    assert [e.step for e in log.entries] == [1, 2, 3]


def test_action_log_render_includes_intent_and_status(registry):
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)
    _batch(
        registry,
        tool,
        params_model,
        {"operation": "peek", "handle": handle.name, "intent": "sanity check"},
    )

    rendered = log.render()
    assert "sanity check" in rendered
    assert "peek" in rendered
    assert "ok" in rendered


def test_action_log_render_shows_predicate_as_boolean_logic_not_raw_json(registry):
    handle = registry.create(
        "list",
        [{"name": "Widget", "in_stock": True, "category": "Tools", "price": 25.0}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    nested_predicate = {
        "and": [
            {"field": "in_stock", "op": "==", "value": True},
            {
                "not": {
                    "or": [
                        {"field": "category", "op": "==", "value": "Furniture"},
                        {"field": "price", "op": "<", "value": 10},
                    ]
                }
            },
        ]
    }
    _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": nested_predicate,
            "intent": "filter",
            "name": "filtered_widgets",
            "description": "widgets matching the nested predicate",
        },
    )

    rendered = log.render()
    assert "(in_stock == True AND NOT (category == 'Furniture' OR price < 10))" in rendered
    # The raw nested-dict form should not appear verbatim in the rendered log.
    assert "'and':" not in rendered and '"and":' not in rendered


def test_registry_state_summary_empty_and_populated(registry):
    assert registry_state_summary(registry) == "(none loaded yet)"
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="in-stock widgets", created_by="fixture"
    )
    summary = registry_state_summary(registry)
    assert "widgets" in summary
    assert "list" in summary
    assert "in-stock widgets" in summary


def test_pick_model_prefers_override(monkeypatch):
    monkeypatch.setenv("DALE_MODEL", "anthropic:claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert pick_model() == "anthropic:claude-sonnet-5"


def test_pick_model_prefers_anthropic_over_gemini(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert pick_model() == "anthropic:claude-haiku-4-5"


def test_pick_model_prefers_openai_over_gemini(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "z")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert pick_model() == "openai:gpt-5.6"


def test_pick_model_falls_back_to_gemini(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert pick_model() == "google:gemini-3.6-flash"


def test_pick_model_falls_back_to_moonshotai(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MOONSHOTAI_API_KEY", "w")
    assert pick_model() == "moonshotai:kimi-k2.6"


def test_pick_model_falls_back_to_deepseek(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOTAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "w")
    assert pick_model() == "deepseek:deepseek-v4-flash"


def test_pick_model_falls_back_to_alibaba(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOTAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ALIBABA_API_KEY", "w")
    assert pick_model() == "alibaba:qwen-plus"


def test_pick_model_accepts_dashscope_api_key(monkeypatch):
    # pydantic-ai's AlibabaProvider itself checks ALIBABA_API_KEY then
    # DASHSCOPE_API_KEY (Alibaba's own docs use the latter name) -- no
    # aliasing needed on DALE's side, just recognize either is set.
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOTAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "w")
    assert pick_model() == "alibaba:qwen-plus"


def test_pick_model_raises_without_any_key(monkeypatch):
    monkeypatch.delenv("DALE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOTAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        pick_model()


# --- param serialization: explicit nulls vs. unset optionals ----------------


def _null_x_registry(registry):
    return registry.create(
        "list",
        [{"id": 1, "x": None}, {"id": 2, "x": 5}],
        name="rows_list",
        description="two rows, one with a null x",
        created_by="fixture",
    )


def test_an_explicitly_sent_null_value_survives_a_run_plan_step(registry):
    """`{"field": "x", "op": "==", "value": None}` is schema-valid and is how a
    model asks "which rows have x set to null?". The agent layer used to dump
    params with `exclude_none=True`, which recurses, so `value` was deleted en
    route and dispatch answered `INVALID_PARAMS: predicate.Comparison.value
    Field required` — telling the model to add a field it had already added.
    That is verbatim the malformed filter_where behind the 46-identical-retry
    pilot failure (the evaluation pilot), i.e. a false positive manufactured by
    DALE's own serialization and then policed by DALE's own repetition limit.

    The length-1 case. Its pre-change twin asserted the same thing on the
    standalone named-tool path, which no longer exists; rather than keep a
    byte-identical copy of this test, that one became the mid-batch case
    below."""
    handle = _null_x_registry(registry)
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "find rows whose x is null",
            "handle": handle.name,
            "predicate": {"field": "x", "op": "==", "value": None},
            "name": "null_x_list",
            "description": "rows with a null x",
        },
    )

    assert payload["status"] == "ok"
    assert payload["results"][0]["status"] == "ok"
    assert log.entries[0].params["predicate"] == {"field": "x", "op": "==", "value": None}
    assert registry.materialize("null_x_list") == [{"id": 1, "x": None}]


def test_an_explicitly_sent_null_value_reaches_dispatch_intact_mid_batch(registry):
    """The same guarantee for a step that isn't the first one. `_call_params`
    runs per step inside a loop now, so "the null survived" and "the null
    survived at position 2" are genuinely different claims — the second is the
    one that would catch a serialization rule accidentally keyed to the batch
    rather than the step."""
    handle = _null_x_registry(registry)
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {"operation": "peek", "intent": "look first", "handle": handle.name},
        {
            "operation": "filter_where",
            "intent": "find rows whose x is null",
            "handle": handle.name,
            "predicate": {"field": "x", "op": "==", "value": None},
            "name": "null_x_list",
            "description": "rows with a null x",
        },
    )

    assert payload["status"] == "ok"
    assert log.entries[1].params["predicate"] == {"field": "x", "op": "==", "value": None}
    assert registry.materialize("null_x_list") == [{"id": 1, "x": None}]


def test_optional_params_the_model_never_sent_are_still_stripped(registry):
    """The half `exclude_none` was right about, pinned so the fix can't quietly
    regress it: a field the model never mentioned must not be forwarded to
    dispatch at all — neither as an explicit null (`describe.field`, whose
    absence means "schema summary") nor as a redundant echo of its own default
    (`peek.n`), which the param schema re-applies on validation anyway.

    The only coverage anywhere of `exclude_unset`'s "strip what the model never
    sent" half, and `run_plan` is now its only caller."""
    handle = _null_x_registry(registry)
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {"operation": "describe", "handle": handle.name, "intent": "what fields exist"},
        {"operation": "peek", "handle": handle.name, "intent": "check the shape"},
    )

    assert "field" not in log.entries[0].params
    assert payload["results"][0]["result"]["fields"] == {"id": "int", "x": "int"}
    assert "n" not in log.entries[1].params
    # PeekParams.n default, re-applied at dispatch rather than echoed by DALE.
    assert len(payload["results"][1]["result"]["sample"]) == 1


# --- run_plan ---------------------------------------------------------------


def test_run_plan_executes_multiple_steps_in_order(registry):
    handle = registry.create(
        "list",
        [
            {"name": "Widget", "price": 25.0, "in_stock": True},
            {"name": "Gadget", "price": 15.0, "in_stock": True},
            {"name": "Gizmo", "price": 5.0, "in_stock": False},
        ],
        name="products_list",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "keep in-stock items",
            "handle": handle.name,
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "name": "in_stock_list",
            "description": "in-stock products",
        },
        {
            "operation": "sort_by",
            "intent": "sort by price ascending",
            "handle": "in_stock_list",
            "keys": [{"field": "price", "order": "asc"}],
            "name": "sorted_in_stock_list",
            "description": "in-stock products sorted by price",
        },
    )

    assert payload["status"] == "ok"
    assert payload["steps_completed"] == 2
    assert payload["steps_requested"] == 2
    # Both steps got real, individually-logged ActionLog entries -- a
    # batched call is indistinguishable in the trace from two calls made on
    # separate turns.
    assert len(log.entries) == 2
    assert [e.operation for e in log.entries] == ["filter_where", "sort_by"]
    assert [e.intent for e in log.entries] == [
        "keep in-stock items",
        "sort by price ascending",
    ]
    result = registry.materialize("sorted_in_stock_list")
    assert [r["name"] for r in result] == ["Gadget", "Widget"]


def test_run_plan_stops_at_first_failure_and_returns_partial_results(registry):
    handle = registry.create(
        "list", [{"name": "Widget", "in_stock": True}], name="widgets", description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "keep in-stock items",
            "handle": handle.name,
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "name": "in_stock_list",
            "description": "d",
        },
        {
            "operation": "sort_by",
            "intent": "sort a handle that doesn't exist",
            "handle": "does_not_exist",
            "keys": [{"field": "x", "order": "asc"}],
            "name": "unreachable",
            "description": "d",
        },
        {"operation": "release_handle", "intent": "never reached", "handle": "in_stock_list"},
    )

    assert payload["status"] == "partial"
    assert payload["steps_completed"] == 2
    assert payload["steps_requested"] == 3
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][1]["status"] == "error"
    assert payload["results"][1]["code"] == "HANDLE_NOT_FOUND"
    # Step 3 never ran -- only the first two steps are in the log.
    assert len(log.entries) == 2
    # Step 1's real handle survives -- a partial failure doesn't roll back
    # what already succeeded.
    assert registry.materialize("in_stock_list") == [{"name": "Widget", "in_stock": True}]


def test_run_plan_step_requires_intent(registry):
    _, params_model = _run_plan()
    with pytest.raises(ValidationError):
        params_model(
            steps=[
                {
                    "operation": "peek",
                    # intent omitted
                    "handle": "x",
                }
            ]
        )


def test_run_plan_rejects_unknown_operation_name(registry):
    _, params_model = _run_plan()
    with pytest.raises(ValidationError):
        params_model(
            steps=[{"operation": "not_a_real_operation", "intent": "x", "handle": "x"}]
        )


def test_run_plan_excludes_peek_step_under_privacy_mode(registry):
    _, params_model = _run_plan(privacy_mode=True)
    with pytest.raises(ValidationError):
        params_model(steps=[{"operation": "peek", "intent": "x", "handle": "x"}])


def test_run_plan_splices_auto_inspect_per_step(registry):
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)  # peek_at_every_step defaults to True

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "keep everything",
            "handle": handle.name,
            "predicate": {"field": "name", "op": "is_not_null"},
            "name": "all_widgets_list",
            "description": "d",
        },
    )
    assert "auto_inspect" in payload["results"][0]


def test_each_handle_creating_step_of_a_batch_gets_its_own_auto_inspect(registry):
    """The splice is keyed to the step, not to the batch. Three steps — create,
    non-creating, create — must produce two auto_inspects, each showing *its
    own* new handle's data, and none on the step that made no handle. The
    failure mode a per-step loop invites is the last handle winning, or the
    splice landing on the wrong step's payload."""
    registry.create(
        "list",
        [{"name": "Widget", "keep": True}, {"name": "Gizmo", "keep": False}],
        name="products_list",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "keep the kept ones",
            "handle": "products_list",
            "predicate": {"field": "keep", "op": "==", "value": True},
            "name": "kept_list",
            "description": "d",
        },
        {"operation": "peek", "intent": "look", "handle": "products_list"},
        {
            "operation": "filter_where",
            "intent": "and the dropped ones",
            "handle": "products_list",
            "predicate": {"field": "keep", "op": "==", "value": False},
            "name": "dropped_list",
            "description": "d",
        },
    )

    assert payload["steps_completed"] == 3
    assert payload["results"][0]["auto_inspect"]["peek"]["sample"] == [
        {"name": "Widget", "keep": True}
    ]
    assert "auto_inspect" not in payload["results"][1]  # peek creates no handle
    assert payload["results"][2]["auto_inspect"]["peek"]["sample"] == [
        {"name": "Gizmo", "keep": False}
    ]


# --- repetition_nudge (Tier-1 stuck-detection) ------------------


def test_repetition_nudge_appears_only_after_threshold_identical_failures(registry):
    log = ActionLog()
    tool, params_model = _run_plan(log)  # repetition_nudge defaults to True
    call = _failing_peek(registry, tool, params_model)

    first = call()
    second = call()
    third = call()

    assert first["status"] == "error" and "repetition_warning" not in first
    assert second["status"] == "error" and "repetition_warning" not in second
    assert "repetition_warning" in third
    assert "3 times" in third["repetition_warning"]
    assert "HANDLE_NOT_FOUND" in third["repetition_warning"]


def test_repetition_nudge_absent_when_params_vary(registry):
    log = ActionLog()
    tool, params_model = _run_plan(log)

    for handle_name in ("missing_a", "missing_b", "missing_c"):
        payload = _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "handle": handle_name, "intent": "check"},
        )
        assert "repetition_warning" not in payload["results"][0]


def test_repetition_nudge_absent_for_successful_calls(registry):
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    for _ in range(4):
        payload = _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "handle": "widgets", "intent": "check"},
        )["results"][0]
        assert payload["status"] == "ok"
        assert "repetition_warning" not in payload


def test_repetition_nudge_disabled_via_flag(registry):
    log = ActionLog()
    # repetition_limit=None isolates the nudge flag from the kill: this test
    # makes exactly 5 identical failing calls, which is the default limit, so
    # without disabling it the 5th would terminate before asserting anything.
    tool, params_model = _run_plan(log, repetition_nudge=False, repetition_limit=None)
    call = _failing_peek(registry, tool, params_model)

    for _ in range(5):
        assert "repetition_warning" not in call()


def test_repetition_nudge_counts_across_separate_batches_and_within_one(registry):
    """Repurposed from "across run_plan and direct calls", which no longer
    describes anything: there is one surface now. The invariant that survives
    is still worth pinning, and is arguably sharper —
    `_count_prior_identical_failures` scans the whole log, not the current
    batch, so two failures in separate turns plus one arriving inside a
    multi-step batch is still the third occurrence of the same pair."""
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)
    bad = {"operation": "peek", "intent": "check", "handle": "does_not_exist"}

    _batch(registry, tool, params_model, dict(bad))
    _batch(registry, tool, params_model, dict(bad))

    # ...then the identical call arrives as the second step of a batch -- still
    # the third occurrence of the same (operation, params) pair, so it nudges.
    # Step 1 has to succeed, or the batch stops before reaching step 2.
    payload = _batch(
        registry,
        tool,
        params_model,
        {"operation": "peek", "intent": "look", "handle": "widgets"},
        dict(bad),
    )
    assert payload["results"][0]["status"] == "ok"
    assert "repetition_warning" not in payload["results"][0]
    assert "repetition_warning" in payload["results"][1]


# --- repetition_limit (the kill, escalating the nudge above) ---------------


def _failing_peek(registry, tool, params_model):
    """A callable submitting one length-1 batch of an identical, always-failing
    peek step, returning that step's own payload."""

    def call():
        return _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "handle": "does_not_exist", "intent": "check"},
        )["results"][0]

    return call


def test_repetition_limit_terminates_on_the_nth_identical_failure(registry):
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=5)
    call = _failing_peek(registry, tool, params_model)

    for _ in range(4):
        payload = call()
        assert payload["status"] == "error"

    with pytest.raises(AgentLoopTerminated) as excinfo:
        call()

    exc = excinfo.value
    assert exc.code == "REPETITION_LIMIT_EXCEEDED"
    assert exc.operation == "peek"
    assert exc.attempts == 5
    assert "HANDLE_NOT_FOUND" in str(exc)


def test_repetition_limit_records_the_killing_call_before_raising(registry):
    """The ActionLog stands in for a resume/checkpoint feature,
    so the call that ended the run must be *in* it — losing it would delete
    the single entry a post-mortem most needs."""
    log = ActionLog()
    # 4, not 3: _validate_repetition_limit rejects anything that would stop the
    # run before the model has been warned at least once, and the nudge first
    # appears on attempt 3.
    tool, params_model = _run_plan(log, repetition_limit=4)
    call = _failing_peek(registry, tool, params_model)

    call()
    call()
    call()
    with pytest.raises(AgentLoopTerminated):
        call()

    assert len(log.entries) == 4
    fatal = log.entries[-1]
    assert fatal.operation == "peek"
    assert fatal.status == "error"
    # ...carrying the nudge it was warned with, not a bare error.
    assert "repetition_warning" in fatal.result


def test_repetition_limit_none_never_terminates(registry):
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=None)
    call = _failing_peek(registry, tool, params_model)

    for _ in range(8):
        assert call()["status"] == "error"


def test_repetition_limit_not_triggered_by_varying_params(registry):
    """Exact-equality counting is deliberate: a varying-param loop is left to
    max_tool_calls / pydantic-ai's UsageLimits rather than a fuzzy 'similar
    enough' heuristic."""
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=4)

    for i in range(8):
        payload = _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "handle": f"missing_{i}", "intent": "check"},
        )["results"][0]
        assert payload["status"] == "error"


def test_repetition_limit_counts_across_separate_batches(registry):
    """Repurposed from "across run_plan and direct calls" — see
    test_repetition_nudge_counts_across_separate_batches_and_within_one. The
    counter reads the whole log, so three failures in three separate turns plus
    a fourth inside a later batch still reaches the limit, and takes the whole
    batch down with it."""
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=4)
    call = _failing_peek(registry, tool, params_model)

    call()
    call()
    call()

    with pytest.raises(AgentLoopTerminated):
        call()


def test_run_plan_terminating_mid_batch_keeps_the_steps_that_already_ran(registry):
    """Raising out of step 3 discards run_plan's *return value* — the partial
    results dict the model would normally have seen — but must not discard the
    work steps 1-2 actually did. The ActionLog is what stands in for a
    resume/checkpoint feature, and the handles are real entries
    in a real registry; both outlive the call that was in flight, exactly as
    they would if the run had ended on a standalone call instead.

    A multi-step plan is the whole point here: a one-step plan can't tell the
    difference between "kept what completed" and "there was nothing to keep."""
    registry.create(
        "list",
        [
            {"name": "Widget", "price": 25.0, "in_stock": True},
            {"name": "Gizmo", "price": 5.0, "in_stock": False},
        ],
        name="products",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=4)
    call = _failing_peek(registry, tool, params_model)

    call()
    call()
    call()

    with pytest.raises(AgentLoopTerminated) as excinfo:
        _batch(
            registry,
            tool,
            params_model,
            {
                "operation": "filter_where",
                "intent": "keep in-stock items",
                "handle": "products",
                "predicate": {"field": "in_stock", "op": "==", "value": True},
                "name": "in_stock_list",
                "description": "d",
            },
            {
                "operation": "sort_by",
                "intent": "sort by price",
                "handle": "in_stock_list",
                "keys": [{"field": "price", "order": "asc"}],
                "name": "sorted_list",
                "description": "d",
            },
            {"operation": "peek", "intent": "check", "handle": "does_not_exist"},
        )

    assert excinfo.value.attempts == 4
    # Steps 1-2 are in the log as their own entries, and the killing step is
    # the last one — nothing was rolled back or swallowed with the discarded
    # return value.
    assert [e.operation for e in log.entries] == [
        "peek", "peek", "peek", "filter_where", "sort_by", "peek",
    ]
    assert [e.status for e in log.entries[3:5]] == ["ok", "ok"]
    # Their handles are alive in both the real registry and the log's view of it.
    assert registry.materialize("sorted_list") == [
        {"name": "Widget", "price": 25.0, "in_stock": True}
    ]
    assert {"in_stock_list", "sorted_list"} <= set(log.alive)


def test_repetition_limit_propagates_out_of_a_real_agent_run(registry):
    """A raise inside a tool must end the run, not be handed back to the model
    as a retryable error — the distinction AgentLoopTerminated exists to make
    (contrast ToolCallLimitError, which a broad `except DaleError` swallows).

    FunctionModel, not TestModel: TestModel cannot synthesize a valid
    discriminated-union list item at all, so it can't express the pathology
    under test — or reach the tool. This model is the UC1 pilot failure in
    miniature — the same rejected call, resent verbatim, forever."""
    calls = {"n": 0}

    def always_the_same_bad_call(messages, info) -> ModelResponse:
        calls["n"] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_plan",
                    {
                        "steps": [
                            {
                                "operation": "peek",
                                "handle": "does_not_exist",
                                "intent": "check",
                                "n": 5,
                            }
                        ]
                    },
                )
            ]
        )

    log = ActionLog()
    agent = build_agent(registry, log, model=FunctionModel(always_the_same_bad_call))

    with pytest.raises(AgentLoopTerminated) as excinfo:
        agent.run_sync("inspect something that does not exist", deps=registry)

    assert excinfo.value.attempts == 5  # the default limit
    # Stopped at 5 rather than running to pydantic-ai's request_limit=50 default.
    assert calls["n"] == 5
    assert len(log.entries) == 5


@pytest.mark.parametrize("limit", [0, -1])
def test_a_successful_call_never_terminates_however_low_the_limit(registry, limit):
    """The repetition counter is only assigned for a call that *didn't*
    succeed, so a successful one leaves it at 0 — and the terminal comparison
    used to fire on that zero, ending a working run on its first call with the
    self-contradictory "stopped after 0 identical failed peek calls: UNKNOWN —".

    Reaches past build_tools deliberately: _validate_repetition_limit now
    rejects these values at construction, so the only way to pin the guard at
    the choke point every call actually goes through is to call it directly.
    Both halves matter — the validation stops a caller configuring this, the
    guard stops the choke point trusting its caller."""
    from dale.agent import _execute_and_log_step

    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()

    payload = _execute_and_log_step(
        registry,
        log,
        operation="peek",
        intent="check",
        call_params={"handle": "widgets"},
        peek_at_every_step=False,
        repetition_nudge=True,
        repetition_limit=limit,
        verbosity="quiet",
    )

    assert payload["status"] == "ok"
    assert len(log.entries) == 1


@pytest.mark.parametrize("limit", [-1, 0, 1, 2, 3])
def test_a_repetition_limit_that_would_kill_before_warning_is_rejected(registry, limit):
    """The invariant _REPETITION_LIMIT_DEFAULT's docstring states — "the model
    is always warned before it is stopped" — enforced rather than merely
    documented. The nudge first appears on attempt 3, so anything at or below
    that stops a run the model was never told was repeating itself."""
    with pytest.raises(ValueError, match="repetition_limit"):
        build_tools(ActionLog(), repetition_limit=limit)
    with pytest.raises(ValueError, match="repetition_limit"):
        build_agent(registry, ActionLog(), model="test", repetition_limit=limit)


def test_a_repetition_limit_leaving_room_for_a_warning_is_accepted(registry):
    """The other side of the same boundary: 4 is the smallest limit that still
    warns first, and None (disable the stop entirely) stays valid."""
    assert build_tools(ActionLog(), repetition_limit=4)
    assert build_tools(ActionLog(), repetition_limit=None)
    assert build_agent(registry, ActionLog(), model="test", repetition_limit=4)


# --- repetition against the cost-estimate gate (cost_gate_exceeded) --------


def _registry_with_a_gated_join() -> dale.DataRegistry:
    """The cost-estimation fixture from tests/test_cost_estimation.py: 10 base
    records against a 10-record bucket is an exact 100-row fan-out, over a
    50-row threshold, so join_lookup returns cost_gate_exceeded rather than
    executing."""
    reg = dale.DataRegistry(limits=dale.RegistryLimits(max_result_rows=50))
    reg.create("list", [{"k": "shared"}] * 10, name="base", description="d", created_by="fixture")
    reg.create(
        "list",
        [{"k": "shared", "v": i} for i in range(10)],
        name="events",
        description="d",
        created_by="fixture",
    )
    dale.call_operation(
        reg,
        "group_by",
        {"handle": "events", "key_fields": ["k"], "name": "grouped", "description": "d"},
    )
    return reg


_GATED_JOIN_STEP = {
    "operation": "join_lookup",
    "intent": "attach each account's events",
    "base_handle": "base",
    "index_handle": "grouped",
    "on": ["k"],
    "how": "inner",
    "name": "joined",
    "description": "d",
}


def test_repeated_cost_gate_exceeded_is_described_as_the_gate_not_as_a_failure():
    """A model that keeps asking for an expensive op and never sets
    confirm=True is genuinely stuck, so it's counted — but a cost_gate_exceeded
    payload carries an `estimate`, not a code/message, so the error wording
    rendered the literal "UNKNOWN — " and called a correct safety response a
    failure. That contradicts DALE's own semantics (eval/harness.py excludes
    cost_gate_exceeded from wasted_turn_rate for exactly this reason) and
    withholds the one fact that would unstick the model: the remedy here is
    confirm=True, not a different approach."""
    registry = _registry_with_a_gated_join()
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=4)

    def call():
        return _batch(registry, tool, params_model, dict(_GATED_JOIN_STEP))["results"][0]

    assert call()["status"] == "cost_gate_exceeded"
    assert call()["status"] == "cost_gate_exceeded"
    warned = call()
    with pytest.raises(AgentLoopTerminated) as excinfo:
        call()

    nudge = warned["repetition_warning"]
    assert "cost-estimate gate" in nudge
    assert "confirm=True" in nudge
    assert "UNKNOWN" not in nudge and "failed" not in nudge

    exc = excinfo.value
    assert exc.code == "REPETITION_LIMIT_EXCEEDED"
    assert exc.attempts == 4
    assert "cost-estimate gate refused" in str(exc)
    assert "confirm=True was never set" in str(exc)
    assert "UNKNOWN" not in str(exc)


# --- max_tool_calls (the session's total-call ceiling) ---------------------


def test_tool_call_limit_terminates_the_run():
    """RegistryLimits.max_tool_calls is documented as the in-process
    runaway-loop backstop. It used not to be one: ToolCallLimitError is a
    DaleError, so the broad handler turned it into a payload the model could
    ignore and keep calling past."""
    registry = dale.DataRegistry(limits=dale.RegistryLimits(max_tool_calls=2))
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    def call():
        return _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "handle": "widgets", "intent": "check"},
        )["results"][0]

    assert call()["status"] == "ok"
    assert call()["status"] == "ok"

    with pytest.raises(AgentLoopTerminated) as excinfo:
        call()

    assert excinfo.value.code == "TOOL_CALL_LIMIT_EXCEEDED"
    assert excinfo.value.attempts is None  # no single call was repeated
    # The rejected call is still in the log — it's what explains why the run
    # stopped, and record_call() fires before the operation runs, so the entry
    # correctly records a call that never executed.
    assert len(log.entries) == 3
    assert log.entries[-1].status == "error"
    assert log.entries[-1].result["code"] == "TOOL_CALL_LIMIT_EXCEEDED"


def test_tool_call_limit_wins_when_both_terminal_conditions_fire_at_once():
    """The two stops are independent counters that can come due on the very
    same call, and only one exception gets raised — so which one it is has to
    be a decision, not an accident of statement order. It's the tool-call
    limit: that one is the registry refusing to run the call at all (nothing
    executed, and no later call will either), whereas repetition describes the
    call that did run. Reporting the repetition here would name a symptom while
    the session's hard ceiling was the actual reason nothing more can happen."""
    registry = dale.DataRegistry(limits=dale.RegistryLimits(max_tool_calls=3))
    log = ActionLog()
    tool, params_model = _run_plan(log, repetition_limit=4)
    call = _failing_peek(registry, tool, params_model)

    for _ in range(3):
        assert call()["status"] == "error"

    # Call 4 is simultaneously the 4th identical failure (repetition_limit=4)
    # and one past max_tool_calls=3.
    with pytest.raises(AgentLoopTerminated) as excinfo:
        call()

    assert excinfo.value.code == "TOOL_CALL_LIMIT_EXCEEDED"
    assert excinfo.value.attempts is None
    assert log.entries[-1].result["code"] == "TOOL_CALL_LIMIT_EXCEEDED"


def test_tool_call_limit_catches_a_loop_repetition_limit_cannot():
    """The reason both layers exist: repetition_limit only sees byte-identical
    resends, so a loop that varies its params walks straight past it. This is
    that loop."""
    registry = dale.DataRegistry(limits=dale.RegistryLimits(max_tool_calls=4))
    calls = {"n": 0}

    def always_a_different_bad_call(messages, info) -> ModelResponse:
        calls["n"] += 1
        # Never the same params twice -> repetition_limit never triggers.
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_plan",
                    {
                        "steps": [
                            {
                                "operation": "peek",
                                "handle": f"missing_{calls['n']}",
                                "intent": "check",
                                "n": 5,
                            }
                        ]
                    },
                )
            ]
        )

    log = ActionLog()
    agent = build_agent(registry, log, model=FunctionModel(always_a_different_bad_call))

    with pytest.raises(AgentLoopTerminated) as excinfo:
        agent.run_sync("keep looking for things", deps=registry)

    assert excinfo.value.code == "TOOL_CALL_LIMIT_EXCEEDED"
    # Every entry is a distinct HANDLE_NOT_FOUND, so no repetition_warning was
    # ever emitted — this loop was invisible to repetition_limit by design.
    assert not any("repetition_warning" in e.result for e in log.entries)


# --- privacy_mode / peek_at_every_step ------------------------------------


def test_peek_is_absent_from_the_step_union_under_privacy_mode():
    """The carve-out survives its move from "drop a tool" to "drop a union
    variant". Asserted on the published schema, both halves — a variant left in
    `$defs` but missing from the discriminator mapping (or vice versa) is
    exactly the half-done version of this change that a tool-name assertion
    could no longer see, since the name set is `{"run_plan"}` either way."""
    tool, _ = _run_plan(privacy_mode=True)

    assert "peek" not in _step_mapping(tool)
    assert "peek" not in _step_defs(tool)
    assert "PeekParamsPlanStep" not in _published(tool)["$defs"]
    # Nothing else was dropped along with it.
    assert set(_step_mapping(tool)) == set(dale.list_operations()) - {"peek"}
    # describe stays: its aggregate statistics were never "individual real
    # values" under this design's own definition.
    assert "describe" in _step_mapping(tool)


def test_peek_present_in_the_step_union_by_default():
    tool, _ = _run_plan()
    assert "peek" in _step_mapping(tool)
    assert "peek" in _step_defs(tool)


def test_auto_inspect_added_after_handle_creating_call(registry):
    handle = registry.create(
        "list",
        [{"name": "Widget", "in_stock": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)  # peek_at_every_step defaults to True

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "intent": "keep in-stock items",
            "name": "in_stock_widgets",
            "description": "widgets in stock",
        },
    )["results"][0]

    assert "auto_inspect" in payload
    assert "peek" in payload["auto_inspect"]
    assert "describe" in payload["auto_inspect"]
    assert payload["auto_inspect"]["peek"]["sample"] == [
        {"name": "Widget", "in_stock": True}
    ]


def test_auto_inspect_of_a_nested_handle_stays_bounded(registry):
    """The path that made peek's cap load-bearing rather than advisory: nobody
    asked for this peek. peek_at_every_step splices one into every
    handle-creating call, and default_system_prompt splices one per pre-loaded
    handle into the prompt itself, so an unbounded peek of a nested document
    detonated before the model had taken a turn — 531KB measured on a
    load_json of an ordinary top-level JSON object. Asserted end to end, in
    bytes, through the two places DALE calls peek on the model's behalf."""
    registry.create(
        "list",
        [{"id": i, "doc": {"rows": [{"n": j, "t": "z" * 80} for j in range(200)]}}
         for i in range(5)],
        name="docs_list",
        description="nested documents",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)  # peek_at_every_step defaults to True

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "index_by",
            "handle": "docs_list",
            "key_fields": ["id"],
            "intent": "index the documents by id",
            "name": "docs_dict",
            "description": "documents keyed by id",
        },
    )["results"][0]

    inspected = json.dumps(payload["auto_inspect"], default=str)
    assert len(inspected.encode("utf-8")) < 10_000
    assert payload["auto_inspect"]["peek"]["truncated"] is True
    # And the same peek spliced into the system prompt, which no tool call
    # gates and no model choice avoids.
    assert len(default_system_prompt(registry).encode("utf-8")) < 20_000


def test_auto_inspect_absent_when_peek_at_every_step_disabled(registry):
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log, peek_at_every_step=False)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": {"field": "name", "op": "==", "value": "Widget"},
            "intent": "keep",
            "name": "filtered",
            "description": "d",
        },
    )["results"][0]

    assert "auto_inspect" not in payload


def test_auto_inspect_absent_under_privacy_mode_even_if_requested():
    registry = dale.DataRegistry(privacy_mode=True)
    handle = registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log, peek_at_every_step=True, privacy_mode=True)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "handle": handle.name,
            "predicate": {"field": "name", "op": "==", "value": "Widget"},
            "intent": "keep",
            "name": "filtered",
            "description": "d",
        },
    )["results"][0]

    assert "auto_inspect" not in payload


def test_build_agent_system_prompt_includes_privacy_note_only_under_privacy_mode():
    off_registry = dale.DataRegistry()
    off_registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    on_registry = dale.DataRegistry(privacy_mode=True)
    on_registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )

    off_agent = build_agent(off_registry, ActionLog(), model="test")
    on_agent = build_agent(on_registry, ActionLog(), model="test")

    off_prompt = off_agent._system_prompts[0]
    on_prompt = on_agent._system_prompts[0]
    assert "privacy_mode is enabled" not in off_prompt
    assert "privacy_mode is enabled" in on_prompt
    assert "peek() is not available" in on_prompt


def test_build_agent_system_prompt_includes_initial_inspect_block():
    registry = dale.DataRegistry()
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    agent = build_agent(registry, ActionLog(), model="test")
    prompt = agent._system_prompts[0]
    assert "Initial peek/describe" in prompt
    assert "Widget" in prompt


def test_build_agent_custom_system_prompt_used_verbatim():
    registry = dale.DataRegistry(privacy_mode=True)
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    agent = build_agent(registry, ActionLog(), model="test", system_prompt="custom prompt")
    assert agent._system_prompts[0] == "custom prompt"


_THREE_STEP_BATCH = [
    {
        "operation": "filter_where",
        "intent": "keep in-stock items",
        "handle": "widgets",
        "predicate": {"field": "in_stock", "op": "==", "value": True},
        "name": "in_stock_list",
        "description": "d",
    },
    {
        "operation": "sort_by",
        "intent": "sort by price",
        "handle": "in_stock_list",
        "keys": [{"field": "price", "order": "asc"}],
        "name": "sorted_list",
        "description": "d",
    },
    {"operation": "peek", "intent": "check the result", "handle": "sorted_list"},
]


def _widget_registry(registry):
    registry.create(
        "list",
        [{"name": "Widget", "price": 25.0, "in_stock": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    return registry


def test_build_agent_runs_a_full_loop_over_one_batch(registry):
    """Replaces the old TestModel walk over every registered tool: with one
    tool there is nothing for TestModel to walk, and it can't synthesize a
    valid step anyway. The invariant that matters is unchanged — every step
    the model submits becomes one correctly-attributed ActionLog entry,
    dispatched for real through dale.call_operation, not a mocked path."""
    _widget_registry(registry)
    log = ActionLog()
    agent = build_agent(
        registry, log, model=_plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    )

    result = agent.run_sync("do something", deps=registry)

    assert result.output
    assert [e.operation for e in log.entries] == ["filter_where", "sort_by", "peek"]
    assert [e.step for e in log.entries] == [1, 2, 3]
    assert all(e.status == "ok" for e in log.entries)
    # Both handle-creating steps produced real handles in the real registry.
    assert registry.materialize("sorted_list") == [
        {"name": "Widget", "price": 25.0, "in_stock": True}
    ]


# --- raw messages / run_agent ----------------------------------------------


def test_render_raw_messages_labels_each_part_kind():
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    messages = [
        ModelRequest(parts=[SystemPromptPart(content="be helpful"), UserPromptPart(content="hi")]),
        ModelResponse(parts=[ToolCallPart(tool_name="peek", args={"handle": "x"})]),
        ModelRequest(parts=[ToolReturnPart(tool_name="peek", content={"status": "ok"})]),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    rendered = render_raw_messages(messages)
    assert "[SYSTEM PROMPT]\nbe helpful" in rendered
    assert "[USER PROMPT]\nhi" in rendered
    assert '[MODEL -> TOOL CALL] peek({"handle": "x"})' in rendered
    assert "[TOOL -> MODEL RETURN] peek:" in rendered
    assert "[MODEL TEXT] done" in rendered


def test_run_agent_prints_raw_tail_after_run(registry, capsys):
    """`run_plan_fn`'s raw block is now the only copy of the live raw-printing
    code — the per-operation tool's duplicate is gone — so this is the only
    guard standing between that seam and silence. It asserts both halves: the
    live per-call print of the model's own tool-call args, and run_agent's
    post-run flush of the trailing messages nothing else would ever show."""
    _widget_registry(registry)
    log = ActionLog()
    model = _plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    agent = build_agent(registry, log, model=model, verbosity="raw")
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="raw")

    assert outcome.success
    captured = capsys.readouterr().out
    # Printed live, from inside the tool call, before DALE's own view of it.
    assert "[MODEL -> TOOL CALL] run_plan(" in captured
    # The trailing tool return(s) + final model output never trigger another
    # tool call, so they'd never print live -- confirm run_agent's post-run
    # flush actually shows them.
    assert "[TOOL -> MODEL RETURN]" in captured
    assert log.raw_messages_seen == len(outcome.result.all_messages())


def test_run_agent_prints_success_result_section(registry, capsys):
    _widget_registry(registry)
    log = ActionLog()
    model = _plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    agent = build_agent(registry, log, model=model, verbosity="normal")
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="normal")

    assert outcome.success
    captured = capsys.readouterr().out
    assert "Result: Success" in captured
    assert "note:" in captured


def test_run_agent_quiet_prints_nothing_extra(registry, capsys):
    _widget_registry(registry)
    log = ActionLog()
    model = _plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    agent = build_agent(registry, log, model=model, verbosity="quiet")
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="quiet")

    assert outcome.success
    captured = capsys.readouterr().out
    assert captured == ""


def test_run_agent_reports_failure_without_raising(registry, capsys):
    log = ActionLog()

    class ExplodingAgent:
        # Signature mirrors the real Agent.run_sync, including the `usage=`
        # accumulator run_agent hands it so a failed run still reports spend.
        model = None

        def run_sync(self, task, deps, usage=None):
            raise RuntimeError("boom")

    outcome = run_agent(ExplodingAgent(), "task", deps=registry, action_log=log, verbosity="normal")

    assert outcome.success is False
    assert outcome.result is None
    assert "boom" in outcome.error
    captured = capsys.readouterr().out
    assert "Result: Failure" in captured
    assert "boom" in captured


def test_run_agent_prints_the_timing_split_and_populates_wall_clock(registry, capsys):
    """The host-compute vs. waiting-on-the-model line prints on every
    non-quiet run, not just under a profiler — that ratio (a measured
    10,005-row run: 0.07% host) is the most counter-intuitive fact about
    operating DALE, and it's only counter-intuitive if you see it."""
    _widget_registry(registry)
    log = ActionLog()
    model = _plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    agent = build_agent(registry, log, model=model, verbosity="normal")
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="normal")

    # The timing line prints on a *failed* run too, so without this the test
    # would read green off a batch that never executed -- e.g. if a step-schema
    # change quietly stopped _THREE_STEP_BATCH validating. The figures below are
    # only about anything if there was work to measure.
    assert outcome.success
    assert len(log.entries) == 3
    captured = capsys.readouterr().out
    assert "  time: " in captured
    assert "host compute" in captured
    assert "model + network" in captured
    assert outcome.wall_clock_s > 0


def test_run_agent_omits_the_timing_line_when_quiet(registry, capsys):
    _widget_registry(registry)
    log = ActionLog()
    model = _plan_model(_THREE_STEP_BATCH, handle="sorted_list")
    agent = build_agent(registry, log, model=model, verbosity="quiet")
    outcome = run_agent(agent, "do something", deps=registry, action_log=log, verbosity="quiet")

    # Same reason as the test above: "nothing was printed" is also what a run
    # that never reached the tool looks like.
    assert outcome.success
    assert len(log.entries) == 3
    assert "time:" not in capsys.readouterr().out
    # Still measured, just not printed — the caller can read it off the outcome.
    assert outcome.wall_clock_s > 0


class _LoggingAgent:
    """Stands in for a real Agent by doing the one thing that matters to the
    timing line — recording entries on the ActionLog as a run proceeds — then
    failing, which is the path that needs no AgentRunResult to be faked.
    Signature mirrors Agent.run_sync, including the `usage=` accumulator."""

    model = None

    def __init__(self, action_log, *, elapsed_ms: float, auto_inspect_ms: float) -> None:
        self._log = action_log
        self._elapsed_ms = elapsed_ms
        self._auto_inspect_ms = auto_inspect_ms

    def run_sync(self, task, deps, usage=None):
        self._log.record(
            intent="i",
            operation="filter_where",
            params={},
            result={"status": "ok"},
            elapsed_ms=self._elapsed_ms,
            auto_inspect_ms=self._auto_inspect_ms,
        )
        raise RuntimeError("boom")


def test_run_agent_attributes_auto_inspect_inside_the_host_compute_figure(registry, capsys):
    """peek_at_every_step's cost is inside the host-compute number (it is host
    compute) but named separately on a continuation line, so a convenience
    feature can't hide inside the operations it wraps."""
    log = ActionLog()
    agent = _LoggingAgent(log, elapsed_ms=3.0, auto_inspect_ms=2.5)

    run_agent(agent, "task", deps=registry, action_log=log, verbosity="normal")

    captured = capsys.readouterr().out
    assert "5.5 ms host compute" in captured
    assert "(host compute includes 2.5 ms of auto-inspect)" in captured


def test_run_agent_timing_covers_only_its_own_run_of_a_reused_log(registry, capsys):
    """`wall_clock_s` measures one run while an ActionLog accumulates over its
    whole life, so the split has to be taken against a snapshot from this run's
    start. Nothing stops a caller reusing one log across two run_agent calls (a
    multi-turn session is a reasonable thing to want) — without the snapshot
    the second run is charged for the first one's compute, which here would
    report 9,004 ms of host time against a sub-second wall clock."""
    log = ActionLog()
    log.record(
        intent="an earlier run's work",
        operation="group_by",
        params={},
        result={"status": "ok"},
        elapsed_ms=9_000.0,
        auto_inspect_ms=1.0,
    )
    agent = _LoggingAgent(log, elapsed_ms=3.0, auto_inspect_ms=2.5)

    run_agent(agent, "task", deps=registry, action_log=log, verbosity="normal")

    captured = capsys.readouterr().out
    assert "5.5 ms host compute" in captured
    assert "9,00" not in captured and "9004" not in captured
    assert "(host compute includes 2.5 ms of auto-inspect)" in captured


def _peeking_registry(*, privacy_mode: bool) -> dale.DataRegistry:
    registry = dale.DataRegistry(privacy_mode=privacy_mode)
    registry.create(
        "list",
        [{"name": "Widget", "price": 25.0, "in_stock": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    return registry


def _always_peek_model() -> FunctionModel:
    def always_peek(messages, info) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_plan",
                    {"steps": [{"operation": "peek", "handle": "widgets", "intent": "look"}]},
                )
            ]
        )

    return FunctionModel(always_peek)


def test_build_agent_under_privacy_mode_never_calls_peek():
    """A model that *tries* to peek under privacy_mode never reaches dispatch:
    the step variant doesn't exist, so validation rejects the whole batch and
    nothing is logged. Not "peek ran and redacted" — peek never ran.

    The second half is a positive control, and it is not optional. "The run
    failed and the log is empty" is also what a batch rejected for some reason
    that has nothing to do with privacy_mode looks like — a renamed step field,
    a changed discriminator — so on its own the first half would keep passing
    while measuring nothing. The control pins that this exact emission does
    reach dispatch when privacy_mode is off, which makes the difference between
    the two runs attributable to the carve-out and nothing else."""
    private = _peeking_registry(privacy_mode=True)
    private_log = ActionLog()
    private_agent = build_agent(private, private_log, model=_always_peek_model())
    outcome = run_agent(private_agent, "do something", deps=private, action_log=private_log)

    # The run dies on the retry budget rather than succeeding, which is the
    # honest outcome for a model that will only ever ask for the one thing it
    # isn't offered -- but nothing was executed, and nothing was logged.
    assert outcome.success is False
    assert private_log.entries == []

    # The control: same model, same emission, privacy_mode off. peek runs.
    public = _peeking_registry(privacy_mode=False)
    public_log = ActionLog()
    public_agent = build_agent(public, public_log, model=_always_peek_model())
    run_agent(public_agent, "do something", deps=public, action_log=public_log)

    assert [e.operation for e in public_log.entries][:1] == ["peek"]
    assert public_log.entries[0].status == "ok"


# --- schema hygiene: what actually goes on the wire ------------------------


def test_every_plan_step_variant_carries_its_operations_docstring():
    """The one thing that must not be skipped. While each operation also had
    its own named tool, that tool carried the docstring and these variants all
    shipped `description: None` — a 7,474-token *information deficit* the
    moment the named tools go away, since the model would then be selecting
    among 17 operations on name and parameter shape alone.

    Full docstrings, not first lines: that is exact information parity with
    what the named tools published, and is what the 9,502-token figure was
    measured against. Truncating is a different change needing its own
    measurement.

    Compared against `inspect.cleandoc`, not `.strip()`, which is what the code
    passes: pydantic normalizes a docstring's continuation indentation on its
    way into `description`. Same words, same order, minus the leading spaces a
    tool description carried for no reason. That is the only difference between
    what these variants publish and what the named tools published, and it is
    in the right direction."""
    tool, _ = _run_plan()
    variants = _step_defs(tool)

    # Exactly the catalog -- a silently dropped variant fails here too.
    assert set(variants) == set(dale.list_operations())
    for name, schema in variants.items():
        doc = dale.get_operation(name).fn.__doc__ or ""
        assert doc.strip(), f"{name} has no docstring to publish"
        assert schema["description"] == inspect.cleandoc(doc), name
        # Not truncated to a first line: every operation whose docstring has
        # more than one line must publish more than one line.
        if len(doc.strip().splitlines()) > 1:
            assert "\n" in schema["description"], name


def test_no_plan_step_variant_ships_a_null_description():
    """The cheap, obvious-failure-message form of the above."""
    tool, _ = _run_plan()
    nulls = [n for n, d in _step_defs(tool).items() if not d.get("description")]
    assert nulls == []


def test_published_schema_carries_no_title_keys():
    """`title` is pydantic bookkeeping — the model class's own name, restating
    what the surrounding schema already says — and 823 tokens of it on every
    request. Stripped by `Tool(schema_generator=...)`, which is why this must
    read the *published* schema: `model_json_schema()` never goes through it,
    so an assertion there would pass while the wire kept every title."""
    tool, _ = _run_plan()

    def titles(node, path="") -> list[str]:
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "title" and isinstance(value, str):
                    found.append(path)
                elif key in ("properties", "$defs", "patternProperties") and isinstance(
                    value, dict
                ):
                    for k, v in value.items():
                        found += titles(v, f"{path}/{key}/{k}")
                else:
                    found += titles(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                found += titles(value, f"{path}[{i}]")
        return found

    assert titles(_published(tool)) == []


def test_null_comparison_ships_no_description():
    """Its 969-char docstring described a *rejected* design and, because
    `NullComparison` is reachable from `Predicate`, shipped 3-4 times per
    request — 1,101 tokens to explain something DALE deliberately doesn't do.
    Moved to a `#` comment: every word kept, none of it on the wire."""
    tool, _ = _run_plan()
    defs = _published(tool)["$defs"]
    assert "NullComparison" in defs  # still part of the grammar
    assert "description" not in defs["NullComparison"]


def test_the_system_prompt_states_what_intent_is_for():
    """The other half of dropping `intent`'s 17 per-field descriptions: the
    explanation has to exist somewhere, exactly once."""
    from dale.agent import DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT.count("`intent`") == 1
    assert "why you're making that" in DEFAULT_SYSTEM_PROMPT
    assert "action log" in DEFAULT_SYSTEM_PROMPT


def test_the_only_tool_declares_a_retry_budget():
    """17 per-tool-name retry budgets collapsed into one when the tools did, so
    pydantic-ai's inherited default of 1 would mean a single malformed batch
    ends the run. It has to be a number DALE chose."""
    from dale.agent import _REPETITION_LIMIT_DEFAULT, _TOOL_MAX_RETRIES

    tool, _ = _run_plan()
    assert tool.max_retries == _TOOL_MAX_RETRIES
    assert _TOOL_MAX_RETRIES > 1
    assert _TOOL_MAX_RETRIES < _REPETITION_LIMIT_DEFAULT


def _malformed_then_good_model(*, bad_batches: int) -> tuple[FunctionModel, dict]:
    """Emits `bad_batches` schema-invalid run_plan calls, then a valid one-step
    batch, then the final output. `sort_by` without `keys`/`name`/`description`
    is rejected by pydantic before run_plan_fn is entered at all — the whole
    point of the tradeoff being measured."""
    sent = {"n": 0}

    def emit(messages, info) -> ModelResponse:
        i = sent["n"]
        sent["n"] += 1
        if i < bad_batches:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_plan",
                        {"steps": [{"operation": "sort_by", "intent": "malformed", "handle": "widgets"}]},
                    )
                ]
            )
        if i == bad_batches:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_plan",
                        {"steps": [{"operation": "peek", "intent": "recovered", "handle": "widgets"}]},
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("final_result", {"note": "done"})])

    return FunctionModel(emit), sent


def test_a_malformed_batch_is_retried_rather_than_ending_the_run(registry):
    """The behaviour `_TOOL_MAX_RETRIES` exists for, not just its value.

    All N steps validate as one object, so one malformed step kills the whole
    tool call before anything executes — and under batch-only that is the only
    tool there is. With pydantic-ai's inherited default of 1 the first such
    batch would end the run outright. The number being > 1 is asserted above;
    this asserts the consequence, which is the thing anyone actually cares
    about: the model gets the validation error back and a corrected batch on
    the next turn still runs."""
    from dale.agent import _TOOL_MAX_RETRIES

    _widget_registry(registry)
    log = ActionLog()
    model, sent = _malformed_then_good_model(bad_batches=_TOOL_MAX_RETRIES - 1)
    agent = build_agent(registry, log, model=model)

    outcome = run_agent(agent, "do something", deps=registry, action_log=log)

    assert outcome.success, outcome.error
    # Nothing from the malformed batches reached dispatch; only the good one did.
    assert [e.operation for e in log.entries] == ["peek"]
    assert log.entries[0].intent == "recovered"


def test_the_retry_budget_is_finite_and_the_run_ends_when_it_is_spent(registry):
    """The other side of the same number. A budget that never runs out is a
    runaway loop by another name — this change's whole subject — so the model
    that never sends a valid batch has to be stopped, and stopped at the
    declared budget rather than at pydantic-ai's request_limit=50."""
    from dale.agent import _TOOL_MAX_RETRIES

    _widget_registry(registry)
    log = ActionLog()
    model, sent = _malformed_then_good_model(bad_batches=_TOOL_MAX_RETRIES + 1)
    agent = build_agent(registry, log, model=model)

    outcome = run_agent(agent, "do something", deps=registry, action_log=log)

    assert outcome.success is False
    assert "max retries" in (outcome.error or "")
    assert log.entries == []
    # One initial call plus _TOOL_MAX_RETRIES retries, then it gives up -- it
    # never got as far as the valid batch this model would have sent next.
    assert sent["n"] == _TOOL_MAX_RETRIES + 1


def test_raw_verbosity_prints_the_models_own_tool_call_args_before_stripping(
    registry, capsys
):
    """The seam directly, without a whole agent run. `run_plan_fn`'s raw block
    is the only surviving copy of that code, and it is the one place the
    model's *own* arguments are shown — `intent` included, before DALE strips
    it — as opposed to DALE's own Action/Result view of the same call.

    This is also why FakeCtx now defines `messages`: nothing drove
    `run_plan_fn` at verbosity="raw" through it before, because the duplicate
    copy in the per-operation tool made this seam feel covered."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log, verbosity="raw")

    ctx = FakeCtx(registry)
    ctx.messages = [ModelRequest(parts=[UserPromptPart(content="do the thing")])]
    tool.function(
        ctx,
        params_model(steps=[{"operation": "peek", "handle": "widgets", "intent": "check"}]),
    )

    captured = capsys.readouterr().out
    assert "[USER PROMPT]\ndo the thing" in captured
    # Only-what's-new bookkeeping, so the next call doesn't reprint the run.
    assert log.raw_messages_seen == 1
    # ...and DALE's own view of the same call is printed too, not instead.
    assert "widgets.peek(" in captured


# --- failure semantics inside a batch --------------------------------------


def test_a_failing_middle_step_banks_earlier_work_and_skips_later_steps(registry):
    """"never waste a turn", now the universal path rather than
    run_plan's special case. Extends the older stop-at-first-failure test with
    the two things it didn't check: that step 1's own *result* is still in the
    payload the model gets back, and that its handle is materializable — not
    just that a handle exists."""
    registry.create(
        "list",
        [{"name": "Widget", "in_stock": True}, {"name": "Gizmo", "in_stock": False}],
        name="products_list",
        description="d",
        created_by="fixture",
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {
            "operation": "filter_where",
            "intent": "keep in-stock",
            "handle": "products_list",
            "predicate": {"field": "in_stock", "op": "==", "value": True},
            "name": "in_stock_list",
            "description": "d",
        },
        {"operation": "peek", "intent": "look at nothing", "handle": "does_not_exist"},
        {"operation": "release_handle", "intent": "never reached", "handle": "in_stock_list"},
    )

    assert payload["status"] == "partial"
    assert payload["steps_completed"] == 2
    assert payload["steps_requested"] == 3
    # Step 1's real result is returned, not just its side effect.
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["handle"]["name"] == "in_stock_list"
    assert payload["results"][1]["code"] == "HANDLE_NOT_FOUND"
    assert len(log.entries) == 2
    # ...and the work it did is real and still there.
    assert registry.materialize("in_stock_list") == [{"name": "Widget", "in_stock": True}]


def test_a_cost_gated_step_halts_the_batch_like_an_error_does():
    """`cost_gate_exceeded` is not `"ok"`, so it stops a plan exactly as an
    error does. That was only ever asserted for HANDLE_NOT_FOUND; under
    batch-only it is the universal path for every cost-gated call, and the
    distinction matters — the model must get back a *partial* result and an
    estimate it can act on with confirm=True, not a dead batch."""
    registry = _registry_with_a_gated_join()
    log = ActionLog()
    tool, params_model = _run_plan(log)

    payload = _batch(
        registry,
        tool,
        params_model,
        {"operation": "peek", "intent": "look first", "handle": "base"},
        dict(_GATED_JOIN_STEP),
        {"operation": "peek", "intent": "never reached", "handle": "base"},
    )

    assert payload["status"] == "partial"
    assert payload["steps_completed"] == 2
    assert payload["steps_requested"] == 3
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][1]["status"] == "cost_gate_exceeded"
    assert "estimate" in payload["results"][1]
    assert [e.operation for e in log.entries] == ["peek", "join_lookup"]


def test_a_malformed_step_fails_the_whole_batch_before_anything_runs(registry):
    """The documented tradeoff (paper.md Section 3.12), now universal: all N
    steps validate as one Pydantic object, so a schema-invalid step 3 means
    steps 1-2 never execute at all. Previously only implied — the existing
    tests assert validation rejects the call, not that nothing ran."""
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    with pytest.raises(ValidationError):
        _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "intent": "fine", "handle": "widgets"},
            {"operation": "describe", "intent": "also fine", "handle": "widgets"},
            # sort_by needs `keys`, `name` and `description`.
            {"operation": "sort_by", "intent": "malformed", "handle": "widgets"},
        )

    assert log.entries == []
    assert {m.name for m in registry.list_handles()} == {"widgets"}


def test_max_tool_calls_terminates_mid_batch():
    """The session ceiling reaching through a batch: step 1 runs and is logged,
    step 2 is refused by the registry, and the whole run ends. `max_tool_calls`
    was only ever exercised on the standalone path."""
    registry = dale.DataRegistry(limits=dale.RegistryLimits(max_tool_calls=1))
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log)

    with pytest.raises(AgentLoopTerminated) as excinfo:
        _batch(
            registry,
            tool,
            params_model,
            {"operation": "peek", "intent": "one", "handle": "widgets"},
            {"operation": "peek", "intent": "two", "handle": "widgets"},
            {"operation": "peek", "intent": "three", "handle": "widgets"},
        )

    assert excinfo.value.code == "TOOL_CALL_LIMIT_EXCEEDED"
    # Step 1's real entry is kept; step 2's rejection is logged too (it is what
    # explains the stop); step 3 never happened.
    assert [e.status for e in log.entries] == ["ok", "error"]
    assert log.entries[-1].result["code"] == "TOOL_CALL_LIMIT_EXCEEDED"


# --- max_steps_per_call (replaces enable_run_plan) --------------------------


def test_max_steps_per_call_is_published_in_the_schema():
    """Visible to the model as `maxItems`, not merely enforced at runtime: a
    ceiling the model can't see is a ceiling it discovers by wasting a call."""
    tool, _ = _run_plan(max_steps_per_call=3)
    assert _published(tool)["properties"]["steps"]["maxItems"] == 3
    assert _published(tool)["properties"]["steps"]["minItems"] == 1

    unbounded, _ = _run_plan()
    assert "maxItems" not in _published(unbounded)["properties"]["steps"]


def test_max_steps_per_call_of_one_reproduces_the_unbatched_condition(registry):
    """paper.md Section 4.2 part (F)'s ablation arm. `enable_run_plan=False`
    used to express it by withholding a second tool; with one tool it is a
    ceiling on batch size instead, which is the same experiment — one operation
    per round trip — and names what it actually does."""
    registry.create(
        "list", [{"name": "Widget"}], name="widgets", description="d", created_by="fixture"
    )
    log = ActionLog()
    tool, params_model = _run_plan(log, max_steps_per_call=1)
    step = {"operation": "peek", "intent": "check", "handle": "widgets"}

    assert _batch(registry, tool, params_model, dict(step))["status"] == "ok"
    with pytest.raises(ValidationError):
        _batch(registry, tool, params_model, dict(step), dict(step))


@pytest.mark.parametrize("bad", [0, -1])
def test_a_max_steps_per_call_below_one_is_rejected(registry, bad):
    """`maxItems: 0` publishes a tool the model is unable to legally call at
    all — a silent, total failure. Rejected at construction, on both entry
    points, the same way an out-of-range repetition_limit is."""
    with pytest.raises(ValueError, match="max_steps_per_call"):
        build_tools(ActionLog(), max_steps_per_call=bad)
    with pytest.raises(ValueError, match="max_steps_per_call"):
        build_agent(registry, ActionLog(), model="test", max_steps_per_call=bad)


# --- operations= allowlist --------------------------------------------------


def test_an_unknown_operation_name_is_rejected_at_build_time(registry):
    """A typo must not silently shrink the catalog: dropped operations surface
    much later as an inexplicable model failure, with nothing pointing back
    here."""
    with pytest.raises(ValueError, match="flter_where"):
        build_tools(ActionLog(), operations=["filter_where", "flter_where"])
    with pytest.raises(ValueError, match="flter_where"):
        build_agent(
            registry, ActionLog(), model="test", operations=["filter_where", "flter_where"]
        )


def test_the_allowlist_restricts_what_the_model_is_offered(registry):
    tool, params_model = _run_plan(operations=["peek", "describe"])

    assert set(_step_mapping(tool)) == {"peek", "describe"}
    assert set(_step_defs(tool)) == {"peek", "describe"}
    with pytest.raises(ValidationError):
        params_model(
            steps=[
                {
                    "operation": "filter_where",
                    "intent": "x",
                    "handle": "h",
                    "predicate": {"field": "a", "op": "is_null"},
                    "name": "n",
                    "description": "d",
                }
            ]
        )


def test_the_allowlist_does_not_restrict_call_operation(registry):
    """"This governs what the *model* is offered, not what the host may run."
    An eval harness loading its own fixtures through dale.call_operation is not
    the party being constrained."""
    registry.create(
        "list",
        [{"name": "Widget", "keep": True}],
        name="widgets",
        description="d",
        created_by="fixture",
    )
    build_tools(ActionLog(), operations=["peek", "describe"])

    out = dale.call_operation(
        registry,
        "filter_where",
        {
            "handle": "widgets",
            "predicate": {"field": "keep", "op": "==", "value": True},
            "name": "kept_list",
            "description": "d",
        },
    )
    assert out.status == "ok"
    assert registry.materialize("kept_list") == [{"name": "Widget", "keep": True}]


def test_operations_none_is_todays_behaviour():
    """The backward-compatible default, pinned."""
    tool, _ = _run_plan(operations=None)
    assert set(_step_mapping(tool)) == set(dale.list_operations())


def test_the_allowlist_and_privacy_mode_compose():
    """Two filters on the same dimension is exactly where an and/or mistake
    hides. Both must apply: peek is allowed by the caller and forbidden by
    privacy_mode, and privacy_mode wins."""
    tool, _ = _run_plan(operations=["peek", "describe"], privacy_mode=True)
    assert set(_step_mapping(tool)) == {"describe"}


def test_an_allowlist_that_selects_nothing_is_rejected_with_a_legible_error(registry):
    """Left to fall through, an empty selection surfaces as "TypeError: Cannot
    take a Union of no types" from inside `typing` — a message that names
    neither operations nor privacy_mode. Both routes here are plausible
    configuration mistakes, and the second is the nastier one: two individually
    reasonable filters that jointly leave nothing."""
    with pytest.raises(ValueError, match="selects no operations"):
        build_tools(ActionLog(), operations=[])
    with pytest.raises(ValueError, match="privacy_mode"):
        build_tools(ActionLog(), operations=["peek"], privacy_mode=True)
    with pytest.raises(ValueError, match="selects no operations"):
        build_agent(registry, ActionLog(), model="test", operations=[])


def _schema_a_real_request_carries(registry, **kwargs) -> dict:
    """`run_plan`'s published schema, read off an actual `build_agent` run.

    Every other schema assertion in this file goes through `build_tools`
    directly, which leaves one link untested: whether `build_agent` hands its
    own knobs on. It does not error if it drops one — it just builds a
    differently-configured agent, and every `build_tools`-level test keeps
    passing. Captured from `AgentInfo.function_tools` (pydantic-ai's own view of
    what the request will carry) and then aborted inside the first request, so
    this stays offline and costs nothing. Same technique as
    eval/baseline.py's tool_schemas."""

    class _Captured(Exception):
        pass

    captured: dict = {}

    def capture(messages, info) -> ModelResponse:
        captured.update(
            {t.name: t.parameters_json_schema for t in info.function_tools}
        )
        raise _Captured

    agent = build_agent(registry, ActionLog(), model=FunctionModel(capture), **kwargs)
    try:
        agent.run_sync(".", deps=registry)
    except _Captured:
        pass
    assert list(captured) == ["run_plan"]
    return captured["run_plan"]


def test_build_agent_hands_the_allowlist_and_step_ceiling_to_the_wire(registry):
    """The knobs reach the schema a request actually carries, not just
    `build_tools`.

    Asserted here because the existing coverage of both parameters on
    `build_agent` is only that a *bad* value raises — and a `build_agent` that
    validated its arguments and then forgot to forward them would satisfy every
    one of those tests while shipping the full catalog and an unbounded plan to
    the model. That is the failure this pins: silent, invisible offline, and
    only ever observed as a bill."""
    published = _schema_a_real_request_carries(
        registry, operations=["peek", "describe"], max_steps_per_call=2
    )
    steps = published["properties"]["steps"]

    assert set(steps["items"]["discriminator"]["mapping"]) == {"peek", "describe"}
    assert steps["maxItems"] == 2
    # And the hygiene the schema_generator is responsible for survives the trip
    # through build_agent too -- it is the constructor everything real uses.
    assert "title" not in published


def test_the_allowlist_ignores_caller_ordering_and_duplicates():
    """Read as a set against the catalog's own order, so a caller's ordering
    can't reorder the published union (and a duplicate can't double a
    variant) — the schema a request carries should depend on which operations
    were asked for, not on how they were typed."""
    a, _ = _run_plan(operations=["sort_by", "peek", "peek"])
    b, _ = _run_plan(operations=["peek", "sort_by"])
    assert _published(a) == _published(b)


# --- model_settings ---------------------------------------------------------


def test_parallel_tool_calls_is_disabled_on_the_constructed_agent(registry):
    """One tool call per response is what this whole design rests on now that
    all multiplicity lives inside `steps`. Previously assumed; now asked for."""
    agent = build_agent(registry, ActionLog(), model="test")
    assert agent.model_settings["parallel_tool_calls"] is False


def test_anthropic_models_get_tool_definition_caching(registry, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    agent = build_agent(registry, ActionLog(), model="anthropic:claude-haiku-4-5")
    assert agent.model_settings["anthropic_cache_tool_definitions"] is True
    assert agent.model_settings["parallel_tool_calls"] is False


def test_non_anthropic_models_do_not_get_the_anthropic_cache_setting(registry, monkeypatch):
    """Applied unconditionally, this is a setting some providers reject at
    request time — i.e. a failure that only shows up live, against a provider
    the developer probably wasn't testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    agent = build_agent(registry, ActionLog(), model="openai:gpt-5.6")
    assert "anthropic_cache_tool_definitions" not in agent.model_settings
    assert agent.model_settings["parallel_tool_calls"] is False


def test_caller_model_settings_merge_over_dales_defaults_without_dropping_them(registry):
    """The trap this ordering exists to avoid: a caller passing one unrelated
    knob must not silently lose `parallel_tool_calls=False`."""
    agent = build_agent(registry, ActionLog(), model="test", model_settings={"temperature": 0})
    assert agent.model_settings["temperature"] == 0
    assert agent.model_settings["parallel_tool_calls"] is False


def test_a_caller_can_deliberately_override_parallel_tool_calls(registry):
    """The other side of the same decision. Overriding it reintroduces exactly
    the multiplicity this change removes — which is the caller's call to make,
    explicitly, and only explicitly."""
    agent = build_agent(
        registry, ActionLog(), model="test", model_settings={"parallel_tool_calls": True}
    )
    assert agent.model_settings["parallel_tool_calls"] is True


# --- trace equivalence against the frozen pre-change baseline ---------------


def _baseline_module():
    """`scripts/capture_trace_baseline.py`, loaded by path.

    Imported rather than reimplemented so the normalizers here are provably the
    same ones that produced the frozen files — a test that re-derives "scrub the
    timings" can drift from the capture and then compares two different
    normalizations of two different things. Only the normalizers and the fixture
    data are used; the script's own pipelines call the named per-operation tools
    and no longer run, which is the whole point."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "capture_trace_baseline.py"
    spec = importlib.util.spec_from_file_location("capture_trace_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_registry(module):
    registry = dale.DataRegistry(files=dale.FileRegistry())
    registry.create(
        "list",
        list(module._PRODUCTS),
        name="products_list",
        description="products",
        created_by="fixture",
    )
    return registry


def _read_baseline(name):
    from pathlib import Path

    data = Path(__file__).parent / "data"
    return (
        (data / f"trace_baseline_{name}_entries.json").read_text(),
        (data / f"trace_baseline_{name}_render.txt").read_text(),
    )


_FILTER_STEP = {
    "operation": "filter_where",
    "handle": "products_list",
    "predicate": {"field": "in_stock", "op": "==", "value": True},
    "intent": "keep in-stock",
    "name": "in_stock_list",
    "description": "in-stock products",
}


def test_a_length_one_batch_produces_the_pre_change_trace():
    """paper.md Section 3.12's "indistinguishable in the resulting audit trail"
    claim, asserted for the first time — against a snapshot frozen on the
    two-surface tree, since once the standalone surface is gone there is no
    live "before" left to diff against.

    A `steps` list of one has to reproduce, byte for byte, what a standalone
    call produced. Not a special case: the trivial case."""
    module = _baseline_module()
    registry = _baseline_registry(module)
    log = ActionLog()
    log.seed_from_registry(registry)
    tool, params_model = _run_plan(log)

    _batch(registry, tool, params_model, dict(_FILTER_STEP))

    expected_entries, expected_render = _read_baseline("one_step")
    assert module.normalize_entries(log) == expected_entries
    assert module.normalize_render(log) == expected_render


def test_a_multi_step_batch_produces_the_pre_change_trace():
    """The same, for a four-step pipeline chosen to reach every rendering
    branch and every record() side effect: a predicate, an aliased `as` field
    (by_alias=True in _call_params), a list-of-dicts param, and a
    non-handle-creating call with a defaulted param. Catches step numbering,
    alive_before/alive_after and _render_assignment regressions in one shot."""
    module = _baseline_module()
    registry = _baseline_registry(module)
    log = ActionLog()
    log.seed_from_registry(registry)
    tool, params_model = _run_plan(log)

    _batch(
        registry,
        tool,
        params_model,
        dict(_FILTER_STEP),
        {
            "operation": "compute_field",
            "handle": "in_stock_list",
            "as": "value",
            "op": "multiply",
            "left": {"field": "qty"},
            "right": {"field": "price"},
            "intent": "compute value",
            "name": "valued_list",
            "description": "with value",
        },
        {
            "operation": "sort_by",
            "handle": "valued_list",
            "keys": [{"field": "value", "order": "desc"}],
            "intent": "rank by value",
            "name": "ranked_list",
            "description": "ranked",
        },
        {"operation": "peek", "handle": "ranked_list", "intent": "check", "n": 2},
    )

    expected_entries, expected_render = _read_baseline("four_step")
    assert module.normalize_entries(log) == expected_entries
    assert module.normalize_render(log) == expected_render
