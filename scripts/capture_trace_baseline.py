"""Capture the pre-change ActionLog trace baseline.

Run ONCE, on the two-tool-surface tree, *before* `run_plan` becomes the only tool.

`paper.md` Section 3.12 claims a batched call is indistinguishable in the resulting
trace from the same call issued on its own turn, and `agent.py`'s own docstrings
(`_execute_and_log_step`, `_call_params`) restate it as the reason both surfaces
share one choke point. Nothing ever asserted it. Once the standalone per-primitive
tools are gone there is no second surface left to compare against, so the claim
becomes unassertable unless the "before" side is frozen first — which is what this
does.

The two artifacts serve different purposes and both are kept:
  - trace_baseline_entries.json — every field of every ActionLog entry, result
    payloads included. The strict guard: catches a changed status, a dropped
    auto_inspect splice, a params key that stopped being serialized.
  - trace_baseline_render.txt — ActionLog.render() with timings scrubbed. The
    legible guard: when the strict one fails, this is the diff a human reads.

Timing fields are excluded rather than frozen: they are real wall-clock
measurements and vary per run. Everything else was probed to be byte-deterministic
across repeated runs.

Regenerating this baseline deliberately destroys the evidence it exists to
preserve, so don't — unless the trace format is being changed on purpose, in which
case the diff is the review artifact.

That is now enforced by construction as well as by this paragraph: the pipelines
below drive the named per-primitive tools, which `build_tools` no longer returns,
so `main()` raises rather than silently overwriting the frozen files. What still
works, and is what the regression test imports, is `_PRODUCTS` and the two
normalizers — deliberately shared rather than reimplemented in the test, so the
"before" and "after" sides are provably normalized the same way.
"""

import inspect
import json
import re

import dale
from dale.agent import ActionLog, build_tools

# Four steps, chosen to reach every rendering branch and every record() side
# effect: a predicate (exercises _render_params' predicate rendering), an aliased
# `as` field (exercises by_alias=True in _call_params), a list-of-dicts param, and
# one non-handle-creating call with a defaulted param (exercises _render_call's
# receiver detection and the "no assignment prefix" branch of _render_assignment).
_PRODUCTS = [
    {"sku": "A", "qty": 3, "price": 10.0, "in_stock": True},
    {"sku": "B", "qty": 1, "price": 50.0, "in_stock": True},
    {"sku": "C", "qty": 9, "price": 2.0, "in_stock": False},
]


def _fixture_registry() -> dale.DataRegistry:
    registry = dale.DataRegistry(files=dale.FileRegistry())
    registry.create(
        "list",
        list(_PRODUCTS),
        name="products_list",
        description="products",
        created_by="fixture",
    )
    return registry


def _caller(tools):
    """Invoke a named tool the way pydantic-ai would, without a live model.

    The param model is read back off the tool function's own annotation rather
    than rebuilt here, so this calls exactly the schema build_tools published.
    """
    by_name = {t.name: t for t in tools}

    class _Ctx:
        def __init__(self, registry):
            self.deps = registry

    # Positional-only: handle-creating primitives take their own `name` kwarg,
    # which would otherwise collide with the tool name.
    def call(registry, primitive, /, **kwargs):
        tool = by_name[primitive]
        params_model = inspect.signature(tool.function).parameters["params"].annotation
        return tool.function(_Ctx(registry), params_model(**kwargs))

    return call


def four_step_pipeline() -> ActionLog:
    registry = _fixture_registry()
    log = ActionLog()
    log.seed_from_registry(registry)
    call = _caller(build_tools(log))

    call(
        registry,
        "filter_where",
        handle="products_list",
        predicate={"field": "in_stock", "op": "==", "value": True},
        intent="keep in-stock",
        name="in_stock_list",
        description="in-stock products",
    )
    call(
        registry,
        "compute_field",
        handle="in_stock_list",
        op="multiply",
        left={"field": "qty"},
        right={"field": "price"},
        intent="compute value",
        name="valued_list",
        description="with value",
        **{"as": "value"},
    )
    call(
        registry,
        "sort_by",
        handle="valued_list",
        keys=[{"field": "value", "order": "desc"}],
        intent="rank by value",
        name="ranked_list",
        description="ranked",
    )
    call(registry, "peek", handle="ranked_list", intent="check", n=2)
    return log


def one_step_pipeline() -> ActionLog:
    """The length-1 case, frozen separately.

    After the change a single primitive call is a `steps` list of one, and this is
    what it has to reproduce — the claim being that batching of size 1 is not a
    special case but the trivial case.
    """
    registry = _fixture_registry()
    log = ActionLog()
    log.seed_from_registry(registry)
    call = _caller(build_tools(log))
    call(
        registry,
        "filter_where",
        handle="products_list",
        predicate={"field": "in_stock", "op": "==", "value": True},
        intent="keep in-stock",
        name="in_stock_list",
        description="in-stock products",
    )
    return log


_TIMING = re.compile(r"\(<?[\d.]+ ms(?: \+ [\d.]+ ms inspect)?\)")


def normalize_entries(log: ActionLog) -> str:
    return json.dumps(
        [e.model_dump(exclude={"elapsed_ms", "auto_inspect_ms"}) for e in log.entries],
        indent=2,
        default=str,
        sort_keys=True,
    )


def normalize_render(log: ActionLog) -> str:
    return _TIMING.sub("(TIMING)", log.render())


def main() -> None:
    for name, log in (("four_step", four_step_pipeline()), ("one_step", one_step_pipeline())):
        entries = f"tests/data/trace_baseline_{name}_entries.json"
        render = f"tests/data/trace_baseline_{name}_render.txt"
        with open(entries, "w") as fh:
            fh.write(normalize_entries(log))
        with open(render, "w") as fh:
            fh.write(normalize_render(log))
        print(f"{name}: {len(log.entries)} entries -> {entries}, {render}")


if __name__ == "__main__":
    main()
