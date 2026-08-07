"""The primitive catalog and the build-time extensibility mechanism.

Built-in primitives register via the same `@primitive(...)` decorator /
`register_primitive()` call a third-party developer would use later
(objections #11) — there is no separate "built-in" code path. This is what
makes concrete the principle that the LLM's action space grows only in size,
never in kind: the catalog can grow, but every entry is developer-authored,
reviewed Python registered ahead of time, never chosen or imported by the LLM
at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from dale.cost import CostEstimate
from dale.errors import PrimitiveNotFoundError
from dale.registry import DataRegistry, HandleMeta


class ConfirmableParams(BaseModel):
    """Base class for params of any primitive with a cost_estimator — provides
    the `confirm` field dispatch checks before honoring cost_gate_exceeded."""

    confirm: bool = False


class PrimitiveOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal["ok", "cost_gate_exceeded"]
    handle: HandleMeta | None = None
    result: Any = None
    estimate: CostEstimate | None = None


PrimitiveFn = Callable[[DataRegistry, BaseModel], PrimitiveOutput]
CostEstimatorFn = Callable[[DataRegistry, BaseModel], CostEstimate]


class PrimitiveSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    fn: PrimitiveFn
    param_schema: type[BaseModel]
    cost_estimator: CostEstimatorFn | None = None
    bounded_by_input: bool = False
    """True for primitives whose output is provably <= input size
    (filter_where, compute_field, sort_by) — documented as safe by
    construction rather than silently missing an estimator (objections #4)."""
    creates_handle: bool = False
    """True for primitives whose PrimitiveOutput.handle is a newly created
    handle (as opposed to peek/describe, which return via `result`, or
    release_handle, which destroys one). Lets dale.agent decide which
    primitives get the LLM-supplied name/description fields injected,
    without hardcoding a primitive-name list at the agent layer — the
    catalog is the single source of truth for what a primitive does."""


_CATALOG: dict[str, PrimitiveSpec] = {}


def register_primitive(
    name: str,
    fn: PrimitiveFn,
    param_schema: type[BaseModel],
    *,
    cost_estimator: CostEstimatorFn | None = None,
    bounded_by_input: bool = False,
    creates_handle: bool = False,
) -> None:
    if name in _CATALOG:
        raise ValueError(f"primitive already registered: {name!r}")
    _CATALOG[name] = PrimitiveSpec(
        name=name,
        fn=fn,
        param_schema=param_schema,
        cost_estimator=cost_estimator,
        bounded_by_input=bounded_by_input,
        creates_handle=creates_handle,
    )


def primitive(
    name: str,
    param_schema: type[BaseModel],
    *,
    cost_estimator: CostEstimatorFn | None = None,
    bounded_by_input: bool = False,
    creates_handle: bool = False,
) -> Callable[[PrimitiveFn], PrimitiveFn]:
    """Decorator sugar over register_primitive — identical usage for built-ins
    and for a developer's own `register_primitive` calls (objections #11).

    Writing one? The obligation that is easy to miss: validate every
    assumption you make about a handle you did not create — element type,
    field presence, value shape — and raise a typed `DaleError` naming what
    was violated. Anything you dereference without checking becomes a bare
    Python exception, which `dispatch` sanitizes into a content-free
    `INTERNAL_ERROR` the model cannot act on. See principles.md, "Any
    `INTERNAL_ERROR` Is a Missing Precondition", and reuse the shared checks
    in `dale/keys.py` rather than hand-rolling them."""

    def deco(fn: PrimitiveFn) -> PrimitiveFn:
        register_primitive(
            name,
            fn,
            param_schema,
            cost_estimator=cost_estimator,
            bounded_by_input=bounded_by_input,
            creates_handle=creates_handle,
        )
        return fn

    return deco


def get_primitive(name: str) -> PrimitiveSpec:
    try:
        return _CATALOG[name]
    except KeyError:
        raise PrimitiveNotFoundError(
            f"no such primitive: {name!r}", details={"name": name}
        ) from None


def list_primitives() -> list[str]:
    return sorted(_CATALOG)


def _reset_catalog_for_tests() -> None:
    """Test-only helper. Not part of the public API."""
    _CATALOG.clear()
