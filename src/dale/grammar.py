"""The declarative grammar the LLM's tool-call arguments are restricted to.

Closed and non-Turing-complete by construction: comparison predicates, boolean
combinators, computed fields, and priority orders — structured data, never code.
See DESIGN.md Section 3.2.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from dale.errors import DivisionByZeroError, TypeMismatchError

_ValueOp = Literal[
    "==", "!=", "<", "<=", ">", ">=",
    "in", "not_in",
    "starts_with", "ends_with", "contains",
]
_NullOp = Literal["is_null", "is_not_null"]
Op = Union[_ValueOp, _NullOp]
"""The full operator vocabulary across both comparison shapes below —
exported for reference/typing, not used directly as a single field's type
(see Comparison/NullComparison for why it's split in two)."""

_ORDERING_OPS = {"<", "<=", ">", ">="}
_STRING_OPS = {"starts_with", "ends_with", "contains"}

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, list[JsonScalar]]
"""The closed value vocabulary an LLM may put in a predicate's `value` or a
computed field's `const` — a JSON scalar, or a flat list of them for
`in`/`not_in`.

Was `Any` until a live OpenAI run rejected every DALE tool outright:
`Invalid schema for function 'compute_field': In context=('properties',
'const'), schema must have a 'type' key`. Pydantic renders `Any` as a bare
`{"title": "Const"}` — no `type`, no `anyOf` — which is legal JSON Schema
("any value") but which OpenAI's function-schema validator deliberately
refuses. That refusal is enforced at request time against the whole tool
list, so all three affected primitives (filter_where/window_flag via
Comparison, compute_field via ConstRef) took every other tool down with
them, before a single token of inference.

Typing it is the fix, not pinning to an older model that still accepts
untyped properties: the constraint is intentional on OpenAI's side, and
`Any` was never the real contract anyway — nothing in DALE ever put a
non-JSON value here. Every other provider (Anthropic, DeepSeek, Kimi, Qwen,
Grok) accepted the untyped version and accepts this one too; verified
offline against each one's own pydantic-ai JSON-schema transformer, none of
which raises or drops the field, and all of which keep `value` in
`required` — see NullComparison for why that requiredness is load-bearing.

Deliberately excludes dicts and lists-of-dicts: comparing a field directly
against a nested object is not something DALE supports today (flatten_json
is the documented path for nested structure), so the narrower vocabulary
costs nothing real and keeps the action space closed — the same argument
NullComparison itself was added on. A model that tries it now gets a
legible InvalidParamsError at dispatch rather than a silent dict compare."""


# --------------------------------------------------------------------------
# Predicate grammar: Comparison | NullComparison | And | Or | Not
# --------------------------------------------------------------------------


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: _ValueOp
    value: JsonValue


# NOTE — deliberately a `#` comment and not a docstring. Pydantic promotes a
# model's `__doc__` into its JSON-Schema `description`, and `NullComparison` is
# reachable from `Predicate`, which the tool schema publishes 3-4 times per
# request. As a docstring this history cost 1,101 tokens on every single request
# to describe a design that was *rejected* — the most expensive comment in the
# codebase, paid for by the model rather than the reader. Every word of it is
# kept, because it is the account of a real failure and of why the obvious
# simplification is wrong; it just doesn't belong on the wire.
#
# `field IS NULL` / `field IS NOT NULL` — deliberately its own type rather than
# a `Comparison` with an optional `value`, so the LLM-visible JSON Schema keeps
# `value` genuinely required for every other operator (present in `Comparison`'s
# own `required` list) instead of merely conventionally required, which a model
# can't see from the schema alone — a plain `value: Any = None` was tried first
# and rejected after a live TestModel run showed exactly this: making `value`
# schema-optional for *all* ops (to allow the null-check case) let a synthesized
# `==` call omit it too. A model literally cannot omit `value` on `Comparison`
# without failing schema validation now; `NullComparison`'s own schema has no
# `value` property at all. Added after a real, recurring live-model failure:
# `filter_where({"field": "x", "op": "=="})`, omitting `value` when the model
# meant "check for missing/null" — a value-less predicate for a value-less
# question is exactly what this type now offers.
class NullComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: _NullOp


class And(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    and_: list["Predicate"] = Field(alias="and")


class Or(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    or_: list["Predicate"] = Field(alias="or")


class Not(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    not_: "Predicate" = Field(alias="not")


Predicate = Union[Comparison, NullComparison, And, Or, Not]

And.model_rebuild()
Or.model_rebuild()
Not.model_rebuild()


def render_predicate(predicate: Predicate) -> str:
    """Human-readable infix rendering of a Predicate tree — e.g. for
    reviewing what an LLM actually constructed instead of reading the raw
    nested dict/JSON it produced. Not used by matches() or any primitive;
    a debugging/audit aid only."""
    if isinstance(predicate, Comparison):
        return f"{predicate.field} {predicate.op} {predicate.value!r}"
    if isinstance(predicate, NullComparison):
        return f"{predicate.field} IS NULL" if predicate.op == "is_null" else f"{predicate.field} IS NOT NULL"
    if isinstance(predicate, And):
        return "(" + " AND ".join(render_predicate(p) for p in predicate.and_) + ")"
    if isinstance(predicate, Or):
        return "(" + " OR ".join(render_predicate(p) for p in predicate.or_) + ")"
    if isinstance(predicate, Not):
        return f"NOT {render_predicate(predicate.not_)}"
    raise TypeMismatchError(f"unknown predicate type: {type(predicate)!r}")


def matches(record: dict[str, Any], predicate: Predicate) -> bool:
    """Evaluate a predicate against a record. The single evaluator reused by
    every predicate-consuming primitive (filter_where now; window_flag and
    graph_walk_resolve later)."""
    if isinstance(predicate, Comparison):
        return _match_comparison(record, predicate)
    if isinstance(predicate, NullComparison):
        actual = record.get(predicate.field)
        return actual is None if predicate.op == "is_null" else actual is not None
    if isinstance(predicate, And):
        return all(matches(record, p) for p in predicate.and_)
    if isinstance(predicate, Or):
        return any(matches(record, p) for p in predicate.or_)
    if isinstance(predicate, Not):
        return not matches(record, predicate.not_)
    raise TypeMismatchError(f"unknown predicate type: {type(predicate)!r}")


def _match_comparison(record: dict[str, Any], comp: Comparison) -> bool:
    actual = record.get(comp.field)
    op = comp.op
    value = comp.value

    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "in":
        return actual in value
    if op == "not_in":
        return actual not in value

    if op in _ORDERING_OPS:
        # Missing/None fields cannot be ordered — a structured, catchable error
        # rather than a Python TypeError leaking through.
        if actual is None:
            raise TypeMismatchError(
                f"cannot compare missing/None field {comp.field!r} with operator {op!r}",
                details={"field": comp.field, "op": op},
            )
        try:
            if op == "<":
                return actual < value
            if op == "<=":
                return actual <= value
            if op == ">":
                return actual > value
            return actual >= value
        except TypeError as exc:
            raise TypeMismatchError(
                f"incompatible types comparing field {comp.field!r} with operator {op!r}",
                details={"field": comp.field, "op": op},
            ) from exc

    if op in _STRING_OPS:
        # Bounded literal string ops — approved in place of regex, which was
        # explicitly rejected for ReDoS risk.
        if actual is None:
            return False
        actual_str = str(actual)
        if op == "starts_with":
            return actual_str.startswith(str(value))
        if op == "ends_with":
            return actual_str.endswith(str(value))
        return str(value) in actual_str

    raise TypeMismatchError(f"unsupported operator {op!r}")


# --------------------------------------------------------------------------
# ComputedField grammar
# --------------------------------------------------------------------------


class FieldRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str


class ConstRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    const: JsonValue


ValueRef = Union[FieldRef, ConstRef]


class ComputedField(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    as_: str = Field(alias="as")
    op: Literal["add", "subtract", "multiply", "divide"]
    left: ValueRef
    right: ValueRef


def resolve_value_ref(record: dict[str, Any], ref: ValueRef) -> Any:
    if isinstance(ref, FieldRef):
        return record.get(ref.field)
    return ref.const


def apply_computed_field(record: dict[str, Any], computed: ComputedField) -> Any:
    left = resolve_value_ref(record, computed.left)
    right = resolve_value_ref(record, computed.right)

    if left is None or right is None:
        raise TypeMismatchError(
            f"cannot compute {computed.as_!r}: operand is missing/None",
            details={"as": computed.as_, "op": computed.op},
        )

    try:
        if computed.op == "add":
            return left + right
        if computed.op == "subtract":
            return left - right
        if computed.op == "multiply":
            return left * right
        if computed.op == "divide":
            if right == 0:
                raise DivisionByZeroError(
                    f"division by zero computing {computed.as_!r}",
                    details={"as": computed.as_},
                )
            return left / right
    except TypeError as exc:
        raise TypeMismatchError(
            f"incompatible operand types computing {computed.as_!r}",
            details={"as": computed.as_, "op": computed.op},
        ) from exc

    raise TypeMismatchError(f"unsupported computed-field operator {computed.op!r}")


# --------------------------------------------------------------------------
# Priority grammar — reserved for graph_walk_resolve (deferred, no consumer yet)
# --------------------------------------------------------------------------

Priority = list[Any]


def resolve_priority(values: list[Any], priority: Priority) -> Any:
    """Return the highest-priority value (per `priority`, ordered high to low)
    present in `values`. Raises ValueError if none of `values` appear in `priority`."""
    for candidate in priority:
        if candidate in values:
            return candidate
    raise ValueError(f"none of {values!r} found in priority order {priority!r}")
