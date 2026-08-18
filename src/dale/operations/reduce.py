"""Deduplicate a list to one record per key, under a stated ordering.

Replaces the earlier `priority_reduce`, which fused this mechanism with a
single hardcoded policy (rank by position in an explicit value list). The
mechanism -- keep one record per group under an ordering -- is the general
one: SQL's `ROW_NUMBER() OVER (PARTITION BY key ORDER BY ...) = 1`, argmax
per group. The explicit ranking is just one way to define that ordering,
alongside the two more common ones (`max` by a field, latest by timestamp)
that `priority_reduce` could not express at all.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from dale.catalog import OperationOutput, operation
from dale.errors import FieldNotFoundError, TypeMismatchError
from dale.keys import make_key
from dale.registry import DataRegistry


class OrderKey(BaseModel):
    """One ordering key. Deliberately not `sort.SortKey`, even though the
    first two fields match: `ranking` is meaningless for a whole-list sort,
    and `run_plan`'s discriminated-union schema rides on every request, so a
    field added to a shared type is paid for by every operation that uses it.

    `ranking` lists values from highest priority to lowest. A value not in
    the list -- and a missing or None field -- ranks below every listed one,
    the same NULLS-LAST rule `sort_by` already applies. That is what makes an
    unranked value resolvable rather than an error: the group still has a
    winner, and a group whose only value is unranked wins by default."""

    field: str
    order: Literal["asc", "desc"] = "asc"
    ranking: list[Any] | None = Field(
        None,
        min_length=1,
        description="Values from highest priority to lowest, e.g. "
        '["gold", "silver", "bronze"]. Anything not listed ranks last. '
        "Omit to order by the field's own values.",
    )


class ReduceByParams(BaseModel):
    handle: str
    key_fields: list[str] = Field(..., min_length=1)
    order_by: list[OrderKey] = Field(
        ...,
        min_length=1,
        description="Ordering keys, most significant first. The first record "
        "per key under this ordering wins.",
    )
    value_field: str | None = Field(
        None,
        description="Keep only this field's value from the winning record "
        "instead of the whole record. Use when the result is compared against "
        "another single-valued dict (dict_diff) rather than read field-wise.",
    )
    name: str
    description: str


_UNRANKED = (1, 0)
"""Sort tuple for a value that has no rank: (1, _) always sorts after any
(0, rank). Comparing ints only, so a str value and an int value in the same
column can never raise the TypeError a bare comparison would."""


def _rank(value: Any, ranking: list[Any]) -> tuple[int, int]:
    try:
        return (0, ranking.index(value))
    except ValueError:
        return _UNRANKED


def _order_once(records: list[dict], key: OrderKey) -> list[dict]:
    """One stable pass. Split rather than sort-with-sentinel so that records
    missing the field never reach the comparison at all -- the same reason
    sort_by._sort_by_single_key splits."""
    descending = key.order == "desc"

    if key.ranking is not None:
        return sorted(
            records, key=lambda r: _rank(r.get(key.field), key.ranking), reverse=descending
        )

    with_value = [r for r in records if r.get(key.field) is not None]
    without_value = [r for r in records if r.get(key.field) is None]
    try:
        with_value.sort(key=lambda r: r[key.field], reverse=descending)
    except TypeError as exc:
        raise TypeMismatchError(
            f"cannot order by field {key.field!r}: incomparable values",
            details={"field": key.field},
        ) from exc
    return with_value + without_value


@operation(
    "reduce_by",
    ReduceByParams,
    io_signature="list → dict",
    summary=(
        '`index_by`, but for duplicate keys instead of erroring: keeps one '
        'record per key — the first under `order_by`. Order by a field '
        '(`{field, order}`: highest score, latest timestamp) or by an '
        'explicit `ranking` of values (`["gold","silver","bronze"]`); '
        'unranked and missing values sort last. Returns whole records, or '
        'just one field\'s value with `value_field`. The "group + argmax" '
        "pattern (SQL's `ROW_NUMBER() … = 1`)"
    ),
    bounded_by_input=True,
    creates_handle=True,
)
def reduce_by(registry: DataRegistry, params: ReduceByParams) -> OperationOutput:
    """Group by key_fields and keep exactly one record per key -- the first
    under `order_by`. Where `index_by` rejects a duplicate key and `group_by`
    keeps every record, this resolves the duplicates.

    Returns key -> winning record, or key -> that record's `value_field` when
    one is named."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"reduce_by requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )

    source = registry.get(params.handle)
    if params.value_field is not None:
        for record in source:
            if params.value_field not in record:
                raise FieldNotFoundError(
                    f"field {params.value_field!r} not present on a record in reduce_by",
                    details={"field": params.value_field, "handle": params.handle},
                )

    # Order the whole list once, then take the first occurrence of each key --
    # O(N log N) total rather than a sort per group. Least-significant key
    # first, so Python's stable sort leaves the most significant one on top.
    ordered = list(source)
    for key in reversed(params.order_by):
        ordered = _order_once(ordered, key)

    resolved: dict = {}
    for record in ordered:
        key = make_key(record, params.key_fields, handle=params.handle)
        if key in resolved:
            continue
        resolved[key] = (
            record if params.value_field is None else record[params.value_field]
        )

    new_meta = registry.create(
        "dict",
        resolved,
        name=params.name,
        description=params.description,
        created_by="reduce_by",
        source_handles=[params.handle],
        value_shape="one",
        key_arity=len(params.key_fields),
    )
    return OperationOutput(status="ok", handle=new_meta)
