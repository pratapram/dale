"""window_flag: sliding-window occurrence counting (DESIGN.md's "Sliding
Window / Two-Pointer" pattern, Use Case 2 — log-stream sessionization).

Implemented via bisect over a per-group sorted list of qualifying timestamps
rather than literal two pointers — same O(N log N) complexity (dominated by
the sort), simpler to get right. The window is always evaluated per-group and
trailing: for record r, it counts qualifying records in the same group whose
window_field falls in (r.window_field - window_size, r.window_field] —
i.e. strictly after (window_field - window_size) and up to and including r's
own timestamp. Every record gets a count and flag, including records that
don't themselves match `predicate` — this is deliberate: it's what lets a
single successful login immediately following a burst of failed attempts
still get flagged, without a separate correlation step.

Sorts internally rather than requiring pre-sorted input (unlike DESIGN.md's
illustrative pipeline description, which lists sort_by as a separate prior
step) — consistent with every other primitive in this catalog not imposing
ordering preconditions on its caller.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dale.catalog import PrimitiveOutput, primitive
from dale.errors import TypeMismatchError
from dale.grammar import Predicate, matches
from dale.keys import make_key
from dale.registry import DataRegistry


class WindowFlagParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    handle: str
    group_by: list[str]
    window_field: str
    window_size: float
    threshold: int
    predicate: Predicate | None = None
    as_: str = Field(default="flagged", alias="as")
    name: str
    description: str


def _window_value(raw: Any, field: str) -> Any:
    """Coerce a window_field value to something orderable/subtractable.
    Numeric values are used as-is; ISO 8601 strings are parsed to datetime
    (window_size is then interpreted as seconds)."""
    if isinstance(raw, bool):
        raise TypeMismatchError(
            f"window_field {field!r} must be numeric or an ISO 8601 string, got bool",
            details={"field": field},
        )
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise TypeMismatchError(
                f"window_field {field!r} value {raw!r} is not a valid ISO 8601 timestamp",
                details={"field": field, "value": raw},
            ) from exc
    raise TypeMismatchError(
        f"window_field {field!r} must be numeric or an ISO 8601 string, got {type(raw).__name__}",
        details={"field": field},
    )


def _window_delta(sample_value: Any, window_size: float) -> Any:
    return timedelta(seconds=window_size) if isinstance(sample_value, datetime) else window_size


@primitive("window_flag", WindowFlagParams, bounded_by_input=True, creates_handle=True)
def window_flag(registry: DataRegistry, params: WindowFlagParams) -> PrimitiveOutput:
    """Flag records with >= threshold qualifying occurrences (matching an
    optional predicate) within a trailing window over an orderable field,
    grouped by one or more key fields. Returns a new list handle with two
    fields added per record: `as` (bool) and `{as}_count` (int). Output order
    is grouped, not guaranteed to match input order — sort_by afterward if a
    specific order matters."""
    meta = registry.meta(params.handle)
    if meta.kind != "list":
        raise TypeMismatchError(
            f"window_flag requires a list handle, got {meta.kind!r}",
            details={"handle": params.handle, "kind": meta.kind},
        )

    source = registry.get(params.handle)
    count_field = f"{params.as_}_count"

    groups: dict[Any, list[dict]] = {}
    for record in source:
        key = make_key(record, params.group_by, handle=params.handle)
        groups.setdefault(key, []).append(record)

    result: list[dict] = []
    for records in groups.values():
        window_values = [
            _window_value(r.get(params.window_field), params.window_field) for r in records
        ]
        try:
            order = sorted(range(len(records)), key=lambda i: window_values[i])
        except TypeError as exc:
            raise TypeMismatchError(
                f"cannot order window_field {params.window_field!r}: incomparable values "
                "(mixed numeric/timestamp types within one group)",
                details={"field": params.window_field},
            ) from exc
        records = [records[i] for i in order]
        window_values = [window_values[i] for i in order]

        qualifies = [
            params.predicate is None or matches(r, params.predicate) for r in records
        ]
        eligible = sorted(v for v, q in zip(window_values, qualifies) if q)

        delta = _window_delta(window_values[0], params.window_size) if window_values else 0.0

        for record, value in zip(records, window_values):
            upper = bisect.bisect_right(eligible, value)
            lower = bisect.bisect_right(eligible, value - delta)
            count = upper - lower

            new_record = dict(record)
            new_record[params.as_] = count >= params.threshold
            new_record[count_field] = count
            result.append(new_record)

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="window_flag",
        source_handles=[params.handle],
    )
    return PrimitiveOutput(status="ok", handle=new_meta)
