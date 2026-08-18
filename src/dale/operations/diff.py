"""dict_diff: compare two dict handles keyed the same way and report which
keys are new, removed, changed, or unchanged (
license tier reconciliation against the previous run's assignment)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import TypeMismatchError
from dale.registry import DataRegistry


class DictDiffParams(BaseModel):
    current_handle: str
    previous_handle: str
    name: str
    description: str


def _status(key: Any, current: dict, previous: dict) -> str:
    in_current = key in current
    in_previous = key in previous
    if in_current and not in_previous:
        return "new"
    if in_previous and not in_current:
        return "removed"
    return "changed" if current[key] != previous[key] else "unchanged"


@operation("dict_diff", DictDiffParams, bounded_by_input=True, creates_handle=True)
def dict_diff(registry: DataRegistry, params: DictDiffParams) -> OperationOutput:
    """Compare two dict handles (any two — reduce_by's output, a plain
    load_json dict, anything keyed the same way) and return a new list
    handle, one row per key across the union of both, each tagged "new"/
    "removed"/"changed"/"unchanged". All four statuses are always included,
    not just deltas — a caller that only wants changes can filter_where
    afterward; a caller that also wants "unchanged" can't recover it if it
    was never returned."""
    current_meta = registry.meta(params.current_handle)
    previous_meta = registry.meta(params.previous_handle)
    if current_meta.type != "dict":
        raise TypeMismatchError(
            f"dict_diff current_handle must be a dict, got {current_meta.type!r}",
            details={"handle": params.current_handle, "type": current_meta.type},
        )
    if previous_meta.type != "dict":
        raise TypeMismatchError(
            f"dict_diff previous_handle must be a dict, got {previous_meta.type!r}",
            details={"handle": params.previous_handle, "type": previous_meta.type},
        )

    current = registry.get(params.current_handle)
    previous = registry.get(params.previous_handle)

    result = []
    for key in {**current, **previous}:
        result.append(
            {
                "key": key,
                "status": _status(key, current, previous),
                "previous_value": previous.get(key),
                "current_value": current.get(key),
            }
        )

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="dict_diff",
        source_handles=[params.current_handle, params.previous_handle],
    )
    return OperationOutput(status="ok", handle=new_meta)
