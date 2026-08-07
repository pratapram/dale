from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from dale.catalog import PrimitiveOutput, primitive
from dale.errors import TypeMismatchError
from dale.registry import DataRegistry


class SortKey(BaseModel):
    field: str
    order: Literal["asc", "desc"] = "asc"


class SortByParams(BaseModel):
    handle: str
    keys: list[SortKey]
    name: str
    description: str


def _sort_by_single_key(records: list[dict], field: str, descending: bool) -> list[dict]:
    # Records missing the field (or with a None value) always sort last,
    # regardless of ascending/descending — a simple, consistent NULLS-LAST rule.
    with_value = [r for r in records if r.get(field) is not None]
    without_value = [r for r in records if r.get(field) is None]
    try:
        with_value.sort(key=lambda r: r[field], reverse=descending)
    except TypeError as exc:
        raise TypeMismatchError(
            f"cannot sort by field {field!r}: incomparable values",
            details={"field": field},
        ) from exc
    return with_value + without_value


@primitive("sort_by", SortByParams, bounded_by_input=True, creates_handle=True)
def sort_by(registry: DataRegistry, params: SortByParams) -> PrimitiveOutput:
    """Stable multi-key sort of a list handle. Records missing a sort field
    are sorted last regardless of ascending/descending. Returns a new handle."""
    meta = registry.meta(params.handle)
    if meta.kind != "list":
        raise TypeMismatchError(
            f"sort_by requires a list handle, got {meta.kind!r}",
            details={"handle": params.handle, "kind": meta.kind},
        )

    result = list(registry.get(params.handle))
    # Stable multi-key sort: apply keys least-significant-first so Python's
    # stable sort preserves ties correctly for the most significant key.
    for key in reversed(params.keys):
        result = _sort_by_single_key(result, key.field, key.order == "desc")

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="sort_by",
        source_handles=[params.handle],
    )
    return PrimitiveOutput(status="ok", handle=new_meta)
