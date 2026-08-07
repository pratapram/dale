from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import PrimitiveOutput, primitive
from dale.errors import TypeMismatchError
from dale.grammar import Predicate, matches
from dale.registry import DataRegistry


class FilterWhereParams(BaseModel):
    handle: str
    predicate: Predicate
    name: str
    description: str


@primitive("filter_where", FilterWhereParams, bounded_by_input=True, creates_handle=True)
def filter_where(registry: DataRegistry, params: FilterWhereParams) -> PrimitiveOutput:
    """Keep only the records in a list handle matching a predicate (field
    comparisons combined with and/or/not). Returns a new list handle."""
    meta = registry.meta(params.handle)
    if meta.kind != "list":
        raise TypeMismatchError(
            f"filter_where requires a list handle, got {meta.kind!r}",
            details={"handle": params.handle, "kind": meta.kind},
        )

    source = registry.get(params.handle)
    filtered = [record for record in source if matches(record, params.predicate)]

    new_meta = registry.create(
        "list",
        filtered,
        name=params.name,
        description=params.description,
        created_by="filter_where",
        source_handles=[params.handle],
    )
    return PrimitiveOutput(status="ok", handle=new_meta)
