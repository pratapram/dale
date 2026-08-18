from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import TypeMismatchError
from dale.grammar import Predicate, matches
from dale.registry import DataRegistry


class FilterWhereParams(BaseModel):
    handle: str
    predicate: Predicate
    name: str
    description: str


@operation(
    "filter_where",
    FilterWhereParams,
    io_signature="list → list",
    summary="Keep records matching a predicate (comparisons, `and`/`or`/`not`)",
    bounded_by_input=True,
    creates_handle=True,
)
def filter_where(registry: DataRegistry, params: FilterWhereParams) -> OperationOutput:
    """Keep only the records in a list handle matching a predicate (field
    comparisons combined with and/or/not). Returns a new list handle."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"filter_where requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
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
    return OperationOutput(status="ok", handle=new_meta)
