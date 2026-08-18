from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import DuplicateKeyError, TypeMismatchError
from dale.keys import make_key
from dale.registry import DataRegistry


class IndexByParams(BaseModel):
    handle: str
    key_fields: list[str]
    name: str
    description: str


@operation(
    "index_by",
    IndexByParams,
    io_signature="list → dict",
    summary=(
        'Build a unique-keyed `dict` (composite key → single record); errors '
        'on duplicates'
    ),
    bounded_by_input=True,
    creates_handle=True,
)
def index_by(registry: DataRegistry, params: IndexByParams) -> OperationOutput:
    """Build a unique-keyed dict (composite key -> single record). Duplicate
    keys are a data-integrity error, not silently overwritten."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"index_by requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )

    source = registry.get(params.handle)
    indexed: dict = {}
    for record in source:
        key = make_key(record, params.key_fields, handle=params.handle)
        if key in indexed:
            raise DuplicateKeyError(
                f"duplicate key {key!r} for key_fields={params.key_fields!r}",
                details={"key_fields": params.key_fields, "handle": params.handle},
            )
        indexed[key] = record

    new_meta = registry.create(
        "dict",
        indexed,
        name=params.name,
        description=params.description,
        created_by="index_by",
        source_handles=[params.handle],
        value_shape="one",
        key_arity=len(params.key_fields),
    )
    return OperationOutput(status="ok", handle=new_meta)


class GroupByParams(BaseModel):
    handle: str
    key_fields: list[str]
    name: str
    description: str


@operation(
    "group_by",
    GroupByParams,
    io_signature="list → dict",
    summary="Build a bucketed `dict` (composite key → list of records)",
    bounded_by_input=True,
    creates_handle=True,
)
def group_by(registry: DataRegistry, params: GroupByParams) -> OperationOutput:
    """Build a bucketed dict (composite key -> list of matching records)."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"group_by requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )

    source = registry.get(params.handle)
    grouped: dict = {}
    for record in source:
        key = make_key(record, params.key_fields, handle=params.handle)
        grouped.setdefault(key, []).append(record)

    new_meta = registry.create(
        "dict",
        grouped,
        name=params.name,
        description=params.description,
        created_by="group_by",
        source_handles=[params.handle],
        value_shape="many",
        key_arity=len(params.key_fields),
    )
    return OperationOutput(status="ok", handle=new_meta)
