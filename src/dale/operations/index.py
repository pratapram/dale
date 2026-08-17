from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import DuplicateKeyError, FieldNotFoundError, TypeMismatchError
from dale.grammar import Priority, resolve_priority
from dale.keys import make_key
from dale.registry import DataRegistry


class IndexByParams(BaseModel):
    handle: str
    key_fields: list[str]
    name: str
    description: str


@operation("index_by", IndexByParams, bounded_by_input=True, creates_handle=True)
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


@operation("group_by", GroupByParams, bounded_by_input=True, creates_handle=True)
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


class PriorityReduceParams(BaseModel):
    handle: str
    key_fields: list[str]
    value_field: str
    priority: Priority
    name: str
    description: str


@operation("priority_reduce", PriorityReduceParams, bounded_by_input=True, creates_handle=True)
def priority_reduce(registry: DataRegistry, params: PriorityReduceParams) -> OperationOutput:
    """index_by, but for duplicate keys instead of erroring: groups records by
    key_fields, and for each group resolves value_field's values via a
    priority order (grammar.resolve_priority — the same function
    graph_walk_resolve already uses internally), keeping only the winning
    *value*, not the whole winning record. If the full winning record is
    needed too, join_lookup can pull it back in afterward — kept out of this
    operation's own scope on purpose."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"priority_reduce requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )

    source = registry.get(params.handle)
    groups: dict = {}
    for record in source:
        if params.value_field not in record:
            raise FieldNotFoundError(
                f"field {params.value_field!r} not present on a record in priority_reduce",
                details={"field": params.value_field, "handle": params.handle},
            )
        key = make_key(record, params.key_fields, handle=params.handle)
        groups.setdefault(key, []).append(record[params.value_field])

    resolved: dict = {}
    for key, values in groups.items():
        try:
            resolved[key] = resolve_priority(values, params.priority)
        except ValueError as exc:
            raise TypeMismatchError(
                f"none of {values!r} (key {key!r}) found in priority order {params.priority!r}",
                details={"key": key, "value_field": params.value_field},
            ) from exc

    new_meta = registry.create(
        "dict",
        resolved,
        name=params.name,
        description=params.description,
        created_by="priority_reduce",
        source_handles=[params.handle],
        value_shape="one",
        key_arity=len(params.key_fields),
    )
    return OperationOutput(status="ok", handle=new_meta)
