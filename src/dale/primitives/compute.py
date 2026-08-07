from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dale.catalog import PrimitiveOutput, primitive
from dale.errors import TypeMismatchError
from dale.grammar import ComputedField, ValueRef, apply_computed_field
from dale.registry import DataRegistry


class ComputeFieldParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    handle: str
    as_: str = Field(alias="as")
    op: Literal["add", "subtract", "multiply", "divide"]
    left: ValueRef
    right: ValueRef
    name: str
    description: str


@primitive("compute_field", ComputeFieldParams, bounded_by_input=True, creates_handle=True)
def compute_field(registry: DataRegistry, params: ComputeFieldParams) -> PrimitiveOutput:
    """Add a derived field to every record in a list handle, computed from
    two operands (each a field reference or a constant) and an arithmetic
    operator. Returns a new list handle with the extra field added."""
    meta = registry.meta(params.handle)
    if meta.kind != "list":
        raise TypeMismatchError(
            f"compute_field requires a list handle, got {meta.kind!r}",
            details={"handle": params.handle, "kind": meta.kind},
        )

    source = registry.get(params.handle)
    computed_spec = ComputedField(
        **{"as": params.as_, "op": params.op, "left": params.left, "right": params.right}
    )

    result = []
    for record in source:
        new_record = dict(record)
        new_record[params.as_] = apply_computed_field(record, computed_spec)
        result.append(new_record)

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="compute_field",
        source_handles=[params.handle],
    )
    return PrimitiveOutput(status="ok", handle=new_meta)
