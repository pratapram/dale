"""flatten_json: explode a nested array field into its own rows, carrying
selected parent fields down onto each one — nested/wrapped enterprise API
response normalization (DESIGN.md Section 3, Use Case 7).

Deliberately scoped small per a real design discussion, grounded against
live-fetched real JSON (GitHub/PokeAPI/npm/GeoJSON samples):
`path` supports exactly one level today, not arbitrary nesting depth. The
multi-level case (e.g. Salesforce's Contacts.records inside records) needs a
`peek()` path extension to discover safely first — building that blind, before
any real multi-level use case forced the question, would be guessing at a
design nothing has validated yet.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import FieldCollisionError, InvalidParamsError, TypeMismatchError
from dale.registry import DataRegistry


class FlattenJsonParams(BaseModel):
    handle: str
    path: list[str]
    carry_fields: list[str] = []
    name: str
    description: str


def _explode_record(
    record: dict[str, Any], field: str, carry_fields: list[str]
) -> list[dict[str, Any]]:
    if field not in record or record[field] is None:
        return []
    value = record[field]
    if not isinstance(value, list):
        raise TypeMismatchError(
            f"flatten_json: field {field!r} is not a list (got {type(value).__name__}) — "
            "path must point at an array-valued field",
            details={"field": field},
        )
    if not value:
        return []
    if not all(isinstance(item, dict) for item in value):
        raise TypeMismatchError(
            f"flatten_json: field {field!r} is an array of non-object values — "
            "only arrays of objects can be exploded into rows",
            details={"field": field},
        )

    carried = {k: record[k] for k in carry_fields if k in record}
    rows = []
    for item in value:
        collisions = carried.keys() & item.keys()
        if collisions:
            raise FieldCollisionError(
                f"flatten_json: carry_fields {sorted(collisions)!r} collide with field(s) "
                f"already present on the exploded {field!r} object",
                details={"field": field, "collisions": sorted(collisions)},
            )
        rows.append({**carried, **item})
    return rows


@operation("flatten_json", FlattenJsonParams, creates_handle=True)
def flatten_json(registry: DataRegistry, params: FlattenJsonParams) -> OperationOutput:
    """Explode a nested array field into one row per element, carrying
    selected parent fields down onto each row. A record whose path field is
    absent, null, or an empty array contributes zero rows — not an error;
    this is what makes "skip records with nothing to explode" free. Only a
    single-element path is supported today. Returns a new list handle."""
    meta = registry.meta(params.handle)
    if meta.type != "list":
        raise TypeMismatchError(
            f"flatten_json requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )
    if len(params.path) != 1:
        raise InvalidParamsError(
            f"flatten_json: path must have exactly one element (multi-level paths not yet "
            f"supported), got {params.path!r}",
            details={"path": params.path},
        )
    field = params.path[0]

    source = registry.get(params.handle)
    result: list[dict[str, Any]] = []
    for record in source:
        result.extend(_explode_record(record, field, params.carry_fields))

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="flatten_json",
        source_handles=[params.handle],
    )
    return OperationOutput(status="ok", handle=new_meta)
