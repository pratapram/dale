"""Composite-key construction shared by index_by/group_by and join_lookup
(objections #14 — multi-field keys produce a tuple, not a fourth primary
type), plus the shape checks a primitive owes any handle it did not create
itself (see principles.md, "Any INTERNAL_ERROR Is a Missing Precondition")."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dale.errors import FieldNotFoundError, TypeMismatchError

if TYPE_CHECKING:  # import cycle: registry imports errors, not keys
    from dale.registry import DataRegistry


def require_dict_handle(
    registry: DataRegistry, handle: str, *, param: str, primitive: str
) -> Any:
    """Reject a non-dict handle where a keyed lookup is required, naming the
    parameter the caller actually passed rather than a generic 'wrong type'."""
    meta = registry.meta(handle)
    if meta.kind != "dict":
        raise TypeMismatchError(
            f"{primitive} {param} must be a dict handle (built via index_by/group_by/"
            f"priority_reduce), got {meta.kind!r}",
            details={"handle": handle, "param": param, "kind": meta.kind},
        )
    return meta


def require_record_valued(
    registry: DataRegistry, handle: str, *, param: str, primitive: str
) -> Any:
    """Reject a dict handle whose values are bare scalars where the caller
    needs to read fields off them.

    The check `require_dict_handle` alone does not cover: `index_by` and
    `priority_reduce` both produce dicts with value_shape="scalar", but the
    former's values are records and the latter's are single field values.
    Reading a field off the latter used to raise a bare Python TypeError
    inside join_lookup, which dispatch sanitized into an unactionable
    INTERNAL_ERROR. Call this whenever you intend to subscript a dict
    handle's values."""
    meta = require_dict_handle(registry, handle, param=param, primitive=primitive)
    if meta.value_type == "scalar":
        raise TypeMismatchError(
            f"{primitive} {param} {handle!r} holds single values, not records — "
            "its values came from something like priority_reduce, so there are no "
            "fields to read off them. Use an index built by index_by or group_by, "
            "or name exactly one target field to bind the value to.",
            details={"handle": handle, "param": param, "value_type": meta.value_type},
        )
    return meta


def make_key(record: dict[str, Any], key_fields: list[str], *, handle: str) -> Any:
    values = []
    for field in key_fields:
        if field not in record:
            raise FieldNotFoundError(
                f"field {field!r} not present on a record in handle {handle!r}",
                details={"field": field, "handle": handle},
            )
        values.append(record[field])

    key = values[0] if len(key_fields) == 1 else tuple(values)
    try:
        hash(key)
    except TypeError as exc:
        raise TypeMismatchError(
            f"key fields {key_fields!r} produced an unhashable key on handle {handle!r}",
            details={"key_fields": key_fields, "handle": handle},
        ) from exc
    return key
