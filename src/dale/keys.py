"""Composite-key construction shared by index_by/group_by and join_lookup, plus the shape checks an operation owes any handle it did not create
itself (an `INTERNAL_ERROR` always means a missing precondition)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dale.errors import FieldNotFoundError, TypeMismatchError

if TYPE_CHECKING:  # import cycle: registry imports errors, not keys
    from dale.registry import DataRegistry


def require_dict_handle(
    registry: DataRegistry, handle: str, *, param: str, operation: str
) -> Any:
    """Reject a non-dict handle where a keyed lookup is required, naming the
    parameter the caller actually passed rather than a generic 'wrong type'."""
    meta = registry.meta(handle)
    if meta.type != "dict":
        raise TypeMismatchError(
            f"{operation} {param} must be a dict handle (built via index_by/group_by/"
            f"reduce_by), got {meta.type!r}",
            details={"handle": handle, "param": param, "type": meta.type},
        )
    return meta


def require_record_valued(
    registry: DataRegistry, handle: str, *, param: str, operation: str
) -> Any:
    """Reject a dict handle whose values are bare single values where the caller
    needs to read fields off them.

    The check `require_dict_handle` alone does not cover: `index_by` and
    `reduce_by` both produce dicts with value_shape="one", but the
    former's values are records and the latter's are single field values.
    Reading a field off the latter used to raise a bare Python TypeError
    inside join_lookup, which dispatch sanitized into an unactionable
    INTERNAL_ERROR. Call this whenever you intend to subscript a dict
    handle's values."""
    meta = require_dict_handle(registry, handle, param=param, operation=operation)
    if meta.element_type == "value":
        raise TypeMismatchError(
            f"{operation} {param} {handle!r} holds single values, not records — "
            "its values came from something like reduce_by with a value_field, so there are no "
            "fields to read off them. Use an index built by index_by or group_by, "
            "or name exactly one target field to bind the value to.",
            details={"handle": handle, "param": param, "element_type": meta.element_type},
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
