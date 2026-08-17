"""join_lookup: the one operation in this increment with genuine fan-out risk,
and therefore the one that needs a real cost estimator. Row
count is estimated *exactly* from the already-built index's bucket sizes,
before the join runs; byte count is a deliberate over-estimate."""

from __future__ import annotations

from typing import Literal

from dale.catalog import ConfirmableParams, OperationOutput, operation
from dale.cost import CostEstimate, make_estimate
from dale.errors import TypeMismatchError
from dale.keys import make_key
from dale.registry import DataRegistry


class JoinLookupParams(ConfirmableParams):
    base_handle: str
    index_handle: str
    on: list[str]
    fields: list[str] | None = None
    how: Literal["left", "inner"] = "left"
    name: str
    description: str


def _bucket_size(index_meta_value_shape: str | None, matched: object) -> int:
    if index_meta_value_shape == "many":
        return len(matched)  # type: ignore[arg-type]
    return 1


def _validate_join_handles(registry: DataRegistry, params: JoinLookupParams) -> None:
    """Shared by the cost estimator and the operation itself — the estimator
    runs first (dispatch.py), so validation cannot live only in the latter."""
    base_meta = registry.meta(params.base_handle)
    index_meta = registry.meta(params.index_handle)
    if base_meta.type != "list":
        raise TypeMismatchError(
            f"join_lookup base_handle must be a list, got {base_meta.type!r}",
            details={"handle": params.base_handle, "type": base_meta.type},
        )
    if index_meta.type != "dict":
        raise TypeMismatchError(
            f"join_lookup index_handle must be a dict (built via index_by/group_by), "
            f"got {index_meta.type!r}",
            details={"handle": params.index_handle, "type": index_meta.type},
        )
    # A scalar-valued index (priority_reduce) has no fields to read off its
    # values, so the caller must say what to call the value it's merging in.
    # Checked here rather than mid-loop so the model learns before any work
    # happens, and so the cost estimator rejects it on the same terms.
    if index_meta.element_type == "value" and (
        params.fields is None or len(params.fields) != 1
    ):
        raise TypeMismatchError(
            f"join_lookup index_handle {params.index_handle!r} holds single values, "
            "not records — pass exactly one name in `fields` to bind that value to "
            f"(got {params.fields!r}). An index_by/group_by index can merge several "
            "fields; a priority_reduce index carries only one value per key.",
            details={
                "handle": params.index_handle,
                "element_type": index_meta.element_type,
                "fields": params.fields,
            },
        )


def _estimate_join_rows(registry: DataRegistry, params: JoinLookupParams) -> int:
    _validate_join_handles(registry, params)
    index_meta = registry.meta(params.index_handle)
    base = registry.get(params.base_handle)
    index = registry.get(params.index_handle)

    total = 0
    for record in base:
        key = make_key(record, params.on, handle=params.base_handle)
        matched = index.get(key)
        if matched is not None:
            total += _bucket_size(index_meta.value_shape, matched)
        elif params.how == "left":
            total += 1
        # inner + no match -> contributes 0
    return total


def join_cost_estimator(registry: DataRegistry, params: JoinLookupParams) -> CostEstimate:
    estimated_rows = _estimate_join_rows(registry, params)

    base_meta = registry.meta(params.base_handle)
    index_meta = registry.meta(params.index_handle)
    avg_bytes = None
    if base_meta.avg_record_bytes is not None and index_meta.avg_record_bytes is not None:
        # Deliberate over-estimate: the true merged record is a subset of
        # base + matched fields, never their full sum.
        avg_bytes = base_meta.avg_record_bytes + index_meta.avg_record_bytes

    return make_estimate(estimated_rows, avg_bytes, registry.limits.max_result_rows)


@operation(
    "join_lookup", JoinLookupParams, cost_estimator=join_cost_estimator, creates_handle=True
)
def join_lookup(registry: DataRegistry, params: JoinLookupParams) -> OperationOutput:
    """Merge a list handle against a dict handle built by index_by/group_by,
    matching on one or more fields. how="left" keeps unmatched base records
    as-is; how="inner" drops them. May require confirm=True if the estimated
    output size exceeds the registry's threshold (see join_cost_estimator)."""
    _validate_join_handles(registry, params)
    index_meta = registry.meta(params.index_handle)

    base = registry.get(params.base_handle)
    index = registry.get(params.index_handle)

    result = []
    for record in base:
        key = make_key(record, params.on, handle=params.base_handle)
        matched = index.get(key)

        if matched is None:
            if params.how == "left":
                result.append(dict(record))
            continue

        bucket = matched if index_meta.value_shape == "many" else [matched]
        for matched_record in bucket:
            merged = dict(record)
            if isinstance(matched_record, dict):
                fields = (
                    params.fields
                    if params.fields is not None
                    else [k for k in matched_record if k not in params.on]
                )
                for f in fields:
                    if f in matched_record:
                        merged[f] = matched_record[f]
            else:
                # Scalar-valued index: bind the value itself under the single
                # name the caller supplied. _validate_join_handles has already
                # guaranteed exactly one field when element_type is "value";
                # this isinstance check is what makes the guarantee unnecessary
                # to trust -- a mixed-value dict (element_type None) lands here
                # too, and still cannot raise a bare TypeError.
                if not params.fields or len(params.fields) != 1:
                    raise TypeMismatchError(
                        f"join_lookup: index {params.index_handle!r} produced a "
                        f"non-record value for key {key!r}; pass exactly one name in "
                        "`fields` to bind it to.",
                        details={"handle": params.index_handle, "fields": params.fields},
                    )
                merged[params.fields[0]] = matched_record
            result.append(merged)

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="join_lookup",
        source_handles=[params.base_handle, params.index_handle],
    )
    return OperationOutput(status="ok", handle=new_meta)
