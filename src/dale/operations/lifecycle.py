from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.registry import DataRegistry


class ReleaseHandleParams(BaseModel):
    handle: str


@operation(
    "release_handle",
    ReleaseHandleParams,
    io_signature="any → —",
    summary="Explicit cleanup of a handle no longer needed",
)
def release_handle(registry: DataRegistry, params: ReleaseHandleParams) -> OperationOutput:
    """Release a handle you no longer need — call this once you've built the
    next handle from it, so intermediate results don't accumulate."""
    registry.release(params.handle)
    return OperationOutput(
        status="ok",
        result={"released": params.handle, "handles_remaining": registry.handle_count()},
    )
