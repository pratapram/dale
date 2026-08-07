from __future__ import annotations

from pydantic import BaseModel

from dale.catalog import PrimitiveOutput, primitive
from dale.registry import DataRegistry


class ReleaseHandleParams(BaseModel):
    handle: str


@primitive("release_handle", ReleaseHandleParams)
def release_handle(registry: DataRegistry, params: ReleaseHandleParams) -> PrimitiveOutput:
    """Release a handle you no longer need — call this once you've built the
    next handle from it, so intermediate results don't accumulate."""
    registry.release(params.handle)
    return PrimitiveOutput(
        status="ok",
        result={"released": params.handle, "handles_remaining": registry.handle_count()},
    )
