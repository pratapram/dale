"""call_primitive: the single choke point every primitive call goes through —
now, direct test/dev calls; later, an agent loop. Handles the tool-call limit,
schema validation, the cost-estimation confirm-gate, and sanitizing any
unexpected exception before it can leak.
"""

from __future__ import annotations

import logging

import pydantic

from dale.catalog import PrimitiveOutput, get_primitive
from dale.errors import DaleError, InternalError, InvalidParamsError
from dale.registry import DataRegistry

logger = logging.getLogger("dale.dispatch")


def call_primitive(registry: DataRegistry, name: str, params: dict) -> PrimitiveOutput:
    registry.record_call()
    spec = get_primitive(name)

    try:
        parsed = spec.param_schema.model_validate(params)
    except pydantic.ValidationError as exc:
        if registry.privacy_mode:
            # Strip pydantic's per-error `input`/`ctx` (echoes the caller's
            # own supplied value back verbatim) — keep only field path and
            # error type, consistent with privacy_mode's "no real content in
            # error messages" guarantee.
            redacted = [
                {"loc": err["loc"], "type": err["type"]} for err in exc.errors()
            ]
            detail = f"invalid parameters for primitive {name!r}: {redacted}"
        else:
            detail = f"invalid parameters for primitive {name!r}: {exc}"
        raise InvalidParamsError(detail, details={"primitive": name}) from exc

    # Cost estimation and execution are sanitized together — an estimator
    # that raises unexpectedly must not leak any more than the primitive
    # itself would.
    try:
        if spec.cost_estimator is not None:
            estimate = spec.cost_estimator(registry, parsed)
            if estimate.exceeds_threshold and not getattr(parsed, "confirm", False):
                return PrimitiveOutput(status="cost_gate_exceeded", estimate=estimate)

        return spec.fn(registry, parsed)
    except DaleError:
        raise
    except Exception as exc:  # intentional catch-all — sanitized before it leaves this function
        logger.exception("primitive %r raised an unexpected exception", name)
        raise InternalError("primitive execution failed") from exc
