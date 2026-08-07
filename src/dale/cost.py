"""Pre-execution cost estimation (objections #4).

Because the primitive set is closed and known in advance — not arbitrary
generated code — the cost of an operation can often be computed or tightly
bounded *before* it runs. Row-count estimates for join-shaped operations are
exact (built from already-materialized index bucket sizes); byte estimates are
an approximation and documented as such rather than asserted as precise.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimated_rows: int
    estimated_bytes: int | None
    threshold_rows: int
    exceeds_threshold: bool


def make_estimate(
    estimated_rows: int,
    avg_record_bytes: int | None,
    threshold_rows: int,
) -> CostEstimate:
    """Build a CostEstimate. `estimated_bytes` is a deliberate approximation —
    see join.py's estimator for why it over-estimates rather than under."""
    estimated_bytes = (
        int(estimated_rows * avg_record_bytes) if avg_record_bytes is not None else None
    )
    return CostEstimate(
        estimated_rows=estimated_rows,
        estimated_bytes=estimated_bytes,
        threshold_rows=threshold_rows,
        exceeds_threshold=estimated_rows > threshold_rows,
    )
