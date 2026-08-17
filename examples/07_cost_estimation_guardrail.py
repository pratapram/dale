"""Pre-execution cost estimation: an operation that would blow past a
configured size limit is refused *before* it runs, with an exact estimate —
not caught after the fact by an exception.

Run: uv run python examples/07_cost_estimation_guardrail.py
"""

from __future__ import annotations

import dale
from dale.registry import RegistryLimits

# A deliberately low ceiling to make the guardrail trigger in a small example.
registry = dale.DataRegistry(limits=RegistryLimits(max_result_rows=20))

# 10 base records all sharing one key...
base = registry.create(
    "list",
    [{"category": "electronics"} for _ in range(10)],
    name="base",
    description="base records, all one category",
    created_by="example_script",
)

# ...joined against a group_by bucket of 10 records for that same key. The
# join's output cardinality is 10 x 10 = 100 rows: a real fan-out, computable
# exactly from the already-built index before the join executes.
tags = registry.create(
    "list",
    [{"category": "electronics", "tag": f"tag{i}"} for i in range(10)],
    name="tags",
    description="tags for that category",
    created_by="example_script",
)
tag_groups = dale.call_operation(
    registry,
    "group_by",
    {
        "handle": tags.name,
        "key_fields": ["category"],
        "name": "tag_groups",
        "description": "tags bucketed by category",
    },
)

result = dale.call_operation(
    registry,
    "join_lookup",
    {
        "base_handle": base.name,
        "index_handle": tag_groups.handle.name,
        "on": ["category"],
        "how": "inner",
        "name": "joined",
        "description": "base joined against every tag in its category",
    },
)

print(f"status: {result.status}")
print(f"estimated_rows: {result.estimate.estimated_rows} "
      f"(threshold: {result.estimate.threshold_rows})")
print(f"handles in registry right now: {registry.handle_count()} "
      "— no handle was created for the refused operation")

# Explicit confirm=True proceeds anyway. The resulting handle's actual size
# matches the earlier estimate exactly — this is the "exact for row count"
# property the estimator provides, not an approximation.
confirmed = dale.call_operation(
    registry,
    "join_lookup",
    {
        "base_handle": base.name,
        "index_handle": tag_groups.handle.name,
        "on": ["category"],
        "how": "inner",
        "confirm": True,
        "name": "joined",
        "description": "base joined against every tag in its category",
    },
)
print(f"\nafter confirm=True: status={confirmed.status}, actual size={confirmed.handle.size}")
