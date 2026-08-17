"""Validates the "exact for row count" claim empirically, not just in prose — a deliberately adversarial fan-out fixture plus a companion
common-case fixture asserting no false-positive trigger.
"""

from __future__ import annotations

from dale.dispatch import call_operation
from dale.registry import DataRegistry, RegistryLimits


def _make_registry(max_result_rows: int) -> DataRegistry:
    return DataRegistry(limits=RegistryLimits(max_result_rows=max_result_rows))


def test_join_fanout_triggers_cost_gate_exceeded_with_exact_estimate():
    reg = _make_registry(max_result_rows=50)

    base = reg.create(
        "list", [{"k": "shared"}] * 10, name="base", description="d", created_by="fixture"
    )
    events = reg.create(
        "list",
        [{"k": "shared", "v": i} for i in range(10)],
        name="events",
        description="d",
        created_by="fixture",
    )
    grouped = call_operation(
        reg,
        "group_by",
        {"handle": events.name, "key_fields": ["k"], "name": "grouped", "description": "d"},
    )

    handles_before = reg.handle_count()
    out = call_operation(
        reg,
        "join_lookup",
        {
            "base_handle": base.name,
            "index_handle": grouped.handle.name,
            "on": ["k"],
            "how": "inner",
            "name": "joined",
            "description": "d",
        },
    )

    assert out.status == "cost_gate_exceeded"
    assert out.estimate.estimated_rows == 100  # 10 base records x 10-record bucket each
    assert out.estimate.exceeds_threshold is True
    assert reg.handle_count() == handles_before  # no handle was created


def test_join_fanout_confirm_produces_actual_size_matching_estimate():
    reg = _make_registry(max_result_rows=50)

    base = reg.create(
        "list", [{"k": "shared"}] * 10, name="base", description="d", created_by="fixture"
    )
    events = reg.create(
        "list",
        [{"k": "shared", "v": i} for i in range(10)],
        name="events",
        description="d",
        created_by="fixture",
    )
    grouped = call_operation(
        reg,
        "group_by",
        {"handle": events.name, "key_fields": ["k"], "name": "grouped", "description": "d"},
    )

    out = call_operation(
        reg,
        "join_lookup",
        {
            "base_handle": base.name,
            "index_handle": grouped.handle.name,
            "on": ["k"],
            "how": "inner",
            "confirm": True,
            "name": "joined",
            "description": "d",
        },
    )

    assert out.status == "ok"
    assert out.handle.size == 100  # matches the earlier exact estimate


def test_unique_keyed_join_does_not_false_positive():
    reg = _make_registry(max_result_rows=50)

    base = reg.create(
        "list",
        [{"k": f"key{i}"} for i in range(20)],
        name="base",
        description="d",
        created_by="fixture",
    )
    catalog = reg.create(
        "list",
        [{"k": f"key{i}", "name": f"n{i}"} for i in range(20)],
        name="catalog",
        description="d",
        created_by="fixture",
    )
    indexed = call_operation(
        reg,
        "index_by",
        {"handle": catalog.name, "key_fields": ["k"], "name": "indexed", "description": "d"},
    )

    out = call_operation(
        reg,
        "join_lookup",
        {
            "base_handle": base.name,
            "index_handle": indexed.handle.name,
            "on": ["k"],
            "how": "left",
            "name": "joined",
            "description": "d",
        },
    )

    assert out.status == "ok"
    assert out.handle.size == 20


def test_bounded_by_input_operations_have_no_estimator():
    """filter_where/compute_field/sort_by are provably output<=input — they
    must be marked bounded_by_input rather than silently missing coverage."""
    from dale.catalog import get_operation

    for name in ("filter_where", "compute_field", "sort_by", "index_by", "group_by", "window_flag"):
        spec = get_operation(name)
        assert spec.bounded_by_input is True
        assert spec.cost_estimator is None

    for name in ("join_lookup", "graph_walk_resolve"):
        spec = get_operation(name)
        assert spec.cost_estimator is not None
        assert spec.bounded_by_input is False
