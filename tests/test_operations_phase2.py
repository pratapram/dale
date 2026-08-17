"""Unit coverage for the two operations added to unblock Use Cases 2
and 4: window_flag (sliding-window occurrence flagging) and
graph_walk_resolve (single-parent ancestor-chain rule resolution). Small,
isolated fixtures — not the full sample datasets (see
tests/test_use_case_pipelines.py for those)."""

from __future__ import annotations

import itertools

import pytest

from dale.dispatch import call_operation
from dale.errors import (
    FieldNotFoundError,
    GraphCycleError,
    TypeMismatchError,
)
from dale.registry import DataRegistry, RegistryLimits

# --- window_flag -------------------------------------------------------

_fixture_counter = itertools.count(1)


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load(registry, records, created_by="fixture"):
    return registry.create(
        "list", records, name=_fixture_name(), description="test fixture data", created_by=created_by
    )


def test_window_flag_numeric_field_flags_burst(registry):
    handle = _load(
        registry,
        [
            {"ip": "a", "t": 0, "outcome": "fail"},
            {"ip": "a", "t": 10, "outcome": "fail"},
            {"ip": "a", "t": 20, "outcome": "fail"},
            {"ip": "a", "t": 30, "outcome": "success"},  # right after the burst
            {"ip": "b", "t": 5, "outcome": "fail"},  # isolated, different group
        ],
    )
    out = call_operation(
        registry,
        "window_flag",
        {
            "handle": handle.name,
            "group_by": ["ip"],
            "window_field": "t",
            "window_size": 35,
            "threshold": 3,
            "predicate": {"field": "outcome", "op": "==", "value": "fail"},
            "name": "flagged",
            "description": "d",
        },
    )
    assert out.status == "ok"
    rows = {(r["t"], r["ip"]): r for r in registry.materialize(out.handle.name)}

    assert rows[(0, "a")]["flagged"] is False
    assert rows[(10, "a")]["flagged"] is False
    assert rows[(20, "a")]["flagged"] is True  # 3rd fail within window_size=35
    assert rows[(20, "a")]["flagged_count"] == 3
    # The success right after the burst is still within the window of the 3 fails.
    assert rows[(30, "a")]["flagged"] is True
    assert rows[(30, "a")]["flagged_count"] == 3
    assert rows[(5, "b")]["flagged"] is False  # different group, isolated


def test_window_flag_iso8601_string_field(registry):
    handle = _load(
        registry,
        [
            {"ip": "a", "ts": "2026-01-01T00:00:00"},
            {"ip": "a", "ts": "2026-01-01T00:00:30"},
            {"ip": "a", "ts": "2026-01-01T00:05:01"},  # outside the 60s window of the first two
        ],
    )
    out = call_operation(
        registry,
        "window_flag",
        {
            "handle": handle.name,
            "group_by": ["ip"],
            "window_field": "ts",
            "window_size": 60,
            "threshold": 2,
            "name": "flagged",
            "description": "d",
        },
    )
    rows = {r["ts"]: r for r in registry.materialize(out.handle.name)}
    assert rows["2026-01-01T00:00:00"]["flagged"] is False
    assert rows["2026-01-01T00:00:30"]["flagged"] is True
    assert rows["2026-01-01T00:00:30"]["flagged_count"] == 2
    assert rows["2026-01-01T00:05:01"]["flagged"] is False


def test_window_flag_composite_group_key(registry):
    handle = _load(
        registry,
        [
            {"region": "us", "ip": "a", "t": 0},
            {"region": "us", "ip": "a", "t": 1},
            {"region": "eu", "ip": "a", "t": 0},  # same ip, different region -> different group
        ],
    )
    out = call_operation(
        registry,
        "window_flag",
        {
            "handle": handle.name,
            "group_by": ["region", "ip"],
            "window_field": "t",
            "window_size": 10,
            "threshold": 2,
            "name": "flagged",
            "description": "d",
        },
    )
    rows = registry.materialize(out.handle.name)
    eu_row = next(r for r in rows if r["region"] == "eu")
    assert eu_row["flagged"] is False
    assert eu_row["flagged_count"] == 1


def test_window_flag_wrong_handle_type_raises(registry):
    d = registry.create("dict", {"a": 1}, name=_fixture_name(), description="d", created_by="fixture")
    with pytest.raises(TypeMismatchError):
        call_operation(
            registry,
            "window_flag",
            {
                "handle": d.name,
                "group_by": ["a"],
                "window_field": "a",
                "window_size": 1,
                "threshold": 1,
                "name": "flagged",
                "description": "d",
            },
        )


def test_window_flag_incomparable_window_values_raise(registry):
    handle = _load(registry, [{"ip": "a", "t": 0}, {"ip": "a", "t": "not-a-timestamp"}])
    with pytest.raises(TypeMismatchError):
        call_operation(
            registry,
            "window_flag",
            {
                "handle": handle.name,
                "group_by": ["ip"],
                "window_field": "t",
                "window_size": 10,
                "threshold": 1,
                "name": "flagged",
                "description": "d",
            },
        )


# --- graph_walk_resolve --------------------------------------------------


def _org(registry, records):
    nodes = _load(registry, records)
    out = call_operation(
        registry,
        "index_by",
        {"handle": nodes.name, "key_fields": ["id"], "name": _fixture_name(), "description": "d"},
    )
    return out.handle.name


def test_graph_walk_resolve_deny_overrides_inherited_allow(registry):
    org_idx = _org(
        registry,
        [
            {"id": "root", "manager_id": None},
            {"id": "mid", "manager_id": "root"},
            {"id": "leaf", "manager_id": "mid"},
        ],
    )
    rules = _load(
        registry,
        [
            {"employee_id": "root", "resource": "db", "effect": "Allow"},
            {"employee_id": "leaf", "resource": "db", "effect": "Deny"},
        ],
    )
    out = call_operation(
        registry,
        "graph_walk_resolve",
        {
            "nodes_index_handle": org_idx,
            "parent_field": "manager_id",
            "rules_handle": rules.name,
            "rule_node_field": "employee_id",
            "group_field": "resource",
            "value_field": "effect",
            "priority": ["Deny", "Allow"],
            "name": "resolved",
            "description": "d",
        },
    )
    assert out.status == "ok"
    rows = {r["node_id"]: r["effect"] for r in registry.materialize(out.handle.name)}
    assert rows["root"] == "Allow"
    assert rows["mid"] == "Allow"  # inherits from root
    assert rows["leaf"] == "Deny"  # explicit deny wins over inherited allow


def test_graph_walk_resolve_node_with_no_rules_is_omitted(registry):
    org_idx = _org(registry, [{"id": "solo", "manager_id": None}])
    rules = _load(registry, [])
    out = call_operation(
        registry,
        "graph_walk_resolve",
        {
            "nodes_index_handle": org_idx,
            "parent_field": "manager_id",
            "rules_handle": rules.name,
            "rule_node_field": "employee_id",
            "group_field": "resource",
            "value_field": "effect",
            "priority": ["Deny", "Allow"],
            "name": "resolved",
            "description": "d",
        },
    )
    assert registry.materialize(out.handle.name) == []


def test_graph_walk_resolve_dangling_parent_treated_as_root(registry):
    org_idx = _org(registry, [{"id": "orphan", "manager_id": "ghost_not_in_index"}])
    rules = _load(registry, [{"employee_id": "orphan", "resource": "db", "effect": "Allow"}])
    out = call_operation(
        registry,
        "graph_walk_resolve",
        {
            "nodes_index_handle": org_idx,
            "parent_field": "manager_id",
            "rules_handle": rules.name,
            "rule_node_field": "employee_id",
            "group_field": "resource",
            "value_field": "effect",
            "priority": ["Deny", "Allow"],
            "name": "resolved",
            "description": "d",
        },
    )
    assert out.status == "ok"
    rows = registry.materialize(out.handle.name)
    assert rows == [{"node_id": "orphan", "resource": "db", "effect": "Allow"}]


def test_graph_walk_resolve_cycle_raises(registry):
    org_idx = _org(
        registry,
        [
            {"id": "a", "manager_id": "b"},
            {"id": "b", "manager_id": "a"},
        ],
    )
    rules = _load(registry, [])
    with pytest.raises(GraphCycleError):
        call_operation(
            registry,
            "graph_walk_resolve",
            {
                "nodes_index_handle": org_idx,
                "parent_field": "manager_id",
                "rules_handle": rules.name,
                "rule_node_field": "employee_id",
                "group_field": "resource",
                "value_field": "effect",
                "priority": ["Deny", "Allow"],
                "name": "resolved",
                "description": "d",
            },
        )


def test_graph_walk_resolve_value_not_in_priority_raises(registry):
    org_idx = _org(registry, [{"id": "a", "manager_id": None}])
    rules = _load(registry, [{"employee_id": "a", "resource": "db", "effect": "Maybe"}])
    with pytest.raises(TypeMismatchError):
        call_operation(
            registry,
            "graph_walk_resolve",
            {
                "nodes_index_handle": org_idx,
                "parent_field": "manager_id",
                "rules_handle": rules.name,
                "rule_node_field": "employee_id",
                "group_field": "resource",
                "value_field": "effect",
                "priority": ["Deny", "Allow"],
                "name": "resolved",
                "description": "d",
            },
        )


def test_graph_walk_resolve_missing_rule_field_raises(registry):
    org_idx = _org(registry, [{"id": "a", "manager_id": None}])
    rules = _load(registry, [{"resource": "db", "effect": "Allow"}])  # no employee_id
    with pytest.raises(FieldNotFoundError):
        call_operation(
            registry,
            "graph_walk_resolve",
            {
                "nodes_index_handle": org_idx,
                "parent_field": "manager_id",
                "rules_handle": rules.name,
                "rule_node_field": "employee_id",
                "group_field": "resource",
                "value_field": "effect",
                "priority": ["Deny", "Allow"],
                "name": "resolved",
                "description": "d",
            },
        )


def test_graph_walk_resolve_wrong_handle_types_raise(registry):
    nodes = _load(registry, [{"id": "a"}])  # a list, not an index_by dict
    rules = _load(registry, [])
    with pytest.raises(TypeMismatchError):
        call_operation(
            registry,
            "graph_walk_resolve",
            {
                "nodes_index_handle": nodes.name,
                "parent_field": "manager_id",
                "rules_handle": rules.name,
                "rule_node_field": "employee_id",
                "group_field": "resource",
                "value_field": "effect",
                "priority": ["Deny", "Allow"],
                "name": "resolved",
                "description": "d",
            },
        )


def test_graph_walk_resolve_fanout_triggers_cost_gate_exceeded():
    reg = DataRegistry(limits=RegistryLimits(max_result_rows=2))
    org_idx = _org(
        reg,
        [{"id": f"e{i}", "manager_id": None} for i in range(5)],
    )
    rules = _load(
        reg,
        [{"employee_id": f"e{i}", "resource": r, "effect": "Allow"} for i in range(5) for r in ("db", "crm")],
    )
    out = call_operation(
        reg,
        "graph_walk_resolve",
        {
            "nodes_index_handle": org_idx,
            "parent_field": "manager_id",
            "rules_handle": rules.name,
            "rule_node_field": "employee_id",
            "group_field": "resource",
            "value_field": "effect",
            "priority": ["Deny", "Allow"],
            "name": "resolved",
            "description": "d",
        },
    )
    assert out.status == "cost_gate_exceeded"
    assert out.estimate.estimated_rows == 10  # 5 nodes x 2 distinct resources
