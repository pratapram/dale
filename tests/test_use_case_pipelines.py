"""Regression tests for the DESIGN.md Section 3 use-case sample data
(`examples/data/`, manifest at `examples/data/README.md`). These pin down the
ground truth that was previously only verified by hand, one-off, in a prior
session — turning that anecdotal verification into a real committed test that
guards against future operation-behavior regressions.

Use Case 1 (inventory reconciliation) and Use Case 2 (log sessionization, via
window_flag) run their DESIGN.md-intended pipelines end to end. Use Case 3
(churn/feature usage) runs the *alternative* pipeline documented in
examples/data/README.md, since dict_frequency/set_difference (the intended
operations) are not yet built — window_flag/graph_walk_resolve were
built to unblock 2 and 4 specifically.
Use Case 4 (org permission inheritance) runs via graph_walk_resolve.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

import dale

DATA_DIR = Path(__file__).parent.parent / "examples" / "data"

_step_counter = itertools.count(1)


def _step_name() -> str:
    return f"step_{next(_step_counter)}"


def _load(registry: dale.DataRegistry, path: Path) -> str:
    assert registry.files is not None
    virtual_name = path.name
    registry.files.register(virtual_name, path)
    out = dale.call_operation(
        registry, "load_csv", {"file": virtual_name, "name": _step_name(), "description": "d"}
    )
    return out.handle.name


def test_use_case_1_inventory_reconciliation(registry):
    base = DATA_DIR / "01_inventory_reconciliation"
    meta = _load(registry, base / "product_metadata.csv")
    stock = _load(registry, base / "stock_counts.csv")
    pricing = _load(registry, base / "pricing_overrides.csv")

    stock_idx = dale.call_operation(
        registry,
        "index_by",
        {"handle": stock, "key_fields": ["sku"], "name": _step_name(), "description": "d"},
    ).handle.name
    pricing_idx = dale.call_operation(
        registry,
        "index_by",
        {"handle": pricing, "key_fields": ["sku"], "name": _step_name(), "description": "d"},
    ).handle.name

    joined1 = dale.call_operation(
        registry,
        "join_lookup",
        {
            "base_handle": meta,
            "index_handle": stock_idx,
            "on": ["sku"],
            "how": "left",
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name
    joined2 = dale.call_operation(
        registry,
        "join_lookup",
        {
            "base_handle": joined1,
            "index_handle": pricing_idx,
            "on": ["sku"],
            "how": "left",
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    # Ordering ops (>) raise on a missing field (SKU-1020, untracked stock) —
    # equality-class ops (!=) treat missing as None instead. See the README's
    # edge-case notes.
    in_stock = dale.call_operation(
        registry,
        "filter_where",
        {
            "handle": joined2,
            "predicate": {"field": "stock_count", "op": "!=", "value": 0},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name
    priced = dale.call_operation(
        registry,
        "filter_where",
        {
            "handle": in_stock,
            "predicate": {"field": "cost", "op": "!=", "value": None},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    margin = dale.call_operation(
        registry,
        "compute_field",
        {
            "handle": priced,
            "as": "margin",
            "op": "subtract",
            "left": {"field": "price"},
            "right": {"field": "cost"},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name
    sorted_h = dale.call_operation(
        registry,
        "sort_by",
        {
            "handle": margin,
            "keys": [{"field": "margin", "order": "desc"}],
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    rows = registry.materialize(sorted_h)
    result = [(r["sku"], round(r["margin"], 2)) for r in rows]

    assert result == [
        ("SKU-1010", 239.99),
        ("SKU-1016", 179.99),
        ("SKU-1009", 159.99),
        ("SKU-1004", 109.99),
        ("SKU-1007", 69.99),
        ("SKU-1012", 44.99),
        ("SKU-1019", 34.99),
        ("SKU-1020", 19.99),
        ("SKU-1003", 17.99),
        ("SKU-1006", 16.99),
        ("SKU-1014", 15.99),
        ("SKU-1001", 11.49),
        ("SKU-1018", 8.99),
        ("SKU-1015", 6.49),
    ]
    # Dropped: 5 zero-stock SKUs (1002/1005/1008/1013/1017) + SKU-1011
    # (in stock but no pricing override).
    assert len(rows) == 14


def test_use_case_2_log_sessionization(registry):
    base = DATA_DIR / "02_log_sessionization"
    logs = _load(registry, base / "auth_logs.csv")

    flagged = dale.call_operation(
        registry,
        "window_flag",
        {
            "handle": logs,
            "group_by": ["source_ip"],
            "window_field": "timestamp",
            "window_size": 300,  # 5 minutes
            "threshold": 5,
            "predicate": {"field": "outcome", "op": "==", "value": "fail"},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    flagged_rows = dale.call_operation(
        registry,
        "filter_where",
        {
            "handle": flagged,
            "predicate": {"field": "flagged", "op": "==", "value": True},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    rows = registry.materialize(flagged_rows)
    flagged_ips = sorted({r["source_ip"] for r in rows})

    assert flagged_ips == ["198.51.100.23", "203.0.113.55"]
    # The success immediately following the 203.0.113.55 burst is also
    # flagged — the interesting case documented in examples/data/README.md.
    breach = next(r for r in rows if r["source_ip"] == "203.0.113.55" and r["outcome"] == "success")
    assert breach["username"] == "asmith"
    assert breach["timestamp"] == "2026-07-30T09:17:05"
    assert len(rows) == 8


def test_use_case_3_churn_feature_usage(registry):
    base = DATA_DIR / "03_churn_feature_usage"
    subs = _load(registry, base / "active_subscriptions.csv")
    events = _load(registry, base / "feature_events.csv")

    events_grouped = dale.call_operation(
        registry,
        "group_by",
        {"handle": events, "key_fields": ["account_id"], "name": _step_name(), "description": "d"},
    ).handle.name
    joined = dale.call_operation(
        registry,
        "join_lookup",
        {
            "base_handle": subs,
            "index_handle": events_grouped,
            "on": ["account_id"],
            "how": "left",
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    # Accounts with zero feature events never got a matched bucket merged in,
    # so `feature` is missing on their row — that's the churn-risk signal.
    churn_risk = dale.call_operation(
        registry,
        "filter_where",
        {
            "handle": joined,
            "predicate": {"field": "feature", "op": "==", "value": None},
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    rows = registry.materialize(churn_risk)
    account_ids = sorted(r["account_id"] for r in rows)

    assert account_ids == ["ACC-004", "ACC-009", "ACC-013"]


def test_use_case_4_org_permissions(registry):
    base = DATA_DIR / "04_org_permissions"
    org = _load(registry, base / "org_structure.csv")
    rules = _load(registry, base / "policy_rules.csv")

    org_idx = dale.call_operation(
        registry,
        "index_by",
        {"handle": org, "key_fields": ["employee_id"], "name": _step_name(), "description": "d"},
    ).handle.name

    effective = dale.call_operation(
        registry,
        "graph_walk_resolve",
        {
            "nodes_index_handle": org_idx,
            "parent_field": "manager_id",
            "rules_handle": rules,
            "rule_node_field": "employee_id",
            "group_field": "resource",
            "value_field": "effect",
            "priority": ["Deny", "Allow"],
            "name": _step_name(),
            "description": "d",
        },
    ).handle.name

    rows = registry.materialize(effective)
    by_employee: dict[str, dict[str, str]] = {}
    for r in rows:
        by_employee.setdefault(r["node_id"], {})[r["resource"]] = r["effect"]

    assert by_employee == {
        "E002": {"prod_database": "Allow"},
        "E003": {"prod_database": "Allow"},
        "E004": {"prod_database": "Allow"},
        "E005": {"prod_database": "Allow"},
        "E006": {"prod_database": "Allow", "staging_environment": "Allow"},
        "E007": {"prod_database": "Deny", "staging_environment": "Allow"},
        "E008": {"prod_database": "Allow", "staging_environment": "Allow"},
        "E009": {"customer_crm": "Allow"},
        "E010": {"customer_crm": "Allow"},
        "E011": {"customer_crm": "Deny"},
        "E012": {"customer_crm": "Allow"},
        "E013": {"billing_system": "Allow"},
        "E014": {"billing_system": "Allow"},
        "E015": {"billing_system": "Allow"},
    }
    # E001 (CEO) has no applicable rules in this sample.
    assert "E001" not in by_employee


def test_use_case_6_license_reconciliation(registry):
    """The project's original
    motivating problem. The three tier lists are unioned and tagged by the
    invoker in plain Python before DALE ever sees them (no union/concat
    operation exists yet — same "assembly is the invoker's job" precedent
    DALE already applies to paginated API responses); DALE's own
    job starts at priority_reduce."""
    base = DATA_DIR / "license_reconciliation"

    candidates = []
    for tier in ("gold", "silver", "bronze"):
        for email in json.loads((base / f"{tier}_users.json").read_text()):
            candidates.append({"email": email, "tier": tier})
    candidates_meta = registry.create(
        "list", candidates, name="candidates", description="d", created_by="test"
    )

    resolved = dale.call_operation(
        registry,
        "priority_reduce",
        {
            "handle": candidates_meta.name,
            "key_fields": ["email"],
            "value_field": "tier",
            "priority": ["gold", "silver", "bronze"],
            "name": "resolved",
            "description": "d",
        },
    )
    assert registry.materialize(resolved.handle.name) == {
        "alice@co.com": "gold",
        "bob@co.com": "gold",
        "carol@co.com": "silver",
        "dave@co.com": "silver",
        "erin@co.com": "bronze",
    }

    previous_data = json.loads((base / "previous_assignments.json").read_text())
    previous_meta = registry.create(
        "dict", previous_data, name="previous", description="d", created_by="test"
    )

    diff = dale.call_operation(
        registry,
        "dict_diff",
        {
            "current_handle": resolved.handle.name,
            "previous_handle": previous_meta.name,
            "name": "diff",
            "description": "d",
        },
    )
    rows = {r["key"]: r["status"] for r in registry.materialize(diff.handle.name)}
    assert rows == {
        "alice@co.com": "changed",
        "bob@co.com": "new",
        "carol@co.com": "unchanged",
        "dave@co.com": "changed",
        "erin@co.com": "new",
        "frank@co.com": "removed",
    }


# --- eval checker robustness ---------------------------------------------
#
# A checker grades the answer, not the route. These pin down that a correct
# result still counts when it arrives in dict_diff shape (rows keyed `key`,
# with the original record nested under `previous_value`) rather than as a
# list of records carrying a top-level `account_id` -- the false negative that
# cost 2 of 3 live uc3_large runs. The wrong-answer cases are the other half:
# the fix must not turn the checker into a rubber stamp.


def _uc3_large_registry():
    from eval.use_cases import USE_CASES

    setup, _task, _checker = USE_CASES["uc3_large"]
    registry = dale.DataRegistry(files=dale.FileRegistry())
    setup(registry)
    return registry


def _run_diff_route(registry):
    """The exact 5-call pipeline a live gpt-5.6 run produced."""
    dale.call_operation(
        registry,
        "index_by",
        {"handle": "active_subscriptions", "key_fields": ["account_id"], "name": "subs_idx", "description": "d"},
    )
    dale.call_operation(
        registry,
        "join_lookup",
        {"base_handle": "feature_events", "index_handle": "subs_idx", "on": ["account_id"],
         "fields": ["company"], "how": "inner", "name": "ev_active", "description": "d"},
    )
    dale.call_operation(
        registry,
        "group_by",
        {"handle": "ev_active", "key_fields": ["account_id"], "name": "with_events", "description": "d"},
    )
    dale.call_operation(
        registry,
        "dict_diff",
        {"previous_handle": "subs_idx", "current_handle": "with_events", "name": "diff", "description": "d"},
    )
    return dale.call_operation(
        registry,
        "filter_where",
        {"handle": "diff", "predicate": {"field": "status", "op": "==", "value": "removed"},
         "name": "zero_usage", "description": "d"},
    )


def test_uc3_large_checker_accepts_dict_diff_shaped_answer():
    # eval.use_cases imports dale.agent, which needs the `agent` extra.
    pytest.importorskip("pydantic_ai")
    from eval.use_cases import check_uc3_large

    registry = _uc3_large_registry()
    out = _run_diff_route(registry)
    rows = registry.materialize(out.handle.name)
    # Precondition: the answer really is diff-shaped, not a record list.
    assert "account_id" not in rows[0] and "key" in rows[0]
    assert check_uc3_large(registry, None, "") is True


def test_uc3_large_checker_still_rejects_a_wrong_answer():
    # eval.use_cases imports dale.agent, which needs the `agent` extra.
    pytest.importorskip("pydantic_ai")
    from eval.use_cases import _UC3_LARGE_EXPECTED, check_uc3_large

    registry = _uc3_large_registry()
    partial = sorted(_UC3_LARGE_EXPECTED)[:2]
    registry.create(
        "list",
        [{"account_id": a} for a in partial],
        name="too_few",
        description="two of the three expected accounts",
        created_by="test",
    )
    registry.create(
        "list",
        [{"key": "ACC-9999999"}],
        name="unrelated",
        description="an account that isn't in the answer",
        created_by="test",
    )
    assert check_uc3_large(registry, None, "") is False


def test_uc3_checker_accepts_dict_diff_shaped_answer(registry):
    """Same false negative existed in check_uc3 against the real fixture."""
    # eval.use_cases imports dale.agent, which needs the `agent` extra.
    pytest.importorskip("pydantic_ai")
    from eval.use_cases import _UC3_EXPECTED, check_uc3

    registry.create(
        "list",
        [
            {"key": account, "status": "removed",
             "previous_value": {"account_id": account}, "current_value": None}
            for account in sorted(_UC3_EXPECTED)
        ],
        name="zero_usage_diff",
        description="diff-shaped answer",
        created_by="test",
    )
    assert check_uc3(registry, None, "") is True
