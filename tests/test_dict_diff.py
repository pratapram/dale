"""Unit coverage for dict_diff (testcases.md Test Case 4 — diffing a
resolved license tier assignment against the previous run's)."""

from __future__ import annotations

import itertools

import pytest

from dale.dispatch import call_primitive
from dale.errors import TypeMismatchError
from dale.registry import DataRegistry

_fixture_counter = itertools.count(1)


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load_dict(registry, data, created_by="fixture"):
    return registry.create(
        "dict", data, name=_fixture_name(), description="test fixture data", created_by=created_by
    )


def _diff(registry, current_handle, previous_handle, **extra):
    return call_primitive(
        registry,
        "dict_diff",
        {
            "current_handle": current_handle,
            "previous_handle": previous_handle,
            "name": _fixture_name(),
            "description": "d",
            **extra,
        },
    )


def _by_key(rows):
    return {r["key"]: r for r in rows}


def test_the_worked_example_all_four_statuses(registry):
    current = _load_dict(
        registry,
        {
            "alice@co.com": "gold",
            "bob@co.com": "gold",
            "carol@co.com": "silver",
            "dave@co.com": "silver",
            "erin@co.com": "bronze",
        },
    )
    previous = _load_dict(
        registry,
        {
            "alice@co.com": "silver",
            "carol@co.com": "silver",
            "dave@co.com": "bronze",
            "frank@co.com": "bronze",
        },
    )
    out = _diff(registry, current.handle, previous.handle)
    assert out.status == "ok"
    rows = _by_key(registry.materialize(out.handle.handle))

    assert rows["bob@co.com"] == {
        "key": "bob@co.com", "status": "new", "previous_value": None, "current_value": "gold"
    }
    assert rows["erin@co.com"]["status"] == "new"
    assert rows["frank@co.com"] == {
        "key": "frank@co.com", "status": "removed", "previous_value": "bronze", "current_value": None
    }
    assert rows["alice@co.com"] == {
        "key": "alice@co.com", "status": "changed", "previous_value": "silver", "current_value": "gold"
    }
    assert rows["dave@co.com"] == {
        "key": "dave@co.com", "status": "changed", "previous_value": "bronze", "current_value": "silver"
    }
    assert rows["carol@co.com"] == {
        "key": "carol@co.com", "status": "unchanged", "previous_value": "silver", "current_value": "silver"
    }
    assert len(rows) == 6


def test_both_empty(registry):
    current = _load_dict(registry, {})
    previous = _load_dict(registry, {})
    out = _diff(registry, current.handle, previous.handle)
    assert registry.materialize(out.handle.handle) == []


def test_completely_disjoint(registry):
    current = _load_dict(registry, {"a": 1})
    previous = _load_dict(registry, {"b": 2})
    out = _diff(registry, current.handle, previous.handle)
    rows = _by_key(registry.materialize(out.handle.handle))
    assert rows["a"]["status"] == "new"
    assert rows["b"]["status"] == "removed"


def test_identical_dicts_all_unchanged(registry):
    current = _load_dict(registry, {"a": 1, "b": 2})
    previous = _load_dict(registry, {"a": 1, "b": 2})
    out = _diff(registry, current.handle, previous.handle)
    rows = registry.materialize(out.handle.handle)
    assert all(r["status"] == "unchanged" for r in rows)
    assert len(rows) == 2


def test_wrong_handle_kind_raises(registry):
    current = registry.create(
        "list", [1, 2], name=_fixture_name(), description="d", created_by="fixture"
    )
    previous = _load_dict(registry, {"a": 1})
    with pytest.raises(TypeMismatchError):
        _diff(registry, current.handle, previous.handle)
