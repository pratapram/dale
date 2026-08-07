"""Unit coverage for priority_reduce (testcases.md Test Case 4 — license
tier reconciliation, the project's original motivating problem)."""

from __future__ import annotations

import itertools

import pytest

from dale.dispatch import call_primitive
from dale.errors import FieldNotFoundError, TypeMismatchError
from dale.registry import DataRegistry

_fixture_counter = itertools.count(1)


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load_records(registry, records, created_by="fixture"):
    return registry.create(
        "list", records, name=_fixture_name(), description="test fixture data", created_by=created_by
    )


def _reduce(registry, handle, key_fields, value_field, priority, **extra):
    return call_primitive(
        registry,
        "priority_reduce",
        {
            "handle": handle,
            "key_fields": key_fields,
            "value_field": value_field,
            "priority": priority,
            "name": _fixture_name(),
            "description": "d",
            **extra,
        },
    )


def test_the_worked_example_gold_silver_bronze(registry):
    candidates = [
        {"email": "alice@co.com", "tier": "gold"},
        {"email": "bob@co.com", "tier": "gold"},
        {"email": "bob@co.com", "tier": "silver"},
        {"email": "carol@co.com", "tier": "silver"},
        {"email": "dave@co.com", "tier": "silver"},
        {"email": "dave@co.com", "tier": "bronze"},
        {"email": "erin@co.com", "tier": "bronze"},
    ]
    meta = _load_records(registry, candidates)
    out = _reduce(
        registry, meta.handle, ["email"], "tier", ["gold", "silver", "bronze"]
    )
    assert out.status == "ok"
    resolved = registry.materialize(out.handle.handle)
    assert resolved == {
        "alice@co.com": "gold",
        "bob@co.com": "gold",
        "carol@co.com": "silver",
        "dave@co.com": "silver",
        "erin@co.com": "bronze",
    }


def test_single_membership_wins_trivially(registry):
    meta = _load_records(registry, [{"email": "solo@co.com", "tier": "bronze"}])
    out = _reduce(registry, meta.handle, ["email"], "tier", ["gold", "silver", "bronze"])
    assert registry.materialize(out.handle.handle) == {"solo@co.com": "bronze"}


def test_composite_key_fields(registry):
    meta = _load_records(
        registry,
        [
            {"region": "us", "email": "a@co.com", "tier": "silver"},
            {"region": "us", "email": "a@co.com", "tier": "gold"},
            {"region": "eu", "email": "a@co.com", "tier": "bronze"},
        ],
    )
    out = _reduce(registry, meta.handle, ["region", "email"], "tier", ["gold", "silver", "bronze"])
    assert out.handle.key_arity == 2
    resolved = registry.materialize(out.handle.handle)
    assert resolved[("us", "a@co.com")] == "gold"
    assert resolved[("eu", "a@co.com")] == "bronze"


def test_value_field_missing_raises(registry):
    meta = _load_records(registry, [{"email": "a@co.com"}])
    with pytest.raises(FieldNotFoundError):
        _reduce(registry, meta.handle, ["email"], "tier", ["gold", "silver", "bronze"])


def test_value_not_in_priority_order_raises(registry):
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "platinum"}])
    with pytest.raises(TypeMismatchError):
        _reduce(registry, meta.handle, ["email"], "tier", ["gold", "silver", "bronze"])


def test_wrong_handle_kind_raises(registry):
    meta = registry.create(
        "dict", {"a": 1}, name=_fixture_name(), description="d", created_by="fixture"
    )
    with pytest.raises(TypeMismatchError):
        _reduce(registry, meta.handle, ["a"], "tier", ["gold", "silver", "bronze"])
