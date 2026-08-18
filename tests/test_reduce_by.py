"""Unit coverage for reduce_by -- deduplicate a list to one record per key
under a stated ordering.

Replaces test_priority_reduce.py. The gold/silver/bronze case it covered is
still here (it is the project's original motivating problem, UC6), now as one
of four ordering policies rather than the only one the operation can express.
"""

from __future__ import annotations

import itertools

import pytest

from dale.dispatch import call_operation
from dale.errors import FieldNotFoundError, InvalidParamsError, TypeMismatchError
from dale.registry import DataRegistry

_fixture_counter = itertools.count(1)

TIERS = ["gold", "silver", "bronze"]


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load_records(registry, records, created_by="fixture"):
    return registry.create(
        "list", records, name=_fixture_name(), description="test fixture data", created_by=created_by
    )


def _reduce(registry, handle, key_fields, order_by, **extra):
    return call_operation(
        registry,
        "reduce_by",
        {
            "handle": handle,
            "key_fields": key_fields,
            "order_by": order_by,
            "name": _fixture_name(),
            "description": "d",
            **extra,
        },
    )


_DEFAULT = object()


def _ranked(field="tier", ranking=_DEFAULT, **extra):
    # Sentinel, not `ranking or TIERS`: an empty ranking is a real thing to
    # pass here (one of the rejection tests does exactly that) and falsy.
    return [{"field": field, "ranking": TIERS if ranking is _DEFAULT else ranking, **extra}]


# --- policy 1: explicit ranking (the old priority_reduce behaviour) ---------


def test_the_worked_example_gold_silver_bronze(registry):
    candidates = [
        {"email": "alice@co.com", "tier": "gold"},
        {"email": "alice@co.com", "tier": "silver"},
        {"email": "bob@co.com", "tier": "silver"},
        {"email": "bob@co.com", "tier": "bronze"},
        {"email": "carol@co.com", "tier": "bronze"},
    ]
    meta = _load_records(registry, candidates)
    out = _reduce(registry, meta.name, ["email"], _ranked(), value_field="tier")
    assert registry.get(out.handle.name) == {
        "alice@co.com": "gold",
        "bob@co.com": "silver",
        "carol@co.com": "bronze",
    }


def test_single_membership_wins_trivially(registry):
    meta = _load_records(registry, [{"email": "solo@co.com", "tier": "bronze"}])
    out = _reduce(registry, meta.name, ["email"], _ranked(), value_field="tier")
    assert registry.get(out.handle.name) == {"solo@co.com": "bronze"}


def test_composite_key_fields(registry):
    rows = [
        {"org": "acme", "email": "a@co.com", "tier": "silver"},
        {"org": "acme", "email": "a@co.com", "tier": "gold"},
        {"org": "beta", "email": "a@co.com", "tier": "bronze"},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["org", "email"], _ranked(), value_field="tier")
    assert registry.get(out.handle.name) == {
        ("acme", "a@co.com"): "gold",
        ("beta", "a@co.com"): "bronze",
    }


def test_unranked_value_ranks_last_but_still_resolves(registry):
    """The behaviour change from priority_reduce, which raised here. An
    unranked value loses to any ranked one -- but does not abort the run."""
    rows = [
        {"email": "a@co.com", "tier": "platinum"},
        {"email": "a@co.com", "tier": "bronze"},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["email"], _ranked(), value_field="tier")
    assert registry.get(out.handle.name) == {"a@co.com": "bronze"}


def test_only_unranked_values_wins_by_default(registry):
    """A group with nothing ranked still has a winner. Under priority_reduce
    this was the 'none of [...] found in priority order' error."""
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "platinum"}])
    out = _reduce(registry, meta.name, ["email"], _ranked(), value_field="tier")
    assert registry.get(out.handle.name) == {"a@co.com": "platinum"}


def test_desc_reverses_the_ranking(registry):
    rows = [{"email": "a@co.com", "tier": "gold"}, {"email": "a@co.com", "tier": "bronze"}]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["email"], _ranked(order="desc"), value_field="tier")
    assert registry.get(out.handle.name) == {"a@co.com": "bronze"}


# --- policies 2-4: orderings priority_reduce could not express -------------


def test_max_by_numeric_field(registry):
    rows = [
        {"account": "ACC-1", "score": 10},
        {"account": "ACC-1", "score": 42},
        {"account": "ACC-2", "score": 7},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(
        registry, meta.name, ["account"], [{"field": "score", "order": "desc"}], value_field="score"
    )
    assert registry.get(out.handle.name) == {"ACC-1": 42, "ACC-2": 7}


def test_latest_by_timestamp(registry):
    rows = [
        {"id": "x", "ts": "2026-01-01T00:00:00Z", "status": "open"},
        {"id": "x", "ts": "2026-03-01T00:00:00Z", "status": "closed"},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(
        registry, meta.name, ["id"], [{"field": "ts", "order": "desc"}], value_field="status"
    )
    assert registry.get(out.handle.name) == {"x": "closed"}


def test_multi_key_ordering_is_most_significant_first(registry):
    rows = [
        {"id": "x", "tier": "silver", "score": 99},
        {"id": "x", "tier": "gold", "score": 1},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(
        registry,
        meta.name,
        ["id"],
        [{"field": "tier", "ranking": TIERS}, {"field": "score", "order": "desc"}],
    )
    # tier is the significant key, so gold wins despite the lower score.
    assert registry.get(out.handle.name)["x"]["score"] == 1


# --- output shape ----------------------------------------------------------


def test_omitting_value_field_keeps_the_whole_record(registry):
    rows = [
        {"email": "a@co.com", "tier": "silver", "seats": 3},
        {"email": "a@co.com", "tier": "gold", "seats": 9},
    ]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["email"], _ranked())
    assert registry.get(out.handle.name) == {
        "a@co.com": {"email": "a@co.com", "tier": "gold", "seats": 9}
    }


def test_record_valued_output_is_joinable(registry):
    """The carve-out in join.py only applies to value_field output. A
    whole-record reduce_by index has fields to read, like index_by."""
    rows = [{"sku": "S1", "tier": "gold", "price": 10}, {"sku": "S1", "tier": "bronze", "price": 4}]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["sku"], _ranked())
    assert registry.meta(out.handle.name).element_type == "record"


def test_value_field_output_is_value_typed(registry):
    rows = [{"sku": "S1", "tier": "gold"}]
    meta = _load_records(registry, rows)
    out = _reduce(registry, meta.name, ["sku"], _ranked(), value_field="tier")
    assert registry.meta(out.handle.name).element_type == "value"


# --- rejections ------------------------------------------------------------


def test_empty_order_by_is_rejected_by_schema(registry):
    """An ordering with no keys cannot resolve anything for any input, so it
    is a malformed call -- caught before execution, not per key during it."""
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "gold"}])
    with pytest.raises(InvalidParamsError):
        _reduce(registry, meta.name, ["email"], [], value_field="tier")


def test_empty_ranking_is_rejected_by_schema(registry):
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "gold"}])
    with pytest.raises(InvalidParamsError):
        _reduce(registry, meta.name, ["email"], _ranked(ranking=[]), value_field="tier")


def test_empty_key_fields_is_rejected_by_schema(registry):
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "gold"}])
    with pytest.raises(InvalidParamsError):
        _reduce(registry, meta.name, [], _ranked(), value_field="tier")


def test_value_field_missing_raises(registry):
    meta = _load_records(registry, [{"email": "a@co.com", "tier": "gold"}])
    with pytest.raises(FieldNotFoundError):
        _reduce(registry, meta.name, ["email"], _ranked(), value_field="nope")


def test_incomparable_order_values_raise(registry):
    rows = [{"id": "x", "v": 1}, {"id": "x", "v": "one"}]
    meta = _load_records(registry, rows)
    with pytest.raises(TypeMismatchError):
        _reduce(registry, meta.name, ["id"], [{"field": "v", "order": "desc"}])


def test_wrong_handle_type_raises(registry):
    meta = registry.create(
        "dict", {"a": 1}, name=_fixture_name(), description="d", created_by="fixture"
    )
    with pytest.raises(TypeMismatchError):
        _reduce(registry, meta.name, ["email"], _ranked(), value_field="tier")
