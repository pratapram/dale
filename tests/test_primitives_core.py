from __future__ import annotations

import itertools
import json

import pytest
from pydantic import BaseModel

from dale.catalog import register_primitive
from dale.dispatch import call_primitive
from dale.errors import (
    DaleError,
    DuplicateKeyError,
    ExportError,
    FieldNotFoundError,
    FileNotRegisteredError,
    HandleNotFoundError,
    InternalError,
    InvalidParamsError,
    LoadError,
    PrimitiveNotFoundError,
    TypeMismatchError,
)
from dale.primitives.inspect import _DESCRIBE_MAX_BYTES, _PEEK_MAX_BYTES
from dale.registry import DataRegistry

_fixture_counter = itertools.count(1)


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load_records(registry, records, created_by="fixture", description="test fixture data"):
    return registry.create(
        "list", records, name=_fixture_name(), description=description, created_by=created_by
    )


# --- load_csv --------------------------------------------------------------


def test_load_csv_infers_types(registry, write_csv):
    file = write_csv(
        "data.csv",
        [
            {"id": "1", "name": "a", "price": "9.99", "note": ""},
            {"id": "2", "name": "b", "price": "5", "note": "x"},
        ],
    )
    out = call_primitive(
        registry, "load_csv", {"file": file, "name": "rows", "description": "loaded csv"}
    )
    assert out.status == "ok"
    rows = registry.materialize(out.handle.handle)
    assert rows[0] == {"id": 1, "name": "a", "price": 9.99, "note": None}
    assert rows[1] == {"id": 2, "name": "b", "price": 5, "note": "x"}


def test_load_csv_unregistered_name_raises(registry):
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            registry, "load_csv", {"file": "nope", "name": "rows", "description": "d"}
        )


def test_load_csv_no_file_registry_configured_raises():
    bare_registry = DataRegistry()  # no files=FileRegistry() at all
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            bare_registry, "load_csv", {"file": "anything", "name": "rows", "description": "d"}
        )


def test_load_csv_registered_file_later_missing_raises_load_error(registry, write_csv, tmp_path):
    file = write_csv("gone.csv", [{"id": "1"}])
    (tmp_path / "gone.csv").unlink()  # registered, but the real file is gone by load time
    with pytest.raises(LoadError):
        call_primitive(
            registry, "load_csv", {"file": file, "name": "rows", "description": "d"}
        )


# --- load_json ---------------------------------------------------------


def test_load_json_top_level_array_becomes_list_handle(registry, write_json):
    file = write_json("orders.json", [{"id": 1, "total": 9.99}, {"id": 2, "total": 5}])
    out = call_primitive(
        registry, "load_json", {"file": file, "name": "orders", "description": "orders"}
    )
    assert out.status == "ok"
    assert out.handle.kind == "list"
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"id": 1, "total": 9.99}, {"id": 2, "total": 5}]


def test_load_json_top_level_object_becomes_dict_handle(registry, write_json):
    file = write_json("config.json", {"a": 1, "b": {"nested": True}})
    out = call_primitive(
        registry, "load_json", {"file": file, "name": "config", "description": "config"}
    )
    assert out.status == "ok"
    assert out.handle.kind == "dict"
    data = registry.materialize(out.handle.handle)
    assert data == {"a": 1, "b": {"nested": True}}


def test_load_json_preserves_nested_structure(registry, write_json):
    file = write_json(
        "orders.json",
        [{"id": 1, "customer": {"email": "a@example.com"}, "items": [{"sku": "x"}]}],
    )
    out = call_primitive(
        registry, "load_json", {"file": file, "name": "orders", "description": "orders"}
    )
    rows = registry.materialize(out.handle.handle)
    assert rows[0]["customer"] == {"email": "a@example.com"}
    assert rows[0]["items"] == [{"sku": "x"}]


def test_load_json_scalar_top_level_raises_load_error(registry, write_json):
    file = write_json("scalar.json", 42)
    with pytest.raises(LoadError):
        call_primitive(
            registry, "load_json", {"file": file, "name": "scalar", "description": "d"}
        )


def test_load_json_malformed_raises_load_error(registry, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    registry.files.register("bad.json", path)
    with pytest.raises(LoadError):
        call_primitive(
            registry, "load_json", {"file": "bad.json", "name": "bad", "description": "d"}
        )


def test_load_json_unregistered_name_raises(registry):
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            registry, "load_json", {"file": "nope", "name": "n", "description": "d"}
        )


def test_load_json_no_file_registry_configured_raises():
    bare_registry = DataRegistry()  # no files=FileRegistry() at all
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            bare_registry, "load_json", {"file": "anything", "name": "n", "description": "d"}
        )


def test_load_json_registered_file_later_missing_raises_load_error(registry, write_json, tmp_path):
    file = write_json("gone.json", [{"id": 1}])
    (tmp_path / "gone.json").unlink()
    with pytest.raises(LoadError):
        call_primitive(
            registry, "load_json", {"file": file, "name": "n", "description": "d"}
        )


def test_load_json_remove_envelope_unwraps_single_list_key(registry, write_json):
    file = write_json(
        "sf.json",
        {"totalSize": 2, "done": True, "records": [{"Id": "1"}, {"Id": "2"}]},
    )
    out = call_primitive(
        registry,
        "load_json",
        {"file": file, "remove_envelope": True, "name": "records", "description": "d"},
    )
    assert out.status == "ok"
    assert out.handle.kind == "list"
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"Id": "1"}, {"Id": "2"}]


def test_load_json_remove_envelope_false_keeps_dict_handle(registry, write_json):
    file = write_json("sf.json", {"totalSize": 1, "records": [{"Id": "1"}]})
    out = call_primitive(
        registry, "load_json", {"file": file, "name": "sf", "description": "d"}
    )
    assert out.handle.kind == "dict"


def test_load_json_remove_envelope_on_bare_array_is_a_no_op(registry, write_json):
    file = write_json("orders.json", [{"id": 1}])
    out = call_primitive(
        registry,
        "load_json",
        {"file": file, "remove_envelope": True, "name": "orders", "description": "d"},
    )
    assert out.handle.kind == "list"
    assert registry.materialize(out.handle.handle) == [{"id": 1}]


def test_load_json_remove_envelope_multiple_list_keys_raises(registry, write_json):
    file = write_json("bad.json", {"gold": ["a", "b"], "silver": ["c"]})
    with pytest.raises(LoadError):
        call_primitive(
            registry,
            "load_json",
            {"file": file, "remove_envelope": True, "name": "bad", "description": "d"},
        )


def test_load_json_remove_envelope_no_list_key_raises(registry, write_json):
    file = write_json("bad.json", {"id": "1", "name": "a"})
    with pytest.raises(LoadError):
        call_primitive(
            registry,
            "load_json",
            {"file": file, "remove_envelope": True, "name": "bad", "description": "d"},
        )


def test_load_json_remove_envelope_nested_dict_sibling_raises(registry, write_json):
    file = write_json("ticket.json", {"id": "T-1", "comments": ["hi"], "reporter": {"name": "a"}})
    with pytest.raises(LoadError):
        call_primitive(
            registry,
            "load_json",
            {"file": file, "remove_envelope": True, "name": "ticket", "description": "d"},
        )


# --- export_handle -----------------------------------------------------


def test_export_handle_writes_csv_for_list(registry, tmp_path):
    handle = _load_records(registry, [{"name": "Alice", "age": 34}, {"name": "Bob", "age": 29}])
    dest = tmp_path / "out.csv"
    registry.files.register_output("out.csv", dest)

    out = call_primitive(
        registry, "export_handle", {"handle": handle.handle, "destination": "out.csv"}
    )
    assert out.status == "ok"
    assert out.result == {
        "handle": handle.handle,
        "destination": "out.csv",
        "format": "csv",
        "rows": 2,
    }
    assert dest.read_text() == "name,age\nAlice,34\nBob,29\n"


def test_export_handle_infers_json_from_destination_extension_for_list(registry, tmp_path):
    handle = _load_records(registry, [{"name": "Alice"}])
    dest = tmp_path / "out.json"
    registry.files.register_output("out.json", dest)

    out = call_primitive(
        registry, "export_handle", {"handle": handle.handle, "destination": "out.json"}
    )
    assert out.result["format"] == "json"
    assert '"name": "Alice"' in dest.read_text()


def test_export_handle_defaults_to_json_for_dict_handle(registry, tmp_path):
    handle = registry.create(
        "dict",
        {"a": 1, "b": 2},
        name=_fixture_name(),
        description="d",
        created_by="fixture",
        value_shape="scalar",
        key_arity=1,
    )
    dest = tmp_path / "out.data"  # no recognized extension -> falls back to kind default
    registry.files.register_output("out.data", dest)

    out = call_primitive(
        registry, "export_handle", {"handle": handle.handle, "destination": "out.data"}
    )
    assert out.result["format"] == "json"
    assert dest.read_text() == '{\n  "a": 1,\n  "b": 2\n}'


def test_export_handle_csv_requested_for_dict_raises(registry, tmp_path):
    handle = registry.create(
        "dict",
        {"a": 1},
        name=_fixture_name(),
        description="d",
        created_by="fixture",
        value_shape="scalar",
        key_arity=1,
    )
    registry.files.register_output("out.csv", tmp_path / "out.csv")
    with pytest.raises(TypeMismatchError):
        call_primitive(
            registry,
            "export_handle",
            {"handle": handle.handle, "destination": "out.csv", "format": "csv"},
        )


def test_export_handle_unregistered_destination_raises(registry):
    handle = _load_records(registry, [{"name": "Alice"}])
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            registry, "export_handle", {"handle": handle.handle, "destination": "nope.csv"}
        )


def test_export_handle_no_file_registry_configured_raises():
    bare_registry = DataRegistry()
    handle = _load_records(bare_registry, [{"name": "Alice"}])
    with pytest.raises(FileNotRegisteredError):
        call_primitive(
            bare_registry, "export_handle", {"handle": handle.handle, "destination": "out.csv"}
        )


def test_export_handle_write_failure_raises_export_error(registry, tmp_path):
    handle = _load_records(registry, [{"name": "Alice"}])
    # Register a destination whose parent exists now but is removed before export runs,
    # so the write itself fails (distinct from the destination never being registered).
    subdir = tmp_path / "gone"
    subdir.mkdir()
    dest = subdir / "out.csv"
    registry.files.register_output("out.csv", dest)
    subdir.rmdir()

    with pytest.raises(ExportError):
        call_primitive(registry, "export_handle", {"handle": handle.handle, "destination": "out.csv"})


# --- filter_where ------------------------------------------------------


def test_filter_where(registry):
    meta = _load_records(registry, [{"qty": 5}, {"qty": 0}, {"qty": 10}])
    out = call_primitive(
        registry,
        "filter_where",
        {
            "handle": meta.handle,
            "predicate": {"field": "qty", "op": ">", "value": 0},
            "name": "positive_qty",
            "description": "d",
        },
    )
    assert out.status == "ok"
    assert registry.materialize(out.handle.handle) == [{"qty": 5}, {"qty": 10}]


def test_filter_where_is_null(registry):
    meta = _load_records(registry, [{"qty": 5}, {"qty": None}, {}])
    out = call_primitive(
        registry,
        "filter_where",
        {
            "handle": meta.handle,
            "predicate": {"field": "qty", "op": "is_null"},
            "name": "missing_qty",
            "description": "d",
        },
    )
    assert out.status == "ok"
    assert registry.materialize(out.handle.handle) == [{"qty": None}, {}]


def test_filter_where_on_dict_handle_raises(registry):
    meta = registry.create(
        "dict", {"a": 1}, name=_fixture_name(), description="d", created_by="fixture"
    )
    with pytest.raises(TypeMismatchError):
        call_primitive(
            registry,
            "filter_where",
            {
                "handle": meta.handle,
                "predicate": {"field": "x", "op": "==", "value": 1},
                "name": "n",
                "description": "d",
            },
        )


# --- compute_field -------------------------------------------------------


def test_compute_field(registry):
    meta = _load_records(registry, [{"price": 10, "cost": 4}])
    out = call_primitive(
        registry,
        "compute_field",
        {
            "handle": meta.handle,
            "as": "margin",
            "op": "subtract",
            "left": {"field": "price"},
            "right": {"field": "cost"},
            "name": "with_margin",
            "description": "d",
        },
    )
    assert registry.materialize(out.handle.handle) == [
        {"price": 10, "cost": 4, "margin": 6}
    ]


# --- sort_by -------------------------------------------------------------


def test_sort_by_single_key_desc(registry):
    meta = _load_records(registry, [{"x": 1}, {"x": 3}, {"x": 2}])
    out = call_primitive(
        registry,
        "sort_by",
        {
            "handle": meta.handle,
            "keys": [{"field": "x", "order": "desc"}],
            "name": "sorted",
            "description": "d",
        },
    )
    assert [r["x"] for r in registry.materialize(out.handle.handle)] == [3, 2, 1]


def test_sort_by_nulls_last_regardless_of_direction(registry):
    meta = _load_records(registry, [{"x": 2}, {"x": None}, {"x": 1}])
    out = call_primitive(
        registry,
        "sort_by",
        {
            "handle": meta.handle,
            "keys": [{"field": "x", "order": "asc"}],
            "name": "sorted",
            "description": "d",
        },
    )
    assert [r["x"] for r in registry.materialize(out.handle.handle)] == [1, 2, None]


def test_sort_by_multi_key_stable():
    reg = DataRegistry()
    meta = _load_records(
        reg, [{"a": 1, "b": 2}, {"a": 1, "b": 1}, {"a": 0, "b": 5}]
    )
    out = call_primitive(
        reg,
        "sort_by",
        {
            "handle": meta.handle,
            "keys": [{"field": "a", "order": "asc"}, {"field": "b", "order": "asc"}],
            "name": "sorted",
            "description": "d",
        },
    )
    result = [(r["a"], r["b"]) for r in reg.materialize(out.handle.handle)]
    assert result == [(0, 5), (1, 1), (1, 2)]


# --- index_by / group_by (composite keys) -----------------------------


def test_index_by_single_key(registry):
    meta = _load_records(registry, [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 2}])
    out = call_primitive(
        registry,
        "index_by",
        {"handle": meta.handle, "key_fields": ["sku"], "name": "by_sku", "description": "d"},
    )
    assert out.handle.value_shape == "scalar"
    assert out.handle.key_arity == 1
    indexed = registry.materialize(out.handle.handle)
    assert indexed["A"] == {"sku": "A", "qty": 1}


def test_index_by_composite_key(registry):
    meta = _load_records(
        registry,
        [
            {"supplier": "acme", "sku": "A", "price": 10},
            {"supplier": "acme", "sku": "B", "price": 20},
        ],
    )
    out = call_primitive(
        registry,
        "index_by",
        {
            "handle": meta.handle,
            "key_fields": ["supplier", "sku"],
            "name": "by_supplier_sku",
            "description": "d",
        },
    )
    assert out.handle.key_arity == 2
    indexed = registry.materialize(out.handle.handle)
    assert indexed[("acme", "A")]["price"] == 10


def test_index_by_duplicate_key_raises(registry):
    meta = _load_records(registry, [{"sku": "A"}, {"sku": "A"}])
    with pytest.raises(DuplicateKeyError):
        call_primitive(
            registry,
            "index_by",
            {"handle": meta.handle, "key_fields": ["sku"], "name": "by_sku", "description": "d"},
        )


def test_index_by_missing_field_raises(registry):
    meta = _load_records(registry, [{"sku": "A"}, {"other": "B"}])
    with pytest.raises(FieldNotFoundError):
        call_primitive(
            registry,
            "index_by",
            {"handle": meta.handle, "key_fields": ["sku"], "name": "by_sku", "description": "d"},
        )


def test_group_by_composite_key(registry):
    meta = _load_records(
        registry,
        [
            {"region": "west", "sku": "A", "qty": 1},
            {"region": "west", "sku": "A", "qty": 2},
            {"region": "east", "sku": "A", "qty": 5},
        ],
    )
    out = call_primitive(
        registry,
        "group_by",
        {
            "handle": meta.handle,
            "key_fields": ["region", "sku"],
            "name": "by_region_sku",
            "description": "d",
        },
    )
    assert out.handle.value_shape == "list"
    grouped = registry.materialize(out.handle.handle)
    assert len(grouped[("west", "A")]) == 2
    assert len(grouped[("east", "A")]) == 1


# --- join_lookup -----------------------------------------------------


def test_join_lookup_left_keeps_unmatched(registry):
    base = _load_records(registry, [{"sku": "A", "qty": 5}, {"sku": "B", "qty": 3}])
    catalog = _load_records(registry, [{"sku": "A", "name": "Widget"}])
    idx = call_primitive(
        registry,
        "index_by",
        {"handle": catalog.handle, "key_fields": ["sku"], "name": "catalog_idx", "description": "d"},
    )
    out = call_primitive(
        registry,
        "join_lookup",
        {
            "base_handle": base.handle,
            "index_handle": idx.handle.handle,
            "on": ["sku"],
            "how": "left",
            "name": "joined",
            "description": "d",
        },
    )
    result = registry.materialize(out.handle.handle)
    assert result[0] == {"sku": "A", "qty": 5, "name": "Widget"}
    assert result[1] == {"sku": "B", "qty": 3}


def test_join_lookup_inner_drops_unmatched(registry):
    base = _load_records(registry, [{"sku": "A"}, {"sku": "B"}])
    catalog = _load_records(registry, [{"sku": "A", "name": "Widget"}])
    idx = call_primitive(
        registry,
        "index_by",
        {"handle": catalog.handle, "key_fields": ["sku"], "name": "catalog_idx", "description": "d"},
    )
    out = call_primitive(
        registry,
        "join_lookup",
        {
            "base_handle": base.handle,
            "index_handle": idx.handle.handle,
            "on": ["sku"],
            "how": "inner",
            "name": "joined",
            "description": "d",
        },
    )
    result = registry.materialize(out.handle.handle)
    assert len(result) == 1
    assert result[0]["sku"] == "A"


def test_join_lookup_with_group_by_fanout(registry):
    base = _load_records(registry, [{"sku": "A"}])
    events = _load_records(
        registry, [{"sku": "A", "event": "click"}, {"sku": "A", "event": "view"}]
    )
    grouped = call_primitive(
        registry,
        "group_by",
        {"handle": events.handle, "key_fields": ["sku"], "name": "events_by_sku", "description": "d"},
    )
    out = call_primitive(
        registry,
        "join_lookup",
        {
            "base_handle": base.handle,
            "index_handle": grouped.handle.handle,
            "on": ["sku"],
            "how": "inner",
            "fields": ["event"],
            "name": "joined",
            "description": "d",
        },
    )
    result = registry.materialize(out.handle.handle)
    assert len(result) == 2
    assert {r["event"] for r in result} == {"click", "view"}


def test_join_lookup_wrong_index_kind_raises(registry):
    base = _load_records(registry, [{"sku": "A"}])
    not_a_dict = _load_records(registry, [{"sku": "A"}])
    with pytest.raises(TypeMismatchError):
        call_primitive(
            registry,
            "join_lookup",
            {
                "base_handle": base.handle,
                "index_handle": not_a_dict.handle,
                "on": ["sku"],
                "name": "joined",
                "description": "d",
            },
        )


# --- join_lookup against a scalar-valued index ---------------------------
#
# Regression set for the bug where join_lookup assumed every index value was
# a record. A priority_reduce index holds bare values, so `if f in matched`
# raised TypeError, which dispatch sanitized into an unactionable
# INTERNAL_ERROR -- see principles.md, "Any INTERNAL_ERROR Is a Missing
# Precondition".


def _usage_marker_index(registry, present_keys, name_suffix="idx"):
    """The exact shape priority_reduce produces: one bare value per key."""
    events = _load_records(registry, [{"account_id": k, "mark": 1} for k in present_keys])
    out = call_primitive(
        registry,
        "priority_reduce",
        {
            "handle": events.handle,
            "key_fields": ["account_id"],
            "value_field": "mark",
            "priority": [1],
            "name": f"usage_{name_suffix}",
            "description": "one marker per account",
        },
    )
    return out.handle


def test_registry_infers_value_type_record_vs_scalar(registry):
    records = _load_records(registry, [{"account_id": "A", "mark": 1}])
    by_key = call_primitive(
        registry,
        "index_by",
        {"handle": records.handle, "key_fields": ["account_id"], "name": "i", "description": "d"},
    )
    assert by_key.handle.value_type == "record"

    scalar_index = _usage_marker_index(registry, ["A"], name_suffix="vt")
    assert scalar_index.value_type == "scalar"
    # value_shape says arity and is identical for both -- which is precisely
    # why value_type has to exist separately.
    assert by_key.handle.value_shape == scalar_index.value_shape == "scalar"


def test_join_lookup_binds_scalar_index_value_to_named_field(registry):
    base = _load_records(registry, [{"account_id": "A"}, {"account_id": "B"}])
    index = _usage_marker_index(registry, ["A"])

    out = call_primitive(
        registry,
        "join_lookup",
        {
            "base_handle": base.handle,
            "index_handle": index.handle,
            "on": ["account_id"],
            "fields": ["has_usage"],
            "how": "left",
            "name": "joined",
            "description": "d",
        },
    )
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"account_id": "A", "has_usage": 1}, {"account_id": "B"}]


@pytest.mark.parametrize("fields", [None, ["a", "b"]])
def test_join_lookup_scalar_index_needs_exactly_one_field(registry, fields):
    base = _load_records(registry, [{"account_id": "A"}])
    index = _usage_marker_index(registry, ["A"])
    params = {
        "base_handle": base.handle,
        "index_handle": index.handle,
        "on": ["account_id"],
        "how": "left",
        "name": "joined",
        "description": "d",
    }
    if fields is not None:
        params["fields"] = fields

    with pytest.raises(TypeMismatchError):
        call_primitive(registry, "join_lookup", params)


# --- producer x consumer conformance -------------------------------------
#
# The real guard against this class of bug: every primitive that emits a dict
# handle, fed to every primitive that consumes one. A pair may succeed or
# raise a typed DaleError -- it may never surface InternalError, which by
# definition means a missing precondition check rather than a bad call.


def _dict_producers(registry):
    records = _load_records(
        registry, [{"k": "A", "v": 1}, {"k": "A", "v": 2}, {"k": "B", "v": 3}]
    )
    common = {"handle": records.handle, "key_fields": ["k"], "description": "d"}
    yield "group_by", call_primitive(
        registry, "group_by", {**common, "name": "p_group"}
    ).handle
    yield "priority_reduce", call_primitive(
        registry,
        "priority_reduce",
        {**common, "value_field": "v", "priority": [1, 2, 3], "name": "p_reduce"},
    ).handle
    unique = _load_records(registry, [{"k": "A", "v": 1}, {"k": "B", "v": 3}])
    yield "index_by", call_primitive(
        registry,
        "index_by",
        {"handle": unique.handle, "key_fields": ["k"], "name": "p_index", "description": "d"},
    ).handle


def _dict_consumer_params(consumer, base_handle, index_handle, name):
    if consumer == "join_lookup":
        return {
            "base_handle": base_handle,
            "index_handle": index_handle,
            "on": ["k"],
            "fields": ["v"],
            "how": "left",
            "name": name,
            "description": "d",
        }
    if consumer == "graph_walk_resolve":
        return {
            "handle": base_handle,
            "index_handle": index_handle,
            "start_field": "k",
            "next_field": "v",
            "name": name,
            "description": "d",
        }
    return {  # dict_diff
        "previous_handle": index_handle,
        "current_handle": index_handle,
        "name": name,
        "description": "d",
    }


@pytest.mark.parametrize("consumer", ["join_lookup", "graph_walk_resolve", "dict_diff"])
def test_dict_consumers_never_raise_internal_error(registry, consumer):
    """A wrong-but-plausible pairing must always be a typed, actionable
    DaleError. InternalError here means a primitive dereferenced something it
    never checked -- the model cannot act on that, so it is a DALE defect."""
    base = _load_records(registry, [{"k": "A"}, {"k": "B"}])
    for i, (producer, index_meta) in enumerate(_dict_producers(registry)):
        params = _dict_consumer_params(
            consumer, base.handle, index_meta.handle, f"out_{consumer}_{i}"
        )
        try:
            call_primitive(registry, consumer, params)
        except InternalError as exc:  # pragma: no cover - the failure we're guarding
            pytest.fail(
                f"{consumer} raised InternalError on a {producer} index "
                f"(value_type={index_meta.value_type!r}): {exc}"
            )
        except DaleError:
            pass  # typed and actionable -- the acceptable outcome


# --- peek / describe ----------------------------------------------------


def test_peek_caps_n_at_hard_maximum(registry):
    meta = _load_records(registry, [{"x": i} for i in range(100)])
    out = call_primitive(registry, "peek", {"handle": meta.handle, "n": 1000})
    assert len(out.result["sample"]) == 50


def _grouped_dict(registry, *, rows=10_000, buckets=3, name="by_region_dict"):
    """A group_by dict whose buckets are far larger than any sample should be
    — the shape that made peek return the whole handle."""
    records = [
        {"id": i, "region": f"r{i % buckets}", "note": "x" * 40} for i in range(rows)
    ]
    source = _load_records(registry, records)
    call_primitive(
        registry,
        "group_by",
        {
            "handle": source.handle,
            "key_fields": ["region"],
            "name": name,
            "description": "rows grouped by region",
        },
    )
    return name


def _sample_bytes(result) -> int:
    return len(json.dumps(result["sample"]).encode("utf-8"))


def _nested_dict(registry, *, name, keys=6, per_key=200):
    """A dict handle whose *values* are single nested records rather than
    buckets — the shape a record cap has nothing to say about, and the one
    `load_json` produces from an ordinary top-level JSON object."""
    records = [
        {"id": i, "doc": {"rows": [{"n": j, "text": "z" * 80} for j in range(per_key)]}}
        for i in range(keys)
    ]
    source = _load_records(registry, records)
    call_primitive(
        registry,
        "index_by",
        {"handle": source.handle, "key_fields": ["id"], "name": name, "description": "d"},
    )
    return name


def test_peek_on_a_dict_handle_is_bounded_by_bytes_not_keys(registry):
    """The regression that started this: `peek(n=3)` on a group_by dict capped
    keys and nothing else, so each of the 3 keys handed back its entire bucket
    — 852,329 bytes (~213k tokens) measured on this exact shape, against 309
    for the same peek on the underlying list."""
    handle = _grouped_dict(registry)
    out = call_primitive(registry, "peek", {"handle": handle, "n": 3})

    assert set(out.result["sample"]) == {"'r0'", "'r1'", "'r2'"}
    assert _sample_bytes(out.result) <= _PEEK_MAX_BYTES


def test_peek_is_bounded_when_the_value_is_nested_rather_than_a_bucket(registry):
    """The hole a record cap left open, and the reason the ceiling is now
    measured in bytes: an index_by/load_json dict whose values are single
    nested records was passed through untouched, because "one item" was read
    as a bound on size when it is only a bound on count. Measured at 179,509
    bytes before, with nothing in the payload admitting it."""
    handle = _nested_dict(registry, name="docs_dict")
    out = call_primitive(registry, "peek", {"handle": handle, "n": 3})

    assert _sample_bytes(out.result) <= _PEEK_MAX_BYTES
    assert out.result["truncated"] is True


def test_peek_is_bounded_when_a_single_leaf_is_enormous(registry):
    """Same argument one level further down: a lone 200KB string field is one
    item too. A leaf that doesn't fit is trimmed and says by how much."""
    meta = _load_records(registry, [{"id": 1, "blob": "q" * 200_000}])
    out = call_primitive(registry, "peek", {"handle": meta.handle, "n": 1})

    assert _sample_bytes(out.result) <= _PEEK_MAX_BYTES
    assert out.result["truncated"] is True
    assert out.result["sample"][0]["blob"].endswith("more chars)")
    assert out.result["sample"][0]["id"] == 1  # the rest of the record survives


def test_peek_stays_bounded_at_max_n_over_many_large_buckets(registry):
    """The budget holds at the top of the range too — 50 keys asked for, each
    a 200-record bucket, still fits the same ceiling."""
    handle = _grouped_dict(registry, rows=10_000, buckets=50)
    out = call_primitive(registry, "peek", {"handle": handle, "n": 1000})

    assert _sample_bytes(out.result) <= _PEEK_MAX_BYTES
    assert len(out.result["sample"]) > 1  # not one key hogging the whole budget


def test_peek_preserves_bucket_shape_while_truncating(registry):
    """A truncated bucket is still a *list* of records, because the shape
    (key -> list, vs key -> record, vs key -> scalar) is what peek exists to
    teach. A cap that collapsed the bucket to a single record would leave the
    model unable to tell a group_by from an index_by. The first two assertions
    pin that truncation really happened — without them this test passes just
    as happily against the unbounded version it exists to guard."""
    handle = _grouped_dict(registry)
    out = call_primitive(registry, "peek", {"handle": handle, "n": 3})

    bucket = out.result["sample"]["'r0'"]
    assert out.result["truncated"] is True
    assert bucket[-1].startswith("+")  # the in-place marker, not a record
    assert isinstance(bucket, list)
    assert all(isinstance(r, dict) and "region" in r for r in bucket[:-1])


def test_peek_marks_every_truncation_in_place_with_its_real_size(registry):
    """A shortened collection that looks complete is worse than the original
    bug: the model would conclude "r0 has 3 rows" from a sample it had no way
    to know was partial. Every cut says how much it cut, next to where it cut,
    so the answer is unambiguous at any depth."""
    small = _grouped_dict(registry, rows=9, buckets=3)  # 3 rows per bucket
    untruncated = call_primitive(registry, "peek", {"handle": small, "n": 3})
    assert "truncated" not in untruncated.result  # nothing was cut, nothing claimed

    handle = _grouped_dict(registry, rows=9_000, buckets=3, name="big_dict")
    out = call_primitive(registry, "peek", {"handle": handle, "n": 3})

    bucket = out.result["sample"]["'r0'"]
    assert bucket[-1] == f"+{3_000 - 3} more items"  # 3 shown of a 3,000-row bucket
    assert out.result["truncated"] is True


def test_peek_says_how_many_keys_it_did_not_show(registry):
    """Key truncation used to be the silent one: 3 of 1,000 keys, with nothing
    saying so. For buckets a missing count costs a size estimate; for keys it
    costs the answer, since "there are 3 regions" is exactly the conclusion a
    3-key sample invites."""
    source = _load_records(registry, [{"id": i, "v": i} for i in range(1_000)])
    call_primitive(
        registry,
        "index_by",
        {"handle": source.handle, "key_fields": ["id"], "name": "wide_dict", "description": "d"},
    )
    out = call_primitive(registry, "peek", {"handle": "wide_dict", "n": 3})

    assert out.result["sample"]["..."] == "+997 more keys"
    assert out.result["truncated"] is True


def test_peek_on_an_index_by_dict_keeps_one_record_per_key(registry):
    """index_by values are single records, not buckets — each key still maps to
    one whole record, with no per-record truncation and no list wrapper."""
    source = _load_records(registry, [{"id": i, "v": i} for i in range(3)])
    call_primitive(
        registry,
        "index_by",
        {"handle": source.handle, "key_fields": ["id"], "name": "by_id", "description": "d"},
    )
    out = call_primitive(registry, "peek", {"handle": "by_id", "n": 3})

    assert out.result["sample"] == {
        "0": {"id": 0, "v": 0},
        "1": {"id": 1, "v": 1},
        "2": {"id": 2, "v": 2},
    }
    assert "truncated" not in out.result


def test_peek_on_a_priority_reduce_dict_is_unchanged(registry):
    """priority_reduce values are bare scalars — they must stay bare scalars,
    not be wrapped in a list or a truncation envelope."""
    source = _load_records(
        registry,
        [{"k": "a", "s": "gold"}, {"k": "a", "s": "silver"}, {"k": "b", "s": "silver"}],
    )
    call_primitive(
        registry,
        "priority_reduce",
        {
            "handle": source.handle,
            "key_fields": ["k"],
            "value_field": "s",
            "priority": ["gold", "silver"],
            "name": "best_dict",
            "description": "d",
        },
    )
    out = call_primitive(registry, "peek", {"handle": "best_dict", "n": 5})

    assert out.result["sample"] == {"'a'": "gold", "'b'": "silver"}
    assert "truncated" not in out.result


def test_peek_on_list_and_set_handles_is_unchanged(registry):
    """These two were always honest — n records, no markers. `n` is a request,
    not a cut: the handle's full size is in its own metadata, so a list peek
    claims nothing about what it didn't show."""
    records = _load_records(registry, [{"x": i} for i in range(100)])
    list_out = call_primitive(registry, "peek", {"handle": records.handle, "n": 4})
    assert list_out.result["sample"] == [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]
    assert "truncated" not in list_out.result

    registry.create(
        "set", {"a", "b", "c"}, name="tags_set", description="d", created_by="fixture"
    )
    set_out = call_primitive(registry, "peek", {"handle": "tags_set", "n": 2})
    assert len(set_out.result["sample"]) == 2
    assert "truncated" not in set_out.result


def test_peek_leaks_no_counts_and_no_keys_under_privacy_mode():
    """peek must not hand back what describe withholds. Under privacy_mode
    describe keeps distinct_count and drops top_k entirely — the value -> count
    pairs are precisely what it suppresses — so a peek reporting "'cancer': 100
    rows, 'diabetes': 200" would reconstruct it verbatim from a different tool.
    Keys go too, and that closes a leak older than the byte budget: redaction
    recursed into values but never into keys, so an index_by on a patient id
    printed every real identifier as the key beside its redacted record."""
    registry = DataRegistry(privacy_mode=True)
    records = [
        {"patient": f"NHS-{1000 + i}", "dx": "cancer" if i < 100 else "diabetes"}
        for i in range(300)
    ]
    source = _load_records(registry, records)
    call_primitive(
        registry,
        "group_by",
        {"handle": source.handle, "key_fields": ["dx"], "name": "by_dx", "description": "d"},
    )
    call_primitive(
        registry,
        "index_by",
        {
            "handle": source.handle,
            "key_fields": ["patient"],
            "name": "by_patient",
            "description": "d",
        },
    )

    grouped = call_primitive(registry, "peek", {"handle": "by_dx", "n": 3}).result
    indexed = call_primitive(registry, "peek", {"handle": "by_patient", "n": 2}).result
    serialized = json.dumps(grouped) + json.dumps(indexed)

    assert set(grouped["sample"]) == {"<key 1>", "<key 2>"}  # not 'cancer'/'diabetes'
    assert list(indexed["sample"])[0] == "<key 1>"  # not 'NHS-1000'
    assert "NHS-" not in serialized and "cancer" not in serialized
    # No bucket size, by any route: not as a count, not as a "+N more".
    assert "100" not in serialized and "200" not in serialized
    assert grouped["sample"]["<key 1>"][-1] == "<truncated>"  # still visibly partial
    assert grouped["sample"]["<key 1>"][0] == {"patient": "<str>", "dx": "<str>"}
    assert "privacy_mode is enabled" in grouped["note"]


def _adversarial_handles(registry):
    """One handle per shape that has broken, or could break, a peek cap —
    kept in one place so the ceiling is asserted against the *class* of
    payload rather than against the two examples that happened to be
    reported. Every entry here is reachable through documented primitives."""
    deep = {"level": 0}
    node = deep
    for i in range(1, 40):
        node["child"] = {"level": i}
        node = node["child"]
    node["leaf"] = "d" * 50_000

    registry.create(
        "list",
        [{"id": i, "orders": [{"o": j, "pad": "y" * 60} for j in range(200)]} for i in range(40)],
        name="nested_list", description="records carrying arrays", created_by="fixture",
    )
    registry.create(
        "dict",
        {f"k{i}": [{"a": "x" * 100, "b": j} for j in range(300)] for i in range(40)},
        name="bucket_dict", description="group_by-shaped", created_by="fixture",
        value_shape="list",
    )
    registry.create(
        "dict",
        {f"k{i}": {"doc": {"rows": [{"n": j} for j in range(300)]}} for i in range(40)},
        name="nested_value_dict", description="index_by/load_json-shaped", created_by="fixture",
    )
    registry.create(
        "dict", {f"k{i}": "v" * 20_000 for i in range(40)},
        name="scalar_dict", description="priority_reduce-shaped, huge scalars",
        created_by="fixture",
    )
    registry.create(
        "dict", {f"k{i}": i for i in range(100_000)},
        name="wide_dict", description="very many keys", created_by="fixture",
    )
    registry.create(
        "dict", {"x" * 5_000: {"v": 1}},
        name="long_key_dict", description="one enormous key", created_by="fixture",
    )
    registry.create(
        "list", [{"id": 1, "blob": "q" * 500_000}],
        name="huge_leaf_list", description="one enormous scalar", created_by="fixture",
    )
    registry.create(
        "list", [deep], name="deep_list", description="deeply nested", created_by="fixture",
    )
    registry.create(
        "set", {"t" * 2_000 + str(i) for i in range(500)},
        name="big_set", description="long set members", created_by="fixture",
    )
    return [
        "nested_list", "bucket_dict", "nested_value_dict", "scalar_dict", "wide_dict",
        "long_key_dict", "huge_leaf_list", "deep_list", "big_set",
    ]


@pytest.mark.parametrize("n", [0, 1, 3, 50, 1000])
@pytest.mark.parametrize("privacy_mode", [False, True])
def test_peek_stays_within_its_ceiling_for_every_shape(n, privacy_mode):
    """The guard that makes the cap a property rather than a patch. Both
    previous versions of this cap were correct for the shape that had been
    reported and unbounded for the next one — keys, then records — so the
    ceiling is asserted here against every shape at once, at both ends of the
    n range and under both privacy settings. A new shape that escapes it fails
    this test rather than a production run."""
    registry = DataRegistry(privacy_mode=privacy_mode)
    for handle in _adversarial_handles(registry):
        out = call_primitive(registry, "peek", {"handle": handle, "n": n})
        assert _sample_bytes(out.result) <= _PEEK_MAX_BYTES, handle


def test_describe_top_k_values_are_bounded_by_bytes(registry):
    """_DESCRIBE_TOP_K caps how many values come back and says nothing about
    how large they are — 20 records holding 100KB strings returned a
    1,000,370-byte describe. Same budget, same in-place marker as peek."""
    meta = _load_records(registry, [{"blob": "q" * 100_000}] * 20)
    out = call_primitive(registry, "describe", {"handle": meta.handle, "field": "blob"})

    assert len(json.dumps(out.result).encode("utf-8")) <= _DESCRIBE_MAX_BYTES + 512
    assert out.result["truncated"] is True
    assert out.result["top_k"][0]["value"].endswith("more chars)")
    assert out.result["top_k"][0]["count"] == 20  # the count itself is untouched


def test_describe_numeric_field(registry):
    meta = _load_records(registry, [{"x": 1}, {"x": 2}, {"x": 3}, {"x": None}])
    out = call_primitive(registry, "describe", {"handle": meta.handle, "field": "x"})
    assert out.result["min"] == 1
    assert out.result["max"] == 3
    assert out.result["count"] == 3
    assert out.result["null_rate"] == pytest.approx(0.25)


def test_describe_categorical_field_top_k(registry):
    meta = _load_records(registry, [{"cat": "a"}] * 5 + [{"cat": "b"}] * 2)
    out = call_primitive(registry, "describe", {"handle": meta.handle, "field": "cat"})
    assert out.result["distinct_count"] == 2
    assert out.result["top_k"][0] == {"value": "a", "count": 5}


def test_describe_schema_summary_when_no_field_given(registry):
    meta = _load_records(registry, [{"a": 1, "b": "x"}])
    out = call_primitive(registry, "describe", {"handle": meta.handle})
    assert out.result["fields"] == {"a": "int", "b": "str"}


# --- privacy_mode ---------------------------------------------------------


def test_peek_redacts_values_under_privacy_mode():
    registry = DataRegistry(privacy_mode=True)
    meta = _load_records(registry, [{"sku": "A1", "price": 9.99, "in_stock": True}])
    out = call_primitive(registry, "peek", {"handle": meta.handle, "n": 1})
    assert out.result["sample"] == [{"sku": "<str>", "price": "<float>", "in_stock": "<bool>"}]
    assert "note" in out.result


def test_peek_unredacted_when_privacy_mode_off(registry):
    meta = _load_records(registry, [{"sku": "A1"}])
    out = call_primitive(registry, "peek", {"handle": meta.handle, "n": 1})
    assert out.result["sample"] == [{"sku": "A1"}]
    assert "note" not in out.result


def test_describe_numeric_stats_still_real_under_privacy_mode():
    registry = DataRegistry(privacy_mode=True)
    meta = _load_records(registry, [{"x": 1}, {"x": 2}, {"x": 3}])
    out = call_primitive(registry, "describe", {"handle": meta.handle, "field": "x"})
    # Numeric aggregate stats are never individual-record content -- stay real.
    assert out.result["min"] == 1
    assert out.result["max"] == 3


def test_describe_categorical_top_k_redacted_under_privacy_mode():
    registry = DataRegistry(privacy_mode=True)
    meta = _load_records(registry, [{"cat": "a"}] * 5 + [{"cat": "b"}] * 2)
    out = call_primitive(registry, "describe", {"handle": meta.handle, "field": "cat"})
    assert out.result["distinct_count"] == 2  # a count, not a value -- stays
    assert "top_k" not in out.result
    assert "note" in out.result


def test_invalid_params_error_sanitized_under_privacy_mode():
    registry = DataRegistry(privacy_mode=True)
    meta = _load_records(registry, [{"x": 1}])
    with pytest.raises(InvalidParamsError) as exc_info:
        call_primitive(
            registry,
            "filter_where",
            {"handle": meta.handle, "predicate": {"field": "x", "op": "=="}},
        )
    assert "input_value" not in str(exc_info.value)


# --- release_handle -----------------------------------------------------


def test_release_handle(registry):
    meta = _load_records(registry, [1, 2, 3])
    out = call_primitive(registry, "release_handle", {"handle": meta.handle})
    assert out.result["handles_remaining"] == 0
    with pytest.raises(HandleNotFoundError):
        registry.get(meta.handle)


# --- dispatch-level cross-cutting behavior -------------------------------


def test_unknown_primitive_raises(registry):
    with pytest.raises(PrimitiveNotFoundError):
        call_primitive(registry, "does_not_exist", {})


def test_invalid_params_raises(registry):
    meta = _load_records(registry, [{"x": 1}])
    with pytest.raises(InvalidParamsError):
        call_primitive(registry, "filter_where", {"handle": meta.handle})  # missing predicate


def test_unexpected_exception_sanitized_to_internal_error(registry):
    """A primitive that raises a raw, unexpected exception (not a DaleError)
    must never leak its original message back to the caller (objections #6)."""

    class _BrokenParams(BaseModel):
        pass

    def _broken(reg, params):
        raise RuntimeError("leaked internal detail: /etc/shadow contents")

    register_primitive("_test_broken_primitive", _broken, _BrokenParams)

    with pytest.raises(InternalError) as exc_info:
        call_primitive(registry, "_test_broken_primitive", {})

    assert "leaked internal detail" not in str(exc_info.value)
    assert "/etc/shadow" not in str(exc_info.value)
