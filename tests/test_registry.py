from __future__ import annotations

import pytest
from pydantic import ValidationError

from dale.errors import (
    DuplicateHandleError,
    HandleNotFoundError,
    InvalidParamsError,
    RegistryLimitError,
    ToolCallLimitError,
)
from dale.registry import DataHandle, DataRegistry, RegistryLimits


def test_create_and_get():
    reg = DataRegistry()
    meta = reg.create("list", [1, 2, 3], name="nums", description="three numbers", created_by="test")
    assert meta.type == "list"
    assert meta.size == 3
    assert meta.name == "nums"
    assert reg.get(meta.name) == [1, 2, 3]


def test_meta_matches_get():
    reg = DataRegistry()
    meta = reg.create(
        "list", [{"a": 1}], name="records", description="one record", created_by="test"
    )
    assert reg.meta(meta.name) == meta


def test_handle_is_the_caller_supplied_name():
    reg = DataRegistry()
    m1 = reg.create("list", [1], name="a_list", description="d", created_by="t")
    m2 = reg.create("list", [2], name="b_list", description="d", created_by="t")
    m3 = reg.create("dict", {"a": 1}, name="a_dict", description="d", created_by="t")
    assert m1.name == "a_list"
    assert m2.name == "b_list"
    assert m3.name == "a_dict"


def test_duplicate_name_raises_duplicate_handle_error():
    reg = DataRegistry()
    reg.create("list", [1], name="taken", description="d", created_by="t")
    with pytest.raises(DuplicateHandleError):
        reg.create("list", [2], name="taken", description="d", created_by="t")


def test_name_that_is_not_a_python_identifier_raises():
    reg = DataRegistry()
    with pytest.raises(InvalidParamsError):
        reg.create("list", [1], name="not-an-identifier", description="d", created_by="t")


def test_name_that_is_a_python_keyword_raises():
    reg = DataRegistry()
    with pytest.raises(InvalidParamsError):
        reg.create("list", [1], name="class", description="d", created_by="t")


def test_release_frees_name_for_reuse():
    reg = DataRegistry()
    meta = reg.create("list", [1, 2], name="reused", description="d", created_by="test")
    reg.release(meta.name)
    with pytest.raises(HandleNotFoundError):
        reg.get(meta.name)
    with pytest.raises(HandleNotFoundError):
        reg.meta(meta.name)
    with pytest.raises(HandleNotFoundError):
        reg.release(meta.name)
    reg.create("list", [3], name="reused", description="d", created_by="test")


def test_get_unknown_handle_raises():
    reg = DataRegistry()
    with pytest.raises(HandleNotFoundError):
        reg.get("nonexistent")


def test_handle_count_tracks_create_and_release():
    reg = DataRegistry()
    assert reg.handle_count() == 0
    m1 = reg.create("list", [1], name="a", description="d", created_by="t")
    assert reg.handle_count() == 1
    reg.release(m1.name)
    assert reg.handle_count() == 0


def test_max_handles_limit_enforced():
    reg = DataRegistry(limits=RegistryLimits(max_handles=2))
    reg.create("list", [1], name="a", description="d", created_by="t")
    reg.create("list", [2], name="b", description="d", created_by="t")
    with pytest.raises(RegistryLimitError):
        reg.create("list", [3], name="c", description="d", created_by="t")


def test_max_tool_calls_limit_enforced():
    reg = DataRegistry(limits=RegistryLimits(max_tool_calls=2))
    reg.record_call()
    reg.record_call()
    with pytest.raises(ToolCallLimitError):
        reg.record_call()


def test_avg_record_bytes_sampled_for_nonempty_list():
    reg = DataRegistry()
    meta = reg.create(
        "list", [{"a": 1}, {"a": 2}], name="a", description="d", created_by="t"
    )
    assert meta.avg_record_bytes is not None
    assert meta.avg_record_bytes > 0


def test_avg_record_bytes_none_for_empty_list():
    reg = DataRegistry()
    meta = reg.create("list", [], name="a", description="d", created_by="t")
    assert meta.avg_record_bytes is None


def test_materialize_returns_full_value():
    reg = DataRegistry()
    meta = reg.create("list", [1, 2, 3], name="a", description="d", created_by="t")
    assert reg.materialize(meta.name) == [1, 2, 3]


# --- the serialized wire shape of a DataHandle -----------------------------
#
# `name`/`type` are not merely the Python attribute names: a DataHandle is
# model_dump()ed into every OperationOutput the agent layer sees, into every
# ActionLog entry's `result`, and from there into the frozen golden trace
# fixtures. Two layers read those *string keys* out of a plain dict rather than
# off the model — ActionLog.record() (which handle went alive) and
# ActionLog._render_assignment() (the `name = op(...)` prefix). Both silently
# do nothing when the key they look for is absent, so a serialization-key
# regression is invisible to every type checker and to every test that goes
# through the model object. These pin the keys directly.


def test_data_handle_serializes_with_name_and_type_keys():
    """The dict key is `name`, not the pre-rename `handle`; and `type`, not
    `kind`. If this fails, the agent layer stops recognizing that a call
    created a handle: Registry State goes empty, the `name = ` assignment
    prefix disappears, and both golden trace fixtures go red — but nothing
    raises, because both readers use `.get()`."""
    reg = DataRegistry()
    meta = reg.create("list", [{"a": 1}], name="rows", description="d", created_by="t")

    dumped = meta.model_dump()
    assert dumped["name"] == "rows"
    assert dumped["type"] == "list"
    assert "handle" not in dumped
    assert "kind" not in dumped

    # ...and through JSON too, which is the form that actually reaches the
    # model and the fixtures.
    assert "name" in meta.model_dump_json()
    assert set(DataHandle.model_json_schema()["properties"]) >= {"name", "type"}


def test_create_takes_type_as_its_first_positional_arg():
    """`create(type, value, *, name=...)` — the handle *type* is positional and
    first, the caller-supplied *name* is keyword-only. The rename moved `kind`
    to `type` in that first slot while `name` stayed a keyword, so a caller
    that had been passing them positionally in the old order would now be
    silently storing a name in the type field. Pinning the positional contract
    is what stops that from being discovered at the model."""
    reg = DataRegistry()
    meta = reg.create("dict", {"k": {"v": 1}}, name="by_k", description="d", created_by="t")
    assert meta.type == "dict"
    assert meta.name == "by_k"
    # The returned name is the registry key, not a copy that could drift.
    assert reg.meta("by_k") is meta
    assert reg.get(meta.name) == {"k": {"v": 1}}


def test_value_shape_and_element_type_reject_the_pre_rename_literals():
    """The two fields once shared the string `"scalar"`, meaning two unrelated
    things — arity for `value_shape`, value-vs-record for `element_type` — and
    join_lookup crashed on an index because of exactly that ambiguity. The
    rename split them into disjoint vocabularies (`one`/`many` and
    `record`/`value`), so neither can ever again be satisfied by the other's
    string. Pinned as a *rejection* because the field names did not change:
    only the values did, so nothing else would notice an old literal being
    written back in."""
    common = dict(name="h", type="dict", size=1, description="d", created_by="t")

    for stale in ("scalar", "list", "record", "value"):
        with pytest.raises(ValidationError):
            DataHandle(**common, value_shape=stale)

    for stale in ("scalar", "one", "many"):
        with pytest.raises(ValidationError):
            DataHandle(**common, element_type=stale)

    # The new vocabularies, in full, do validate.
    for shape in ("one", "many"):
        assert DataHandle(**common, value_shape=shape).value_shape == shape
    for element in ("record", "value"):
        assert DataHandle(**common, element_type=element).element_type == element


def test_list_handles_returns_all_current_metadata():
    reg = DataRegistry()
    assert reg.list_handles() == []
    m1 = reg.create("list", [1, 2], name="a", description="d", created_by="t")
    m2 = reg.create("dict", {"a": 1}, name="b", description="d", created_by="t")
    assert {m.name for m in reg.list_handles()} == {m1.name, m2.name}
    reg.release(m1.name)
    assert [m.name for m in reg.list_handles()] == [m2.name]
