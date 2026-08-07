from __future__ import annotations

import pytest

from dale.errors import (
    DuplicateHandleError,
    HandleNotFoundError,
    InvalidParamsError,
    RegistryLimitError,
    ToolCallLimitError,
)
from dale.registry import DataRegistry, RegistryLimits


def test_create_and_get():
    reg = DataRegistry()
    meta = reg.create("list", [1, 2, 3], name="nums", description="three numbers", created_by="test")
    assert meta.kind == "list"
    assert meta.size == 3
    assert meta.handle == "nums"
    assert reg.get(meta.handle) == [1, 2, 3]


def test_meta_matches_get():
    reg = DataRegistry()
    meta = reg.create(
        "list", [{"a": 1}], name="records", description="one record", created_by="test"
    )
    assert reg.meta(meta.handle) == meta


def test_handle_is_the_caller_supplied_name():
    reg = DataRegistry()
    m1 = reg.create("list", [1], name="a_list", description="d", created_by="t")
    m2 = reg.create("list", [2], name="b_list", description="d", created_by="t")
    m3 = reg.create("dict", {"a": 1}, name="a_dict", description="d", created_by="t")
    assert m1.handle == "a_list"
    assert m2.handle == "b_list"
    assert m3.handle == "a_dict"


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
    reg.release(meta.handle)
    with pytest.raises(HandleNotFoundError):
        reg.get(meta.handle)
    with pytest.raises(HandleNotFoundError):
        reg.meta(meta.handle)
    with pytest.raises(HandleNotFoundError):
        reg.release(meta.handle)
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
    reg.release(m1.handle)
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
    assert reg.materialize(meta.handle) == [1, 2, 3]


def test_list_handles_returns_all_current_metadata():
    reg = DataRegistry()
    assert reg.list_handles() == []
    m1 = reg.create("list", [1, 2], name="a", description="d", created_by="t")
    m2 = reg.create("dict", {"a": 1}, name="b", description="d", created_by="t")
    assert {m.handle for m in reg.list_handles()} == {m1.handle, m2.handle}
    reg.release(m1.handle)
    assert [m.handle for m in reg.list_handles()] == [m2.handle]
