from __future__ import annotations

import pytest

from dale.files import FileRegistry


def test_register_and_resolve(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    registry = FileRegistry()
    registry.register("data.csv", path)
    assert registry.resolve("data.csv") == path.resolve()
    assert registry.list_names() == ["data.csv"]


def test_register_nonexistent_file_raises(tmp_path):
    registry = FileRegistry()
    with pytest.raises(ValueError):
        registry.register("missing.csv", tmp_path / "missing.csv")


def test_resolve_unregistered_name_returns_none():
    registry = FileRegistry()
    assert registry.resolve("nope") is None


def test_register_output_and_resolve(tmp_path):
    dest = tmp_path / "out.csv"  # need not exist yet
    registry = FileRegistry()
    registry.register_output("out.csv", dest)
    assert registry.resolve_output("out.csv") == dest.resolve()
    assert registry.list_output_names() == ["out.csv"]


def test_register_output_missing_parent_dir_raises(tmp_path):
    registry = FileRegistry()
    with pytest.raises(ValueError):
        registry.register_output("out.csv", tmp_path / "no_such_dir" / "out.csv")


def test_resolve_output_unregistered_name_returns_none():
    registry = FileRegistry()
    assert registry.resolve_output("nope") is None


def test_input_and_output_registrations_are_independent(tmp_path):
    """register()/register_output() are separate maps — registering a name
    as a readable input doesn't make it resolvable as an output destination,
    and vice versa."""
    readable = tmp_path / "in.csv"
    readable.write_text("a\n1\n")
    registry = FileRegistry()
    registry.register("shared_name", readable)
    registry.register_output("shared_name", tmp_path / "out.csv")
    assert registry.resolve("shared_name") == readable.resolve()
    assert registry.resolve_output("shared_name") == (tmp_path / "out.csv").resolve()
