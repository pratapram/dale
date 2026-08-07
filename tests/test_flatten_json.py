"""Unit + integration coverage for flatten_json (DESIGN.md/JSON_FEATURE.md
§3.4, USECASES.md UC7). Test list previewed and agreed on in testcases.md
before this file was written — case numbers below match that list.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from dale.dispatch import call_primitive
from dale.errors import FieldCollisionError, InvalidParamsError, TypeMismatchError
from dale.registry import DataRegistry

DATA_DIR = Path(__file__).parent.parent / "examples" / "data" / "json_flatten_github_issues"

_fixture_counter = itertools.count(1)


def _fixture_name() -> str:
    return f"fixture_{next(_fixture_counter)}"


def _load_records(registry, records, created_by="fixture"):
    return registry.create(
        "list", records, name=_fixture_name(), description="test fixture data", created_by=created_by
    )


def _flatten(registry, handle, path, carry_fields=None, **extra):
    return call_primitive(
        registry,
        "flatten_json",
        {
            "handle": handle,
            "path": path,
            "carry_fields": carry_fields or [],
            "name": _fixture_name(),
            "description": "d",
            **extra,
        },
    )


# --- 1. basic explode ------------------------------------------------------


def test_basic_explode(registry):
    meta = _load_records(
        registry,
        [
            {
                "number": 66594,
                "title": "BUG: read_csv ...",
                "labels": [
                    {"id": 1, "name": "Bug", "color": "e10c02"},
                    {"id": 2, "name": "IO CSV", "color": "5319e7"},
                ],
            }
        ],
    )
    out = _flatten(registry, meta.handle, path=["labels"], carry_fields=["number", "title"])
    assert out.status == "ok"
    rows = registry.materialize(out.handle.handle)
    assert rows == [
        {"number": 66594, "title": "BUG: read_csv ...", "id": 1, "name": "Bug", "color": "e10c02"},
        {
            "number": 66594,
            "title": "BUG: read_csv ...",
            "id": 2,
            "name": "IO CSV",
            "color": "5319e7",
        },
    ]


# --- 2. mixed populated/empty records --------------------------------------


def test_mixed_populated_and_empty_records(registry):
    meta = _load_records(
        registry,
        [
            {"number": 1, "labels": [{"name": "Bug"}]},
            {"number": 2, "labels": []},
            {"number": 3, "labels": [{"name": "Docs"}]},
        ],
    )
    out = _flatten(registry, meta.handle, path=["labels"], carry_fields=["number"])
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"number": 1, "name": "Bug"}, {"number": 3, "name": "Docs"}]


# --- 3. field entirely absent on some records -------------------------------


def test_field_absent_on_some_records_contributes_zero_rows(registry):
    meta = _load_records(
        registry,
        [
            {"number": 1, "labels": [{"name": "Bug"}]},
            {"number": 2},  # no "labels" key at all — e.g. like pull_request on a plain issue
        ],
    )
    out = _flatten(registry, meta.handle, path=["labels"], carry_fields=["number"])
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"number": 1, "name": "Bug"}]


# --- 4. field present but wrong type ----------------------------------------


def test_field_wrong_type_raises(registry):
    meta = _load_records(registry, [{"number": 1, "labels": "not-a-list"}])
    with pytest.raises(TypeMismatchError):
        _flatten(registry, meta.handle, path=["labels"])


# --- 5/6. invalid path shapes ------------------------------------------------


def test_multi_element_path_raises(registry):
    meta = _load_records(registry, [{"a": {"b": []}}])
    with pytest.raises(InvalidParamsError):
        _flatten(registry, meta.handle, path=["a", "b"])


def test_empty_path_raises(registry):
    meta = _load_records(registry, [{"labels": []}])
    with pytest.raises(InvalidParamsError):
        _flatten(registry, meta.handle, path=[])


# --- 7. carry_fields naming a missing parent field --------------------------


def test_carry_field_missing_on_parent_is_silently_skipped(registry):
    meta = _load_records(registry, [{"number": 1, "labels": [{"name": "Bug"}]}])
    out = _flatten(registry, meta.handle, path=["labels"], carry_fields=["number", "title"])
    rows = registry.materialize(out.handle.handle)
    assert rows == [{"number": 1, "name": "Bug"}]  # "title" silently absent, no error


# --- 8. carry_fields colliding with a child field ---------------------------


def test_carry_field_collision_with_child_field_raises(registry):
    meta = _load_records(
        registry, [{"id": 66594, "labels": [{"id": 1, "name": "Bug"}]}]
    )
    with pytest.raises(FieldCollisionError):
        _flatten(registry, meta.handle, path=["labels"], carry_fields=["id"])


# --- 9. wrong handle kind ----------------------------------------------------


def test_wrong_handle_kind_raises(registry):
    meta = registry.create(
        "dict", {"a": 1}, name=_fixture_name(), description="d", created_by="fixture"
    )
    with pytest.raises(TypeMismatchError):
        _flatten(registry, meta.handle, path=["labels"])


# --- 10. array of non-dict elements -----------------------------------------


def test_array_of_non_dict_elements_raises(registry):
    meta = _load_records(registry, [{"number": 1, "keywords": ["flask", "python", "web"]}])
    with pytest.raises(TypeMismatchError):
        _flatten(registry, meta.handle, path=["keywords"])


# --- 11. integration test against real, committed data ----------------------


def test_real_github_issues_label_explosion(registry):
    """Real data fetched live from api.github.com/repos/pandas-dev/pandas/issues
    during design, trimmed to number/title/labels and committed to
    examples/data/json_flatten_github_issues/ — see testcases.md."""
    path = DATA_DIR / "pandas_issues.json"
    assert registry.files is not None
    registry.files.register("pandas_issues.json", path)
    loaded = call_primitive(
        registry,
        "load_json",
        {"file": "pandas_issues.json", "name": "pandas_issues", "description": "d"},
    )

    out = _flatten(
        registry,
        loaded.handle.handle,
        path=["labels"],
        carry_fields=["number", "title"],
    )
    rows = registry.materialize(out.handle.handle)

    # Only issue #66594 has any labels in this real snapshot — everything
    # else contributes zero rows.
    assert len(rows) == 2
    assert {r["number"] for r in rows} == {66594}
    assert {r["name"] for r in rows} == {"Bug", "IO CSV"}
    assert rows[0]["title"].startswith("BUG: read_csv")


def test_real_github_issues_dispatch_end_to_end(registry):
    """Same real fixture, but every step (including flatten_json itself)
    routed through dale.call_primitive with a raw params dict, matching how
    an LLM tool call would actually shape the request."""
    path = DATA_DIR / "pandas_issues.json"
    assert registry.files is not None
    registry.files.register("pandas_issues.json", path)
    loaded = call_primitive(
        registry,
        "load_json",
        {"file": "pandas_issues.json", "name": "issues", "description": "pandas issues"},
    )
    out = call_primitive(
        registry,
        "flatten_json",
        {
            "handle": loaded.handle.handle,
            "path": ["labels"],
            "carry_fields": ["number", "title"],
            "name": "issue_labels",
            "description": "one row per label, with issue number/title attached",
        },
    )
    assert out.status == "ok"
    assert out.handle.handle == "issue_labels"
    rows = registry.materialize(out.handle.handle)
    assert len(rows) == 2
