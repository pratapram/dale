from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dale.files import FileRegistry
from dale.registry import DataRegistry

# `dale` is installed (src layout); `eval` is a plain repo-root package that
# isn't, so tests covering the eval harness/checkers can't import it without
# the repo root on the path.
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture
def registry() -> DataRegistry:
    return DataRegistry(files=FileRegistry())


@pytest.fixture
def write_csv(registry: DataRegistry, tmp_path: Path):
    """Writes a CSV and registers it under a virtual name (the filename
    itself) on `registry`'s FileRegistry — returns that virtual name, not a
    path, matching load_csv's `file` parameter (never a raw path)."""

    def _write(name: str, rows: list[dict], fieldnames: list[str] | None = None) -> str:
        path = tmp_path / name
        fieldnames = fieldnames or list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        assert registry.files is not None
        registry.files.register(name, path)
        return name

    return _write


@pytest.fixture
def write_json(registry: DataRegistry, tmp_path: Path):
    """Writes arbitrary JSON-serializable data and registers it under a
    virtual name on `registry`'s FileRegistry — returns that virtual name,
    matching load_json's `file` parameter (never a raw path)."""

    def _write(name: str, data: Any) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        assert registry.files is not None
        registry.files.register(name, path)
        return name

    return _write
