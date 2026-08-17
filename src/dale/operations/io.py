"""Local file loading — never a network connector, and
never an LLM-constructed path (see FileRegistry in files.py): `file` is a
virtual name the invoker registered ahead of time, resolved through
DataRegistry.files rather than accepted as a raw path directly."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import FileNotRegisteredError, LoadError
from dale.registry import DataRegistry


class LoadCsvParams(BaseModel):
    file: str
    name: str
    description: str


def _infer_scalar(value: str) -> Any:
    """Deterministic, closed type inference — not guessing: try int, then
    float, else keep the string; empty string becomes None."""
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_registered_file(registry: DataRegistry, file: str) -> Path:
    """Shared virtual-name -> real Path resolution for every local-file
    loader (load_csv, load_json, ...) — never a raw path (see FileRegistry)."""
    if registry.files is None:
        raise FileNotRegisteredError(
            f"no file registered as {file!r} (no FileRegistry configured)",
            details={"file": file},
        )
    path = registry.files.resolve(file)
    if path is None:
        raise FileNotRegisteredError(f"no file registered as {file!r}", details={"file": file})
    if not path.is_file():
        # The virtual name was registered, but the underlying file is gone by
        # the time the loader actually runs -- still never reveal the real path.
        raise LoadError(f"registered file {file!r} no longer exists", details={"file": file})
    return path


@operation("load_csv", LoadCsvParams, creates_handle=True)
def load_csv(registry: DataRegistry, params: LoadCsvParams) -> OperationOutput:
    """Load a CSV file, referenced by an invoker-registered virtual name (not
    a raw path — see FileRegistry), into a new list-of-records handle, with
    deterministic type inference (int, then float, else string; blank ->
    null). Local files only — never a network or database connection."""
    path = _resolve_registered_file(registry, params.file)

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                {key: _infer_scalar(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error) as exc:
        raise LoadError(
            f"failed to load CSV {params.file!r}: {exc}", details={"file": params.file}
        ) from exc

    meta = registry.create(
        "list", rows, name=params.name, description=params.description, created_by="load_csv"
    )
    return OperationOutput(status="ok", handle=meta)


class LoadJsonParams(BaseModel):
    file: str
    remove_envelope: bool = False
    name: str
    description: str


def _unwrap_envelope(data: dict[str, Any], file: str) -> list[Any]:
    """Find the single list-valued key in an envelope-shaped dict (e.g.
    Salesforce's {"records": [...], "totalSize":.., "done":..} or a plain
    {"data": [...]}) and return its value. Only called when the model has
    explicitly asserted remove_envelope=True -- it knows the source system,
    DALE doesn't -- so an ambiguous or absent match is a legible LoadError,
    never a silent guess among candidates."""
    list_keys = [k for k, v in data.items() if isinstance(v, list)]
    other_keys_scalar = all(
        not isinstance(v, (list, dict)) for k, v in data.items() if k not in list_keys
    )
    if len(list_keys) != 1 or not other_keys_scalar:
        raise LoadError(
            f"remove_envelope=True but {file!r} isn't a single-list envelope "
            "(expected exactly one list-valued key, with every other key scalar)",
            details={"file": file},
        )
    return data[list_keys[0]]


@operation("load_json", LoadJsonParams, creates_handle=True)
def load_json(registry: DataRegistry, params: LoadJsonParams) -> OperationOutput:
    """Load a JSON file, referenced by an invoker-registered virtual name (not
    a raw path — see FileRegistry), into a new handle. A top-level JSON array
    becomes a list handle, a top-level JSON object becomes a dict handle —
    JSON is a loading path, not a new handle type.
    Many enterprise APIs (Salesforce, ServiceNow, Stripe...) wrap their real
    payload in an envelope object, e.g. {"records": [...]}; if the model
    recognizes this shape it can pass remove_envelope=True to unwrap straight
    to a list handle instead of a dict handle wrapping one. DALE never guesses
    this on its own — remove_envelope defaults to False, and a top-level array
    ignores the flag entirely (nothing to unwrap). Nested/irregular structure
    inside the result is otherwise preserved as-is; other operations (peek,
    flatten_json) are how the model inspects and reshapes it. Local files
    only — never a network or database connection."""
    path = _resolve_registered_file(registry, params.file)

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadError(
            f"failed to load JSON {params.file!r}: {exc}", details={"file": params.file}
        ) from exc

    if isinstance(data, list):
        meta = registry.create(
            "list", data, name=params.name, description=params.description, created_by="load_json"
        )
    elif isinstance(data, dict):
        if params.remove_envelope:
            unwrapped = _unwrap_envelope(data, params.file)
            meta = registry.create(
                "list",
                unwrapped,
                name=params.name,
                description=params.description,
                created_by="load_json",
            )
        else:
            meta = registry.create(
                "dict", data, name=params.name, description=params.description, created_by="load_json"
            )
    else:
        raise LoadError(
            f"JSON file {params.file!r} must contain an array or object at the "
            f"top level, got {type(data).__name__}",
            details={"file": params.file},
        )

    return OperationOutput(status="ok", handle=meta)
