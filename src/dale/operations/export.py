"""export_handle: delivers a handle's real content straight to a file,
without it ever passing through the LLM's context — the
strict-privacy design, pulled forward on its own (the rest of #12 — redacted
peek/describe, sanitized error content — remains deferred). The LLM sees only
a confirmation (row/byte count, success/failure), never the exported values,
symmetric to how load_csv never lets the LLM see a real path: `destination`
is a virtual name the invoker registered ahead of time via
FileRegistry.register_output, resolved server-side.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.errors import ExportError, FileNotRegisteredError, TypeMismatchError
from dale.registry import DataRegistry

Format = Literal["csv", "json"]


class ExportHandleParams(BaseModel):
    handle: str
    destination: str
    format: Format | None = None
    """Defaults to whatever `destination`'s own extension implies (.csv/.json)
    — inferred from the virtual name the LLM supplied, not the real resolved
    path, since that's the only part of the destination the LLM actually
    sees/chose. Falls back to the handle's type (csv for list, json for dict/set — not
    tabular, csv would lose structure) only if the name has neither
    extension."""


def _resolve_format(requested: Format | None, destination: str, handle_type: str) -> Format:
    if requested is not None:
        return requested
    suffix = destination.rsplit(".", 1)[-1].lower() if "." in destination else ""
    if suffix in ("csv", "json"):
        return suffix  # type: ignore[return-value]
    return "csv" if handle_type == "list" else "json"


def _write_csv(path: Path, value: list[dict]) -> int:
    if not all(isinstance(r, dict) for r in value):
        raise TypeMismatchError(
            "csv export requires a list of records (dicts), got non-dict elements"
        )
    fieldnames: list[str] = []
    for record in value:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(value)
    return len(value)


def _write_json(path: Path, value: Any, handle_type: str) -> int:
    serializable = list(value) if handle_type == "set" else value
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    return path.stat().st_size


@operation("export_handle", ExportHandleParams)
def export_handle(registry: DataRegistry, params: ExportHandleParams) -> OperationOutput:
    """Write a handle's real content to a registered output destination.
    Returns only a confirmation (format, row/byte count) — never the
    exported values themselves, so final results can leave the system
    without a single real value passing through the LLM's context."""
    if registry.files is None:
        raise FileNotRegisteredError(
            f"no destination registered as {params.destination!r} "
            "(no FileRegistry configured)",
            details={"destination": params.destination},
        )
    path = registry.files.resolve_output(params.destination)
    if path is None:
        raise FileNotRegisteredError(
            f"no destination registered as {params.destination!r}",
            details={"destination": params.destination},
        )

    meta = registry.meta(params.handle)
    value = registry.get(params.handle)
    fmt = _resolve_format(params.format, params.destination, meta.type)
    if fmt == "csv" and meta.type != "list":
        raise TypeMismatchError(
            f"csv export requires a list handle, got {meta.type!r}",
            details={"handle": params.handle, "type": meta.type},
        )

    try:
        if fmt == "csv":
            count = _write_csv(path, value)
            unit = "rows"
        else:
            count = _write_json(path, value, meta.type)
            unit = "bytes"
    except OSError as exc:
        raise ExportError(
            f"failed to write to destination {params.destination!r}: {exc}",
            details={"destination": params.destination},
        ) from exc

    return OperationOutput(
        status="ok",
        result={
            "handle": params.handle,
            "destination": params.destination,
            "format": fmt,
            unit: count,
        },
    )
