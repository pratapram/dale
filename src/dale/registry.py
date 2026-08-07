"""DataRegistry: host-side storage for native list/dict/set collections,
referenced by opaque handles. One instance per invocation —
not a global singleton, not shared across sessions, no concurrency to guard
against (tool calls are strictly sequential).
"""

from __future__ import annotations

import json
import keyword
from itertools import islice
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from dale.errors import (
    DuplicateHandleError,
    HandleNotFoundError,
    InvalidParamsError,
    RegistryLimitError,
    ToolCallLimitError,
)
from dale.files import FileRegistry

HandleKind = Literal["list", "dict", "set"]
ValueShape = Literal["scalar", "list"]
ValueType = Literal["record", "scalar"]

_SAMPLE_SIZE = 50


class RegistryLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_handles: int = 10_000
    max_result_rows: int = 1_000_000
    max_result_bytes: int = 500_000_000
    max_tool_calls: int | None = None


class HandleMeta(BaseModel):
    """Metadata surfaced to the LLM — never the underlying data itself. Also what cost estimation (cost.py) reads from.

    `handle` is the single identifier — there is no separate internal id.
    It's supplied by the caller at create() time (an LLM tool call or an
    invoker's own registry.create()), must read like a Python variable name,
    and is rejected outright on collision with an already-alive handle,
    never auto-suffixed. `description` is mandatory for the same reason:
    an honest "unknown, uninspected data" is a complete, useful answer when
    true — see DESIGN.md Section 2's Pointer-Based State Management."""

    model_config = ConfigDict(frozen=True)

    handle: str
    kind: HandleKind
    size: int
    description: str
    value_shape: ValueShape | None = None
    value_type: ValueType | None = None
    """For a dict handle, whether its values are whole records or bare
    single values — `"record"` for index_by/group_by, `"scalar"` for
    priority_reduce. Deliberately separate from `value_shape`, which states
    *arity* (one value per key vs a list of them) and nothing about type:
    index_by and priority_reduce both declare value_shape="scalar" while
    holding completely different things, and join_lookup once read that
    field as if it implied "record", crashing on a priority_reduce index
    (an `INTERNAL_ERROR` always means a missing precondition).

    Inferred by create() from the same sample it already takes for
    avg_record_bytes, rather than passed in by each producing primitive —
    an author who forgets to pass it is exactly the failure mode this field
    exists to prevent. `None` when unknowable (an empty handle, or a
    non-dict kind)."""
    key_arity: int | None = None
    avg_record_bytes: int | None = None
    created_by: str
    source_handles: tuple[str, ...] = ()


def _infer_value_type(kind: HandleKind, sample: list[Any]) -> ValueType | None:
    """Classify a dict handle's values as whole records or bare scalars, from
    the same sample create() already collects. Only meaningful for dicts —
    a list handle's elements and a set's members aren't "values keyed by
    something", so this returns None for them.

    Inferred rather than declared on purpose: making each producing
    primitive pass its own value_type would put the burden on exactly the
    author most likely to forget, which is how join_lookup came to assume
    every index held records. Mixed samples (some records, some not) fall
    back to None — "unknown", which consumers must handle defensively —
    rather than guessing from the majority."""
    if kind != "dict" or not sample:
        return None
    head = sample[:_SAMPLE_SIZE]
    if all(isinstance(v, dict) for v in head):
        return "record"
    if not any(isinstance(v, dict) for v in head):
        return "scalar"
    return None


def _estimate_avg_bytes(sample: list[Any]) -> int | None:
    if not sample:
        return None
    total = 0
    n = 0
    for item in islice(sample, _SAMPLE_SIZE):
        try:
            total += len(json.dumps(item, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            total += len(str(item).encode("utf-8"))
        n += 1
    return round(total / n) if n else None


class DataRegistry:
    def __init__(
        self,
        limits: RegistryLimits | None = None,
        files: FileRegistry | None = None,
        *,
        privacy_mode: bool = False,
    ) -> None:
        self._limits = limits or RegistryLimits()
        self._files = files
        self._privacy_mode = privacy_mode
        self._storage: dict[str, Any] = {}
        self._meta: dict[str, HandleMeta] = {}
        self._call_count = 0

    @property
    def limits(self) -> RegistryLimits:
        return self._limits

    @property
    def files(self) -> FileRegistry | None:
        return self._files

    @property
    def privacy_mode(self) -> bool:
        """Opt-in strict-privacy flag (default off,
        DESIGN.md's Optional Strict-Privacy Mode). When True, `peek`/
        `describe` redact real values (see src/dale/primitives/inspect.py)
        and `dispatch.call_primitive` sanitizes validation-error messages —
        the session-wide switch every privacy-sensitive read path checks."""
        return self._privacy_mode

    def create(
        self,
        kind: HandleKind,
        value: Any,
        *,
        name: str,
        description: str,
        created_by: str,
        source_handles: tuple[str, ...] | list[str] = (),
        value_shape: ValueShape | None = None,
        key_arity: int | None = None,
    ) -> HandleMeta:
        """Register a new native collection under caller-supplied `name`,
        which becomes the handle itself — no separate internal id. Transform
        primitives always create a *new* handle rather than mutating a
        source in place — this is what makes release_handle meaningful."""
        if not name.isidentifier() or keyword.iskeyword(name):
            raise InvalidParamsError(
                f"handle name {name!r} must be a valid Python identifier and not a "
                "reserved keyword",
                details={"name": name},
            )
        if name in self._storage:
            raise DuplicateHandleError(
                f"handle {name!r} already exists", details={"name": name}
            )
        if len(self._storage) >= self._limits.max_handles:
            raise RegistryLimitError(
                f"registry handle limit exceeded ({self._limits.max_handles})",
                details={"limit": self._limits.max_handles},
            )

        handle = name
        size = len(value)

        if kind == "list":
            sample: list[Any] = value
        elif kind == "dict":
            values = list(islice(value.values(), _SAMPLE_SIZE))
            if value_shape == "list":
                # group_by-style: each value is itself a list of records —
                # sample flattened records, not the buckets themselves.
                sample = []
                for bucket in values:
                    sample.extend(bucket if isinstance(bucket, list) else [bucket])
            else:
                sample = values
        else:  # set
            sample = list(islice(value, _SAMPLE_SIZE))

        meta = HandleMeta(
            handle=handle,
            kind=kind,
            size=size,
            description=description,
            value_shape=value_shape,
            value_type=_infer_value_type(kind, sample),
            key_arity=key_arity,
            avg_record_bytes=_estimate_avg_bytes(sample),
            created_by=created_by,
            source_handles=tuple(source_handles),
        )
        self._storage[handle] = value
        self._meta[handle] = meta
        return meta

    def get(self, handle: str) -> Any:
        if handle not in self._storage:
            raise HandleNotFoundError(f"no such handle: {handle!r}", details={"handle": handle})
        return self._storage[handle]

    def meta(self, handle: str) -> HandleMeta:
        if handle not in self._meta:
            raise HandleNotFoundError(f"no such handle: {handle!r}", details={"handle": handle})
        return self._meta[handle]

    def release(self, handle: str) -> None:
        if handle not in self._storage:
            raise HandleNotFoundError(f"no such handle: {handle!r}", details={"handle": handle})
        del self._storage[handle]
        del self._meta[handle]

    def handle_count(self) -> int:
        return len(self._storage)

    def list_handles(self) -> list[HandleMeta]:
        """All current handles' metadata — never the underlying data. Used to
        summarize registry state (e.g. for an initial system-prompt context),
        not exposed as an LLM-facing primitive in this increment."""
        return list(self._meta.values())

    def record_call(self) -> None:
        """Cheap in-process partial mitigation for the resource-governance
        'runaway loop' concern. The full OS-level
        backstop (ulimit/cgroup + agent-loop max_turns) is deferred."""
        self._call_count += 1
        limit = self._limits.max_tool_calls
        if limit is not None and self._call_count > limit:
            raise ToolCallLimitError(
                f"tool call limit exceeded ({limit})",
                details={"limit": limit},
            )

    def materialize(self, handle: str) -> Any:
        """Test/dev-only full-value accessor. NOT part of the LLM-facing
        primitive surface — bypasses peek/describe output caps. Used only to
        assert ground truth in tests."""
        return self.get(handle)
