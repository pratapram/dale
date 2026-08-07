"""Structured, sanitized error surface.

Every error the LLM-facing dispatch layer can raise is a DaleError with a stable
``code`` and a human-readable ``message`` that never embeds a raw traceback, a
filesystem path, or (per the strict-privacy design intent) real data values.
Anything unexpected is caught by dispatch and re-raised as InternalError with a
generic message — see dispatch.call_primitive.
"""

from __future__ import annotations

from typing import Any


class DaleError(Exception):
    """Base class for all errors DALE returns to a caller."""

    code: str = "DALE_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        """Shape suitable for returning to an LLM tool call as a structured error."""
        return {
            "status": "error",
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class HandleNotFoundError(DaleError):
    code = "HANDLE_NOT_FOUND"


class FieldNotFoundError(DaleError):
    code = "FIELD_NOT_FOUND"


class TypeMismatchError(DaleError):
    code = "TYPE_MISMATCH"


class DuplicateKeyError(DaleError):
    code = "DUPLICATE_KEY"


class DuplicateHandleError(DaleError):
    """Raised when registry.create() is given a name already in use by a
    live handle. Handle identity is deny-on-collision, never auto-suffixed —
    see DESIGN.md Section 2's Pointer-Based State Management."""

    code = "DUPLICATE_HANDLE"


class DivisionByZeroError(DaleError):
    code = "DIVISION_BY_ZERO"


class RegistryLimitError(DaleError):
    code = "REGISTRY_LIMIT_EXCEEDED"


class ToolCallLimitError(DaleError):
    code = "TOOL_CALL_LIMIT_EXCEEDED"


class LoadError(DaleError):
    code = "LOAD_ERROR"


class FileNotRegisteredError(DaleError):
    """Raised for a load_csv/export_handle call naming a virtual file the
    invoker never registered (or no FileRegistry configured at all) — never
    reveals anything about the real filesystem, only the LLM-visible virtual
    name."""

    code = "FILE_NOT_REGISTERED"


class ExportError(DaleError):
    """Raised when export_handle fails to write to a registered destination
    (e.g. a permissions error) — mirrors LoadError for the write direction.
    Never embeds the real path, only the LLM-visible virtual name."""

    code = "EXPORT_ERROR"


class InvalidParamsError(DaleError):
    code = "INVALID_PARAMS"


class FieldCollisionError(DaleError):
    """Raised by flatten_json when a carry_fields name collides with a field
    already present on the exploded child object — rejected outright rather
    than silently picking a winner, the same deny-don't-guess precedent
    DuplicateHandleError already set for handle-name collisions."""

    code = "FIELD_COLLISION"


class PrimitiveNotFoundError(DaleError):
    code = "PRIMITIVE_NOT_FOUND"


class GraphCycleError(DaleError):
    code = "GRAPH_CYCLE"


class InternalError(DaleError):
    """Catch-all for unexpected exceptions. Message is always generic —
    the original exception is logged server-side, never returned to the caller."""

    code = "INTERNAL_ERROR"
