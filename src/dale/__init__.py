"""DALE — Declarative Algorithmic Logic Engine.

An LLM manipulates in-memory data (list/dict/set, held behind opaque handles
in a DataRegistry) exclusively through a small, closed set of declarative
operations — never by generating or executing code. See README.md and
DESIGN.md for the full architecture and design rationale.

Importing this package registers every built-in operation into the catalog
(via `dale.operations`), so `dale.call_operation(...)` works immediately.
"""

from dale import operations  # noqa: F401  (import side effect: registers builtins)
from dale.catalog import (
    ConfirmableParams,
    OperationOutput,
    OperationSpec,
    get_operation,
    list_operations,
    operation,
    register_operation,
)
from dale.cost import CostEstimate
from dale.dispatch import call_operation
from dale.errors import (
    DaleError,
    DivisionByZeroError,
    DuplicateHandleError,
    DuplicateKeyError,
    ExportError,
    FieldCollisionError,
    FieldNotFoundError,
    FileNotRegisteredError,
    GraphCycleError,
    HandleNotFoundError,
    InternalError,
    InvalidParamsError,
    LoadError,
    OperationNotFoundError,
    RegistryLimitError,
    ToolCallLimitError,
    TypeMismatchError,
)
from dale.files import FileRegistry
from dale.grammar import (
    And,
    Comparison,
    ComputedField,
    ConstRef,
    FieldRef,
    Not,
    NullComparison,
    Op,
    Or,
    Predicate,
    Priority,
    ValueRef,
    apply_computed_field,
    matches,
    render_predicate,
    resolve_priority,
    resolve_value_ref,
)
from dale.registry import DataHandle, DataRegistry, RegistryLimits

__all__ = [
    "DataRegistry",
    "DataHandle",
    "RegistryLimits",
    "FileRegistry",
    "call_operation",
    "register_operation",
    "operation",
    "get_operation",
    "list_operations",
    "OperationOutput",
    "OperationSpec",
    "ConfirmableParams",
    "CostEstimate",
    # grammar
    "Predicate",
    "Comparison",
    "NullComparison",
    "And",
    "Or",
    "Not",
    "Op",
    "ComputedField",
    "FieldRef",
    "ConstRef",
    "ValueRef",
    "Priority",
    "matches",
    "render_predicate",
    "apply_computed_field",
    "resolve_value_ref",
    "resolve_priority",
    # errors
    "DaleError",
    "HandleNotFoundError",
    "FieldNotFoundError",
    "TypeMismatchError",
    "DuplicateKeyError",
    "DuplicateHandleError",
    "FieldCollisionError",
    "DivisionByZeroError",
    "RegistryLimitError",
    "ToolCallLimitError",
    "LoadError",
    "ExportError",
    "InvalidParamsError",
    "OperationNotFoundError",
    "InternalError",
    "GraphCycleError",
    "FileNotRegisteredError",
]
