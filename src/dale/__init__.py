"""DALE — Declarative Algorithmic Logic Engine.

An LLM manipulates in-memory data (list/dict/set, held behind opaque handles
in a DataRegistry) exclusively through a small, closed set of declarative
primitives — never by generating or executing code. See DESIGN.md and
DESIGN.md for the full architecture and design rationale.

Importing this package registers every built-in primitive into the catalog
(via `dale.primitives`), so `dale.call_primitive(...)` works immediately.
"""

from dale import primitives  # noqa: F401  (import side effect: registers builtins)
from dale.catalog import (
    ConfirmableParams,
    PrimitiveOutput,
    PrimitiveSpec,
    get_primitive,
    list_primitives,
    primitive,
    register_primitive,
)
from dale.cost import CostEstimate
from dale.dispatch import call_primitive
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
    PrimitiveNotFoundError,
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
from dale.registry import DataRegistry, HandleMeta, RegistryLimits

__all__ = [
    "DataRegistry",
    "HandleMeta",
    "RegistryLimits",
    "FileRegistry",
    "call_primitive",
    "register_primitive",
    "primitive",
    "get_primitive",
    "list_primitives",
    "PrimitiveOutput",
    "PrimitiveSpec",
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
    "PrimitiveNotFoundError",
    "InternalError",
    "GraphCycleError",
    "FileNotRegisteredError",
]
