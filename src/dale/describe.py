"""Human-readable renderings of the operation catalog.

Three formats over one source of truth. `full` is the reference dump — every
operation with its whole purpose and every parameter. `compact` is the same
catalog at a glance, one screen for all of it. `markdown` is the table in
GUIDE.md, which is generated from here rather than hand-maintained (see
`scripts/sync_guide_operations.py`).

Everything is read through the public catalog API — `list_operations()` and
`get_operation()` — never `_CATALOG` directly, so an operation a third party
registered with `@operation` prints alongside the built-ins with no change
here. Parameters come from `param_schema.model_json_schema()` rather than
`model_fields` because the wire names are what a caller actually writes, and
they differ from the Python attribute names in at least one place
(`window_flag`'s `as` is `as_` in Python).

These functions build strings and print nothing. The caller decides where the
text goes: stdout for `python -m dale operations`, the live trace for
`dale.agent` at `verbosity="raw"`, GUIDE.md for the sync script.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Iterable, Literal, Sequence

from dale.catalog import OperationSpec, get_operation, list_operations

__all__ = ["CatalogFormat", "render_catalog"]

CatalogFormat = Literal["full", "compact", "markdown"]

_GUIDE_ORDER: tuple[str, ...] = (
    # The reading order of GUIDE.md's table: load, then transform, then index,
    # then combine, then inspect, then lifecycle, then export. Sorted order --
    # what `list_operations()` returns -- interleaves those groups and makes the
    # table harder to read than the pipeline it describes. Asserted complete
    # against the catalog in tests/test_describe.py, so a new built-in has to be
    # placed here deliberately rather than landing wherever the alphabet puts it.
    "load_csv",
    "load_json",
    "filter_where",
    "compute_field",
    "sort_by",
    "index_by",
    "group_by",
    "reduce_by",
    "dict_diff",
    "join_lookup",
    "window_flag",
    "graph_walk_resolve",
    "flatten_json",
    "peek",
    "describe",
    "release_handle",
    "export_handle",
)


# --------------------------------------------------------------------------
# shared bits
# --------------------------------------------------------------------------


def _resolve(names: Sequence[str] | None) -> list[str]:
    """`None` means the whole catalog in `list_operations()` order. An explicit
    sequence is honored in the order given -- that is what lets `dale.agent`
    print the operations one agent was actually offered, in the order it
    offers them, rather than the whole catalog."""
    if names is None:
        return list(list_operations())
    # get_operation raises OperationNotFoundError, which is the right error for
    # a caller naming something that isn't registered.
    for name in names:
        get_operation(name)
    return list(names)


def _flags(spec: OperationSpec, *, short: bool) -> list[str]:
    out = []
    if spec.creates_handle:
        out.append("new" if short else "creates handle")
    if spec.bounded_by_input:
        out.append("bounded" if short else "output <= input")
    if spec.cost_estimator is not None:
        out.append("$" if short else "cost-gated")
    return out


def _purpose(spec: OperationSpec) -> str:
    """`summary` when the operation declares one, the docstring otherwise.

    The fallback is what keeps a third-party operation renderable: it never
    had to know this field exists."""
    if spec.summary:
        return spec.summary
    return " ".join((inspect.getdoc(spec.fn) or "").split())


def _io(spec: OperationSpec) -> str:
    return spec.io_signature or ""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"


# --------------------------------------------------------------------------
# JSON Schema -> a type a human reads
# --------------------------------------------------------------------------


def _type_of(node: dict[str, Any]) -> str:
    """Render one JSON-Schema node. Deliberately partial: it covers the shapes
    pydantic actually emits for these param models (`$ref`, `anyOf`, `enum`,
    `array` + `items`, and the scalars) and degrades to the bare type name for
    anything else rather than guessing."""
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in node:
        return " | ".join(_type_of(alt) for alt in node["anyOf"])
    if "enum" in node:
        return " | ".join(repr(v) for v in node["enum"])
    kind = node.get("type")
    if kind == "array":
        items = node.get("items")
        return f"list[{_type_of(items)}]" if items else "list"
    if kind == "object":
        extra = node.get("additionalProperties")
        return f"dict[str, {_type_of(extra)}]" if isinstance(extra, dict) else "dict"
    return kind or "any"


def _params(spec: OperationSpec) -> list[dict[str, Any]]:
    """Wire-level parameters: name, rendered type, whether required, default,
    and the schema description if the field carries one."""
    schema = spec.param_schema.model_json_schema()
    required = set(schema.get("required", ()))
    out = []
    for name, node in schema.get("properties", {}).items():
        out.append(
            {
                "name": name,
                "type": _type_of(node),
                "required": name in required,
                "default": node.get("default"),
                "description": " ".join(node.get("description", "").split()),
            }
        )
    return out


def _signature(spec: OperationSpec) -> str:
    """`(handle, field, top_k=10)` -- the call shape on one line.

    Required first, optional after, each group keeping its schema order. These
    are keyword arguments in a JSON object, so the wire does not care; a reader
    does, and schema order alone puts `confirm=False` -- inherited from
    `ConfirmableParams` -- ahead of the required params on every cost-gated
    operation, which reads like a mistake."""
    params = _params(spec)
    ordered = [p for p in params if p["required"]] + [p for p in params if not p["required"]]
    parts = [p["name"] if p["required"] else f"{p['name']}={p['default']!r}" for p in ordered]
    return "(" + ", ".join(parts) + ")"


# --------------------------------------------------------------------------
# the three formats
# --------------------------------------------------------------------------


def _render_full(names: Iterable[str], width: int) -> str:
    blocks = []
    for name in names:
        spec = get_operation(name)
        head = name
        if io := _io(spec):
            head += f"  {io}"
        if flags := _flags(spec, short=False):
            head += f"  [{', '.join(flags)}]"

        lines = [head]
        lines += textwrap.wrap(
            _purpose(spec), width - 2, initial_indent="  ", subsequent_indent="  "
        )

        params = _params(spec)
        if params:
            lines.append("  params:")
            pad = max(len(p["name"]) for p in params)
            tpad = max(len(p["type"]) for p in params)
            for p in params:
                state = "required" if p["required"] else f"default={p['default']!r}"
                lines.append(f"    {p['name']:<{pad}}  {p['type']:<{tpad}}  {state}")
                if p["description"]:
                    lines += textwrap.wrap(
                        p["description"],
                        width - 6,
                        initial_indent="      ",
                        subsequent_indent="      ",
                    )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_compact(names: Iterable[str], width: int) -> str:
    """Three lines per operation: what it is, what it does, how it's called.

    Putting the purpose on its own full-width line rather than in a fourth
    column is what makes this readable -- the columns eat ~50 characters
    before the prose starts, which leaves too little of an 88-column line to
    say anything. One clipped line of prose each still fits the whole catalog
    on a screen."""
    names = list(names)
    specs = [get_operation(n) for n in names]
    if not specs:
        return ""
    npad = max(len(n) for n in names)
    iopad = max((len(_io(s)) for s in specs), default=0)

    lines = []
    for name, spec in zip(names, specs):
        flags = _flags(spec, short=True)
        cell = f"[{', '.join(flags)}]" if flags else ""
        lines.append(f"{name:<{npad}}  {_io(spec):<{iopad}}  {cell}".rstrip())
        lines.append("  " + _clip(_purpose(spec), width - 2))
        lines += textwrap.wrap(
            _signature(spec), width - 2, initial_indent="  ", subsequent_indent="      "
        )
    return "\n".join(lines)


def _render_markdown(names: Iterable[str]) -> str:
    # Curated order first; anything the catalog has that _GUIDE_ORDER doesn't
    # know about (a third-party registration) follows, so it is never dropped.
    names = list(names)
    ordered = [n for n in _GUIDE_ORDER if n in names]
    ordered += [n for n in names if n not in _GUIDE_ORDER]

    rows = ["| Operation | In → Out | Purpose |", "|---|---|---|"]
    for name in ordered:
        spec = get_operation(name)
        # The pipe is the cell separator, so any pipe inside a cell has to be
        # escaped on the way out. It is stored unescaped, so the other two
        # formats show a clean `list | dict`.
        io = _io(spec).replace("|", r"\|")
        purpose = _purpose(spec).replace("|", r"\|")
        rows.append(f"| `{name}` | {io} | {purpose} |")
    return "\n".join(rows)


def render_catalog(
    names: Sequence[str] | None = None,
    *,
    format: CatalogFormat = "full",
    width: int = 88,
) -> str:
    """Render the operation catalog as text.

    `names=None` renders everything in `list_operations()` order; an explicit
    sequence renders exactly those, in the order given, and raises
    `OperationNotFoundError` for a name that isn't registered.

    `width` is a target for the wrapped prose in `full` and `compact`; the
    `markdown` format ignores it, since a table cell wraps at display time and
    a hard-wrapped row would only make the source harder to diff."""
    selected = _resolve(names)
    if format == "full":
        return _render_full(selected, width)
    if format == "compact":
        return _render_compact(selected, width)
    if format == "markdown":
        return _render_markdown(selected)
    raise ValueError(
        f"unknown catalog format: {format!r} (expected 'full', 'compact', or 'markdown')"
    )
