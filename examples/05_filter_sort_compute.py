"""Basic pipeline: load data, filter, compute a derived field, sort, inspect.

Run: uv run python examples/05_filter_sort_compute.py

Demonstrates the core idea: every step is a declarative primitive call, never
generated code. The "LLM" in a real deployment would be choosing these same
primitive names and structured arguments — here we call them directly.
"""

from __future__ import annotations

from pydantic import TypeAdapter

import dale

_predicate_adapter = TypeAdapter(dale.Predicate)

registry = dale.DataRegistry()

# In a real pipeline this would come from load_csv(file=...) against a
# virtual name registered on a FileRegistry (never a raw path — see "The
# DALE environment" in README.md); here we create the handle directly to
# keep the example self-contained.
products = registry.create(
    "list",
    [
        {"name": "Widget", "price": 25.0, "cost": 10.0, "in_stock": True},
        {"name": "Gadget", "price": 40.0, "cost": 35.0, "in_stock": True},
        {"name": "Gizmo", "price": 15.0, "cost": 20.0, "in_stock": False},
        {"name": "Doohickey", "price": 60.0, "cost": 22.0, "in_stock": True},
    ],
    name="products",
    description="product catalog with price/cost/stock status",
    created_by="example_script",
)

# filter_where: keep only in-stock products. The predicate is data, not code
# — dale.render_predicate shows the human-readable form of what was actually
# sent, for review, the same rendering ActionLog.render() uses to display
# predicates an LLM constructs (see examples/04).
predicate = {"field": "in_stock", "op": "==", "value": True}
print(f"predicate: {dale.render_predicate(_predicate_adapter.validate_python(predicate))}")
in_stock = dale.call_primitive(
    registry,
    "filter_where",
    {
        "handle": products.handle,
        "predicate": predicate,
        "name": "in_stock",
        "description": "products currently in stock",
    },
)
print(f"in_stock: {in_stock.handle.size} of {products.size} products")

# compute_field: derive profit margin. `left`/`right` are tagged so there's
# never ambiguity between "a field name" and "a literal value".
with_margin = dale.call_primitive(
    registry,
    "compute_field",
    {
        "handle": in_stock.handle.handle,
        "as": "margin",
        "op": "subtract",
        "left": {"field": "price"},
        "right": {"field": "cost"},
        "name": "with_margin",
        "description": "in-stock products with a computed margin field",
    },
)

# sort_by: highest margin first.
sorted_result = dale.call_primitive(
    registry,
    "sort_by",
    {
        "handle": with_margin.handle.handle,
        "keys": [{"field": "margin", "order": "desc"}],
        "name": "sorted_by_margin",
        "description": "in-stock products sorted by margin, descending",
    },
)

# peek: a small sample, never the full dataset — this is what an LLM would
# see if it wanted to sanity-check the result shape.
sample = dale.call_primitive(registry, "peek", {"handle": sorted_result.handle.handle, "n": 10})
print("\nResult (sorted by margin, descending):")
for row in sample.result["sample"]:
    print(f"  {row['name']:<10} margin={row['margin']:.2f}")

# describe: aggregate statistics, never individual values dumped in bulk.
stats = dale.call_primitive(
    registry, "describe", {"handle": sorted_result.handle.handle, "field": "margin"}
)
print(f"\nmargin stats: min={stats.result['min']:.2f} max={stats.result['max']:.2f} "
      f"mean={stats.result['mean']:.2f}")

# release_handle: explicit cleanup once a handle is no longer needed.
for h in (products.handle, in_stock.handle.handle, with_margin.handle.handle):
    dale.call_primitive(registry, "release_handle", {"handle": h})
print(f"\nhandles remaining after cleanup: {registry.handle_count()}")
