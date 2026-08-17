"""Composite keys and joins: index_by/group_by with multi-field keys, then
join_lookup to merge two datasets.

Run: uv run python examples/06_composite_key_join.py
"""

from __future__ import annotations

import dale

registry = dale.DataRegistry()

# Two suppliers report overlapping products under a composite (supplier, sku) key.
price_feed = registry.create(
    "list",
    [
        {"supplier": "acme", "sku": "A1", "price": 9.99},
        {"supplier": "acme", "sku": "A2", "price": 14.50},
        {"supplier": "globex", "sku": "A1", "price": 8.75},
    ],
    name="price_feed",
    description="per-supplier-and-sku price feed",
    created_by="example_script",
)

# index_by with key_fields=[...] (plural) builds a tuple-keyed dict when more
# than one field is given — a composite key, not a fourth primary type.
price_index = dale.call_operation(
    registry,
    "index_by",
    {
        "handle": price_feed.name,
        "key_fields": ["supplier", "sku"],
        "name": "price_index",
        "description": "price_feed indexed by (supplier, sku)",
    },
)
print(f"price_index: {price_index.handle.size} entries, key_arity={price_index.handle.key_arity}")

orders = registry.create(
    "list",
    [
        {"supplier": "acme", "sku": "A1", "qty": 3},
        {"supplier": "acme", "sku": "A2", "qty": 1},
        {"supplier": "globex", "sku": "A1", "qty": 5},
        {"supplier": "acme", "sku": "A9", "qty": 2},  # no matching price entry
    ],
    name="orders",
    description="orders by supplier and sku",
    created_by="example_script",
)

# join_lookup: merge price info into orders by the same composite key.
# how="left" keeps unmatched rows (A9 stays without a price); how="inner"
# would drop them instead.
joined = dale.call_operation(
    registry,
    "join_lookup",
    {
        "base_handle": orders.name,
        "index_handle": price_index.handle.name,
        "on": ["supplier", "sku"],
        "how": "left",
        "name": "joined",
        "description": "orders with price attached where available",
    },
)

print("\nJoined orders:")
for row in registry.materialize(joined.handle.name):
    price = row.get("price", "—")
    print(f"  {row['supplier']:<8} {row['sku']:<4} qty={row['qty']:<3} price={price}")

# group_by (also composite-key-aware) buckets records instead of requiring
# uniqueness — useful when a key legitimately has multiple matches.
events = registry.create(
    "list",
    [
        {"supplier": "acme", "sku": "A1", "event": "restock"},
        {"supplier": "acme", "sku": "A1", "event": "price_change"},
        {"supplier": "globex", "sku": "A1", "event": "restock"},
    ],
    name="events",
    description="per-supplier-and-sku event log",
    created_by="example_script",
)
grouped = dale.call_operation(
    registry,
    "group_by",
    {
        "handle": events.name,
        "key_fields": ["supplier", "sku"],
        "name": "events_grouped",
        "description": "events bucketed by (supplier, sku)",
    },
)
bucket = registry.materialize(grouped.handle.name)[("acme", "A1")]
print(f"\nEvents for (acme, A1): {[e['event'] for e in bucket]}")
