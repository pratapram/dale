# Changelog

## Unreleased — breaking

### `priority_reduce` → `reduce_by`

`priority_reduce` fused a general mechanism with a single hardcoded policy. The
mechanism — keep one record per key, under an ordering — is SQL's
`ROW_NUMBER() OVER (PARTITION BY key ORDER BY ...) = 1`, or argmax per group.
Ranking by position in an explicit value list is only *one* way to define that
ordering, and it was the only one `priority_reduce` could express: "highest
score wins" and "latest timestamp wins" were both unreachable, despite being
more common than the ranked-enum case it was built for.

`reduce_by` keeps the mechanism and makes the policy a parameter. No
back-compat shim, per the 0.2.0 precedent — `priority_reduce` is gone.

```python
# before
call_operation(registry, "priority_reduce", {
    "handle": "candidates", "key_fields": ["email"],
    "value_field": "tier", "priority": ["gold", "silver", "bronze"],
    "name": "resolved", "description": "...",
})

# after
call_operation(registry, "reduce_by", {
    "handle": "candidates", "key_fields": ["email"],
    "value_field": "tier",
    "order_by": [{"field": "tier", "ranking": ["gold", "silver", "bronze"]}],
    "name": "resolved", "description": "...",
})
```

Three behaviour changes beyond the rename:

- **`order_by` replaces `priority`**, and takes a list of keys, most
  significant first. Each key is `{field, order: asc|desc}` — order by the
  field's own values — or `{field, ranking: [...]}` for an explicit
  highest-to-lowest list of values. `sort_by`'s `SortKey` is deliberately *not*
  reused: `ranking` is meaningless for a whole-list sort, and `run_plan`'s
  schema rides on every request, so a field added to a shared type is paid for
  by every operation that uses it.
- **An unranked value no longer aborts the call.** Values not in `ranking` —
  and missing or `None` values — now rank below every listed one, the same
  NULLS-LAST rule `sort_by` already applies. A group whose only value is
  unranked wins by default. Previously this raised
  `TYPE_MISMATCH: none of [...] found in priority order [...]`, which was
  observed in live trials as a recoverable but avoidable wasted turn.
- **`value_field` is now optional.** Omit it and the whole winning record is
  kept (`key -> record`, like `index_by`); name it and only that field's value
  is kept (`key -> value`, the old behaviour, which is what `dict_diff`
  against a single-valued dict needs).

`key_fields`, `order_by` and `ranking` are all `min_length=1`. An empty
ordering cannot resolve anything for any input, so it is now rejected by schema
validation before execution rather than surfacing per key at runtime.

## 0.2.0 — breaking

A vocabulary rename, with **no back-compat shims**. Every name below changed at
once; nothing is deprecated, aliased, or forwarded. Code written against 0.1.1
will not run on 0.2.0 without the edits described here.

Two words were wrong. **"Primitive"** reads as a noun — in Python it means `int`,
`str`, `bool` — so it suggested "the data types DALE knows about" rather than
"the things DALE can do". It was also inaccurate: `graph_walk_resolve`,
`window_flag` and `priority_reduce` are coarse-grained composite algorithms,
deliberately chunked to avoid N-turn loops, which is the opposite of what
"primitive" promises. **"Kind"** was a euphemism for "type" that stopped paying
for itself once the registry began widening past `list`/`dict`/`set`.

### Catalog: `primitive` → `operation`

| 0.1.1 | 0.2.0 |
|---|---|
| `dale.call_primitive()` | `dale.call_operation()` |
| `dale.register_primitive()` | `dale.register_operation()` |
| `dale.get_primitive()` | `dale.get_operation()` |
| `dale.list_primitives()` | `dale.list_operations()` |
| `@dale.primitive` | `@dale.operation` |
| `PrimitiveOutput` | `OperationOutput` |
| `PrimitiveSpec` | `OperationSpec` |
| `PrimitiveFn` | `OperationFn` |
| `PrimitiveNotFoundError` | `OperationNotFoundError` |
| `dale.primitives` (module) | `dale.operations` |
| `build_agent(primitives=[...])` | `build_agent(operations=[...])` |
| `build_tools(primitives=[...])` | `build_tools(operations=[...])` |

### Registry: handle metadata

| 0.1.1 | 0.2.0 | Note |
|---|---|---|
| `HandleMeta` | `DataHandle` | |
| `HandleMeta.handle` | `DataHandle.name` | kills the `out.handle.handle` stutter — it is now `out.handle.name` |
| `HandleKind` | `HandleType` | |
| `.kind` | `.type` | |
| `ValueType` | `ElementType` | it was a second axis also called "type" |
| `.value_type` | `.element_type` | |
| `registry.create(kind, ...)` | `registry.create(type, ...)` | first positional arg |

**`OperationOutput.handle` is unchanged** and still holds the `DataHandle`
object. Operation *input* parameters are still called `handle`
(`peek(handle=...)`, `export_handle(handle=...)`). Only the string-id field on
the metadata class became `.name`.

### Literal value renames

Field names are unchanged; the values they hold changed.

| Field | 0.1.1 values | 0.2.0 values |
|---|---|---|
| `value_shape` | `"scalar"` / `"list"` | `"one"` / `"many"` |
| `element_type` (was `value_type`) | `"record"` / `"scalar"` | `"record"` / `"value"` |

`"scalar"` previously meant two different things across these two fields, which
is what made `join_lookup` once read `value_shape` as if it implied "record".
`value_shape` states *arity*; `element_type` states what the value *is*.

### Serialization and wire format

These are not Python identifiers, so a type checker will not catch them. They
affect anyone reading `model_dump()`, an `ActionLog` JSON dump, or a raw tool
payload.

- A serialized `DataHandle` now has keys **`name`** and **`type`** (was `handle`
  and `kind`).
- The `run_plan` step discriminator field is **`operation`** (was `primitive`).
  A step sent with the old field name is rejected by validation.
- `peek`'s result key `kind` → `type`.
- `ActionLogEntry.primitive` → `.operation`; `HandleLabel.primitive` →
  `.operation`; `HandleLabel.kind` → `.type`. `HandleLabel.handle` is unchanged
  — it names *which* handle a log record is about, so `.handle` still reads
  correctly there.
- `DaleError.details` keys: `"primitive"` → `"operation"`, `"kind"` → `"type"`,
  `"value_type"` → `"element_type"`.
- Error code `PRIMITIVE_NOT_FOUND` → `OPERATION_NOT_FOUND`.
- `INTERNAL_ERROR` message `"primitive execution failed"` → `"operation
  execution failed"`.

### System prompt

`DEFAULT_SYSTEM_PROMPT` now says "operations" rather than "primitives", and asks
the model to suffix a handle name with its *type* rather than its *kind*. If you
pass your own `system_prompt`, update it to match — the wire field is
`operation`, and a prompt that still says "primitive" will describe a field the
model cannot use.

### Migration

The mechanical part is a substring replace of `primitive` → `operation` (plurals
and compounds follow from the stem). The part that needs care is `.handle`,
which meant three different things in 0.1.1:

- `meta.handle` → `meta.name`
- `out.handle` → **unchanged** (the `DataHandle` object)
- `params.handle` → **unchanged** (an operation input parameter)

Rename the field first and let a type checker or your test suite find the call
sites; do not `sed` this one.

### Unchanged

The operation catalog itself — all 17 names (`filter_where`, `index_by`,
`join_lookup`, …), their parameters, and their behavior. This release renames
vocabulary; it adds and removes no capability.

Evaluated before and after against `eval/run_trials.py` on uc1–uc4: success rate
unchanged on every use case.

## 0.1.1

Published as `dale-engine`.
