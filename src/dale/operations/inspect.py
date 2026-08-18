"""peek/describe — secondary sanity-check tools, not load-bearing (see
#2: schema is given upfront in the task, not discovered). Hard output caps
here are a small piece of the deferred strict_privacy mode
pulled forward now: cheap to do correctly from the start, awkward to retrofit.
"""

from __future__ import annotations

import json
from itertools import islice
from typing import Any

from pydantic import BaseModel

from dale.catalog import OperationOutput, operation
from dale.registry import DataRegistry

_PEEK_MAX_N = 50
"""Ceiling on how many *items* a sample may contain — records from a list or
set handle, keys from a dict handle, and elements of any nested collection
inside them. A count cap alone is not a bound on what reaches the model (see
_PEEK_MAX_BYTES); it stays because 50 tiny records read better than 500, not
because it makes anything safe."""

_PEEK_MAX_BYTES = 4_096
"""The real ceiling: a peek sample never serializes to more than this, for any
handle, any shape, any nesting depth.

Stated in bytes because bytes are what DALE's central claim is actually about
— the LLM never sees the underlying data (DESIGN.md Section 2,
#2) — and because every count-based cap tried here has failed on a shape
nobody thought to enumerate. Capping keys let a group_by hand back whole
buckets (852KB measured on 10,000 rows in 3 regions). Capping records then let
a single *nested* value through untouched: an index_by whose records carry an
array field returned 179KB, a group_by whose records are nested 341KB, and a
load_json of an ordinary top-level JSON object — the shape load_json's own
docstring advertises — 531KB, with nothing in the payload admitting it. Each
of those was one cap short in a dimension the previous fix had not thought of.
A byte budget has no such dimension left: it is the quantity being bounded,
so it is shape-agnostic by construction.

This matters unprompted, not just when the model asks: peek_at_every_step
(default on) fires peek(n=3) after every handle-creating call, and
_initial_inspect_summary splices one into the system prompt for every
pre-loaded handle, so a load_json of a nested document used to detonate before
the model had taken a single turn.

4 KB is roughly a thousand tokens — enough to show real structure several
levels deep, small enough that a per-step auto-inspect is affordable in a
context window this project's whole argument is about keeping bounded."""

_MARKER_RESERVE = 32
"""Room, in bytes, for one truncation marker — and so also the smallest budget
worth descending into.

Both halves of that are load-bearing. Held back from every budget, it stops the
marker from being the overshoot: a sample that filled its budget exactly would
otherwise exceed it by the length of the note saying so, and the ceiling would
hold everywhere except at the moment it binds. Used as a floor on every child
budget, it stops the same overshoot arriving from below — a child given less
than a marker costs can only answer with something bigger than it was allowed,
a few bytes at a time, once per nesting level. Anything that can't be given
this much is not descended into at all; the marker one level up already says it
is missing.

32 is comfortably above the longest marker this file can emit (`"...(+1234567
more chars)"` and the like), which is what makes "every _fit call returns at
most its budget" true by induction rather than by measurement."""

_MORE_KEY = "..."
"""The key a truncation marker is filed under when a *dict* is cut short. A
real data key could collide with it in principle; that costs the model one
ambiguous entry in a sample it is already being told is partial, whereas any
collision-proof sentinel (a UUID, a reserved prefix) is unreadable to the
model, which is the one audience this payload has."""

_DESCRIBE_TOP_K = 10
_DESCRIBE_MAX_BYTES = 2_048
"""Same ceiling, same reason, applied to `describe`'s categorical top_k — the
only part of a describe payload carrying real field *values* rather than
counts. `_DESCRIBE_TOP_K` caps how many values come back and, exactly like
`_PEEK_MAX_N`, says nothing about how large they are: 20 records holding 100KB
strings returned a 1,000,370-byte describe. Not reachable from _auto_inspect
(which calls describe with no field), but the model can call describe directly,
and a cap that only holds for the calls DALE makes on the model's behalf is not
a cap."""

_SCHEMA_SAMPLE_SIZE = 50


class PeekParams(BaseModel):
    handle: str
    n: int = 1


def _scalar_bytes(value: Any) -> int:
    """Serialized size of one leaf, measured the way DataRegistry already
    measures a record (json.dumps to UTF-8, str() for anything json refuses).
    Only ever called on a leaf that has already been trimmed to fit, so it
    never serializes more than a budget's worth of data — measuring the whole
    value first would cost exactly the work the budget exists to avoid."""
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8")) + 2


def _more(count: int, unit: str, *, redact: bool) -> str:
    """The marker left in place of whatever was cut — `"+3331 more items"`,
    or a count-free `"<truncated>"` under privacy_mode.

    Count-free is not squeamishness. `shown + "+N more"` is the exact size of
    the thing that was cut, and for a dict handle's buckets that is a
    value -> count pair: peek would hand back the categorical top_k that
    `describe` deliberately withholds under the same flag.
    The model still learns that it is looking at a partial sample, which is
    the part it needs in order not to be misled; it just doesn't learn the
    number, which is the part that leaks."""
    return "<truncated>" if redact else f"+{count} more {unit}"


def _fit(value: Any, budget: int, *, max_items: int, redact: bool) -> tuple[Any, int, bool]:
    """Return `(fitted_value, bytes_used, was_truncated)` — a copy of `value`
    that serializes to at most `budget` bytes, cut in place wherever it had to
    be, with every cut marked.

    Recursive because the thing being bounded is recursive: the payloads that
    broke every previous cap were not long collections but deep ones — a
    record with an array field, a bucket of such records, a JSON object of such
    buckets. Descending with a shrinking budget bounds all of them without
    naming any of them.

    Three properties this has to keep, none of them negotiable:

    *Shape survives.* A dict stays a dict and a list stays a list, however
    little of it fits, because learning the shape is the entire point of peek
    (key -> list of records vs key -> one record vs key -> scalar). What
    shrinks is breadth and leaf size, never the structure itself.

    *Every cut is visible, in place.* A truncated list ends with `"+N more
    items"`, a truncated dict gains a `"..."` entry, a truncated string ends
    with `"...(+N more chars)"` — always adjacent to what was cut, so at any
    depth there is no ambiguity about which collection was shortened and no
    way to read a partial sample as a complete one. A shortened bucket that
    looks complete is worse than the original bug: the model would draw
    conclusions ("north has 3 orders") from a sample it had no way to know was
    partial.

    *Redaction happens here, in the same pass.* Not afterwards: budgeting the
    real values and then substituting type placeholders would mean the emitted
    payload is not the one that was measured (`1` costs 1 byte, `"<int>"`
    costs 7), so the byte ceiling would hold for every session except the one
    whose whole premise is that nothing escapes. Leaves are replaced by their
    type, dict keys *inside* records are not — those are field names, which
    privacy_mode has always shown, since schema is supplied upfront rather
    than discovered. A dict *handle's* own keys are data,
    not schema, and are handled by _sample_dict, which knows the difference."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        used = 2  # the braces
        truncated = False
        for key, item in value.items():
            # `- _MARKER_RESERVE` on every child budget, not just on the
            # stopping test: the last child admitted is entitled to spend
            # everything left, so without holding the marker's own room back
            # here, the note saying "there was more" is exactly what pushes
            # the sample past the ceiling.
            room = budget - used - _MARKER_RESERVE
            # Two stopping conditions. The count one is _PEEK_MAX_N rather than
            # max_items because `n` is how many *records* the model asked to
            # see and a record's fields are not records — capping them by n
            # turned peek(n=1) on a 3-field row into a 1-field row, deleting
            # the schema signal peek exists to give in the name of the
            # parameter that asked for it. The byte one needs room for a label,
            # a value and the ": " between them, each at least a marker's
            # worth; below that, stop and let the marker below report the rest.
            if len(out) >= _PEEK_MAX_N or room < 2 * _MARKER_RESERVE + 2:
                break
            # Keys are fitted too. Inside a record they are field names, which
            # privacy_mode shows and which are normally short — but "normally"
            # is what every previous version of this cap relied on, so a key
            # gets a quarter of the room at most and is trimmed like anything
            # else if it wants more. redact=False because a key that needed
            # redacting was already replaced before it got here (_sample_dict).
            label, label_bytes, _ = _fit(
                str(key), max(_MARKER_RESERVE, room // 4), max_items=max_items, redact=False
            )
            label_bytes += 2  # the ": " json.dumps puts between key and value
            fitted, item_bytes, cut = _fit(
                item, room - label_bytes, max_items=max_items, redact=redact
            )
            out[label] = fitted
            used += label_bytes + item_bytes + 2  # the ", " between entries
            truncated = truncated or cut
        if (remaining := len(value) - len(out)) > 0:
            marker = _more(remaining, "keys", redact=redact)
            out[_MORE_KEY] = marker
            used += _scalar_bytes(marker) + _scalar_bytes(_MORE_KEY) + 4
            truncated = True
        return out, used, truncated

    if isinstance(value, list):
        items: list[Any] = []
        used = 2  # the brackets
        truncated = False
        for item in value:
            room = budget - used - _MARKER_RESERVE
            if len(items) >= max_items or room < _MARKER_RESERVE:
                break
            fitted, item_bytes, cut = _fit(item, room, max_items=max_items, redact=redact)
            items.append(fitted)
            used += item_bytes + 2  # the ", " between items
            truncated = truncated or cut
        if (remaining := len(value) - len(items)) > 0:
            marker = _more(remaining, "items", redact=redact)
            items.append(marker)
            used += _scalar_bytes(marker) + 2
            truncated = True
        return items, used, truncated

    if redact:
        # The leaf case privacy_mode cares about: type, never content. Size
        # doesn't survive either -- a 100KB string and a 3-character one are
        # both "<str>", so nothing here restates a length the markers above
        # were careful not to give away.
        placeholder = f"<{type(value).__name__}>"
        return placeholder, _scalar_bytes(placeholder), False

    # A leaf is not automatically small. A 200KB text field is the obvious
    # case, but json also hands back unbounded ints, and a leaf returned whole
    # because "scalars are cheap" is the same assumption -- one item, therefore
    # one small item -- that let whole buckets through twice already. Strings
    # are pre-checked in characters, which is O(1) and can only over-estimate
    # the byte length, so the common short string never pays for a measurement
    # and a 200KB one is never serialized in full just to discover its size.
    if not (isinstance(value, str) and len(value) > budget):
        if (size := _scalar_bytes(value)) <= budget:
            return value, size, False

    text = value if isinstance(value, str) else str(value)
    keep = text[: max(0, budget - _MARKER_RESERVE)]
    keep = keep.encode("utf-8")[: max(0, budget - _MARKER_RESERVE)].decode("utf-8", "ignore")
    cut_chars = len(text) - len(keep)
    fitted_text = keep + f"...(+{cut_chars} more chars)"
    return fitted_text, _scalar_bytes(fitted_text), True


def _sample_dict(
    value: dict[Any, Any], n: int, budget: int, *, redact: bool
) -> tuple[dict[str, Any], bool]:
    """Sample a dict *handle* — the one level where keys are data rather than
    field names, and so the one level `_fit` can't handle on its own.

    Two things happen here that don't happen anywhere below. First, keys are
    rendered with `repr()` (so `"'north'"` is visibly a string and `"0"`
    visibly an int) when privacy_mode is off, and replaced with positional
    `"<key 1>"` placeholders when it is on. That second case closes a leak that
    predates the byte budget entirely: `index_by(["patient"])` under
    privacy_mode used to return `{"'NHS-1000'": {...}}` — every value redacted,
    every real identifier printed as the key next to it. Redaction that
    recurses into values but not keys isn't redaction of a handle whose keys
    *are* values.

    Second, the byte budget is split across the keys shown rather than spent
    first-come-first-served, so a peek of a 3-bucket dict doesn't return one
    enormous bucket and two stubs. Each key gets an equal share of whatever is
    left, which lets a cheap key hand its unspent remainder to the next one.

    Key truncation is reported the same way every other cut is — a `"..."`
    entry giving how many keys were not shown. It is not cosmetic: for buckets
    a missing count costs the model a size estimate, but for keys it costs it
    the answer, since "there are 3 regions" is a conclusion a 3-of-1000-key
    sample invites and DALE's own metadata (DataHandle.size) contradicts."""
    n = max(0, min(n, _PEEK_MAX_N))
    # islice, not list(value.keys())[:n] -- the slice materialized the entire
    # keyspace to throw all but n of it away: 4.2 ms on a 1M-key dict against
    # 0.0018 ms here. peek_at_every_step pays that per step, and it lands in
    # ActionLogEntry.auto_inspect_ms, the field that exists to be honest about
    # what automatic inspection costs.
    keys = list(islice(value, n))
    sample: dict[str, Any] = {}
    used = 2
    truncated = False
    for position, key in enumerate(keys):
        room = budget - used - _MARKER_RESERVE
        share = room // (len(keys) - position)
        if share < 2 * _MARKER_RESERVE + 2:
            break
        label_raw = f"<key {position + 1}>" if redact else repr(key)
        label, label_bytes, _ = _fit(
            label_raw, max(_MARKER_RESERVE, share // 4), max_items=n, redact=False
        )
        label_bytes += 2
        fitted, item_bytes, cut = _fit(
            value[key], share - label_bytes, max_items=n, redact=redact
        )
        sample[label] = fitted
        used += label_bytes + item_bytes + 2  # the ", " between entries
        truncated = truncated or cut
    # Counted against what was actually shown, not against `keys`: a key the
    # budget ran out on is a key the model isn't seeing, and the marker is the
    # only thing that says so.
    if (remaining := len(value) - len(sample)) > 0:
        sample[_MORE_KEY] = _more(remaining, "keys", redact=redact)
        truncated = True
    return sample, truncated


@operation(
    "peek",
    PeekParams,
    io_signature="any → sample",
    summary=(
        'A small sample of a handle — a sanity check, not a data-dump '
        'channel. Hard-capped at 50 items **and 4 KB serialized**, whatever '
        'the handle holds and however deeply it nests, so a `peek` of a '
        '10-million-row handle costs the same as one of a 10-row handle. '
        'Anything cut to fit says so in place: a shortened list ends with '
        '`"+N more items"`, a shortened dict gains a `"..."` entry counting '
        'the keys not shown, a shortened string ends with `"...(+N more '
        'chars)"`, and `"truncated": true` sits alongside the sample — a '
        'sample never understates what is there'
    ),
)
def peek(registry: DataRegistry, params: PeekParams) -> OperationOutput:
    """Return a small sample of a handle — a sanity check on shape and fields,
    never a way to read the data. The sample is capped at 50 items and, hard,
    at 4 KB serialized, whatever the handle holds and however deeply it nests;
    a `peek` of a 10-million-row handle and of a 10-row one cost the same.
    Anything cut to fit is marked in place: a shortened list ends with
    `"+N more items"`, a shortened dict gains a `"..."` entry saying how many
    keys are missing, a shortened string ends with `"...(+N more chars)"`, and
    `"truncated": true` appears alongside the sample. So a sample never
    understates what is there — read those markers as "this collection is
    larger than what you are seeing", and use the operations to process data
    rather than peek to read it. Under `privacy_mode`, values are redacted to
    type placeholders and dict keys to positions rather than the call being
    refused — it still shows shape, just never content, and never the counts
    that would let content be reconstructed."""
    meta = registry.meta(params.handle)
    n = max(0, min(params.n, _PEEK_MAX_N))
    value = registry.get(params.handle)
    redact = registry.privacy_mode

    if meta.type == "dict":
        sample, truncated = _sample_dict(value, n, _PEEK_MAX_BYTES, redact=redact)
    else:
        # Sliced to n *before* fitting, so the ordinary "you asked for 3 of
        # 10,000" case adds no marker: that isn't peek cutting anything short,
        # it's peek answering what was asked, and the handle's full size is
        # already in the metadata the model has. A marker here means the bytes
        # ran out, which is the thing it can't otherwise know.
        items = value[:n] if meta.type == "list" else list(islice(value, n))
        sample, _, truncated = _fit(items, _PEEK_MAX_BYTES, max_items=n, redact=redact)

    result: dict[str, Any] = {
        "handle": params.handle,
        "type": meta.type,
        "requested_n": params.n,
        "sample": sample,
    }
    if truncated:
        result["truncated"] = True
    if redact:
        result["note"] = "privacy_mode is enabled: values redacted to type placeholders"

    return OperationOutput(status="ok", result=result)


class DescribeParams(BaseModel):
    handle: str
    field: str | None = None


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _records_from_handle(registry: DataRegistry, handle: str) -> list[Any]:
    meta = registry.meta(handle)
    value = registry.get(handle)

    if meta.type == "list":
        return value
    if meta.type == "dict":
        values = list(value.values())
        if meta.value_shape == "many":
            flattened: list[Any] = []
            for bucket in values:
                flattened.extend(bucket if isinstance(bucket, list) else [bucket])
            return flattened
        return values
    return [{"value": v} for v in value]  # set


@operation(
    "describe",
    DescribeParams,
    io_signature="any → stats",
    summary=(
        'Aggregate statistics for a field (numeric min/max/mean/null-rate, or '
        'categorical distinct-count/top-k) — never individual values dumped '
        "in bulk. `top_k`'s *values* are subject to the same byte cap and the "
        'same in-place markers as `peek`; its counts never are'
    ),
)
def describe(registry: DataRegistry, params: DescribeParams) -> OperationOutput:
    """Aggregate statistics for a field: min/max/mean/null-rate for numeric
    fields, distinct-count/top-k for categorical ones. With no field given,
    returns a schema summary (field names and inferred types) instead of
    individual values. Works fully under `privacy_mode`
    — the whole point of `describe` is that it's already aggregate, not
    individual-record content — except the categorical top-k *values*
    (real field content, unlike a count) are redacted."""
    records = _records_from_handle(registry, params.handle)

    if params.field is None:
        fields: dict[str, str] = {}
        for record in records[:_SCHEMA_SAMPLE_SIZE]:
            if not isinstance(record, dict):
                continue
            for k, v in record.items():
                if k not in fields and v is not None:
                    fields[k] = type(v).__name__
        return OperationOutput(status="ok", result={"handle": params.handle, "fields": fields})

    all_values = [r.get(params.field) for r in records if isinstance(r, dict)]
    non_null = [v for v in all_values if v is not None]
    null_rate = 1 - (len(non_null) / len(all_values)) if all_values else 0.0

    if non_null and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null
    ):
        stats = _numeric_stats(non_null)
        stats["null_rate"] = null_rate
        return OperationOutput(
            status="ok", result={"handle": params.handle, "field": params.field, **stats}
        )

    counts: dict[Any, int] = {}
    for v in non_null:
        counts[v] = counts.get(v, 0) + 1

    result: dict[str, Any] = {
        "handle": params.handle,
        "field": params.field,
        "distinct_count": len(counts),
        "null_rate": null_rate,
    }
    if registry.privacy_mode:
        result["note"] = "privacy_mode is enabled: top_k values redacted"
    else:
        # The one part of a describe payload that carries real field values
        # rather than counts, so the one part that needs peek's byte ceiling
        # too — same budget mechanism, same in-place markers, rather than a
        # second convention for the same problem. _DESCRIBE_TOP_K alone bounds
        # how many values come back and says nothing about how large they are.
        top_k = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:_DESCRIBE_TOP_K]
        # The *values* are fitted, one budget share each; the counts never are.
        # Fitting whole `{"value": ..., "count": ...}` entries would let a long
        # value spend the room its own count needed and drop it -- surrendering
        # the aggregate half of describe, which is the half that is always safe
        # to return and the only half privacy_mode keeps.
        share = _DESCRIBE_MAX_BYTES // max(1, len(top_k))
        entries = []
        truncated = False
        for value, count in top_k:
            fitted, _, cut = _fit(value, share, max_items=_DESCRIBE_TOP_K, redact=False)
            entries.append({"value": fitted, "count": count})
            truncated = truncated or cut
        result["top_k"] = entries
        if truncated:
            result["truncated"] = True

    return OperationOutput(status="ok", result=result)
