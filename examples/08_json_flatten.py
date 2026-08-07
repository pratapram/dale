"""Nested JSON: load a real API response and explode a nested array field
into rows, carrying selected parent fields down onto each one.

Run: uv run python examples/08_json_flatten.py

Data is real, not synthetic: examples/data/json_flatten_github_issues/pandas_issues.json
is a trimmed, live-fetched snapshot of real GitHub issues from pandas-dev/pandas
(number/title/labels only — see examples/data/README.md). Only one of the 8
issues in this snapshot has any labels; the rest have "labels": [] and
correctly contribute zero rows to the result — no separate filter step
needed, see JSON_FEATURE.md / testcases.md for the full design rationale.

Like examples/01-03, this calls primitives directly (no LLM, no API key
needed) — the "LLM" in a real deployment would choose these same primitive
names and structured arguments; here we call them directly to keep the
example self-contained and runnable everywhere. examples/04_llm_orchestrated.py
shows the same kind of pipeline chosen by a real model instead.
"""

from __future__ import annotations

from pathlib import Path

import dale

DATA_FILE = Path(__file__).parent / "data" / "json_flatten_github_issues" / "pandas_issues.json"

files = dale.FileRegistry()
files.register("pandas_issues.json", DATA_FILE)
registry = dale.DataRegistry(files=files)

# load_json: a top-level JSON array becomes a list handle, same as load_csv.
loaded = dale.call_primitive(
    registry,
    "load_json",
    {
        "file": "pandas_issues.json",
        "name": "pandas_issues",
        "description": "real pandas-dev/pandas GitHub issues (number, title, labels)",
    },
)
print(f"loaded: {loaded.handle.size} issues")

# flatten_json: explode the nested "labels" array on each issue into its own
# row, carrying "number"/"title" down from the parent issue onto each one.
# An issue with an empty "labels" array contributes zero rows — nothing
# extra to do for that.
flattened = dale.call_primitive(
    registry,
    "flatten_json",
    {
        "handle": loaded.handle.handle,
        "path": ["labels"],
        "carry_fields": ["number", "title"],
        "name": "issue_labels",
        "description": "one row per label, with the issue number/title attached",
    },
)
print(f"flattened: {flattened.handle.size} label rows (from {loaded.handle.size} issues)\n")

print("Result (one row per label):")
for row in registry.materialize(flattened.handle.handle):
    print(f"  #{row['number']}  {row['name']:<8} {row['title'][:60]}")

# release_handle: explicit cleanup once a handle is no longer needed.
for h in (loaded.handle.handle, flattened.handle.handle):
    dale.call_primitive(registry, "release_handle", {"handle": h})
print(f"\nhandles remaining after cleanup: {registry.handle_count()}")
