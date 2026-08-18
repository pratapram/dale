"""Regenerate GUIDE.md's operation table from the catalog.

The table used to be hand-written, which meant nothing stopped it drifting
from the code — a new operation simply went undocumented, and a changed one
kept its old description forever. Its two columns that were not derivable
(`In → Out`, and the Purpose prose) now live on `OperationSpec` as
`io_signature` and `summary`, so the table can be generated from the catalog
instead.

Same shape as `capture_trace_baseline.py`, and for the same reason: a script
regenerates the artifact, a test asserts it is current.

    uv run python scripts/sync_guide_operations.py            # rewrite the block
    uv run python scripts/sync_guide_operations.py --check    # exit 1 if stale

Only the rows between the markers are touched. The prose around the table —
including the "The table gives purpose; the library gives parameters"
paragraph below it — stays hand-written.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

import dale

GUIDE = pathlib.Path(__file__).resolve().parent.parent / "GUIDE.md"

BEGIN = "<!-- BEGIN GENERATED: operations. Regenerate with scripts/sync_guide_operations.py -->"
END = "<!-- END GENERATED: operations -->"


def _split(text: str) -> tuple[str, str, str]:
    """(before, current_block, after). Raises if the markers are missing or
    out of order — silently doing nothing would be the worst outcome for a
    script whose whole job is keeping a file current."""
    try:
        head, rest = text.split(BEGIN, 1)
        block, tail = rest.split(END, 1)
    except ValueError:
        raise SystemExit(
            f"{GUIDE.name}: could not find the generated-operations markers.\n"
            f"  expected {BEGIN}\n"
            f"       and {END}"
        ) from None
    return head, block, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="don't write; exit 1 with a diff if the table is out of date",
    )
    args = parser.parse_args(argv)

    text = GUIDE.read_text()
    head, current, tail = _split(text)
    generated = "\n" + dale.render_catalog(format="markdown") + "\n"

    if current == generated:
        if args.check:
            print(f"{GUIDE.name}: operation table is up to date")
        return 0

    if args.check:
        print(f"{GUIDE.name}: operation table is out of date.", file=sys.stderr)
        print(
            "Run: uv run python scripts/sync_guide_operations.py\n", file=sys.stderr
        )
        diff = difflib.unified_diff(
            current.splitlines(),
            generated.splitlines(),
            fromfile=f"{GUIDE.name} (committed)",
            tofile="generated from the catalog",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        return 1

    GUIDE.write_text(head + BEGIN + generated + END + tail)
    print(f"{GUIDE.name}: operation table regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
