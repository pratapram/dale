"""`python -m dale` — read the operation catalog from the terminal.

Deliberately tiny. It prints; it does not write files, run models, or touch a
registry. Regenerating GUIDE.md's table is a repo maintenance job and lives in
`scripts/sync_guide_operations.py`, not here: a library people pip-install
should not grow a subcommand that rewrites files on disk.

Because it reads the live catalog rather than a baked-in list, an operation a
downstream developer registered with `@operation` shows up next to the
built-ins without this file knowing anything about it.
"""

from __future__ import annotations

import argparse
import sys

import dale
from dale.errors import DaleError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dale",
        description="Inspect the DALE operation catalog.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ops = sub.add_parser(
        "operations",
        help="print the operation catalog",
        description=(
            "Print the operation catalog. With no NAME, prints every registered "
            "operation; with one or more, prints just those, in the order given."
        ),
    )
    ops.add_argument("name", nargs="*", help="operation name(s) to print")
    ops.add_argument(
        "--format",
        choices=("full", "compact", "markdown"),
        default="full",
        help=(
            "full: every operation with its whole purpose and parameters "
            "(default). compact: the catalog at a glance. markdown: the table "
            "as it appears in GUIDE.md."
        ),
    )
    ops.add_argument(
        "--width",
        type=int,
        default=88,
        help="target line width for wrapped prose (default: 88)",
    )

    args = parser.parse_args(argv)

    try:
        print(
            dale.render_catalog(
                args.name or None, format=args.format, width=args.width
            )
        )
    except DaleError as exc:
        # An unknown operation name is the one error a caller can actually hit
        # here. A traceback would bury the useful part.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
