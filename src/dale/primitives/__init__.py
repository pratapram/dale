"""Importing this package registers every built-in primitive into the catalog.
Built-ins use the exact same @primitive decorator a third-party developer's
own register_primitive call would use later — no separate
'built-in' code path."""

from dale.primitives import (  # noqa: F401
    compute,
    diff,
    export,
    filter_,
    flatten,
    graph,
    index,
    inspect,
    io,
    join,
    lifecycle,
    sort,
    window,
)
