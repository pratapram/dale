"""Importing this package registers every built-in operation into the catalog.
Built-ins use the exact same @operation decorator a third-party developer's
own register_operation call would use later — no separate
'built-in' code path."""

from dale.operations import (  # noqa: F401
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
