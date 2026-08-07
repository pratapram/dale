"""FileRegistry: invoker-controlled mapping from LLM-visible virtual file
names to real local filesystem paths.

`load_csv`'s `path: str` parameter used to accept an LLM-constructed string
directly — Path(path) would open whatever it was given, bounded only by the
OS-level file permissions of the running process. That's an open-ended local
file *read* primitive with no restriction, structurally the same shape of
problem objections.md #1 already ruled out for code ("a condition as a string
is eval() with extra steps"), just pointed at the filesystem instead.

FileRegistry closes it the same way DataRegistry already closes the
equivalent problem for in-memory data (objections #2): the LLM never sees or
constructs a real path, only picks among names the invoker explicitly
registered for this session. See objections.md's FileRegistry entry for the
full design rationale.

The write side (`register_output`/`resolve_output`) is the symmetric
mechanism for `export_handle` (objections #12) — a *destination* an invoker
names ahead of time, never a path the LLM constructs. Kept as a separate
name->Path map from the read side rather than reusing `_files`: registering
a path as writable is a materially different trust decision than registering
it as readable (a destination need not exist yet, and conflating the two
would let a single registration accidentally serve both directions).
"""

from __future__ import annotations

from pathlib import Path


class FileRegistry:
    """Register real local files under invoker-chosen virtual names before
    handing the registry to an LLM. `load_csv` (and any future local loader)
    resolves a virtual name through this registry — never a raw path.

    Registration is trusted, invoker-side setup, not an LLM-facing
    operation — validation failures raise plain exceptions, not a DaleError,
    the same way constructing RegistryLimits does.
    """

    def __init__(self) -> None:
        self._files: dict[str, Path] = {}
        self._output_files: dict[str, Path] = {}

    def register(self, name: str, path: str | Path) -> None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(f"not a file: {path!r}")
        self._files[name] = resolved

    def resolve(self, name: str) -> Path | None:
        return self._files.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._files)

    def register_output(self, name: str, path: str | Path) -> None:
        """Register a virtual destination name for export_handle. Unlike
        register(), the target need not exist yet — that's the point, it's
        where output gets written — but its parent directory must, so a
        typo'd invoker-supplied path fails fast at setup time rather than
        inside a live agent run."""
        resolved = Path(path).resolve()
        if not resolved.parent.is_dir():
            raise ValueError(f"parent directory does not exist: {path!r}")
        self._output_files[name] = resolved

    def resolve_output(self, name: str) -> Path | None:
        return self._output_files.get(name)

    def list_output_names(self) -> list[str]:
        return sorted(self._output_files)
