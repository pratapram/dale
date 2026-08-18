from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
from pydantic import BaseModel

import dale
from dale.agent import ALL_OPERATIONS
from dale.catalog import get_operation, register_operation
from dale.describe import _GUIDE_ORDER, _signature, _type_of
from dale.errors import OperationNotFoundError

# Underscore-prefixed names are test-only registrations other modules leave in
# the module-level catalog with no teardown (see
# test_operations_core.test_list_operations_publishes_exactly_the_builtin_catalog).
# Every assertion about "the built-ins" has to exclude them or it depends on
# test ordering.
BUILTINS = [n for n in dale.list_operations() if not n.startswith("_")]


# ---------------------------------------------------------------------------
# the metadata itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BUILTINS)
def test_every_builtin_declares_io_signature_and_summary(name):
    """The drift guard, and the reason this change exists at all. GUIDE.md's
    table is generated from these two fields, so a built-in missing either one
    would silently produce a blank cell in the published documentation. Failing
    here — at the operation, by name — says what to go fix."""
    spec = get_operation(name)
    assert spec.io_signature, f"{name} has no io_signature"
    assert spec.summary, f"{name} has no summary"
    assert "\n" not in spec.summary, f"{name}'s summary must be a single line"


@pytest.mark.parametrize("name", BUILTINS)
def test_every_builtin_has_a_docstring(name):
    """`fn.__doc__` is the model's tool description (agent/tools.py builds the
    `run_plan` step union from it), so an operation without one is invisible to
    the model in a way no type checker catches. It is also `summary`'s
    fallback."""
    assert (get_operation(name).fn.__doc__ or "").strip()


def test_guide_order_is_exactly_the_builtin_catalog():
    """The markdown table renders `_GUIDE_ORDER` first and appends anything it
    doesn't know about, so a missing name would still be published — just
    dumped at the bottom, out of the curated pipeline order. Assert the two
    agree instead of trusting that fallback."""
    assert sorted(_GUIDE_ORDER) == sorted(BUILTINS)
    assert len(_GUIDE_ORDER) == len(set(_GUIDE_ORDER))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["full", "compact", "markdown"])
def test_every_format_renders_the_whole_catalog(fmt):
    out = dale.render_catalog(format=fmt)
    for name in BUILTINS:
        assert name in out, f"{name} missing from the {fmt} rendering"


def test_markdown_is_a_well_formed_table_with_one_row_per_operation():
    lines = dale.render_catalog(format="markdown").splitlines()
    assert lines[0] == "| Operation | In → Out | Purpose |"
    assert lines[1] == "|---|---|---|"
    rows = lines[2:]
    assert len(rows) == len(BUILTINS)
    for row in rows:
        # Three cells means exactly four pipes, so an unescaped pipe inside a
        # cell — which would shift every later cell one column left — fails
        # here rather than silently corrupting the published table.
        assert row.count("|") - row.count(r"\|") == 4, row


def test_markdown_escapes_pipes_that_the_other_formats_show_plainly():
    """`load_json` is `file → list | dict`. Stored unescaped so `full` and
    `compact` read naturally; escaped only on the way into a table cell."""
    assert get_operation("load_json").io_signature == "file → list | dict"
    md = dale.render_catalog(["load_json"], format="markdown")
    assert r"file → list \| dict" in md
    assert "file → list | dict" in dale.render_catalog(["load_json"], format="compact")


def _compact_order(out: str) -> list[str]:
    """The operation names in the order `compact` printed them.

    Read off the header lines rather than with `str.index`, because plenty of
    operation names occur inside other operations' prose and parameter lists —
    `window_flag`'s signature contains `group_by`, for one — and a substring
    search finds those first."""
    return [line.split("  ")[0] for line in out.splitlines() if not line.startswith(" ")]


def test_explicit_names_render_in_the_order_given():
    """`dale.agent` passes the operations one agent was actually offered, and
    that order is the order the model sees them in — not sorted."""
    names = ["sort_by", "load_csv", "peek"]
    assert _compact_order(dale.render_catalog(names, format="compact")) == names


def test_default_order_is_the_catalog_order():
    out = dale.render_catalog(BUILTINS, format="compact")
    assert _compact_order(out) == sorted(BUILTINS)


def test_unknown_name_raises_operation_not_found():
    with pytest.raises(OperationNotFoundError):
        dale.render_catalog(["definitely_not_an_operation"])


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown catalog format"):
        dale.render_catalog(format="yaml")


def test_full_format_shows_parameters_with_required_and_defaults():
    out = dale.render_catalog(["reduce_by"], format="full")
    assert "handle" in out and "required" in out
    assert "value_field" in out and "default=None" in out
    # The pydantic Field(description=...) prose rides along, since that is what
    # the model is told about the parameter.
    assert "Ordering keys, most significant first" in out


def test_signature_puts_required_parameters_before_optional_ones():
    """`confirm` is inherited from `ConfirmableParams` and lands first in schema
    order, which reads like a Python syntax error on every cost-gated
    operation."""
    sig = _signature(get_operation("join_lookup"))
    assert sig.index("base_handle") < sig.index("confirm=False")
    assert sig.startswith("(base_handle")


@pytest.mark.parametrize(
    "node, expected",
    [
        ({"type": "string"}, "string"),
        ({"type": "array", "items": {"type": "string"}}, "list[string]"),
        ({"$ref": "#/$defs/OrderKey"}, "OrderKey"),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "string | null"),
        ({"type": "string", "enum": ["left", "inner"]}, "'left' | 'inner'"),
        ({}, "any"),
    ],
)
def test_type_rendering(node, expected):
    assert _type_of(node) == expected


# ---------------------------------------------------------------------------
# third-party operations
# ---------------------------------------------------------------------------


def test_a_third_party_operation_renders_without_the_new_metadata():
    """The whole catalog is the extensibility mechanism, so an operation
    registered by someone who never heard of `io_signature` or `summary` has to
    render anyway — falling back to its docstring rather than raising or
    printing `None`."""

    class _P(BaseModel):
        handle: str

    def _fn(registry, params):  # pragma: no cover - never called
        """A third-party operation that declared no catalog metadata."""

    register_operation("_test_bare_operation", _fn, _P)

    for fmt in ("full", "compact", "markdown"):
        out = dale.render_catalog(["_test_bare_operation"], format=fmt)
        assert "_test_bare_operation" in out
        assert "A third-party operation that declared no catalog metadata." in out
        assert "None" not in out

    # Unknown to _GUIDE_ORDER, so markdown appends it rather than dropping it.
    md = dale.render_catalog(BUILTINS + ["_test_bare_operation"], format="markdown")
    assert md.splitlines()[-1].startswith("| `_test_bare_operation` |")


# ---------------------------------------------------------------------------
# the agent preamble
# ---------------------------------------------------------------------------

# Not `pytest.importorskip` at module scope: that skips the whole file, and
# everything above this point is core-only and must still run in a
# `pydantic_ai`-free env (GUIDE.md documents `uv sync --extra dev` as a
# supported configuration). Skip just these four.
needs_agent = pytest.mark.skipif(
    importlib.util.find_spec("pydantic_ai") is None,
    reason="the agent layer needs the `agent` extra",
)


def _agent_bits():
    """Imported inside the tests, not at module scope: `dale.agent` pulls in
    pydantic_ai, which the core-only env does not have."""
    from dale.agent import build_agent
    from dale.agent.log import ActionLog

    return build_agent, ActionLog


@needs_agent
@pytest.mark.parametrize("verbosity", ["quiet", "normal", "debug"])
def test_no_preamble_below_raw(capsys, verbosity):
    """`raw` is the only level that prints this. `quiet` in particular is
    asserted to print nothing whatsoever (see
    test_agent.test_run_agent_quiet_prints_nothing_extra), and `normal` and
    `debug` are per-step views that a 50-line catalog dump would bury."""
    build_agent, ActionLog = _agent_bits()
    registry = dale.DataRegistry(files=dale.FileRegistry())
    build_agent(registry, ActionLog(), model="test", verbosity=verbosity)
    assert capsys.readouterr().out == ""


@needs_agent
def test_raw_prints_the_catalog_before_the_run(capsys):
    build_agent, ActionLog = _agent_bits()
    registry = dale.DataRegistry(files=dale.FileRegistry())
    build_agent(
        registry, ActionLog(), model="test", verbosity="raw", operations=ALL_OPERATIONS
    )
    out = capsys.readouterr().out
    assert out.startswith("OPERATIONS AVAILABLE TO THE MODEL (")
    for name in BUILTINS:
        assert name in out


@needs_agent
def test_raw_preamble_says_why_the_default_is_narrowed(capsys):
    """The default narrows the catalog too. A header reporting "9 of 17" with
    no reason reads as a bug rather than a setting."""
    build_agent, ActionLog = _agent_bits()
    registry = dale.DataRegistry(files=dale.FileRegistry())
    build_agent(registry, ActionLog(), model="test", verbosity="raw")
    out = capsys.readouterr().out
    assert "core default" in out
    assert "ALL_OPERATIONS" in out


@needs_agent
def test_raw_preamble_reports_the_selection_not_the_catalog(capsys):
    """The header has to describe what this agent can actually call. An
    allowlist narrows it, and `privacy_mode` withholds `peek` outright — a
    preamble claiming the full catalog would credit the model with calls it
    cannot make."""
    build_agent, ActionLog = _agent_bits()
    registry = dale.DataRegistry(files=dale.FileRegistry())
    build_agent(
        registry,
        ActionLog(),
        model="test",
        verbosity="raw",
        operations=["load_csv", "filter_where", "peek"],
    )
    # Derived, not literal: other tests register `_test_*` operations into the
    # module-level catalog with no teardown, so the total depends on what ran
    # first. The point of the assertion is the ratio and the reason, not 17.
    total = len(dale.list_operations())
    out = capsys.readouterr().out
    assert f"(3 of {total} — allowlist)" in out
    assert "sort_by" not in out

    private = dale.DataRegistry(files=dale.FileRegistry(), privacy_mode=True)
    build_agent(
        private, ActionLog(), model="test", verbosity="raw", operations=ALL_OPERATIONS
    )
    out = capsys.readouterr().out
    assert f"({total - 1} of {total} — peek withheld under privacy_mode)" in out
    assert "\npeek " not in out


# ---------------------------------------------------------------------------
# the CLI and the GUIDE.md sync
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parent.parent


def test_cli_prints_the_catalog():
    """Compared against the built-ins rather than `render_catalog()`: this
    process's catalog also holds the `_test_*` operations other tests register
    into it with no teardown, and a fresh subprocess has only the built-ins."""
    proc = _run("-m", "dale", "operations")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip("\n") == dale.render_catalog(BUILTINS)


def test_cli_rejects_an_unknown_operation_without_a_traceback():
    proc = _run("-m", "dale", "operations", "nope")
    assert proc.returncode == 1
    assert "no such operation" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_guide_operation_table_is_in_sync_with_the_catalog():
    """GUIDE.md's table is generated. If this fails, run
    `uv run python scripts/sync_guide_operations.py` — the diff is in stderr."""
    proc = _run("scripts/sync_guide_operations.py", "--check")
    assert proc.returncode == 0, proc.stderr
