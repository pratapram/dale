"""What the model is told: the system prompt, and automatic inspection.

Two halves of one job. The prompt frames the model as an Algorithmic
Orchestrator and states the rules its tool calls must satisfy; `_auto_inspect`
and `_initial_inspect_summary` are `peek_at_every_step`, which splices a small
peek+describe of each new handle into the result the model sees. Both exist to
tell the model about data it can never look at directly, which is why they
live together rather than with the tool-building or execution code.
"""

import json
from typing import Any

import dale


_AUTO_PEEK_N = 3
"""Sample size for peek_at_every_step's automatic inspection — deliberately
small (unlike a model-initiated peek, which might reasonably ask for more):
this is a shape/sanity-check spliced into every step, not something the
model asked for, so it should stay cheap by default."""


_PRIVACY_MODE_NOTE = """


privacy_mode is enabled for this session: peek() is not available to you at \
all — it isn't one of the steps you can request, so don't spend a call trying \
it. Use \
describe() instead for field-level information: its aggregate statistics \
(min/max/mean/null_rate/distinct_count) are still fully available and are \
your real source of information about field values. Automatic post-step \
inspection is also disabled under privacy_mode, so call describe() yourself \
when you need it."""


def _auto_inspect(registry: dale.DataRegistry, handle: str) -> dict[str, Any] | None:
    """Run peek+describe directly against the catalog function, not through
    dale.call_operation — this is bonus information spliced into an existing
    tool result or the system prompt, not a call the model made, so it
    shouldn't consume a max_tool_calls slot or appear as its own ActionLog
    entry. Returns None if the handle can't be inspected (best-effort
    augmentation, not load-bearing — a failure here shouldn't break the
    call it's attached to). Caller is responsible for checking
    registry.privacy_mode first — this function doesn't gate on it, since
    peek/describe already redact correctly on their own (see
    src/dale/operations/inspect.py) and the "skip when privacy_mode" policy
    is a peek_at_every_step decision, not an inspection-operation one."""
    try:
        peek_spec = dale.get_operation("peek")
        describe_spec = dale.get_operation("describe")
        peek_out = peek_spec.fn(registry, peek_spec.param_schema(handle=handle, n=_AUTO_PEEK_N))
        describe_out = describe_spec.fn(registry, describe_spec.param_schema(handle=handle))
    except dale.DaleError:
        return None
    return {"peek": peek_out.result, "describe": describe_out.result}


def _initial_inspect_summary(registry: dale.DataRegistry) -> str:
    """Auto peek+describe for every handle already in the registry before
    the agent's first turn — part of peek_at_every_step (default on): saves
    the model spending its first turn on exploratory calls it would almost
    certainly make anyway. Callers skip invoking this entirely under
    privacy_mode (see build_agent) rather than relying on it to no-op."""
    lines = []
    for meta in registry.list_handles():
        inspected = _auto_inspect(registry, meta.name)
        if inspected is None:
            continue
        lines.append(
            f"- {meta.name}:\n"
            f"    peek: {json.dumps(inspected['peek'], default=str)}\n"
            f"    describe: {json.dumps(inspected['describe'], default=str)}"
        )
    if not lines:
        return ""
    return (
        "Initial peek/describe for each handle above (peek_at_every_step is enabled):\n"
        + "\n".join(lines)
    )


DEFAULT_SYSTEM_PROMPT = """\
You are an Algorithmic Orchestrator. You solve data-processing tasks by \
calling a small set of declarative operations against data held in host \
memory. You never see the underlying data directly — only structural metadata \
and whatever a call explicitly returns — and you never write or execute code. \
You reach every operation through one tool, `run_plan`, which takes a list of \
`steps`: one step per operation call, run in order. A single call is simply a \
list of one, so batch whenever you already know the shape of what you need. \
Every step must use structured parameters (field/op/value predicates, \
computed-field specs, etc.), never expressions or code strings. Every step \
also takes an `intent` field — one short sentence on why you're making that \
specific call. Always fill it in; it becomes part of an auditable \
action log. Steps that create a new handle also take `name` and `description`. \
`name` becomes that handle's real identifier — every later step \
references it by exactly this name, so it must read like a Python variable \
name and must not already be in use. Suffix it with the handle's type (e.g. \
"in_stock_items_list", "org_dict" — not "list_5") so the type is legible \
from the name alone. `description` is one sentence on what the handle contains, e.g. "products \
currently in stock, sorted by margin" — or, honestly, "unknown, not yet \
inspected" when that's true; that's a complete and useful answer, not one to \
avoid. Always fill both in; they make the action log and the current \
registry state readable without inspecting the data itself.

When you finish, your final output must point at the real result, never \
restate it: set `handle` to the handle holding the answer, or, if the task \
asked you to write output to a file, call export_handle and set \
`exported_to` to that destination instead. `note` is a one-sentence summary \
for a human skimming the log — it is not the source of truth for the \
result, and the invoker will read the handle's real data or the exported \
file directly, not your note.

Data currently available to you:
{registry_state}{privacy_note}"""


def registry_state_summary(registry: dale.DataRegistry) -> str:
    lines = []
    for meta in registry.list_handles():
        extra = f", value_shape={meta.value_shape}" if meta.value_shape else ""
        lines.append(f"- {meta.name}: {meta.type}, size={meta.size}{extra} — {meta.description}")
    return "\n".join(lines) if lines else "(none loaded yet)"


def default_system_prompt(
    registry: dale.DataRegistry, *, peek_at_every_step: bool = True
) -> str:
    """The system prompt build_agent uses when the caller supplies none.

    Public, and extracted from build_agent rather than left inline, because
    measuring DALE's own context footprint (paper.md Section 4.2 part C)
    requires counting the exact prompt a real run would send — reconstructing
    an approximation of it in the eval code would measure the reconstruction
    instead, and would drift the first time this changes. eval/baseline.py
    counts this plus the live tool schemas.

    `peek_at_every_step` is ignored when registry.privacy_mode is on, matching
    build_agent: that flag always wins, and there is nothing unredacted left
    to splice in anyway."""
    state = registry_state_summary(registry)
    if peek_at_every_step and not registry.privacy_mode:
        initial = _initial_inspect_summary(registry)
        if initial:
            state = f"{state}\n\n{initial}"
    return DEFAULT_SYSTEM_PROMPT.format(
        registry_state=state,
        privacy_note=_PRIVACY_MODE_NOTE if registry.privacy_mode else "",
    )
