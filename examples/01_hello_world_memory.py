"""Hello World #1: an LLM against in-memory data.

The shortest possible dale.agent example. Data is created directly in a
DataRegistry — no file, no FileRegistry — and handed to an agent along with
a one-line, plain-English task. The model never sees the rows themselves,
only handle metadata (dale.agent.registry_state_summary) and whatever a
tool call explicitly returns; it solves the task entirely by choosing
primitives and structured parameters, never by writing code.

See examples/02_hello_world_file.py for the file-backed counterpart, and
examples/04_llm_orchestrated.py for a longer, multi-step version of this
same idea.

Requires: uv sync --extra agent
Requires: one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or
          GOOGLE_API_KEY) set — e.g. via .env (see .env.example).

Run:  uv run --env-file .env --extra agent python examples/01_hello_world_memory.py

DALE_VERBOSITY controls how much prints live as the agent works: "quiet"
(default) prints nothing but MODEL/TASK/FINAL RESULT; "normal" adds a CALL
line + one-line RETURN status + REGISTRY STATE per step; "debug" adds each
call's full JSON RETURN payload; "raw" adds the literal exchange with the
model itself (system/user prompt, the model's exact tool-call args before
DALE strips `intent`, tool returns) interleaved with the human-readable
trace — e.g.:
  DALE_VERBOSITY=raw uv run --env-file .env --extra agent python examples/01_hello_world_memory.py
"""

import os
import sys

import dale

try:
    from dale.agent import ActionLog, build_agent, pick_model, run_agent
except ImportError:
    print(
        "Missing the 'pydantic-ai' package. Install the agent extra:\n"
        "  uv sync --extra agent\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    registry = dale.DataRegistry()
    registry.create(
        "list",
        [
            {"name": "Alice", "department": "Engineering", "age": 34},
            {"name": "Bob", "department": "Sales", "age": 29},
            {"name": "Carol", "department": "Engineering", "age": 41},
            {"name": "Dave", "department": "Sales", "age": 37},
            {"name": "Erin", "department": "Engineering", "age": 26},
        ],
        name="people",
        description="employee roster with name, department, and age",
        created_by="hello_world_memory",
    )

    task = "How many people are in Engineering, and what are their names?"

    try:
        model = pick_model()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    verbosity = os.environ.get("DALE_VERBOSITY", "quiet")

    print(f"MODEL: {model}")
    print(f"TASK: {task}\n")

    action_log = ActionLog()
    action_log.seed_from_registry(registry)
    agent = build_agent(registry, action_log, model=model, verbosity=verbosity)

    # run_agent, not a bare agent.run_sync() -- it prints a Result: section
    # for any verbosity != "quiet" and, under "raw", flushes the final leg
    # of the exchange (the last tool return + the model's closing output)
    # that no live per-step hook ever sees on its own. verbosity != "quiet"
    # already prints each step live as it happens, so there's nothing left
    # to print here afterward -- no redundant re-render of the whole trace.
    outcome = run_agent(agent, task, deps=registry, action_log=action_log, verbosity=verbosity)
    if not outcome.success:
        raise SystemExit(1)

    # The agent's output is structured (dale.agent.DaleResult), not free text —
    # `note` is a one-sentence summary for a human, never the source of truth.
    # The real answer is read straight from the registry, not from LLM prose.
    result = outcome.result
    print(f"\nFINAL RESULT:\n  note: {result.output.note}")
    if result.output.handle:
        data = registry.materialize(result.output.handle)
        print(f"  handle: {result.output.handle}")
        print(f"  data (from the registry, not LLM prose): {data}")


if __name__ == "__main__":
    main()
