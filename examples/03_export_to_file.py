"""Hello World #3: an LLM writing its result straight to a file.

Same task and data as examples/01_hello_world_memory.py, but this time the
task asks for the result to be written to a file instead of just answered
in place. The model calls export_handle as its last action — the filtered
records are written directly to disk by DALE, never passed back through the
model's own context. The model's structured output (dale.agent.DaleResult)
then reports `exported_to`, not `handle`, and its `note` never contains the
actual row content, only a summary.

This is the export_handle primitive, the file-writing half
of DALE's strict-privacy design: intermediate steps already never let the
LLM author data content, and export_handle closes the same gap at final
delivery — a task can complete with zero real data values ever reaching the
model, which a code-generation/debug-loop approach structurally cannot do.

export_handle resolves `destination` through FileRegistry.register_output —
a real path the invoker names ahead of time, exactly like load_csv's `file`
param on the read side (see examples/02_hello_world_file.py) — never a path
the model constructs itself.

Requires: uv sync --extra agent
Requires: one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or
          GOOGLE_API_KEY) set — e.g. via .env (see .env.example).

Run:  uv run --env-file .env --extra agent python examples/03_export_to_file.py
"""

import sys
import tempfile
from pathlib import Path

import dale

try:
    from dale.agent import ActionLog, build_agent, pick_model
except ImportError:
    print(
        "Missing the 'pydantic-ai' package. Install the agent extra:\n"
        "  uv sync --extra agent\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    registry = dale.DataRegistry(files=dale.FileRegistry())
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
        created_by="hello_world_export",
    )

    output_path = Path(tempfile.gettempdir()) / "dale_engineering_staff.csv"
    registry.files.register_output("engineering_staff.csv", output_path)

    task = (
        "Filter to just the people in Engineering, then export the result "
        "to the file registered as 'engineering_staff.csv'."
    )

    try:
        model = pick_model()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    print(f"MODEL: {model}")
    print(f"TASK: {task}\n")

    action_log = ActionLog()
    action_log.seed_from_registry(registry)
    agent = build_agent(registry, action_log, model=model)

    result = agent.run_sync(task, deps=registry)

    print("ACTION LOG:")
    print(action_log.render())

    print(f"\nFINAL RESULT:\n  note: {result.output.note}")
    if result.output.exported_to:
        print(f"  exported_to: {result.output.exported_to}")
        print(f"  real file on disk: {output_path}")
        print(f"  file content:\n{output_path.read_text()}")


if __name__ == "__main__":
    main()
