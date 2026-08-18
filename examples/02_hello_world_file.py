"""Hello World #2: an LLM reading from a file and writing back to one.

Same data as examples/01_hello_world_memory.py, but this time both ends are
files, not just the source: the model loads 'people.csv' via load_csv, then
writes its filtered result straight to another file via export_handle — the
filtered rows are written to disk by DALE, never passed back through the
model's own context.

The model never sees a real filesystem path on either end — only the
virtual names we register — and it calls load_csv/export_handle itself,
exactly like every other operation.

FileRegistry.register(name, path) / register_output(name, path) are the
invoker-side mappings from LLM-visible virtual names to real locations,
resolved only inside load_csv/export_handle, never constructed by the model
(see "The DALE environment" in docs/environment.md for the FileRegistry
entry — this mirrors DataRegistry's own opaque-handle treatment of
in-memory data, applied to file access instead). Note this mapping is
intentionally just name -> Path today: FileRegistry itself doesn't know or
care whether "people.csv" ends up meaning a path on local disk, an object
downloaded from S3/GCS ahead of time, or anything else — the abstraction
already supports registering under a virtual name independent of the real
location; only local paths are wired up yet.

Requires: uv sync --extra agent
Requires: one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY (or
          GOOGLE_API_KEY) set — e.g. via .env (see .env.example).

Run:  uv run --env-file .env --extra agent python examples/02_hello_world_file.py

Prints each step live as the agent works (dale.agent.Verbosity's "normal"
level — CALL line, one-line RETURN status, REGISTRY STATE; no raw JSON).
Set DALE_VERBOSITY=quiet for silent mode (nothing but the final result) or
DALE_VERBOSITY=debug to also see each call's full JSON RETURN payload.
"""

import os
import sys
import tempfile
from pathlib import Path

import dale

try:
    from dale.agent import CORE_OPERATIONS, ActionLog, build_agent, pick_model
except ImportError:
    print(
        "Missing the 'pydantic-ai' package. Install the agent extra:\n"
        "  uv sync --extra agent\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

DATA_FILE = Path(__file__).parent / "data" / "hello_world" / "people.csv"


def main() -> None:
    files = dale.FileRegistry()
    files.register("people.csv", DATA_FILE)
    output_path = Path(tempfile.gettempdir()) / "dale_engineering_staff.csv"
    files.register_output("engineering_staff.csv", output_path)
    registry = dale.DataRegistry(files=files)

    task = (
        "Load the file registered as 'people.csv', filter to just the "
        "people in Engineering, then export the result to the file "
        "registered as 'engineering_staff.csv'."
    )

    try:
        model = pick_model()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    verbosity = os.environ.get("DALE_VERBOSITY", "debug")

    print(f"MODEL: {model}")
    print(f"TASK: {task}\n")

    action_log = ActionLog()
    # `operations=` is how a deployment pays for only what it uses: the
    # run_plan schema is ~78% of every request, and the default (CORE_OPERATIONS)
    # is about half the full catalog. load_csv is not in the core, so this task
    # -- which loads its own file -- has to ask for it.
    agent = build_agent(
        registry,
        action_log,
        model=model,
        verbosity=verbosity,
        operations=[*CORE_OPERATIONS, "load_csv"],
    )

    # verbosity != "quiet" prints each step live as it happens (build_tools'
    # own doing) — nothing further to print here, so no redundant re-print
    # of the whole trace after the run.
    result = agent.run_sync(task, deps=registry)

    # The agent's output is structured (dale.agent.DaleResult), not free text —
    # `note` is a one-sentence summary for a human, never the source of truth.
    # The real answer is read straight from the registry, not from LLM prose.
    print(f"\nFINAL RESULT:\n  note: {result.output.note}")
    if result.output.exported_to:
        print(f"  exported_to: {result.output.exported_to}")
        print(f"  real file on disk: {output_path}")
        print(f"  file content:\n{output_path.read_text()}")
    elif result.output.handle:
        data = registry.materialize(result.output.handle)
        print(f"  handle: {result.output.handle}")
        print(f"  data (from the registry, not LLM prose): {data}")


if __name__ == "__main__":
    main()
