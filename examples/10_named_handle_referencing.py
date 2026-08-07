"""Named/semantic handle referencing (USECASES.md UC8), solved by a real LLM.

A developer registers three in-memory datasets before running the agent and
refers to them by name in the task text, the same way they'd name variables
in code — without explaining in the prompt itself which handle ID
corresponds to which dataset. This works because handle identity and name
were unified (Task #38, DESIGN.md Section 2's Pointer-Based State
Management bullet): `people_list` below *is* the real DataRegistry handle,
not a label the model has to resolve to one.

org_list is a list of {department, office} records, not a raw scalar dict
(department -> office string) as UC8 was originally drafted — building this
example surfaced that join_lookup only merges in dict values that are
themselves records (built via index_by/group_by); a bare key->scalar dict
isn't a supported join_lookup input today. See USECASES.md UC8 for that
note in full. The model resolves both org_list and place_list to dicts via
index_by before join_lookup can chain them onto people_list.

Run:  uv run --env-file .env --extra agent python examples/10_named_handle_referencing.py
"""

import json
import sys

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
        name="people_list",
        description="employees with their department and age",
        created_by="example_script",
    )

    registry.create(
        "list",
        [
            {"department": "Engineering", "office": "Building A"},
            {"department": "Sales", "office": "Building B"},
        ],
        name="org_list",
        description="each department's office building",
        created_by="example_script",
    )

    registry.create(
        "list",
        [
            {"office": "Building A", "city": "Austin", "capacity": 120},
            {"office": "Building B", "city": "Denver", "capacity": 60},
        ],
        name="place_list",
        description="each office building's city and seating capacity",
        created_by="example_script",
    )

    task = (
        "Using people_list and org_list, find everyone in Engineering, attach their office "
        "building from org_list, then attach that office's city and capacity from place_list."
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
    if result.output.handle:
        data = registry.materialize(result.output.handle)
        print(f"  handle: {result.output.handle}")
        print(f"  data (from the registry, not LLM prose): {json.dumps(data, indent=2)}")


if __name__ == "__main__":
    main()
