"""License tier reconciliation — DALE's original motivating problem
(USECASES.md UC6, testcases.md Test Case 4), solved by a real LLM.

Three per-tier eligibility lists are regenerated hourly by an upstream
system. A user can appear on more than one list and must be resolved to
their single highest tier (gold beats silver beats bronze). The result is
diffed against the previous hour's assignment to report who's new, who was
removed, and whose tier changed.

The three tier lists are unioned and tagged with their source tier here, in
plain Python, before DALE ever sees the data — no union/concat primitive
exists in the catalog yet, and this kind of mechanical assembly isn't an
interesting decision for the model to make anyway (objections.md already
treats assembling paginated API responses into one document the same way:
the invoker's job, not the LLM's). The model's job starts at priority_reduce:
resolving the tagged candidates to one tier per user, then dict_diff against
previous_assignments.

Run:  uv run --env-file .env --extra agent python examples/09_license_reconciliation.py
"""

import json
import sys
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

DATA_DIR = Path(__file__).parent / "data" / "license_reconciliation"


def main() -> None:
    registry = dale.DataRegistry()

    candidates = []
    for tier in ("gold", "silver", "bronze"):
        for email in json.loads((DATA_DIR / f"{tier}_users.json").read_text()):
            candidates.append({"email": email, "tier": tier})
    registry.create(
        "list",
        candidates,
        name="tier_candidates",
        description="users' tier-list memberships, one row per (email, tier) pair — a user "
        "eligible for more than one tier appears more than once, unresolved",
        created_by="example_script",
    )

    previous = json.loads((DATA_DIR / "previous_assignments.json").read_text())
    registry.create(
        "dict",
        previous,
        name="previous_assignments",
        description="last hour's resolved tier assignment, keyed by email",
        created_by="example_script",
    )

    task = (
        "Using tier_candidates, resolve each user to their single highest eligible tier "
        "(gold beats silver beats bronze). Then compare the resolved assignment against "
        "previous_assignments and tell me who's new, who was removed, and whose tier changed."
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
