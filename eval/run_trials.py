"""CLI entry point for trial runs.

Run:  uv run --env-file .env --extra agent python -m eval.run_trials <use_case> <model> [n]
e.g.  uv run --env-file .env --extra agent python -m eval.run_trials uc1 anthropic:claude-sonnet-5 10

Use `test` as the model to run against pydantic_ai's built-in TestModel — no
API key or cost, structural verification only (does the harness itself work),
not a real correctness signal. Token figures are suppressed for it, since
TestModel reports synthetic usage (see dale.agent.TokenUsage).

Flags:
  --steps-per-call N cap how many operation calls the model may batch into one
                     turn. `1` is the unbatched condition: run a use case with
                     and without it and compare the reported mean request count
                     — that comparison is paper.md Section 4.2 part (F).
  --max-requests N   cap requests per trial (pydantic-ai defaults to 50 when
                     unset). Live-spend insurance for a real batch.
"""

import argparse
import sys

try:
    from pydantic_ai.usage import UsageLimits

    from eval.harness import describe_setup, run_trials
    from eval.use_cases import USE_CASES
except ImportError:
    print(
        "Missing the 'pydantic-ai' package. Install the agent extra:\n"
        "  uv sync --extra agent\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_trials")
    parser.add_argument("use_case", choices=sorted(USE_CASES), metavar="use_case")
    parser.add_argument("model", help="e.g. anthropic:claude-sonnet-5, or 'test'")
    parser.add_argument("n", nargs="?", type=int, default=10, help="trials (default 10)")
    parser.add_argument(
        "--steps-per-call",
        type=int,
        default=None,
        metavar="N",
        help="cap steps per turn; 1 is the unbatched condition (Section 4.2 part F)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        metavar="N",
        help="cap model requests per trial (pydantic-ai's own default is 50)",
    )
    args = parser.parse_args()

    setup, task, checker = USE_CASES[args.use_case]
    # `is not None`, not truthiness: `--max-requests 0` is a real thing to type
    # (as "don't let this batch spend anything"), and a bare `if` silently read
    # it as "unset" and handed the run pydantic-ai's 50-request default — the
    # exact opposite of what was asked, on a flag whose whole job is capping
    # live spend. It isn't a *sensible* value, so it's rejected outright rather
    # than honored: a trial needs at least one request to do anything at all.
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be at least 1 (a trial needs at least one request)")
    # Same reasoning, same trap: `--steps-per-call 0` publishes a `steps` field
    # with maxItems 0, i.e. a tool the model cannot legally call at all.
    if args.steps_per_call is not None and args.steps_per_call < 1:
        parser.error("--steps-per-call must be at least 1 (a plan needs at least one step)")
    usage_limits = (
        UsageLimits(request_limit=args.max_requests) if args.max_requests is not None else None
    )

    print(describe_setup(args.use_case, setup, task), flush=True)
    print(f"\nRunning {args.n} trial(s) against {args.model}...\n", flush=True)
    summary = run_trials(
        args.use_case,
        setup,
        task,
        checker,
        args.model,
        args.n,
        max_steps_per_call=args.steps_per_call,
        usage_limits=usage_limits,
    )
    print(summary.render())


if __name__ == "__main__":
    main()
