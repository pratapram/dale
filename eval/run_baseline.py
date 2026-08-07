"""CLI entry point for the raw-data-in-prompt baseline (paper.md Section 4.2
part (C)).

Run:  uv run --env-file .env --extra agent python -m eval.run_baseline <use_case> [model]
e.g.  uv run --env-file .env --extra agent python -m eval.run_baseline uc3 claude-sonnet-5

Counting is the default and costs **no inference** — `anthropic.messages.count_tokens`
is a real HTTP call (so ANTHROPIC_API_KEY must be set) but runs no model, which
is what makes a whole scaling curve affordable. `--run` opts into the expensive
mode: actually answering the task with the entire dataset in the prompt.

`model` is a bare Anthropic model id (`claude-sonnet-5`), not pydantic-ai's
prefixed form — token counts come from the Anthropic SDK directly. `--run`
adds the `anthropic:` prefix itself. Token counts are model-specific, and this
default is *not* the same as `dale.agent.pick_model`'s live-run default
(`claude-haiku-4-5`): pass the id explicitly to match whatever model the trial
runs you are comparing against actually used.

Flags:
  --format csv|tsv|json   how the baseline serializes records (default csv).
                          CSV is the headline number: for these flat record
                          shapes JSON costs 2-3x more per row, so publishing a
                          JSON curve would inflate the baseline for a reason
                          unrelated to DALE. Pass `json` to show that gap, not
                          to report it as the baseline.
  --count-only            (default) count tokens for both arms, no inference.
  --run                   also run the baseline through the model and grade it
                          against eval.use_cases.EXPECTED_ANSWERS. Costs real
                          money, and at large dataset sizes may not fit.
  --max-requests N        cap model requests per trial when --run is given.

For the scale curve, set $DALE_SCALE_ROWS and use `uc3_large` — same task,
same schema, only the row count changes.
"""

import argparse
import sys

try:
    from pydantic_ai.usage import UsageLimits

    from eval.baseline import FORMATS, AnthropicTokenCounter, count_arms, run_baseline_trial
    from eval.use_cases import USE_CASES
except ImportError as exc:
    # Only pydantic-ai's absence gets the friendly message. This block covers
    # four imports, and reporting "install the agent extra" for a broken
    # eval.baseline or a renamed eval.use_cases would send someone off fixing
    # their environment over a typo in this repo.
    if not (exc.name == "pydantic_ai" or (exc.name or "").startswith("pydantic_ai.")):
        raise
    print(
        "Missing the 'pydantic-ai' package. Install the agent extra:\n"
        "  uv sync --extra agent\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_baseline")
    parser.add_argument("use_case", choices=sorted(USE_CASES), metavar="use_case")
    parser.add_argument(
        "model",
        nargs="?",
        default="claude-sonnet-5",
        help="bare Anthropic model id, e.g. claude-sonnet-5 (no 'anthropic:' prefix)",
    )
    parser.add_argument("n", nargs="?", type=int, default=1, help="trials for --run (default 1)")
    parser.add_argument("--format", choices=FORMATS, default="csv", help="baseline serialization")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--count-only",
        action="store_true",
        help="(default) count tokens for both arms; no inference, no cost",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="also run the baseline through the model and grade it (costs inference)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        metavar="N",
        help="cap model requests per --run trial (pydantic-ai's own default is 50)",
    )
    args = parser.parse_args()

    # Same reasoning as run_trials.py: `--max-requests 0` is a real thing to
    # type and means the opposite of "unset", so it is rejected rather than
    # silently replaced by pydantic-ai's 50-request default.
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be at least 1 (a trial needs at least one request)")

    counter = AnthropicTokenCounter(args.model)
    print(f"Counting context for {args.use_case} against {args.model}...\n", flush=True)
    print(count_arms(args.use_case, counter, fmt=args.format, model=args.model).render())

    if not args.run:
        return

    usage_limits = (
        UsageLimits(request_limit=args.max_requests) if args.max_requests is not None else None
    )
    model = args.model if ":" in args.model else f"anthropic:{args.model}"
    print(f"\nRunning {args.n} baseline trial(s) against {model}...\n", flush=True)
    successes = 0
    for i in range(args.n):
        result = run_baseline_trial(
            i + 1, args.use_case, model, fmt=args.format, usage_limits=usage_limits
        )
        successes += result.success
        outcome = "ok" if result.success else "FAILED"
        print(
            f"  [{i + 1}/{args.n}] {outcome} in {result.wall_clock_s:.1f}s — "
            f"{result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out",
            flush=True,
        )
        if result.error:
            print(f"      error: {result.error}", flush=True)
        elif not result.success:
            print(f"      answer: {result.final_answer}", flush=True)
    print(f"\nbaseline correctness: {successes}/{args.n}")


if __name__ == "__main__":
    main()
