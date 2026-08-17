"""The actual pitch, demonstrated: an LLM turns a natural-language task into
DALE operation calls — never code, never raw data in its context — and we
watch it happen via the action log. Examples 01-03 validate the engine in
isolation; this is the one that shows why that engine exists.

Built on dale.agent (the real integration surface, not a one-off demo
script) which wraps PydanticAI (DESIGN.md Stream 3's stated target) — the
same code runs against Anthropic, OpenAI, or Gemini, whichever API key is
available.

Requires: uv sync --extra agent   (adds pydantic-ai; not a core DALE
          dependency — see pyproject.toml's [agent] extra)
Requires: ANTHROPIC_API_KEY and/or OPENAI_API_KEY and/or GEMINI_API_KEY
          (or GOOGLE_API_KEY) set.
Optional: DALE_MODEL to force a specific model string, e.g.
          "anthropic:claude-sonnet-5", "openai:gpt-5.6", or
          "google:gemini-3.6-flash".

Note: filter_where's predicate parameter (And/Or/Not) is a
recursive schema. Gemini's function-calling format cannot represent
recursive $ref/$defs schemas at all (a confirmed upstream Gemini API
limitation, not a pydantic-ai bug — see pydantic/pydantic-ai#1598) — a task
that needs filter_where will fail immediately against google:* models.
Re-verified against the current gemini-3.6-flash generation (not just the
2.5-flash originally tested), same failure: "ref loops are only supported
if they include optional or nullable property values ... but a ref loop of
required fields was found" — a persistent API-level schema-validation
constraint, not a stale-model artifact. Anthropic and OpenAI's tool-calling
formats both support standard JSON Schema $ref/$defs and have no such
restriction — likewise re-confirmed against Kimi (Moonshot), DeepSeek, and
Qwen (Alibaba/DashScope), all of which handle it correctly too.

Run:  uv run --env-file .env --extra agent python examples/04_llm_orchestrated.py

Compare this to examples/05_filter_sort_compute.py, which solves the same
task by calling the same operations directly, by hand. Here, nothing but
the task description and the tool schemas are handed to the model — it
decides which operations to call and with what parameters, and states its
intent for each call, which dale.agent.ActionLog records.

The printed action log renders any filter_where/window_flag predicate as
human-readable boolean logic (dale.render_predicate), not the raw nested
JSON the model actually produced — see grammar.render_predicate.
"""

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
            {"name": "Widget", "price": 25.0, "cost": 10.0, "in_stock": True},
            {"name": "Gadget", "price": 40.0, "cost": 35.0, "in_stock": True},
            {"name": "Gizmo", "price": 15.0, "cost": 20.0, "in_stock": False},
            {"name": "Doohickey", "price": 60.0, "cost": 22.0, "in_stock": True},
        ],
        name="products",
        description="product catalog with price/cost/stock status",
        created_by="example_script",
    )

    task = (
        "Using the product data available to you, keep only the in-stock "
        "products, add a 'margin' field (price minus cost) to each, and "
        "sort them by margin from highest to lowest. Tell me the resulting "
        "order of product names and their margins."
    )

    try:
        model = pick_model()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    print(f"MODEL: {model}")
    print(f"TASK: {task}\n")

    action_log = ActionLog()
    agent = build_agent(registry, action_log, model=model)

    result = agent.run_sync(task, deps=registry)

    print("ACTION LOG:")
    print(action_log.render())

    # The agent's output is structured (dale.agent.DaleResult), not free text —
    # `note` is a one-sentence summary for a human, never the source of truth.
    # The real answer is read straight from the registry, not from LLM prose.
    print(f"\nFINAL RESULT:\n  note: {result.output.note}")
    if result.output.handle:
        data = registry.materialize(result.output.handle)
        print(f"  handle: {result.output.handle}")
        print(f"  data (from the registry, not LLM prose): {data}")


if __name__ == "__main__":
    main()
