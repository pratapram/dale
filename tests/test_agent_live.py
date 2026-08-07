"""Live-model tests — deselected by default, cost real money.

The one assertion in this change that cannot be made offline: `FunctionModel`
and `TestModel` never touch a provider, so no synthetic run can show that
prompt caching actually engaged.

Three gates, on purpose:
  - `pytest.mark.live` plus `addopts = "-m 'not live'"` in pyproject.toml keeps
    a bare `uv run pytest` at zero live tests;
  - `DALE_LIVE_TESTS=1` stops a developer who types `-m live` with a key
    already in their shell from spending money by accident;
  - a missing `ANTHROPIC_API_KEY` skips rather than fails.

`DALE_LIVE_TESTS=1 uv run --extra agent pytest -m live` is the one deliberate
way to run this.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("pydantic_ai")

import dale
from dale.agent import ActionLog, build_agent, run_agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
    ),
    pytest.mark.skipif(
        os.environ.get("DALE_LIVE_TESTS") != "1", reason="set DALE_LIVE_TESTS=1"
    ),
]

_MODEL = os.environ.get("DALE_LIVE_MODEL", "anthropic:claude-haiku-4-5")


def test_cache_read_tokens_are_nonzero_on_the_second_request():
    """`anthropic_cache_tool_definitions=True` is cost, not speed: measured
    cached TTFT was marginally *slower* than uncached, but it cuts input-token
    spend ~90% after a run's first request, which is what an eval sweep is made
    of. That only holds if the cache actually engages, and nothing offline can
    tell us whether it did.

    A task needing at least two round trips, so there is a second request for
    the first one's tool definitions to be read back into."""
    registry = dale.DataRegistry(files=dale.FileRegistry())
    registry.create(
        "list",
        [{"name": f"item_{i}", "qty": i, "in_stock": i % 2 == 0} for i in range(20)],
        name="items_list",
        description="twenty items",
        created_by="fixture",
    )
    log = ActionLog()
    agent = build_agent(registry, log, model=_MODEL)

    outcome = run_agent(
        agent,
        "Filter items_list to the in-stock ones, then sort them by qty descending, "
        "then tell me which one is first. Do the filter first and look at the result "
        "before deciding how to sort.",
        deps=registry,
        action_log=log,
    )

    assert outcome.success, outcome.error
    assert outcome.usage.available
    assert outcome.usage.requests >= 2, "need a second request for a cache read"
    assert outcome.usage.cache_read_tokens > 0
    # The cached prefix is still counted against the context window (see
    # TokenUsage.from_run), so paper.md Section 4.2 part (C)'s bounded-context
    # figure isn't flattered by exactly the amount the cache saved.
    #
    # Not compared against cache_read_tokens, tempting as that is:
    # cache_read_tokens is *cumulative* across the run and peak_context_tokens
    # is a single request's *max*, so on a 3-request run the cumulative figure
    # is routinely the larger of the two (measured: 20,451 read against a
    # 15,547 peak). That is the same cumulative-vs-peak distinction TokenUsage
    # exists to keep straight, and comparing the two is a category error.
    assert outcome.usage.peak_available
    assert outcome.usage.peak_context_tokens > 0
