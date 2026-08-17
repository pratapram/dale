"""The raw-data-in-prompt baseline arm (eval/baseline.py) — paper.md 4.2 (C).

Everything here runs offline: token counting goes through eval.baseline's
TokenCounter seam, which these tests substitute with a deterministic word
counter. No test in this file may need ANTHROPIC_API_KEY or a network — a
measuring instrument whose tests can only run when someone is paying for them
is an instrument nobody re-runs.

What's pinned here is the set of things that would be easy to get quietly
wrong in a published number: that CSV really is the cheaper serialization the
module claims (design decision 1), that DALE's arm is counted *with* its tool
schemas (decision 2), that an approximate count can never be reported as
exact, and that the grading path agrees with the ground truth both arms share.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pydantic_ai")

import dale
from eval.baseline import (
    ArmCounts,
    ContextCount,
    ContextWindowExceeded,
    CountsAnswer,
    IdCount,
    IdSetAnswer,
    IdValue,
    MappingAnswer,
    MappingEntry,
    OrderedPairsAnswer,
    build_baseline_prompt,
    count_arms,
    fresh_registry,
    grade,
    measure_tokens,
    serialize_value,
    tool_schemas,
)
from eval.use_cases import EXPECTED_ANSWERS, USE_CASES


# --- offline counters ------------------------------------------------------


def _words(**kwargs) -> int:
    """Stand-in token counter: whitespace-separated words across everything the
    request would carry. Not calibrated against any real tokenizer — nothing
    here asserts an absolute token figure, only relationships (bigger/smaller,
    included/excluded) that hold under any monotone counter."""
    text = kwargs.get("user_text") or ""
    if kwargs.get("system"):
        text += " " + kwargs["system"]
    if kwargs.get("tools"):
        text += " " + json.dumps(list(kwargs["tools"]))
    return len(text.split())


# --- serialization ---------------------------------------------------------


def test_csv_serializes_records_with_one_shared_header():
    rows = [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 2}]
    text, used = serialize_value(rows, "csv")
    assert used == "csv"
    assert text == "sku,qty\nA,1\nB,2\n"


def test_tsv_uses_tabs():
    text, used = serialize_value([{"a": 1, "b": 2}], "tsv")
    assert used == "tsv"
    assert text == "a\tb\n1\t2\n"


def test_header_is_the_union_of_every_rows_keys():
    """A ragged dataset must not lose columns — dropping them would understate
    the baseline's cost, the exact direction of error this arm must avoid."""
    text, _ = serialize_value([{"a": 1}, {"a": 2, "b": 3}], "csv")
    assert text.splitlines()[0] == "a,b"
    assert text.splitlines()[2] == "2,3"


def test_nested_values_are_json_encoded_not_dropped():
    text, _ = serialize_value([{"id": 1, "tags": ["x", "y"]}], "csv")
    assert '"[""x"",""y""]"' in text or '[""x"",""y""]' in text


def test_non_record_handles_fall_back_to_json_and_say_so():
    """A dict or set handle has no meaningful delimited form. The returned
    format is the one actually used, so a mixed prompt can't be labelled
    "csv" in a results table when half of it is JSON."""
    text, used = serialize_value({"k": 1}, "csv")
    assert used == "json"
    assert json.loads(text) == {"k": 1}

    text, used = serialize_value({"a", "b"}, "csv")
    assert used == "json"
    assert sorted(json.loads(text)) == ["a", "b"]


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown format"):
        serialize_value([{"a": 1}], "yaml")


def test_csv_is_cheaper_than_json_for_flat_records():
    """Design decision 1, asserted rather than asserted-in-prose: JSON repeats
    every field name on every record, so publishing a JSON baseline curve
    would inflate the baseline for a reason that has nothing to do with DALE."""
    rows = [{"account_id": f"ACC-{i}", "plan": "pro", "mrr": i} for i in range(50)]
    csv_text, _ = serialize_value(rows, "csv")
    json_text, _ = serialize_value(rows, "json")
    # Counted, not measured in characters: the claim in the module docstring is
    # about tokens, and characters are only a proxy for them.
    assert _words(user_text=csv_text) * 2 < _words(user_text=json_text)


# --- prompt assembly -------------------------------------------------------


def test_prompt_contains_every_alive_handles_full_contents_and_the_task():
    registry = dale.DataRegistry()
    registry.create(
        "list",
        [{"sku": "A"}, {"sku": "B"}],
        name="products",
        description="the products",
        created_by="fixture",
    )
    registry.create(
        "list", [{"sku": "A", "qty": 7}], name="stock", description="the stock", created_by="fixture"
    )

    prompt = build_baseline_prompt(registry, "Reconcile them.")

    assert "## products (list, 2 records, csv) — the products" in prompt
    assert "## stock (list, 1 records, csv) — the stock" in prompt
    assert "sku\nA\nB" in prompt
    assert "sku,qty\nA,7" in prompt
    assert prompt.rstrip().endswith("## Task\nReconcile them.")


def test_prompt_grows_with_the_dataset():
    """The claim under test in part (C) is a *scaling* one, so the arm that is
    supposed to grow must actually grow with row count.

    Asserted as a per-row delta rather than a ratio: the framing (preamble,
    per-handle header, task text) is a fixed cost, so 100x the rows is
    deliberately *not* 100x the prompt at these toy sizes. Claiming otherwise
    would be the same overstatement this module exists to avoid."""
    small = dale.DataRegistry()
    small.create("list", [{"a": i} for i in range(10)], name="r", description="d", created_by="f")
    big = dale.DataRegistry()
    big.create("list", [{"a": i} for i in range(1000)], name="r", description="d", created_by="f")

    small_n = _words(user_text=build_baseline_prompt(small, "t"))
    big_n = _words(user_text=build_baseline_prompt(big, "t"))

    assert big_n - small_n >= 990  # at least one row's worth per added row
    assert big_n > 20 * small_n


def test_uc3_prompt_carries_the_real_sample_data():
    registry = fresh_registry("uc3")
    prompt = build_baseline_prompt(registry, USE_CASES["uc3"][1])
    # ACC-004 is one of uc3's expected answers: an active subscriber with no
    # usage events. It must be visible in the baseline's prompt, or the
    # baseline is being asked an unanswerable question.
    assert "ACC-004" in prompt
    assert "active_subscriptions" in prompt and "feature_events" in prompt


# --- token counting seam ---------------------------------------------------


def test_measure_tokens_is_exact_when_the_prompt_fits():
    count = measure_tokens(_words, user_text="one two three")
    assert count == ContextCount(3, exact=True)
    assert count.render() == "3"


def test_oversized_prompts_are_counted_in_chunks_and_marked_approximate():
    """A prompt too large for the model's window is the interesting end of the
    curve, not an error — but a chunked sum is a different measurement from a
    single count, so it is flagged. Reporting it as exact is the one outcome
    this seam exists to prevent."""
    calls: list[str] = []

    def counter(*, user_text, system=None, tools=None):
        calls.append(user_text)
        if len(calls) == 1:
            raise ContextWindowExceeded("prompt too long")
        return _words(user_text=user_text, system=system, tools=tools)

    text = "\n".join(f"row {i}" for i in range(100))
    count = measure_tokens(counter, user_text=text, chunk_chars=100)

    assert count.exact is False
    assert count.tokens == 200  # every word still counted, once
    assert "approximate" in count.render()
    assert count.render().startswith("~")
    assert len(calls) > 2  # first attempt, then several chunks


def test_chunking_splits_on_line_boundaries():
    """A chunk edge inside a CSV row would create a token that exists in
    neither the original nor the chunk."""
    seen: list[str] = []

    def counter(*, user_text, system=None, tools=None):
        if not seen:
            seen.append(user_text)
            raise ContextWindowExceeded("too long")
        seen.append(user_text)
        return 1

    measure_tokens(counter, user_text="aaaa,bbbb\ncccc,dddd\neeee,ffff\n", chunk_chars=12)
    for chunk in seen[1:]:
        assert chunk.endswith("\n")


# --- the two arms ----------------------------------------------------------


def test_dale_arm_includes_tool_schemas():
    """Design decision 2. The schemas are sent on every request and are
    essentially DALE's entire first-turn cost; omitting them would over-flatter
    DALE by roughly an order of magnitude."""
    counts = count_arms("uc3", _words)

    assert counts.tool_schema_tokens > 0
    assert counts.dale.tokens > counts.dale_without_tools.tokens
    # Not a calibrated threshold — just the shape of the finding: schemas
    # dominate, registry state does not.
    assert counts.tool_schema_tokens > counts.dale_without_tools.tokens


def test_tool_schemas_are_every_tool_a_real_request_carries():
    """Including `final_result`, the output tool pydantic-ai synthesizes from
    build_agent's `output_type=DaleResult`. It appears on the wire but in no
    list build_tools returns, and leaving it out undercounted DALE's context by
    ~3% — always in DALE's favour, which is the direction that matters. That
    share is much larger now that `build_tools` returns one tool rather than
    18, which is exactly why the count is read off a request rather than
    reconstructed."""
    schemas = tool_schemas(dale.DataRegistry())
    names = {s["name"] for s in schemas}
    assert names == {"run_plan", "final_result"}
    assert all(s["input_schema"]["type"] == "object" for s in schemas)


def test_counted_schemas_exceed_the_build_tools_list():
    """The regression guard for the same finding, stated as the comparison that
    was wrong before: whatever build_tools returns, the counted payload is
    strictly larger.

    The count comparison alone is now `2 == 1 + 1`, which several wrong
    implementations would also satisfy, so the name-set assertion below carries
    the real weight: the extra one has to be `final_result` specifically."""
    from dale.agent import ActionLog, build_tools

    counted = tool_schemas(dale.DataRegistry())
    built = build_tools(ActionLog())
    assert len(counted) == len(built) + 1
    assert {s["name"] for s in counted} - {t.name for t in built} == {"final_result"}


def test_privacy_mode_drops_peek_from_the_counted_schemas():
    """The counted cost has to be the cost a real run pays under the same
    configuration — `peek` is omitted entirely in privacy mode.

    Asserted on the step union inside `run_plan`'s published schema, not on
    tool names. The name set is `{"run_plan", "final_result"}` either way now,
    so a `"peek" not in names` check would pass green whether or not
    privacy_mode did anything at all — the most dangerous shape a test can
    take, and the shape this one had."""
    schemas = {s["name"]: s for s in tool_schemas(dale.DataRegistry(privacy_mode=True))}
    private = schemas["run_plan"]["input_schema"]
    assert "PeekParamsPlanStep" not in private["$defs"]
    assert "peek" not in private["properties"]["steps"]["items"]["discriminator"]["mapping"]

    # ...and it is there without privacy_mode, or the assertions above are
    # measuring nothing.
    public = {s["name"]: s for s in tool_schemas(dale.DataRegistry())}["run_plan"]
    assert "PeekParamsPlanStep" in public["input_schema"]["$defs"]


def test_count_arms_reports_the_dataset_size_it_measured():
    counts = count_arms("uc3", _words)
    registry = fresh_registry("uc3")
    assert counts.handles == len(registry.list_handles())
    assert counts.records == sum(m.size for m in registry.list_handles())
    assert counts.baseline.tokens > 0


def test_baseline_arm_scales_with_the_data():
    """Half of part (C)'s claim: more rows, bigger baseline prompt."""
    small = count_arms("uc3", _words)

    registry = fresh_registry("uc3")
    inflated = dale.DataRegistry(files=dale.FileRegistry())
    for meta in registry.list_handles():
        rows = registry.materialize(meta.name)
        inflated.create(
            "list", rows * 20, name=meta.name, description=meta.description, created_by="test"
        )
    task = USE_CASES["uc3"][1]
    big_baseline = _words(user_text=build_baseline_prompt(inflated, task))

    assert big_baseline > 5 * small.baseline.tokens


def test_dales_context_does_not_scale_with_row_count():
    """The other half, and the one actually worth asserting: DALE's first-turn
    context is its system prompt plus tool schemas, and the system prompt sees
    handle *metadata*, not rows. Four orders of magnitude of data must move it
    by a couple of digits in a size field, not by row count.

    (The tool-schema half is row-independent by construction — it is built from
    the operation catalog and never sees the data at all — so the system prompt
    is the only part that could have scaled.)"""
    from dale.agent import default_system_prompt

    sizes = [10, 1_000, 100_000]
    counts = []
    for n in sizes:
        registry = dale.DataRegistry()
        registry.create(
            "list",
            [{"account_id": f"ACC-{i}", "plan": "pro"} for i in range(n)],
            name="rows",
            description="d",
            created_by="test",
        )
        counts.append(_words(user_text=default_system_prompt(registry)))

    assert max(counts) - min(counts) <= 2  # only the rendered size/memory fields move


def test_render_never_labels_an_approximate_count_as_exact():
    counts = ArmCounts(
        use_case="uc3",
        model="claude-sonnet-5",
        fmt="csv",
        handles=2,
        records=10_000,
        baseline=ContextCount(900_000, exact=False),
        dale=ContextCount(12_000),
        dale_without_tools=ContextCount(300),
    )
    rendered = counts.render()
    assert "~900,000 (approximate)" in rendered
    assert counts.tool_schema_tokens == 11_700
    assert "75.00x" in rendered
    # Token counts are model-specific; a row without its model id can't be
    # safely put in a table next to another row.
    assert "claude-sonnet-5" in rendered


# --- grading ---------------------------------------------------------------


def test_grades_an_id_set_answer_against_shared_ground_truth():
    expected = EXPECTED_ANSWERS["uc3"]()
    assert grade(expected, IdSetAnswer(ids=["ACC-004", "ACC-009", "ACC-013"])) is True
    assert grade(expected, IdSetAnswer(ids=["ACC-004", "ACC-009"])) is False
    # A superset is wrong too — tolerance in the checkers must never become a
    # rubber stamp for "mentioned the right ones somewhere".
    assert grade(expected, IdSetAnswer(ids=["ACC-004", "ACC-009", "ACC-013", "ACC-001"])) is False


def test_grades_ordered_pairs_and_honours_the_documented_alternate():
    """uc1's untracked-SKU ambiguity is a genuine one this project introduced,
    and `ExpectedAnswer.alternates` accepts both readings — the baseline arm
    has to inherit that, or it would be graded more harshly than DALE."""
    expected = EXPECTED_ANSWERS["uc1"]()
    full = OrderedPairsAnswer(pairs=[IdValue(id=s, value=m) for s, m in expected.value])
    assert grade(expected, full) is True

    without_untracked = OrderedPairsAnswer(
        pairs=[IdValue(id=s, value=m) for s, m in expected.value if s != "SKU-1020"]
    )
    assert grade(expected, without_untracked) is True

    reordered = OrderedPairsAnswer(pairs=list(reversed(full.pairs)))
    assert grade(expected, reordered) is False


def test_grades_uc2_on_ips_and_per_ip_counts_not_just_ips():
    """uc2's task asks "which source IPs were flagged **and how many records
    were flagged for each**". Grading the baseline on the IP set alone let it
    pass by naming two strings, while DALE's checker had to produce the right
    IPs, the right total, and the right breach row — the two arms answering
    different questions."""
    expected = EXPECTED_ANSWERS["uc2"]()
    assert expected.kind == "counts"

    right = CountsAnswer(
        counts=[IdCount(id="198.51.100.23", count=4), IdCount(id="203.0.113.55", count=4)]
    )
    assert grade(expected, right) is True

    # The answer that used to pass: correct IPs, invented counts.
    wrong_counts = CountsAnswer(
        counts=[IdCount(id="198.51.100.23", count=1), IdCount(id="203.0.113.55", count=7)]
    )
    assert grade(expected, wrong_counts) is False


def test_uc2s_dale_checker_still_passes_a_correct_pipeline():
    """Positive control for the constant this finding tightened. `check_uc2`
    now also requires the per-IP counts, and nothing else in the suite runs it
    — a wrong number in `_UC2_EXPECTED_FLAG_COUNTS` would silently fail every
    live uc2 trial and look like a model failure."""
    from dale.agent import ActionLog
    from eval.use_cases import check_uc2

    registry = fresh_registry("uc2")
    logs = registry.list_handles()[0].name
    dale.call_operation(
        registry,
        "window_flag",
        {
            "handle": logs,
            "group_by": ["source_ip"],
            "window_field": "timestamp",
            "window_size": 300,
            "predicate": {"field": "outcome", "op": "==", "value": "fail"},
            "threshold": 5,
            "name": "flagged_logs",
            "description": "d",
        },
    )
    assert check_uc2(registry, ActionLog(), "") is True


def test_uc2s_two_arms_share_one_constant():
    """The counts the baseline is graded on and the total DALE's checker
    requires are the same numbers, derived — not two hand-kept copies that can
    drift into grading two different things."""
    from eval.use_cases import (
        _UC2_EXPECTED_FLAG_COUNTS,
        _UC2_EXPECTED_FLAGGED_ROWS,
        _UC2_EXPECTED_IPS,
    )

    assert EXPECTED_ANSWERS["uc2"]().value == _UC2_EXPECTED_FLAG_COUNTS
    assert _UC2_EXPECTED_FLAGGED_ROWS == sum(_UC2_EXPECTED_FLAG_COUNTS.values())
    assert _UC2_EXPECTED_IPS == sorted(_UC2_EXPECTED_FLAG_COUNTS)


def test_a_self_contradictory_answer_fails_rather_than_being_resolved():
    """Both flat answer shapes could silently take the last value for a
    repeated key. An answer stating two different counts for one IP is
    contradictory, and picking one would grade it on emission order."""
    expected = EXPECTED_ANSWERS["uc2"]()
    contradictory = CountsAnswer(
        counts=[
            IdCount(id="198.51.100.23", count=4),
            IdCount(id="198.51.100.23", count=9),
            IdCount(id="203.0.113.55", count=4),
        ]
    )
    assert grade(expected, contradictory) is False

    uc4 = EXPECTED_ANSWERS["uc4"]()
    entries = [
        MappingEntry(id=emp, key=resource, value=effect)
        for emp, perms in uc4.value.items()
        for resource, effect in perms.items()
    ]
    entries.append(MappingEntry(id="E002", key="prod_database", value="Deny"))
    assert grade(uc4, MappingAnswer(entries=entries)) is False


def test_grades_a_mapping_answer():
    expected = EXPECTED_ANSWERS["uc4"]()
    entries = [
        MappingEntry(id=emp, key=resource, value=effect)
        for emp, perms in expected.value.items()
        for resource, effect in perms.items()
    ]
    assert grade(expected, MappingAnswer(entries=entries)) is True
    assert grade(expected, MappingAnswer(entries=entries[:-1])) is False


def test_every_use_case_has_a_gradeable_answer_shape():
    """The two arms have to cover the same use cases. A use case with no
    output type is one the comparison silently can't include.

    uc3_large is included rather than skipped: its expected set is generated by
    its setup(), so `fresh_registry` has to run first — which is exactly the
    ordering `run_baseline_trial` uses, and the ordering the empty-expectation
    guard below exists to enforce."""
    from eval.baseline import OUTPUT_TYPES

    assert set(EXPECTED_ANSWERS) == set(USE_CASES)
    for name, factory in EXPECTED_ANSWERS.items():
        fresh_registry(name)
        assert factory().kind in OUTPUT_TYPES


def test_an_empty_expectation_refuses_to_construct():
    """`matches_ids([])` against an empty expected set returns True — a grader
    that passes any answer naming nothing. The realistic way to get one is
    reading uc3_large's expected set before its setup() has generated it, so
    that ordering mistake has to fail loudly at construction rather than three
    steps later as a perfect score."""
    from eval.use_cases import ExpectedAnswer

    with pytest.raises(ValueError, match="expected answer is empty"):
        ExpectedAnswer(kind="id_set", value=set())
    with pytest.raises(ValueError, match="did setup"):
        ExpectedAnswer(kind="ordered_pairs", value=[])


def test_missing_api_key_is_reported_as_a_missing_api_key(monkeypatch):
    """Not as the SDK's bare TypeError about an `api_key` argument, which reads
    as a bug in this module rather than as "you forgot --env-file"."""
    from eval.baseline import AnthropicTokenCounter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="--env-file"):
        AnthropicTokenCounter("claude-sonnet-5")


def test_a_baseline_trial_runs_end_to_end_against_testmodel():
    """Structural check only, and free: TestModel answers with garbage, so the
    thing being verified is that the run path holds together — no tools, a
    structured answer, graded rather than parsed, an empty action log, and
    usage correctly marked unavailable rather than reported as a real zero."""
    from eval.baseline import run_baseline_trial

    result = run_baseline_trial(1, "uc3", "test")

    assert result.error is None
    assert result.success is False  # TestModel's answer is not the ground truth
    assert result.action_log.entries == []
    assert result.usage.available is False


def test_unknown_answer_kind_is_loud():
    from eval.use_cases import ExpectedAnswer

    with pytest.raises(ValueError, match="unknown expected-answer kind"):
        grade(ExpectedAnswer(kind="vibes", value={"a"}), IdSetAnswer(ids=["a"]))
