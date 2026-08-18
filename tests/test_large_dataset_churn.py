"""Large-dataset end-to-end test: two ~1M-row CSVs in, one small file out.

Scenario — year-over-year user churn. Given `users_2021.csv` (1,000,000 users)
and `users_2022.csv` (999,900 retained + 150 new), report who joined and who
left. The answer is 250 rows: 150 arrivals, 100 departures.

Pipeline (all real operations, no code generated or executed anywhere):

    load_csv ×2 → index_by ×2 → dict_diff → filter_where → sort_by → export_handle

What this pins down that the small `examples/data/` fixtures can't:

- **Scale.** 2M input rows through load_csv, 2M-entry indexes, a 1,000,150-row
  diff, and a filter over all of it. `ROADMAP.md` Phase 4 wants the use-case
  datasets at larger sizes; the existing regression tests top out at ~40 rows.
- **The funnel.** Handle sizes are asserted at every step, so the test fails
  loudly if a stage ever stops being O(input) or the 250-row answer drifts.
- **The `changed` trap.** 5,000 retained users changed plan in 2022.
  "Who joined and who left" is `status in ("new", "removed")`, *not*
  `status != "unchanged"` — the lazy predicate would return 5,250 rows. The
  test asserts the changed group is counted and excluded.
- **No data through the model's context** (objections #12). The final answer
  reaches disk via `export_handle`, whose `OperationOutput` carries only a
  byte count — asserted here on 250 rows that never appear in any return value.

Ground truth comes from `examples/data/user_churn_1m/generate.py`, which
*chooses* the churn sets and then writes the CSVs to match — so nothing DALE
computes is being compared against something DALE computed.

Measured on the 1M dataset: 2,000,050 input rows (142 MB of CSV) collapse to a
250-row, 81 KB report in ~12s of pipeline time, ~1.7 GB peak RSS. The whole
test (which runs the pipeline twice — once end-to-end, once to count the four
diff statuses) takes ~28s, so it's marked `slow`: `pytest -m "not slow"` skips
it, and the 25k-row variant still covers the same pipeline in under a second.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import dale
from dale.files import FileRegistry

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "data" / "user_churn_1m"))
import generate  # noqa: E402  (path-dependent import of the dataset generator)

DATA_DIR = Path(__file__).parent.parent / "examples" / "data" / "user_churn_1m"


def _run_churn_pipeline(data_dir: Path, out_dir: Path) -> tuple[dict, Path]:
    """Run the full pipeline over the two CSVs in `data_dir`. Returns the
    per-step handle sizes and the path the report was written to."""
    files = FileRegistry()
    files.register("users_2021.csv", data_dir / "users_2021.csv")
    files.register("users_2022.csv", data_dir / "users_2022.csv")
    report_path = out_dir / "churn_report.json"
    files.register_output("churn_report.json", report_path)

    registry = dale.DataRegistry(files=files)

    def call(name: str, params: dict) -> dale.OperationOutput:
        out = dale.call_operation(registry, name, params)
        assert out.status == "ok", f"{name} returned {out.status}"
        return out

    users_2021 = call(
        "load_csv",
        {"file": "users_2021.csv", "name": "users_2021", "description": "2021 user roster"},
    ).handle
    users_2022 = call(
        "load_csv",
        {"file": "users_2022.csv", "name": "users_2022", "description": "2022 user roster"},
    ).handle

    index_2021 = call(
        "index_by",
        {
            "handle": "users_2021",
            "key_fields": ["user_id"],
            "name": "index_2021",
            "description": "2021 roster keyed by user_id",
        },
    ).handle
    index_2022 = call(
        "index_by",
        {
            "handle": "users_2022",
            "key_fields": ["user_id"],
            "name": "index_2022",
            "description": "2022 roster keyed by user_id",
        },
    ).handle

    diff = call(
        "dict_diff",
        {
            "current_handle": "index_2022",
            "previous_handle": "index_2021",
            "name": "year_over_year",
            "description": "2022 vs 2021, one row per user_id",
        },
    ).handle

    # `status in ("new", "removed")` — deliberately not `!= "unchanged"`,
    # which would sweep in the 5,000 plan changes as well.
    churn = call(
        "filter_where",
        {
            "handle": "year_over_year",
            "predicate": {"field": "status", "op": "in", "value": ["new", "removed"]},
            "name": "churn",
            "description": "arrivals and departures only",
        },
    ).handle

    churn_sorted = call(
        "sort_by",
        {
            "handle": "churn",
            "keys": [{"field": "status", "order": "asc"}, {"field": "key", "order": "asc"}],
            "name": "churn_sorted",
            "description": "churn rows, departures then arrivals, by user_id",
        },
    ).handle

    export = call(
        "export_handle",
        {"handle": "churn_sorted", "destination": "churn_report.json"},
    )

    # The model-facing return value of the export is a receipt, not the data:
    # the names it already supplied plus a byte count, nothing else. (JSON
    # exports report bytes; CSV exports report rows.)
    assert export.handle is None
    assert set(export.result) == {"handle", "destination", "format", "bytes"}
    assert export.result["format"] == "json"
    assert export.result["bytes"] == report_path.stat().st_size

    sizes = {
        "users_2021": users_2021.size,
        "users_2022": users_2022.size,
        "index_2021": index_2021.size,
        "index_2022": index_2022.size,
        "diff": diff.size,
        "churn": churn.size,
        "churn_sorted": churn_sorted.size,
    }
    return sizes, report_path


def _assert_matches_ground_truth(sizes: dict, report_path: Path, truth: dict) -> None:
    n_2021, n_2022 = truth["rows_2021"], truth["rows_2022"]
    departed, arrived = truth["departed_ids"], truth["arrived_ids"]

    # Every stage is exactly as big as it should be — nothing dropped, nothing
    # fanned out. The diff spans the union of both rosters.
    assert sizes["users_2021"] == n_2021
    assert sizes["users_2022"] == n_2022
    assert sizes["index_2021"] == n_2021, "user_id must be unique in 2021"
    assert sizes["index_2022"] == n_2022, "user_id must be unique in 2022"
    assert sizes["diff"] == n_2021 + len(arrived)
    assert sizes["churn"] == len(departed) + len(arrived)
    assert sizes["churn_sorted"] == sizes["churn"]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report) == len(departed) + len(arrived)

    new_rows = [r for r in report if r["status"] == "new"]
    removed_rows = [r for r in report if r["status"] == "removed"]
    assert len(new_rows) == len(arrived)
    assert len(removed_rows) == len(departed)

    assert sorted(r["key"] for r in new_rows) == sorted(arrived)
    assert sorted(r["key"] for r in removed_rows) == sorted(departed)

    # A new user has no 2021 side and vice versa — the diff's two value columns
    # are what tells you which direction each row moved.
    assert all(r["previous_value"] is None for r in new_rows)
    assert all(r["current_value"] is not None for r in new_rows)
    assert all(r["current_value"] is None for r in removed_rows)
    assert all(r["previous_value"] is not None for r in removed_rows)

    # sort_by grouped the two statuses ("new" < "removed" ascending) and
    # ordered each group by user_id.
    assert [r["status"] for r in report] == ["new"] * len(arrived) + ["removed"] * len(departed)
    assert [r["key"] for r in report[: len(arrived)]] == sorted(arrived)

    # Two ~70 MB inputs collapse to a file measured in kilobytes.
    assert report_path.stat().st_size < 500_000


def _assert_changed_group_excluded(data_dir: Path, truth: dict) -> None:
    """The 5,000 plan changes are real, are visible to the pipeline, and are
    correctly left out of the churn answer."""
    files = FileRegistry()
    files.register("users_2021.csv", data_dir / "users_2021.csv")
    files.register("users_2022.csv", data_dir / "users_2022.csv")
    registry = dale.DataRegistry(files=files)

    for year in (2021, 2022):
        dale.call_operation(
            registry,
            "load_csv",
            {"file": f"users_{year}.csv", "name": f"u{year}", "description": f"{year} roster"},
        )
        dale.call_operation(
            registry,
            "index_by",
            {
                "handle": f"u{year}",
                "key_fields": ["user_id"],
                "name": f"i{year}",
                "description": f"{year} keyed",
            },
        )
    dale.call_operation(
        registry,
        "dict_diff",
        {
            "current_handle": "i2022",
            "previous_handle": "i2021",
            "name": "d",
            "description": "diff",
        },
    )

    counts = {}
    for status in ("new", "removed", "changed", "unchanged"):
        out = dale.call_operation(
            registry,
            "filter_where",
            {
                "handle": "d",
                "predicate": {"field": "status", "op": "==", "value": status},
                "name": f"only_{status}",
                "description": status,
            },
        )
        counts[status] = out.handle.size

    assert counts["changed"] == truth["changed_count"]
    assert counts["unchanged"] == truth["unchanged_count"]
    assert counts["new"] == len(truth["arrived_ids"])
    assert counts["removed"] == len(truth["departed_ids"])
    # The whole point of the trap: the sloppy predicate is 21× too big.
    assert counts["new"] + counts["removed"] + counts["changed"] == 5_250


def test_churn_pipeline_small_scale(tmp_path):
    """Same pipeline, 25k rows — fast enough for every default test run."""
    data_dir = tmp_path / "data"
    truth = generate.generate(data_dir, rows=25_000)

    sizes, report_path = _run_churn_pipeline(data_dir, tmp_path)
    _assert_matches_ground_truth(sizes, report_path, truth)
    _assert_changed_group_excluded(data_dir, truth)


@pytest.mark.slow
def test_churn_pipeline_one_million_rows(tmp_path):
    """The real thing: 1,000,000 + 1,000,050 input rows → a 250-row report.

    Uses the cached dataset in `examples/data/user_churn_1m/` (generated on
    first run, ~3s, gitignored) rather than regenerating into tmp_path — the
    files are ~70 MB each.
    """
    truth = generate.ensure(DATA_DIR)
    assert truth["rows_2021"] == 1_000_000
    assert truth["rows_2022"] == 1_000_050

    sizes, report_path = _run_churn_pipeline(DATA_DIR, tmp_path)
    _assert_matches_ground_truth(sizes, report_path, truth)
    _assert_changed_group_excluded(DATA_DIR, truth)
