"""Deterministic generator for the year-over-year user-churn dataset — two
~1M-row CSVs in, one small file out (`tests/test_large_dataset_churn.py`).

The two big CSVs are *not* committed (~120 MB together, see `.gitignore`);
this script is, along with the ground truth it derives. Regenerating from a
fixed seed reproduces both files byte for byte, so the committed
`ground_truth.json` stays valid without the data itself living in git.

Ground truth here is computed the way `examples/data/README.md` requires:
by construction in a plain reference script, independent of DALE — the churn
sets are *chosen* first and the CSVs written to match, so the test compares
DALE's pipeline output against something no DALE operation produced.

Run directly to (re)generate into this directory:

    python examples/data/user_churn_1m/generate.py            # 1,000,000 users
    python examples/data/user_churn_1m/generate.py --rows 50000 --out /tmp/small
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent

SEED = 20212022
BASE_ROWS = 1_000_000
"""Users present in the 2021 file."""

DEPARTED = 100
"""2021 users with no 2022 row — the ones who left."""

ARRIVED = 150
"""2022 users with no 2021 row — the new ones."""

CHANGED = 5_000
"""Retained users whose 2022 record differs (plan change + new spend). The
trap this dataset exists to catch: a pipeline that answers "who joined and
who left" with `status != "unchanged"` silently sweeps these 5,000 in too."""

FIELDNAMES = ["user_id", "name", "email", "signup_date", "plan", "region", "monthly_spend"]

PLANS = ["free", "starter", "pro", "enterprise"]
REGIONS = ["us-east", "us-west", "emea", "apac", "latam"]
_PLAN_CHANGE = {"free": "starter", "starter": "pro", "pro": "enterprise", "enterprise": "pro"}
"""Every plan maps to a *different* plan, so all CHANGED users genuinely differ
in 2022. An earlier version sent enterprise -> enterprise, which left ~25% of
the "changed" group byte-identical and quietly overstated `changed_count`."""
_PLAN_SPEND = {"free": 0.0, "starter": 19.0, "pro": 99.0, "enterprise": 499.0}


def _user_id(n: int) -> str:
    return f"U{n:07d}"


def _record(rng: random.Random, n: int, *, year: int) -> dict:
    plan = rng.choice(PLANS)
    return {
        "user_id": _user_id(n),
        "name": f"User {n}",
        "email": f"user{n}@example.com",
        "signup_date": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "plan": plan,
        "region": rng.choice(REGIONS),
        "monthly_spend": _PLAN_SPEND[plan],
    }


def build(rows: int = BASE_ROWS) -> tuple[list[dict], list[dict], dict]:
    """Return (rows_2021, rows_2022, ground_truth). Kept in memory rather than
    streamed because the test needs the ground truth anyway and `rows` is a
    knob — at the 1M default this is a couple of GB peak, which is the point."""
    if rows < DEPARTED + CHANGED + 1:
        raise ValueError(f"rows must be at least {DEPARTED + CHANGED + 1}, got {rows}")

    rng = random.Random(SEED)
    rows_2021 = [_record(rng, n, year=2020) for n in range(1, rows + 1)]

    # Choose the churn sets first; the CSVs are then written to match them.
    picked = rng.sample(range(1, rows + 1), DEPARTED + CHANGED)
    departed_ids = sorted(_user_id(n) for n in picked[:DEPARTED])
    changed_ids = set(_user_id(n) for n in picked[DEPARTED:])
    arrived_ids = [_user_id(rows + i) for i in range(1, ARRIVED + 1)]

    departed_set = set(departed_ids)
    rows_2022 = []
    for record in rows_2021:
        if record["user_id"] in departed_set:
            continue
        if record["user_id"] in changed_ids:
            plan = _PLAN_CHANGE[record["plan"]]
            record = {**record, "plan": plan, "monthly_spend": _PLAN_SPEND[plan]}
        # An unchanged 2022 row is the *same* dict object as its 2021 row —
        # nothing here mutates records, and copying a million of them would
        # double peak memory for no gain.
        rows_2022.append(record)
    rows_2022.extend(_record(rng, rows + i, year=2022) for i in range(1, ARRIVED + 1))

    ground_truth = {
        "seed": SEED,
        "rows_2021": len(rows_2021),
        "rows_2022": len(rows_2022),
        "departed_ids": departed_ids,
        "arrived_ids": arrived_ids,
        "changed_count": len(changed_ids),
        "unchanged_count": len(rows_2021) - DEPARTED - len(changed_ids),
    }
    return rows_2021, rows_2022, ground_truth


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def generate(out_dir: Path = DATA_DIR, rows: int = BASE_ROWS) -> dict:
    """Write users_2021.csv, users_2022.csv and ground_truth.json into
    `out_dir`; return the ground truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_2021, rows_2022, ground_truth = build(rows)
    _write_csv(out_dir / "users_2021.csv", rows_2021)
    _write_csv(out_dir / "users_2022.csv", rows_2022)
    (out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8"
    )
    return ground_truth


def ensure(out_dir: Path = DATA_DIR, rows: int = BASE_ROWS) -> dict:
    """Generate only if the three files aren't already there for this row
    count — the 1M-row build takes ~30s, and the test suite shouldn't pay it
    on every run."""
    truth_path = out_dir / "ground_truth.json"
    csvs = [out_dir / "users_2021.csv", out_dir / "users_2022.csv"]
    if truth_path.is_file() and all(p.is_file() for p in csvs):
        existing = json.loads(truth_path.read_text(encoding="utf-8"))
        if existing.get("rows_2021") == rows and existing.get("seed") == SEED:
            return existing
    return generate(out_dir, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=BASE_ROWS, help="users in the 2021 file")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="output directory")
    args = parser.parse_args()

    truth = generate(args.out, args.rows)
    print(f"wrote {args.out}/users_2021.csv  ({truth['rows_2021']:,} rows)")
    print(f"wrote {args.out}/users_2022.csv  ({truth['rows_2022']:,} rows)")
    print(
        f"  departed={len(truth['departed_ids'])}  arrived={len(truth['arrived_ids'])}  "
        f"changed={truth['changed_count']}  unchanged={truth['unchanged_count']}"
    )


if __name__ == "__main__":
    main()
