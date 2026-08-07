"""Per-use-case setup/task/checker triples for the Phase 4 harness, built on
the DESIGN.md Section 3 sample data (examples/data/) and the same ground truth
already pinned down as regression tests in tests/test_use_case_pipelines.py.

Checkers inspect registry state after the run (any handle still alive,
searched by shape rather than a hardcoded handle name, since an LLM's actual
handle IDs won't match a deterministic pipeline's) rather than parsing the
model's free-text final answer — structured-data checking, not NLP grading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import dale
from dale.agent import ActionLog

DATA_DIR = Path(__file__).parent.parent / "examples" / "data"


def _load(registry: dale.DataRegistry, path: Path) -> str:
    assert registry.files is not None
    virtual_name = path.name
    registry.files.register(virtual_name, path)
    out = dale.call_primitive(
        registry,
        "load_csv",
        {
            "file": virtual_name,
            "name": path.stem,
            "description": f"data loaded from {virtual_name}",
        },
    )
    return out.handle.handle


def _alive_lists(registry: dale.DataRegistry):
    for meta in registry.list_handles():
        if meta.kind != "list":
            continue
        try:
            rows = registry.materialize(meta.handle)
        except Exception:
            continue
        if rows and isinstance(rows[0], dict):
            yield rows


_ID_FIELDS = ("account_id", "key")
_NESTED_FIELDS = ("previous_value", "current_value")


def _identifier_sets(rows: list[Any], id_field: str) -> list[set[Any]]:
    """Every plausible reading of "the identifiers in this handle", so a
    correct answer isn't missed because of the shape it arrived in.

    Checkers grade *the answer*, not the route taken to it — route quality is
    already measured separately by wasted-turn rate and call count. A model
    that reaches the right accounts via dict_diff produces rows keyed `key`
    with the original record nested under `previous_value`, not a top-level
    `account_id`; scoring that as a failure (which it was, in 2 of 3 live
    uc3_large runs) measures pipeline choice while claiming to measure
    correctness.

    Returns candidate sets rather than one merged set: each is compared for
    *exact* equality against the expected answer by the caller, which is what
    keeps this from becoming a rubber stamp. A handle holding precisely the
    expected ids and nothing else is the answer whatever its field is named;
    a superset (a full diff, an unfiltered intermediate) still fails."""
    candidates: list[set[Any]] = []
    for field in _ID_FIELDS:
        if all(isinstance(r, dict) and field in r for r in rows):
            candidates.append({r[field] for r in rows})
    for nested in _NESTED_FIELDS:
        values = [r.get(nested) if isinstance(r, dict) else None for r in rows]
        if all(isinstance(v, dict) and id_field in v for v in values):
            candidates.append({v[id_field] for v in values})
    return candidates


def _answer_matches(registry: dale.DataRegistry, expected: set[Any], id_field: str) -> bool:
    """True when any alive list handle's identifiers are exactly `expected`."""
    return any(
        expected in _identifier_sets(rows, id_field) for rows in _alive_lists(registry)
    )


# --- Use Case 1: inventory reconciliation --------------------------------

def setup_uc1(registry: dale.DataRegistry) -> None:
    base = DATA_DIR / "01_inventory_reconciliation"
    _load(registry, base / "product_metadata.csv")
    _load(registry, base / "stock_counts.csv")
    _load(registry, base / "pricing_overrides.csv")


TASK_UC1 = (
    "You have three product datasets: item metadata, stock counts, and pricing "
    "overrides, all keyed by SKU. Some SKUs may be missing from the stock or "
    "pricing files, and some may have zero stock. Reconcile them by SKU, keep "
    "only items that are in stock and have a price, compute a 'margin' field "
    "(price minus cost) for each, and sort the results by margin from highest "
    "to lowest. Tell me the final list of SKUs and their margins."
)

_UC1_EXPECTED = [
    ("SKU-1010", 239.99), ("SKU-1016", 179.99), ("SKU-1009", 159.99),
    ("SKU-1004", 109.99), ("SKU-1007", 69.99), ("SKU-1012", 44.99),
    ("SKU-1019", 34.99), ("SKU-1020", 19.99), ("SKU-1003", 17.99),
    ("SKU-1006", 16.99), ("SKU-1014", 15.99), ("SKU-1001", 11.49),
    ("SKU-1018", 8.99), ("SKU-1015", 6.49),
]
# SKU-1020 has no row in stock_counts.csv at all (untracked, not zero-stock —
# see examples/data/README.md). Whether "untracked" counts as "in stock" is a
# genuine, defensible ambiguity in the task text, not a reasoning error either
# way: a left-join + `!= 0` filter keeps it (the interpretation the sample
# data's own reference pipeline uses); an inner join across all three
# datasets excludes it (equally defensible — "only items confirmed in stock").
# Accept both rather than penalize a model for resolving an ambiguity we
# introduced — found by a real live trial that produced exactly this second,
# reasonable interpretation (see TODO.md session notes).
_UC1_EXPECTED_EXCLUDING_UNTRACKED = [t for t in _UC1_EXPECTED if t[0] != "SKU-1020"]


def check_uc1(registry: dale.DataRegistry, action_log: ActionLog, final_answer: str) -> bool:
    for rows in _alive_lists(registry):
        if "sku" not in rows[0] or "margin" not in rows[0]:
            continue
        try:
            actual = [(r["sku"], round(r["margin"], 2)) for r in rows]
        except (KeyError, TypeError):
            continue
        if actual in (_UC1_EXPECTED, _UC1_EXPECTED_EXCLUDING_UNTRACKED):
            return True
    return False


# --- Use Case 2: log sessionization ---------------------------------------

def setup_uc2(registry: dale.DataRegistry) -> None:
    base = DATA_DIR / "02_log_sessionization"
    _load(registry, base / "auth_logs.csv")


TASK_UC2 = (
    "You have an authentication log dataset (login attempts with timestamp, "
    "region, source IP, username, and outcome). Detect credential-stuffing "
    "attacks: flag any record where the same source IP has 5 or more failed "
    "login attempts within a trailing 300-second window. Tell me which source "
    "IPs were flagged and how many records were flagged for each."
)


# Lifted out of check_uc2's body so the baseline arm (eval/baseline.py) can
# grade against the identical ground truth. One constant, two arms: DALE is
# checked by inspecting registry state, the baseline by comparing structured
# output. What "correct" means must not vary between them, which it would if
# each arm carried its own copy of these values.
#
# Per-IP flagged-record counts, not just the IP set, because the task text
# asks for both: "which source IPs were flagged *and how many records were
# flagged for each*". An id_set expectation would have let the baseline arm
# score a pass for naming two IPs while DALE's checker had to produce the
# right IPs, the right total, and the right breach row — the two arms graded
# on different questions. Everything below is derived from this one mapping so
# the total and the IP list cannot drift away from it.
_UC2_EXPECTED_FLAG_COUNTS = {"198.51.100.23": 4, "203.0.113.55": 4}
_UC2_EXPECTED_IPS = sorted(_UC2_EXPECTED_FLAG_COUNTS)
_UC2_EXPECTED_FLAGGED_ROWS = sum(_UC2_EXPECTED_FLAG_COUNTS.values())
_UC2_EXPECTED_BREACH_IP = "203.0.113.55"
_UC2_EXPECTED_BREACH_USER = "asmith"


def check_uc2(registry: dale.DataRegistry, action_log: ActionLog, final_answer: str) -> bool:
    # window_flag's `as` output field name is caller-chosen (default "flagged"),
    # and this task's text never pins down a name — a reasonable, more
    # descriptive choice (e.g. "cred_stuffing_flag") is not a mistake. Search
    # every boolean-valued field as a flag candidate rather than hardcoding
    # the default name, which caused a real false-failure the first time this
    # checker ran against live trials (see TODO.md session notes).
    for rows in _alive_lists(registry):
        if "source_ip" not in rows[0]:
            continue
        bool_fields = [k for k, v in rows[0].items() if isinstance(v, bool)]
        for field in bool_fields:
            flagged_rows = [r for r in rows if r.get(field) is True]
            flagged_ips = sorted({r["source_ip"] for r in flagged_rows})
            if flagged_ips != _UC2_EXPECTED_IPS:
                continue
            if len(flagged_rows) != _UC2_EXPECTED_FLAGGED_ROWS:
                continue
            per_ip: dict[str, int] = {}
            for r in flagged_rows:
                per_ip[r["source_ip"]] = per_ip.get(r["source_ip"], 0) + 1
            if per_ip != _UC2_EXPECTED_FLAG_COUNTS:
                continue
            breach = next(
                (
                    r
                    for r in flagged_rows
                    if r["source_ip"] == _UC2_EXPECTED_BREACH_IP
                    and r.get("outcome") == "success"
                ),
                None,
            )
            if breach is not None and breach.get("username") == _UC2_EXPECTED_BREACH_USER:
                return True
    return False


# --- Use Case 3: churn / feature usage ------------------------------------

def setup_uc3(registry: dale.DataRegistry) -> None:
    base = DATA_DIR / "03_churn_feature_usage"
    _load(registry, base / "active_subscriptions.csv")
    _load(registry, base / "feature_events.csv")


TASK_UC3 = (
    "You have two datasets: active subscriptions (with an account ID, company, "
    "plan, MRR, and signup date) and feature usage events (account ID, feature, "
    "and event timestamp) — most active accounts have usage events, but some "
    "active accounts have none at all, which is a churn-risk signal. Note that "
    "not every account with usage events is currently active. Find the active "
    "subscriber account IDs that have zero feature usage events, and tell me "
    "which accounts those are."
)

_UC3_EXPECTED = {"ACC-004", "ACC-009", "ACC-013"}


def check_uc3(registry: dale.DataRegistry, action_log: ActionLog, final_answer: str) -> bool:
    return _answer_matches(registry, _UC3_EXPECTED, "account_id")


# --- Use Case 3 at scale --------------------------------------------------
#
# Same task, same schema, same shape of answer as uc3 — only the row count
# changes. The point isn't a harder reasoning problem: it's that the model's
# side of the work is *identical* at 39 rows and at 10,000, because it only
# ever sees handle metadata plus a 3-record auto_inspect sample. What scales
# is host compute (index_by, dict_diff), not context. The token usage
# difference between uc3 and uc3_large is the measurement worth taking.
#
# Data is generated in memory rather than written to CSV: a large fixture has
# no business living in git, and load_csv's parsing cost isn't what's under
# test here.

SCALE_ROWS_ENV = "DALE_SCALE_ROWS"
_DEFAULT_SCALE_ROWS = 10_000

# Filled in by setup_uc3_large, since the expected answer depends on the
# generated size. Module-level because the harness calls setup and checker
# separately, in that order, within one process.
_UC3_LARGE_EXPECTED: set[str] = set()


def scale_rows() -> int:
    """Row count for uc3_large's feature_events, from $DALE_SCALE_ROWS
    (default 10,000). Kept as a function, not a constant, so the env var can
    be set after import and still take effect."""
    raw = os.environ.get(SCALE_ROWS_ENV)
    if not raw:
        return _DEFAULT_SCALE_ROWS
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{SCALE_ROWS_ENV} must be an integer, got {raw!r}") from None
    if value < 100:
        raise SystemExit(f"{SCALE_ROWS_ENV} must be at least 100, got {value}")
    return value


def setup_uc3_large(registry: dale.DataRegistry) -> None:
    """Generate the uc3 scenario at `scale_rows()` events, deterministically.

    Three properties are constructed on purpose, mirroring the real fixture:
    every active account except a known handful has at least one event; that
    handful (the answer) has exactly zero; and some accounts appear in the
    events without being active, so a naive "distinct event accounts" answer
    is still wrong at this size."""
    rows = scale_rows()
    n_accounts = max(100, rows // 100)

    # The answer: three active accounts, spread across the id space so a
    # partial scan can't stumble onto all of them.
    zero_usage_idx = {7, n_accounts // 2, n_accounts - 1}
    _UC3_LARGE_EXPECTED.clear()
    _UC3_LARGE_EXPECTED.update(f"ACC-{i:07d}" for i in zero_usage_idx)

    subscriptions = [
        {
            "account_id": f"ACC-{i:07d}",
            "company": f"Company {i}",
            "plan": ("enterprise", "pro", "starter")[i % 3],
            "mrr": 100 + (i % 900),
            "signup_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        }
        for i in range(n_accounts)
    ]

    # Every account that should have usage, in a fixed order — indexing into
    # this by i % len() guarantees full coverage once rows >= len(eligible).
    eligible = [i for i in range(n_accounts) if i not in zero_usage_idx]
    features = ("dashboard", "export", "api", "alerts", "search")
    events = [
        {
            "account_id": f"ACC-{eligible[i % len(eligible)]:07d}",
            "feature": features[i % len(features)],
            "event_timestamp": f"2025-0{(i % 9) + 1}-{(i % 28) + 1:02d}T12:00:00Z",
        }
        for i in range(rows)
    ]

    # Churned accounts: present in events, absent from active subscriptions.
    # Without these, "distinct account_ids in events" would be a correct
    # shortcut, and the diff the task is really about would be optional.
    for i in range(5):
        events.append(
            {
                "account_id": f"ACC-CHURNED-{i:04d}",
                "feature": features[i % len(features)],
                "event_timestamp": "2025-01-01T12:00:00Z",
            }
        )

    registry.create(
        "list",
        subscriptions,
        name="active_subscriptions",
        description=(
            f"{n_accounts} active subscriptions with account_id, company, plan, "
            "mrr, signup_date"
        ),
        created_by="eval_scale_generator",
    )
    registry.create(
        "list",
        events,
        name="feature_events",
        description=(
            f"{len(events)} feature usage events with account_id, feature, "
            "event_timestamp"
        ),
        created_by="eval_scale_generator",
    )


def check_uc3_large(
    registry: dale.DataRegistry, action_log: ActionLog, final_answer: str
) -> bool:
    return _answer_matches(registry, _UC3_LARGE_EXPECTED, "account_id")


# --- Use Case 4: org permission inheritance -------------------------------

def setup_uc4(registry: dale.DataRegistry) -> None:
    base = DATA_DIR / "04_org_permissions"
    _load(registry, base / "org_structure.csv")
    _load(registry, base / "policy_rules.csv")


TASK_UC4 = (
    "You have an organization chart (employee ID, name, title, manager ID) and "
    "access policy rules (employee ID, resource, effect — where effect is "
    "Allow or Deny). Permissions are inherited down the management chain from "
    "whoever a rule is attached to, but an explicit Deny always overrides an "
    "inherited Allow for the same resource. Compute the effective permissions "
    "for every employee who has an applicable rule (their own or inherited), "
    "and tell me the resulting resource/effect for each."
)

_UC4_EXPECTED = {
    "E002": {"prod_database": "Allow"},
    "E003": {"prod_database": "Allow"},
    "E004": {"prod_database": "Allow"},
    "E005": {"prod_database": "Allow"},
    "E006": {"prod_database": "Allow", "staging_environment": "Allow"},
    "E007": {"prod_database": "Deny", "staging_environment": "Allow"},
    "E008": {"prod_database": "Allow", "staging_environment": "Allow"},
    "E009": {"customer_crm": "Allow"},
    "E010": {"customer_crm": "Allow"},
    "E011": {"customer_crm": "Deny"},
    "E012": {"customer_crm": "Allow"},
    "E013": {"billing_system": "Allow"},
    "E014": {"billing_system": "Allow"},
    "E015": {"billing_system": "Allow"},
}


def check_uc4(registry: dale.DataRegistry, action_log: ActionLog, final_answer: str) -> bool:
    for rows in _alive_lists(registry):
        if "node_id" not in rows[0]:
            continue
        by_employee: dict[str, dict[str, Any]] = {}
        ok = True
        for r in rows:
            if "resource" not in r or "effect" not in r:
                ok = False
                break
            by_employee.setdefault(r["node_id"], {})[r["resource"]] = r["effect"]
        if ok and by_employee == _UC4_EXPECTED:
            return True
    return False


USE_CASES = {
    "uc1": (setup_uc1, TASK_UC1, check_uc1),
    "uc2": (setup_uc2, TASK_UC2, check_uc2),
    "uc3": (setup_uc3, TASK_UC3, check_uc3),
    # Same task as uc3 at $DALE_SCALE_ROWS events (default 10,000) — the
    # scale comparison, not a new reasoning problem.
    "uc3_large": (setup_uc3_large, TASK_UC3, check_uc3_large),
    "uc4": (setup_uc4, TASK_UC4, check_uc4),
}


# --- arm-independent ground truth (eval/baseline.py) -----------------------
#
# The checkers above grade DALE by inspecting registry state after a run. The
# raw-data-in-prompt baseline produces no registry state at all — it answers
# in one shot — so those checkers would return False for it unconditionally,
# and a 0% baseline that measured nothing would look like a result.
#
# Rejected the tempting shortcut of stuffing the baseline's answer into a
# fresh handle and reusing check_*: _identifier_sets' shape tolerances were
# built to accept the shapes *DALE pipelines* legitimately produce (dict_diff
# rows nested under previous_value, and so on). Applying them to free-form
# model output would quietly change what "correct" means between the two arms
# — the one thing that must not vary in a comparison.
#
# So ground truth is declared once, here, in a shape neither arm owns.


@dataclass(frozen=True)
class ExpectedAnswer:
    """What a correct answer contains, independent of how it was produced.

    `kind` says how to compare, not what the pipeline looked like:
      - "id_set"        : an unordered set of identifiers
      - "ordered_pairs" : [(id, number), ...] where order is part of the claim
      - "counts"        : {id: how many}, when the task asks "how many each"
      - "mapping"       : {id: {key: value}}
    `alternates` holds equally-defensible readings of an ambiguous task (uc1's
    untracked-SKU case), so a model isn't marked wrong for resolving an
    ambiguity this project introduced.
    """

    kind: str
    value: Any
    alternates: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """An empty expectation is refused rather than stored.

        Every matcher here compares for equality, so an empty expected value
        makes a grader that passes any answer that names nothing — a silent
        100% instead of a loud failure. The realistic way to get one is
        uc3_large, whose expected set is generated by its `setup()` and is
        empty until that has run; that ordering bug should surface here, at
        construction, not as a suspiciously perfect result three steps later."""
        if not self.value:
            raise ValueError(
                f"expected answer is empty — did setup() run? (kind={self.kind!r})"
            )

    def _candidates(self) -> tuple[Any, ...]:
        return (self.value,) + self.alternates

    def matches_ids(self, ids: Iterable[Any]) -> bool:
        got = set(ids)
        return any(got == set(c) for c in self._candidates())

    def matches_pairs(self, pairs: Sequence[tuple[Any, float]]) -> bool:
        got = [(k, round(float(v), 2)) for k, v in pairs]
        return any(got == list(c) for c in self._candidates())

    def matches_counts(self, counts: dict[Any, int]) -> bool:
        return any(dict(counts) == dict(c) for c in self._candidates())

    def matches_mapping(self, mapping: dict[Any, Any]) -> bool:
        return any(mapping == c for c in self._candidates())


# Callables, not values: uc3_large's expected set only exists once its setup
# has run and generated the data.
EXPECTED_ANSWERS: dict[str, Callable[[], ExpectedAnswer]] = {
    "uc1": lambda: ExpectedAnswer(
        kind="ordered_pairs",
        value=_UC1_EXPECTED,
        alternates=(_UC1_EXPECTED_EXCLUDING_UNTRACKED,),
    ),
    "uc2": lambda: ExpectedAnswer(kind="counts", value=dict(_UC2_EXPECTED_FLAG_COUNTS)),
    "uc3": lambda: ExpectedAnswer(kind="id_set", value=_UC3_EXPECTED),
    "uc3_large": lambda: ExpectedAnswer(kind="id_set", value=set(_UC3_LARGE_EXPECTED)),
    "uc4": lambda: ExpectedAnswer(kind="mapping", value=_UC4_EXPECTED),
}
