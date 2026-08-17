# Sample datasets — DESIGN.md Use Cases 1-4

Synthetic data for the first four of the five enterprise patterns in `DESIGN.md` Section 3
("Key Enterprise Software Patterns & Use Cases"). Built to actually exercise the engine in
live-agent runs, not toy placeholder rows — and, as of `window_flag`/`graph_walk_resolve` landing,
now also fully wired into DALE's own operation pipelines with committed regression tests, not just
data sitting alongside the engine. Each dataset has deliberate edge cases baked in; a correct
pipeline has to actually handle them, not just run happy-path logic. All files are CSV, since
`load_csv` is currently the only built loader (`load_json` is built too; `load_jsonl` is planned).

Ground-truth results below were computed with a plain reference script (or, where noted, by direct
operation calls through `dale.call_operation`) — independent of any LLM — following the evaluation plan's
"independently-computed ground truth" requirement .

## 1. Multi-Source Inventory & Price Reconciliation — `01_inventory_reconciliation/`

**Scenario** (DESIGN.md Use Case 1): reconcile product feeds from three sources sharing a SKU.
**Pipeline:** `load_csv` ×3 → `index_by` (stock, pricing) → `join_lookup` ×2 → `filter_where`
(drop zero/untracked stock) → `compute_field` (margin) → `sort_by` (margin desc).
**Fully runnable today** — every operation it needs already exists.

| File | Rows | Schema |
|---|---|---|
| `product_metadata.csv` | 20 | `sku, name, category` |
| `stock_counts.csv` | 19 | `sku, stock_count` |
| `pricing_overrides.csv` | 19 | `sku, cost, price` |

**Edge cases:**
- `SKU-1020` (Ring Light) has no row in `stock_counts.csv` at all — an untracked/new item, not a
  zero-stock item. Filtering with `stock_count > 0` (an *ordering* comparison) raises
  `TypeMismatchError` on this row, since ordering ops refuse to compare against a missing field.
  `stock_count != 0` (an *equality*-class comparison) is the correct predicate — it treats the
  missing field as `None`, which is `!= 0`, so the row is correctly kept.
- `SKU-1011` (Desk Lamp) has stock but no row in `pricing_overrides.csv` — in stock, but nothing to
  compute a margin from. A pipeline that runs `compute_field` before filtering these out gets
  `InvalidParamsError` on the missing operand. Correct order: filter out unpriced rows first.
- Five SKUs (`1002, 1005, 1008, 1013, 1017`) have `stock_count = 0` and should be dropped.

**Verified ground truth** (`filter_where(stock_count != 0)` → `filter_where(cost != None)` →
`compute_field(margin = price - cost)` → `sort_by(margin desc)`) — 14 of 20 SKUs survive:

```
SKU-1010  Standing Desk                 margin=239.99
SKU-1016  4K Monitor                    margin=179.99
SKU-1009  Ergonomic Chair               margin=159.99
SKU-1004  27in Monitor                  margin=109.99
SKU-1007  Noise Cancelling Headphones   margin=69.99
SKU-1012  Portable SSD 1TB              margin=44.99
SKU-1019  USB Microphone                margin=34.99
SKU-1020  Ring Light                    margin=19.99
SKU-1003  USB-C Hub                     margin=17.99
SKU-1006  Laptop Stand                  margin=16.99
SKU-1014  Wireless Charger              margin=15.99
SKU-1001  Wireless Mouse                margin=11.49
SKU-1018  Mousepad XL                   margin=8.99
SKU-1015  Cable Organizer               margin=6.49
```

## 2. Multi-Region Security Log Sessionization — `02_log_sessionization/`

**Scenario** (DESIGN.md Use Case 2): detect credential stuffing from auth log dumps.
**Pipeline:** `load_csv` → `window_flag` (groups by `source_ip`, predicate `outcome == "fail"`,
`window_size=300` seconds, `threshold=5`) → `filter_where` (`flagged == True`).
**Fully runnable today** — `window_flag` was built specifically for this use case; no `sort_by`
step needed first, since `window_flag` sorts internally.

| File | Rows | Schema |
|---|---|---|
| `auth_logs.csv` | 40 | `timestamp, region, source_ip, username, outcome` |

**Edge cases:**
- `203.0.113.55` (us-east): 7 failed logins against 7 different usernames in ~2.5 minutes
  (09:14:00–09:16:17), then a **success** at 09:17:05 — a credential-stuffing attempt that appears
  to have worked. `window_flag` flags the success too, not just the failures — every record gets a
  window count computed from *matching* (fail) neighbors regardless of whether the record itself
  matches, so the success inherits the count from the burst immediately preceding it.
- `198.51.100.23` (eu-west): 8 failed logins against 5 distinct usernames (2 retried) in ~2.5
  minutes (14:02:10–14:04:50), no success — attack attempted, apparently unsuccessful.
- Everything else is normal single-user traffic (occasional isolated fail-then-success, e.g.
  `90.187.22.14`/`klein`, `71.12.44.9`/`jdoe`) — correctly **not** flagged.
- `203.0.113.55` reappears later (16:33, ap-south, user `vpatel`, success) — same IP, different
  region/user, hours later, well outside any 300-second window of the earlier burst, so it's
  correctly not flagged even though `window_flag` only groups by `source_ip`, not region.

**Verified ground truth** (`window_flag` + `filter_where(flagged == True)`) — 8 of 40 rows flagged:
the 3 threshold-crossing fails and the trailing success for `203.0.113.55` (5th through 7th fail,
plus the `asmith` success at `09:17:05`), and the 4 threshold-crossing fails for `198.51.100.23`
(5th through 8th fail). See `tests/test_use_case_pipelines.py::test_use_case_2_log_sessionization`.

## 3. Customer Churn Risk & Feature Usage Aggregation — `03_churn_feature_usage/`

**Scenario** (DESIGN.md Use Case 3): find active subscribers with zero product usage.
**Intended pipeline** (DESIGN.md): `index_by` (subscriptions) → `dict_frequency` (event counts) →
`set_difference` → `sort_by`. `dict_frequency`/`set_difference` are not yet built.
**Currently-runnable alternative pipeline**, using only what exists today: `group_by(feature_events,
account_id)` → `join_lookup(active_subscriptions, grouped_events, how="left")` → `filter_where`
(matched field `== None`) isolates accounts with no matching event bucket. Verified by direct
`dale.call_operation` calls — see below.

| File | Rows | Schema |
|---|---|---|
| `active_subscriptions.csv` | 15 | `account_id, company, plan, mrr, signup_date` |
| `feature_events.csv` | 39 | `account_id, feature, event_ts` |

**Edge cases:**
- `ACC-004`, `ACC-009`, `ACC-013` are active subscribers with **zero** rows in
  `feature_events.csv` — the churn-risk accounts a correct pipeline must surface.
- `ACC-099` has feature events but is **not** in `active_subscriptions.csv` (already churned,
  events still trickling in) — noise that a `base_handle=subscriptions` left-join correctly
  ignores, since the join walks from the subscriptions list, not the events.

**Verified ground truth** (via the alternative pipeline above): `ACC-004` (Initech), `ACC-009`
(Soylent Corp), `ACC-013` (Oscorp).

## 4. Organization Chart Role & Permission Inheritance — `04_org_permissions/`

**Scenario** (DESIGN.md Use Case 4): compute effective permissions across an org chart with
Deny-over-Allow conflict resolution.
**Pipeline:** `load_csv` → `index_by(employee_id)` → `graph_walk_resolve` (`parent_field=manager_id`,
`group_field=resource`, `value_field=effect`, `priority=["Deny", "Allow"]`).
**Fully runnable today** — `graph_walk_resolve` was built specifically for this use case, and is
exactly the operation `grammar.py`'s `Priority`/`resolve_priority` were reserved for.

| File | Rows | Schema |
|---|---|---|
| `org_structure.csv` | 15 | `employee_id, name, title, manager_id` (empty for the CEO/root) |
| `policy_rules.csv` | 6 | `employee_id, resource, effect` (`effect` is `Allow`/`Deny`) |

Note: `policy_rules.csv` intentionally has **no** numeric priority column. `dale.grammar.Priority`
is a single ordered list of *values* (e.g. `["Deny", "Allow"]`), not a per-rule numeric score —
conflict resolution is "does `Deny` appear anywhere in the applicable set," not "which rule has the
highest number." A per-rule priority field would misrepresent how `resolve_priority` actually works.

**Edge cases:**
- `E007` (Grace Lin, individual contributor under Eng Manager B) has an explicit `Deny` on
  `prod_database`, while her chain also inherits an `Allow` from `E002` (VP Engineering, four
  levels up). Correct resolution: `Deny` wins — she loses access despite the broad inherited grant.
- `E011` (Sales Rep) has an explicit `Deny` on `customer_crm`, overriding the `Allow` inherited from
  `E009` (VP Sales).
- `E006`'s team additionally inherits a narrower `staging_environment` Allow that doesn't apply to
  the rest of engineering — tests that inheritance is chain-scoped, not org-wide.

**Verified ground truth** — computed independently two ways: a plain Python reference script (walk
each employee's manager chain to the root, union all `(resource, effect)` rules found along it,
resolve conflicts via `["Deny", "Allow"]`) and, separately, the real `graph_walk_resolve` operation
via `dale.call_operation` — both produce an identical result
(`tests/test_use_case_pipelines.py::test_use_case_4_org_permissions`):

```
E002  {'prod_database': 'Allow'}
E003  {'prod_database': 'Allow'}
E004  {'prod_database': 'Allow'}
E005  {'prod_database': 'Allow'}
E006  {'prod_database': 'Allow', 'staging_environment': 'Allow'}
E007  {'prod_database': 'Deny', 'staging_environment': 'Allow'}
E008  {'prod_database': 'Allow', 'staging_environment': 'Allow'}
E009  {'customer_crm': 'Allow'}
E010  {'customer_crm': 'Allow'}
E011  {'customer_crm': 'Deny'}
E012  {'customer_crm': 'Allow'}
E013  {'billing_system': 'Allow'}
E014  {'billing_system': 'Allow'}
E015  {'billing_system': 'Allow'}
```

(E001/CEO has no applicable rules in this sample and is omitted.)

## `flatten_json` regression fixture — `json_flatten_github_issues/`

Not one of DESIGN.md's numbered use cases — a smaller, `flatten_json`-specific fixture (UC7's
own scenario needs multi-level nesting `flatten_json` doesn't support yet).
Real data, not synthetic: fetched live from `api.github.com/repos/pandas-dev/pandas/issues`
during design, trimmed down to `number`/`title`/`labels` (dropped the `*_url` fields, `user`
object, etc. — not relevant to this fixture) with every remaining value exactly as fetched, no
edits to the real content.

**Scenario:** for each issue, one row per label it has, with the issue number/title carried onto
each row — the worked example `flatten_json`'s design was built backward from.
**Pipeline:** `load_json` → `flatten_json(path=["labels"], carry_fields=["number", "title"])`.

| File | Records | Schema |
|---|---|---|
| `pandas_issues.json` | 8 | `number, title, labels` (`labels` is an array of `{id, name, color, description}` objects, empty on 7 of the 8) |

**Edge case baked in:** only issue `#66594` has any labels in this real snapshot (`Bug`, `IO CSV`)
— the other 7 have `"labels": []`. A correct `flatten_json` call produces exactly 2 output rows,
both attributed to `#66594`; the other 7 issues contribute nothing, without needing a separate
filter step (see `tests/test_flatten_json.py::test_real_github_issues_label_explosion`).

## Status summary

All four use cases now run end to end through real DALE operations, each with a committed
regression test in `tests/test_use_case_pipelines.py` asserting independently-computed ground
truth.

| # | Use case | Runnable through DALE today | Notes |
|---|---|---|---|
| 1 | Inventory reconciliation | Yes | Intended pipeline, no gaps |
| 2 | Log sessionization | Yes | Via `window_flag`, built for this use case |
| 3 | Churn / feature usage | Yes, via an alternative pipeline | `dict_frequency`/`set_difference` (still unbuilt) is the *intended* pipeline; the working alternative uses `group_by`+`join_lookup`+`filter_where` instead |
| 4 | Org permission inheritance | Yes | Via `graph_walk_resolve`, built for this use case — general `graph_bfs`/`graph_dfs` were not needed |

Use Case 1 remains the simplest candidate for a live-agent run (no predicate-heavy or newly-built
operations involved). Use Cases 2 and 4 are the more interesting live-agent tests now that
`window_flag`/`graph_walk_resolve` exist — they require the model to correctly select and
parameterize an operation it's likely seen fewer analogous examples of during training than
`filter_where`/`sort_by`, which is exactly the kind of orchestration-correctness question the evaluation is
for. Use Case 3's alternative pipeline remains a good test of whether the agent can improvise a
join-based workaround rather than reaching for the "obvious" (but unbuilt) frequency/set-difference
approach.
