# DALE — Architecture

**Design goal:** let an LLM solve data-processing and algorithmic workflows without hitting context
window limits, generating executable code, or incurring execution security risks.

**The approach in one line:** semantic state and pointer abstraction — the LLM's output is
restricted to non-Turing-complete declarative parameters, and the data it operates on never leaves
the host.

This document covers the architecture and the reasoning behind it. For installation and usage, see
the [README](README.md).

## 1. Executive Summary
Traditional AI agents struggle with processing large datasets (10,000+ items) because they either:

* Load raw data directly into the prompt context, leading to token exhaustion and high latency.  
* Generate raw Python code in a loop, introducing syntax errors, non-deterministic outputs, and severe security risks.

This project introduces an architecture in which **the LLM's output is restricted to a non-Turing-complete declarative grammar** — structured parameters (predicates, thresholds, priority orders), never executable instructions — consumed by a fixed, pre-written set of host-side operations. The LLM operates strictly as an **Algorithmic Orchestrator**, manipulating native, high-performance host data structures (list, dict, set) purely through string handles or pointers. Data remains hidden in native memory, while the LLM receives only structural metadata summaries. This constraint — not code generation, only bounded declarative data — is what makes the safety and pre-execution cost-estimation properties described in later sections possible.

**Non-Goal / Scope Boundary:** DALE targets single-shot, batch data-processing pipelines and simple business-rule workflows (the patterns in Section 3) — not long-horizon, many-turn conversational state mutation (e.g., a banking ledger accumulating transactions turn by turn, or a smart-home controller tracking relative commands across a session). That territory is well served by stateful code-execution frameworks such as CaveAgent ([Ran et al., 2026](https://arxiv.org/abs/2601.01569)), which deliberately keeps a persistent, Turing-complete code-execution kernel to support exactly this kind of interdependent, incremental state management. DALE is not attempting to be a general-purpose stateful agent runtime; it is a narrower tool for getting a large dataset from input to a correct answer in one pass, safely and cheaply.

```
+--------------------------------------------------------+
|                       LLM AGENT                        |
|    (Sees: "var_list_1: List of size 10,000", Logic)    |
+---------------------------+----------------------------+
                            |
              Calls Tools via Handles/IDs
                            v
+--------------------------------------------------------+
|                  STATEFUL DATA ENGINE                  |
|    (Manages _storage = {'var_list_1': [...]}, Logic)   |
+--------------------------------------------------------+
```

## 2. Core Architectural Principles
* **Pointer-Based State Management:** Native Python collections live in host memory. The LLM only sees metadata blocks (e.g., {"handle": "user_list", "size": 10000, "type": "list"}) — never the underlying data itself. A handle's identifier is not an opaque, host-generated counter (`list_7`) with a separate human-readable label bolted on afterward — the two are unified into one field. Every handle-creating call, whether issued directly by the invoker via `registry.create()` or by the LLM through an operation's tool call, must supply a `name` that *becomes* the handle itself, constrained to satisfy Python's own `str.isidentifier()` (and not be a keyword) — the same bar as a variable name, so a handle reads identically everywhere it appears: the LLM's own context, a human's task prompt, and a debugger session, with no translation step between what any of the three call it. A name colliding with an already-alive handle is rejected outright — a structured, catchable error, the same pattern `index_by` already uses for `DuplicateKeyError` — rather than silently disambiguated with an appended suffix, consistent with this project's general bias toward making the caller be explicit rather than having the system guess on their behalf. `description` is mandatory alongside it for the same reason, and critically, an honest "unknown, uninspected data" is a complete, encouraged answer when true, not a burden to avoid — it is itself useful signal, telling the LLM to `peek()`/`describe()` before assuming structure, rather than a confident-sounding guess standing in for the caller's actual uncertainty.  
* **Native C-Speed Execution:** Processing logic runs directly on underlying Python lists and dictionaries, guaranteeing deterministic `O(1)` lookups and `O(N log N)` sorting speeds.  
* **Vectorized Operations:** To eliminate expensive N-turn loops through the context window, the system provides high-level functional tools (filter_where, batch_dedup, merge_collections).  
* **Zero-Sandbox Security:** By restricting the LLM to pre-compiled API methods rather than raw code execution strings, arbitrary system access and malicious prompt injections are inherently blocked.  
* **Single-Tenant Execution Model:** Each agent invocation runs in its own dedicated process/sandbox for the duration of the task, with no state shared across sessions. There is no multi-tenant fairness problem to solve — resource governance relies on this: a single-invocation memory/turn ceiling is sufficient, since no other tenant shares the process.  
* **Build-Time Extensibility, Not Runtime Discovery:** The operation catalog is not fixed forever at the patterns in Section 3 — developers may register new operations before deployment via `register_operation(name, fn, param_schema, cost_estimator)`. `fn` is ordinary Python the developer writes and reviews (wrapping any library — pandas, scipy, a domain-specific calculation); `param_schema` constrains what the LLM may pass to the same declarative grammar used elsewhere (no string-eval'd expressions, no callable references, no module/function selection as a parameter); `cost_estimator` keeps pre-execution cost estimation intact for third-party operations. The LLM's action space grows only in the sense that the fixed set of selectable operations grows — it never selects, imports, or invokes a module/function at runtime. This mirrors the trusted/untrusted distinction SQL databases draw for user-defined functions (e.g., PostgreSQL's sandboxed `plpgsql` vs. superuser-only `plpythonu`): DALE's extension mechanism is the trusted case by design, since letting the LLM choose which module or function to call at runtime is exactly the class of risk that blocks enterprise adoption.  
* **Optional Strict-Privacy Mode:** Because field names and business rules are supplied upfront in the problem statement rather than discovered by inspection (Section 2, Pointer-Based State Management), the LLM does not fundamentally need to see real data values to construct correct pipelines — unlike code-generation alternatives, whose write/execute/debug loop requires inspecting real intermediate values to catch and fix generated code. An opt-in `strict_privacy` mode (default off) disables `peek` (or restricts it to type/shape only), restricts `describe` to aggregate statistics with no individual values, sanitizes error messages to strip real content, and routes final results through a new `export_handle(handle, destination)` operation that delivers pipeline output directly to an external sink without ever passing it through the LLM's context. Under this mode, no real data value reaches the LLM at any point in a session — a claim code-generation-based tools cannot make without abandoning their own debugging mechanism.  
* **Local File Ingestion, Not a Connector Framework:** DALE assumes data arrives already materialized or loadable from a local file — it is not a data-ingestion/ETL framework and ships no live database or API connectors. Built-in loaders cover local, static, already-present input — `load_csv` and `load_json` are built today, with `load_jsonl` and `load_parquet` planned; each parses to one canonical shape, with restructuring into a dict or set handled as an explicit downstream operation step (`index_by`, `group_by`), not an ambiguous decision hidden in the loader. Loaders take a **virtual file name registered ahead of time on a `FileRegistry`, never a raw filesystem path** — the LLM picks among names the invoker explicitly exposed for the session, the same opaque-reference pattern `DataRegistry` already applies to in-memory data (Section 2, Pointer-Based State Management), rather than being handed an open-ended path string it could point anywhere the process can read. Connecting to a live external system (a database, an API) is out of scope for the same reason a runtime code-execution surface is: if the LLM has any say in what gets queried against a live system, that reopens an injection-shaped risk pointed at production infrastructure. Where live-source connectivity is genuinely needed, it is a `register_operation` (Build-Time Extensibility) integration a developer writes and vets, not a capability DALE owns — and as a result, DALE's process needs no network access or credentials in the common case, a stronger boundary than a sandboxed connection.  
* **Data Structure Scope:** Three handle types ship today — `list`, `dict`, `set`. Most of the classical data-structures canon needs no fourth type. A tree or graph is an adjacency-shaped `dict`; a composite key is a `tuple` produced by `index_by`/`group_by` accepting multiple key fields. Others are deliberately internal: a `deque` inside `window_flag`'s sliding window, a heap-equivalent inside `top_k`.

  The registry is expected to widen, and two extensions are live design projects rather than settled exclusions. **Scalars as handles** is justified by strict-privacy mode: a computed total or a salary threshold should not round-trip through the LLM's context just to be compared against. Its cost is a handle-reference form in the predicate grammar (`value: 42` vs `value: {ref: "threshold"}`), which doubles a surface whose smallness is a headline claim. **First-class `graph`/`tree`** would move preconditions from walk time to build time and let the model read `type: "graph"` instead of inferring shape from a description string. Its cost is that a type is only worth having if something enforces its invariant, so it implies a constructor operation that validates parent resolvability and cycles at construction, and it turns `HandleType` from a `Literal` into a build-time type registry — closed to the LLM at runtime, the same trust model as the operation catalog, one level down.

  What stays out of scope is narrower than "everything else": structures built for high-frequency-update/range-query or online/streaming workloads (segment trees, Fenwick trees, exposed priority queues). They serve live, incremental workloads, which the Non-Goal / Scope Boundary above already excludes. That is a deliberate exclusion, not a gap in coverage.

## 3. Key Enterprise Software Patterns & Use Cases
Rather than academic contest puzzles, the single agent orchestrator manages full multi-step data processing pipelines using native pointer tools across common software engineering scenarios:

### Use Case 1: Multi-Source Inventory & Price Reconciliation
* **Scenario:** Reconcile product feed files from three suppliers sharing a SKU identifier.  
* **Pipeline:** Load feed lists into pointers → index primary metadata by SKU → join stock counts and pricing overrides via key lookups → filter out zero-stock items → export a clean catalog sorted by profit margin.

### Use Case 2: Multi-Region Security Log Sessionization
* **Scenario:** Detect credential stuffing attacks from global authentication log dumps.  
* **Pipeline:** Merge log streams → sort records chronologically (`O(N log N)`) → execute a sliding window grouping tool over timestamps → filter for failed attempt thresholds → aggregate flagged IPs into an alert set.

### Use Case 3: Customer Churn Risk & Feature Usage Aggregation
* **Scenario:** Identify churn-risk SaaS users by comparing active subscriptions against click logs.  
* **Pipeline:** Index active subscriptions → aggregate millions of feature event clicks into a frequency map → execute a set difference operation to isolate zero-usage accounts → sort high-value accounts for customer success outreach.

### Use Case 4: Organization Chart Role & Permission Inheritance
* **Scenario:** Compute effective system permissions for 50,000 employees.  
* **Pipeline:** Load org structures into an adjacency graph → read policy rules → execute topological graph traversals (DFS/BFS) to compute inherited permissions → resolve priority conflicts (Deny > Allow) → export effective permission maps.

### Use Case 5: Multi-Tenant Database Migration Diff & Audit
* **Scenario:** Verify multi-tenant database synchronization before cutover.  
* **Pipeline:** Ingest two database snapshots into key-value maps → execute a symmetric difference tool (dict_diff) to locate missing keys and payload hash mismatches → sort discrepancies by tenant ID → generate an audit report.

### Use Case 6: License/Entitlement Tier Reconciliation with Incremental Diff
* **Scenario:** The project's original motivating problem. Three per-tier eligibility lists (e.g. gold/silver/bronze) arrive as separate JSON files on an hourly cadence; a user appearing on more than one list must be resolved to the single highest tier, and the result must be diffed against the *previous* hour's assignment to find joins, leaves, upgrades, and downgrades.  
* **Pipeline:** Load each tier list (→ `load_json`) → union the lists into one candidate set tagged by source tier → resolve each user to a single tier via priority order (gold > silver > bronze, `resolve_priority`) → diff the resolved assignment against the prior run's snapshot (`dict_diff`) → export new/removed/changed assignments. Exercises the "hash map + priority-ordered reduction" family in the Layer 2 mapping table below.

### Use Case 7: Nested/Wrapped Enterprise API Response Normalization
* **Scenario:** JSON responses from enterprise APIs (Salesforce SOQL relationship queries, ServiceNow Table API reference fields, Shopify orders/line items, GitHub issues/labels, Jira custom fields) arrive envelope-wrapped (`{"records": [...]}`, `{"result": [...]}`, `{"data": [...]}`) and nested, not flat — the API-integration analog of Use Case 1's file-based reconciliation. Pagination is explicitly out of scope; the invoker is assumed to have already fetched and assembled one complete document before it reaches DALE.  
* **Pipeline:** Load the raw JSON response, unwrapping the response envelope in the same call if the model recognizes the source system's shape (→ `load_json(file, remove_envelope=True)`; DALE never guesses this on its own) → explode nested child arrays (e.g. line items, related records) into rows, carrying parent-level fields down (→ `flatten_json`) → filter/aggregate with the existing tabular operations exactly as if the data had loaded from CSV.

## 4. System Layers

### Layer 1: Core Library & Data Engine
* The `DataRegistry` manages hidden `list` and `dict` instances in memory.
* Native operations are exposed as catalog entries rather than as raw method access.
* Batch and vectorized operations keep execution cycles single-turn.
* `register_operation(name, fn, param_schema, cost_estimator)` is a build-time-only developer API (never exposed to the LLM), so the operation catalog can grow past the patterns in Section 3 without reopening a runtime code-execution or module-discovery surface (see Section 2, Build-Time Extensibility).
* `export_handle(handle, destination)` and the `strict_privacy` mode flag provide redacted `peek`/`describe`, sanitized error content, and LLM-context-bypassing final delivery (see Section 2, Optional Strict-Privacy Mode).

### Layer 2: Algorithmic Design Patterns
Compile a standard catalog mapping software development data problems to abstract CS operations.
Build status below reflects the current catalog; the operation table in
[GUIDE.md](GUIDE.md#operation-catalog) is the authoritative list of what ships.

| Enterprise Use Case | CS Paradigm / Mechanism | Required Operations |
| :---- | :---- | :---- |
| **Hierarchical Deduplication** | Hash Maps + Priority Ordering | `priority_reduce` (built) — reduces a group of duplicate records to one via priority order. Exercised by Use Case 6 |
| **Log Stream Sessionization** | Sliding Window / Two-Pointer | `window_flag` |
| **Effective Access Resolution** | Adjacency Graph Traversal | `graph_walk_resolve` (built — a bounded single-parent walk, not general traversal); `graph_bfs`/`graph_dfs`/`graph_topological_sort` remain unbuilt and are not required by any current use case |
| **Feature Usage Aggregation** | Frequency Maps + Set Difference | `dict_frequency`, `set_difference` (unbuilt — `group_by` + `join_lookup` + `filter_where` is a working alternative pipeline for this specific case, see `examples/data/README.md`) |
| **Migration Diff & Audit** | Symmetric Difference | `dict_diff` (built), `sort_by`. Needed by Use Case 5 and Use Case 6 (built out — `examples/09_license_reconciliation.py`) |
| **Entity Resolution / Fraud-Ring Detection** | Union-Find / Connected Components | `graph_connected_components`, `resolve_entities` (both unbuilt) |
| **Top-N Reporting** | Sort + Truncate (batch equivalent of a priority queue) | Not a new mechanism — for batch data this reduces to `sort_by` + take-K, already expressible. A `top_k` operation, if ever added, would be sugar over existing operations, not new capability |
| **Multi-Field Join / Dedup** | Composite Keys | `index_by(key_fields=[...])`, `group_by(key_fields=[...])` (built) |
| **Nested JSON Normalization** | Explode + Carry-Down (tabular flattening) | `load_json` (built), `flatten_json` (built). Exercised by Use Case 7 |

### Layer 3: Agent Framework
* Tool methods are wrapped in strict Pydantic models for parameter validation.
* Integration targets type-safe agent frameworks; the shipped implementation uses [PydanticAI](https://ai.pydantic.dev/) (see [`docs/agent.md`](docs/agent.md)).
* System guardrails are enforced at this layer: tool-call count limits, registry memory ceilings, and handle type validation.

## 5. Evaluation

The claims in this document are meant to be tested, not asserted. The evaluation design:

* Synthetic datasets with independently-computed ground truth (a plain reference script, not LLM-generated), for the use cases in Section 3, at multiple sizes (100 / 1,000 / 10,000+ items).
* Each use case run N=10–20 times per size, measuring agent success rate against ground truth — LLM behavior is stochastic, so a single successful run is not sufficient evidence of reliability.
* Failure modes logged and categorized from the intent/action log (wrong operation chosen, malformed predicate, wrong threshold/window size, exhausted turn budget, hit the resource-quota backstop) rather than reported as a bare pass/fail count.
* Context tokens measured across dataset sizes and compared against a naive "raw data pasted into the prompt" baseline, to directly validate the context-window-bounded claim in Section 1.
* The resource-quota mechanism validated: an intentionally oversized operation must trigger `cost_gate_exceeded` before execution, and a runaway tool-call loop must be stopped by the turn cap.
* A comparison against a raw-code-generation baseline agent on 1–2 use cases (correctness rate, safety incidents), to substantiate the comparative claim in Section 1 rather than leaving it asserted.

**Status:** what has actually been run so far is a preliminary pilot — N=5, four of the eight
patterns, a single dataset size — not the full design above. The README states this alongside the
rest of the project's current limitations.

