# DALE — Declarative Algorithmic Logic Engine

**describe a problem, provide the inputs, and let DALE compose algorithms**

DALE lets an LLM compose a fixed catalog of validated, declarative algorithms over data that stays
in host memory behind named handles: the model selects operations and structured parameters, never
sees the full dataset, and never generates or executes code against it.

## Install

```bash
pip install "dale-engine[agent]"
```

DALE requires Python 3.10+ and one supported model-provider API key. The distribution is
`dale-engine`; the Python import is `dale`.

## Quickstart

### Ask a question about in-memory data

```python
import dale
from dale.agent import ActionLog, build_agent, pick_model, run_agent

people = [
    {"name": "Alice", "department": "Engineering", "age": 34},
    {"name": "Bob", "department": "Sales", "age": 29},
    {"name": "Carol", "department": "Engineering", "age": 41},
]

registry = dale.DataRegistry()
registry.create(
    "list",
    people,
    name="people",
    description="employee roster with name, department, and age",
    created_by="quickstart",
)

action_log = ActionLog()
action_log.seed_from_registry(registry)
agent = build_agent(registry, action_log, model=pick_model())

outcome = run_agent(
    agent,
    "How many people are in Engineering, and what are their names?",
    deps=registry,
    action_log=action_log,
)
if not outcome.success:
    raise SystemExit(f"run failed: {outcome.error}")

engineering = registry.materialize(outcome.result.output.handle)
print(engineering)
```

```text
[{'name': 'Alice', 'department': 'Engineering', 'age': 34}, {'name': 'Carol', 'department': 'Engineering', 'age': 41}]
```

### Reconcile service data with in-memory business data

Here the host application has already parsed a microservice's JSON response into a Python
dictionary. DALE reconciles it with existing orders and a product-discount table; it does not make
the network request or mutate the original order list.

```python
import dale
from dale.agent import ActionLog, build_agent, pick_model, run_agent

orders = [
    {
        "order_id": "ORD-1001",
        "product_id": "PROD-A",
        "quantity": 2,
        "unit_price": 50.0,
        "status": "processing",
    },
    {
        "order_id": "ORD-1002",
        "product_id": "PROD-B",
        "quantity": 1,
        "unit_price": 80.0,
        "status": "processing",
    },
]

order_service_response = {
    "updates": [
        {"order_id": "ORD-1001", "status": "shipped"},
        {"order_id": "ORD-1002", "status": "cancelled"},
        {"order_id": "ORD-9999", "status": "shipped"},
    ]
}

product_discounts = [
    {"product_id": "PROD-A", "discount_rate": 0.10},
    {"product_id": "PROD-B", "discount_rate": 0.25},
]

registry = dale.DataRegistry()
for name, data, description in [
    ("orders", orders, "orders currently held by this application"),
    (
        "order_updates",
        order_service_response["updates"],
        "parsed status updates returned by the order service",
    ),
    ("product_discounts", product_discounts, "discount rate for each product"),
]:
    registry.create(
        "list",
        data,
        name=name,
        description=description,
        created_by="host",
    )

action_log = ActionLog()
action_log.seed_from_registry(registry)
agent = build_agent(registry, action_log, model=pick_model())

outcome = run_agent(
    agent,
    """
    Reconcile the existing orders with the order-service updates and product discounts.
    Existing orders are authoritative: update their status by order_id, but do not create
    orders that appear only in order_updates. Match discounts by product_id. For every order,
    add discount_amount (unit_price * discount_rate), discounted_unit_price
    (unit_price - discount_amount), and line_total (discounted_unit_price * quantity).
    Return the reconciled orders.
    """,
    deps=registry,
    action_log=action_log,
)
if not outcome.success:
    raise SystemExit(f"run failed: {outcome.error}")

reconciled_orders = registry.materialize(outcome.result.output.handle)
print(reconciled_orders)
```

Both examples require credentials for a compatible model provider. Set `DALE_MODEL` to choose a
specific provider and model; see [Agent integration](docs/agent.md) for current configuration and
compatibility details.

## Docs

- [Guide](GUIDE.md) — concepts, operations, examples, extension points, and development
- [Design](DESIGN.md) — architecture, scope, and design rationale
- [Agent integration](docs/agent.md) — model setup, execution, observability, and usage
- [Environment and security](docs/environment.md) — deployment boundaries and resource controls
- [Changelog](CHANGELOG.md) — release notes; **0.2.0 renames `primitive` → `operation` with no back-compat shims**

## License

Apache 2.0. See [LICENSE](LICENSE).
