"""graph_walk_resolve: single-parent ancestor-chain rule resolution
(DESIGN.md's "Adjacency Graph Traversal" pattern, Use Case 4 — organization
chart role & permission inheritance). `grammar.Priority`/`resolve_priority`
were built for exactly this consumer — see the comment in grammar.py.

Deliberately not general BFS/DFS/topological sort: every node has at most one
parent (an org chart, not an arbitrary graph), so a bounded upward walk per
node is the whole algorithm — polynomial, no combinatorial-blowup risk
(the same exclusion rule, same discipline as the regex/ReDoS
exclusion applied to the predicate grammar).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from dale.catalog import ConfirmableParams, OperationOutput, operation
from dale.cost import CostEstimate, make_estimate
from dale.errors import FieldNotFoundError, GraphCycleError, TypeMismatchError
from dale.grammar import Priority, resolve_priority
from dale.registry import DataRegistry


class GraphWalkResolveParams(ConfirmableParams):
    model_config = ConfigDict(populate_by_name=True)

    nodes_index_handle: str
    parent_field: str
    rules_handle: str
    rule_node_field: str
    group_field: str
    value_field: str
    priority: Priority
    name: str
    description: str


def _validate_graph_walk_handles(registry: DataRegistry, params: GraphWalkResolveParams) -> None:
    """Shared by the cost estimator and the operation itself — the estimator
    runs first (dispatch.py), so validation cannot live only in the latter."""
    nodes_meta = registry.meta(params.nodes_index_handle)
    rules_meta = registry.meta(params.rules_handle)
    if nodes_meta.type != "dict":
        raise TypeMismatchError(
            f"graph_walk_resolve nodes_index_handle must be a dict (built via index_by), "
            f"got {nodes_meta.type!r}",
            details={"handle": params.nodes_index_handle, "type": nodes_meta.type},
        )
    if rules_meta.type != "list":
        raise TypeMismatchError(
            f"graph_walk_resolve rules_handle must be a list, got {rules_meta.type!r}",
            details={"handle": params.rules_handle, "type": rules_meta.type},
        )


def _index_rules_by_node(rules: list[dict], node_field: str) -> dict[Any, list[dict]]:
    by_node: dict[Any, list[dict]] = {}
    for rule in rules:
        if node_field not in rule:
            raise FieldNotFoundError(
                f"field {node_field!r} not present on a rule in graph_walk_resolve",
                details={"field": node_field},
            )
        by_node.setdefault(rule[node_field], []).append(rule)
    return by_node


def _collect_chain_rules(
    node_id: Any,
    nodes: dict[Any, dict],
    rules_by_node: dict[Any, list[dict]],
    parent_field: str,
) -> dict[Any, list[Any]]:
    """Walk node_id up to the root, collecting each ancestor's applicable
    rules, bucketed by group_field's value on the rule (caller resolves the
    group_field/value_field lookup — this just gathers raw rule dicts)."""
    chain_rules: list[dict] = []
    visited: set[Any] = set()
    current_id = node_id
    while current_id is not None:
        if current_id in visited:
            raise GraphCycleError(
                f"cycle detected walking the parent chain from node {node_id!r} "
                f"(revisited {current_id!r})",
                details={"node_id": node_id, "repeated": current_id},
            )
        visited.add(current_id)
        chain_rules.extend(rules_by_node.get(current_id, ()))

        current_node = nodes.get(current_id)
        if current_node is None:
            break  # dangling parent reference — treat as root
        current_id = current_node.get(parent_field)

    return chain_rules


def _estimate_output_rows(registry: DataRegistry, params: GraphWalkResolveParams) -> int:
    _validate_graph_walk_handles(registry, params)
    nodes_meta = registry.meta(params.nodes_index_handle)
    rules = registry.get(params.rules_handle)
    distinct_groups = {
        r[params.group_field] for r in rules if params.group_field in r
    }
    # Deliberate over-estimate (like join_lookup's byte estimate): the true
    # count depends on which ancestor chains actually reach a rule for each
    # group, which is the graph walk itself — computing it exactly here would
    # mean doing the operation's own work twice.
    return nodes_meta.size * max(len(distinct_groups), 1)


def graph_walk_resolve_cost_estimator(
    registry: DataRegistry, params: GraphWalkResolveParams
) -> CostEstimate:
    estimated_rows = _estimate_output_rows(registry, params)
    nodes_meta = registry.meta(params.nodes_index_handle)
    return make_estimate(estimated_rows, nodes_meta.avg_record_bytes, registry.limits.max_result_rows)


@operation(
    "graph_walk_resolve",
    GraphWalkResolveParams,
    cost_estimator=graph_walk_resolve_cost_estimator,
    creates_handle=True,
)
def graph_walk_resolve(registry: DataRegistry, params: GraphWalkResolveParams) -> OperationOutput:
    """For every node in an index_by-built dict, walk its single-parent chain
    (self, then parent, then parent's parent, ...) collecting rules that
    attach to any node in the chain, grouped by group_field, and resolve each
    group's collected values via resolve_priority. A node with no parent
    (parent_field missing/None) or whose parent isn't in the index is treated
    as a root. Returns a new list handle: one row per (node, group) pair that
    had at least one applicable rule — {"node_id", <group_field>, <value_field>}."""
    _validate_graph_walk_handles(registry, params)

    nodes = registry.get(params.nodes_index_handle)
    rules = registry.get(params.rules_handle)
    rules_by_node = _index_rules_by_node(rules, params.rule_node_field)

    result: list[dict] = []
    for node_id in nodes:
        chain_rules = _collect_chain_rules(node_id, nodes, rules_by_node, params.parent_field)

        collected: dict[Any, list[Any]] = {}
        for rule in chain_rules:
            if params.group_field not in rule or params.value_field not in rule:
                raise FieldNotFoundError(
                    f"rule for node {node_id!r} missing {params.group_field!r} or "
                    f"{params.value_field!r}",
                    details={"group_field": params.group_field, "value_field": params.value_field},
                )
            collected.setdefault(rule[params.group_field], []).append(rule[params.value_field])

        for group_value, values in collected.items():
            try:
                resolved = resolve_priority(values, params.priority)
            except ValueError as exc:
                raise TypeMismatchError(
                    f"none of {values!r} (node {node_id!r}, {params.group_field}={group_value!r}) "
                    f"found in priority order {params.priority!r}",
                    details={"node_id": node_id, params.group_field: group_value},
                ) from exc
            result.append(
                {"node_id": node_id, params.group_field: group_value, params.value_field: resolved}
            )

    new_meta = registry.create(
        "list",
        result,
        name=params.name,
        description=params.description,
        created_by="graph_walk_resolve",
        source_handles=[params.nodes_index_handle, params.rules_handle],
    )
    return OperationOutput(status="ok", handle=new_meta)
