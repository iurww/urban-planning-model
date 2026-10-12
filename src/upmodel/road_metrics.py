"""The paper's road-planning objective, shared by experts and environments.

All geometric inputs use metres and square metres.  Road density is reported
in the conventional ``km / km^2`` unit; this is the same Length/A expression
after converting both numerator and denominator to kilometres.
"""

from __future__ import annotations

import math
from typing import Iterable


# Covering every road endpoint (including tile-boundary portals) needs a dense
# but still non-trivial subset on the real-road graphs.  85% is the conservative
# default used by the portal-connected low-detour expert; callers may override
# it for a deliberate density experiment.
DEFAULT_ROAD_BUDGET_FRAC = 0.85
DEFAULT_SPACING_MIN_M = 100.0
DEFAULT_SPACING_MAX_M = 600.0


def road_budget(num_roads: int, budget_frac: float = DEFAULT_ROAD_BUDGET_FRAC) -> int:
    """Return the maximum number of selected roads without altering the fraction."""
    if num_roads <= 0:
        return 0
    if not 0.0 < float(budget_frac) <= 1.0:
        raise ValueError("road budget fraction must be in (0, 1]")
    return max(1, min(num_roads, int(math.ceil(float(budget_frac) * num_roads))))


def road_structure_metrics(
    endpoint_pairs: list[tuple[int, int]],
    selected: Iterable[int],
    *,
    interior_nodes: Iterable[int] | None = None,
) -> dict[str, float | bool]:
    """Check that deletions preserve the source graph's physical contract."""
    valid_edges = [
        index
        for index, (left, right) in enumerate(endpoint_pairs)
        if left >= 0 and right >= 0 and left != right
    ]
    required_nodes = sorted(
        {
            node
            for index in valid_edges
            for node in endpoint_pairs[index]
        }
    )
    selected_set = {
        int(index)
        for index in selected
        if 0 <= int(index) < len(endpoint_pairs)
        and int(index) in valid_edges
    }
    source_parent = {node: node for node in required_nodes}
    selected_parent = {node: node for node in required_nodes}
    source_degree = {node: 0 for node in required_nodes}
    selected_degree = {node: 0 for node in required_nodes}

    def find(parent: dict[int, int], node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(parent: dict[int, int], left: int, right: int) -> None:
        left_root, right_root = find(parent, left), find(parent, right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in valid_edges:
        left, right = endpoint_pairs[edge]
        source_degree[left] += 1
        source_degree[right] += 1
        union(source_parent, left, right)
        if edge in selected_set:
            selected_degree[left] += 1
            selected_degree[right] += 1
            union(selected_parent, left, right)

    source_components = len(
        {find(source_parent, node) for node in required_nodes}
    )
    selected_components = len(
        {find(selected_parent, node) for node in required_nodes}
    )
    interior = (
        set(required_nodes)
        if interior_nodes is None
        else {int(node) for node in interior_nodes} & set(required_nodes)
    )
    covered_nodes = sum(selected_degree[node] > 0 for node in required_nodes)
    degree_violations = 0
    for node in required_nodes:
        minimum = min(2, source_degree[node]) if node in interior else 1
        degree_violations += selected_degree[node] < minimum
    connected_within_source = (
        covered_nodes == len(required_nodes)
        and selected_components == source_components
    )
    structurally_feasible = connected_within_source and degree_violations == 0
    return {
        "required_road_nodes": float(len(required_nodes)),
        "covered_road_nodes": float(covered_nodes),
        "road_node_coverage": float(
            covered_nodes / max(len(required_nodes), 1)
        ),
        "source_road_components": float(source_components),
        "selected_road_components": float(selected_components),
        "degree_violations": float(degree_violations),
        "connected_within_source_components": bool(connected_within_source),
        "structurally_feasible": bool(structurally_feasible),
    }


def road_traffic_metrics(
    endpoint_pairs: list[tuple[int, int]],
    lengths_m: list[float],
    selected: Iterable[int],
    community_area_m2: float,
    *,
    budget: int | None = None,
    spacing_min_m: float = DEFAULT_SPACING_MIN_M,
    spacing_max_m: float = DEFAULT_SPACING_MAX_M,
    dead_end_nodes: Iterable[int] | None = None,
) -> dict[str, float | bool]:
    """Evaluate exactly the road reward specified in the source paper.

    ``g_R`` contains the selected road edges and their incident vertices.
    The empty plan is assigned zero because ``1 / NCC`` is undefined for it;
    expert optimization and the rollout mask both require at least one road.
    A disconnected non-empty plan retains the paper's finite positive score.
    """
    if len(endpoint_pairs) != len(lengths_m):
        raise ValueError("road endpoints and lengths must have the same size")
    if community_area_m2 <= 0.0:
        raise ValueError("community area must be positive")

    selected_indices = sorted(
        {int(index) for index in selected if 0 <= int(index) < len(endpoint_pairs)}
    )
    length_sum_m = float(sum(float(lengths_m[index]) for index in selected_indices))
    # (metres / 1000) / (square metres / 1e6) = 1000 * metres / square metres.
    density = 1000.0 * length_sum_m / float(community_area_m2)

    parent: dict[int, int] = {}
    degree: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index in selected_indices:
        left, right = endpoint_pairs[index]
        if left < 0 or right < 0 or left == right:
            continue
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
        union(left, right)

    ncc = len({find(node) for node in parent})
    # Boundary portals are clipped external connections, not interior
    # cul-de-sacs.  Callers with portal metadata can restrict this count to
    # true road junctions; legacy callers retain the original all-node metric.
    dead_end_scope = (
        set(degree)
        if dead_end_nodes is None
        else {int(node) for node in dead_end_nodes}
    )
    dead_ends = sum(
        value == 1 for node, value in degree.items() if node in dead_end_scope
    )
    too_long = sum(float(lengths_m[index]) > spacing_max_m for index in selected_indices)
    too_short = sum(float(lengths_m[index]) < spacing_min_m for index in selected_indices)

    if selected_indices and ncc > 0:
        connectivity_ncc = 1.0 / ncc
        connectivity_dead_end = 1.0 / (1.0 + dead_ends)
        connectivity = connectivity_ncc + connectivity_dead_end
        spacing_long = 1.0 / (1.0 + too_long)
        spacing_short = 1.0 / (1.0 + too_short)
        spacing = spacing_long + spacing_short
        traffic = (density + connectivity + spacing) / 3.0
    else:
        connectivity_ncc = 0.0
        connectivity_dead_end = 0.0
        connectivity = 0.0
        spacing_long = 0.0
        spacing_short = 0.0
        spacing = 0.0
        traffic = 0.0

    selected_count = len(selected_indices)
    within_budget = budget is None or selected_count <= int(budget)
    feasible = selected_count >= 1 and within_budget
    return {
        "Tdensity": float(density),
        "Tconnectivity": float(connectivity),
        "Tconnectivity_ncc": float(connectivity_ncc),
        "Tconnectivity_dead_end": float(connectivity_dead_end),
        "Tspacing": float(spacing),
        "Tspacing_long": float(spacing_long),
        "Tspacing_short": float(spacing_short),
        "Traffic": float(traffic),
        "NCC": float(ncc),
        "dead_ends": float(dead_ends),
        "too_long": float(too_long),
        "too_short": float(too_short),
        "length_sum_m": float(length_sum_m),
        "selected_roads": float(selected_count),
        "budget": float(budget if budget is not None else len(endpoint_pairs)),
        "budget_utilization": float(selected_count / max(int(budget or len(endpoint_pairs)), 1)),
        "connected": bool(ncc == 1 and selected_count > 0),
        "feasible": bool(feasible),
    }
