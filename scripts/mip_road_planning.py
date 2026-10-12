#!/usr/bin/env python3
# coding: utf-8

"""静态道路规划 MIP：构造局部可达的道路网络。

专家解覆盖所有真实 ``kind=junction`` 路口及边界 portal，并让每一个原生道路
分量内部连通。每个原本有至少两条候选边的内部路口在解中至少保持度 2；portal
只要求接入网络，不按死胡同计。郑博士论文的 ``Traffic`` reward 是主目标，
局部绕行只用于数值极小的确定性 tie-break；原始候选图中无法由升级动作跨越的
分量如实计入 NCC。
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd

try:
    gp = importlib.import_module("gurobipy")
    GRB = gp.GRB
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("gurobipy is required.") from exc

from scripts.topology_io import load_topology, resolve_topology_path
from upmodel.road_metrics import (
    DEFAULT_ROAD_BUDGET_FRAC,
    DEFAULT_SPACING_MAX_M,
    DEFAULT_SPACING_MIN_M,
    road_budget,
    road_traffic_metrics,
)


def _ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """WGS84 等地理坐标系转为本地 UTM，使 length/area 单位为米。"""
    if gdf.empty or gdf.crs is None:
        return gdf
    if getattr(gdf.crs, "is_geographic", False):
        return gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def _community_area_m2(parcels: gpd.GeoDataFrame) -> float:
    """社区面积：用地块外包矩形（米制 CRS），避免仅用 parcel 面积之和导致 T_density 爆炸。"""
    if parcels.empty:
        return 1.0
    g = _ensure_metric_crs(parcels)
    minx, miny, maxx, maxy = g.total_bounds
    bbox_area = max(float(maxx - minx) * float(maxy - miny), 1.0)
    if "area" in g.columns and g["area"].notna().any():
        parcel_sum = float(g["area"].sum())
    else:
        parcel_sum = float(g.geometry.area.sum())
    return max(bbox_area, parcel_sum, 1.0)


def _road_lengths_m(roads_r: gpd.GeoDataFrame) -> list[float]:
    if "length_m" in roads_r.columns:
        return roads_r["length_m"].astype(float).tolist()
    return roads_r.geometry.length.astype(float).tolist()


@dataclass
class RoadConfig:
    budget_frac: float = DEFAULT_ROAD_BUDGET_FRAC
    max_runtime_sec: int = 20
    mip_gap: float = 1.0e-4
    spacing_min_m: float = DEFAULT_SPACING_MIN_M
    spacing_max_m: float = DEFAULT_SPACING_MAX_M


def build_parcel_road_adjacency(
    parcels: gpd.GeoDataFrame, roads: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, Dict[int, List[int]]]:
    """roads 重置索引为 0..R-1；返回 parcel_id -> 相邻道路行号列表。"""
    roads_r = roads.reset_index(drop=True)
    roads_r = roads_r.copy()
    if "length_m" not in roads_r.columns or roads_r["length_m"].isna().any():
        roads_r["length_m"] = roads_r.geometry.length.astype(float)
    R = len(roads_r)
    adj: Dict[int, List[int]] = {}
    if R == 0:
        for _, row in parcels.iterrows():
            adj[int(row["parcel_id"])] = []
        return roads_r, adj

    sindex = roads_r.sindex
    for _, row in parcels.iterrows():
        pid = int(row["parcel_id"])
        bnd = row.geometry.boundary
        pos = sindex.query(bnd, predicate="intersects")
        adj[pid] = sorted({int(j) for j in pos})
    return roads_r, adj


def _endpoints_keyed(roads_r: gpd.GeoDataFrame, ndigits: int = 1) -> tuple[list[tuple[int, int]], int]:
    """将道路端点离散为节点，返回道路-端点对与节点数。

    新格式（topology.json）带显式 from_iidx/to_iidx 时直接使用拓扑路口编号，
    否则退回旧的几何端点坐标离散。
    """
    if "from_iidx" in roads_r.columns and "to_iidx" in roads_r.columns:
        pairs_explicit = [
            (int(u), int(v))
            for u, v in zip(roads_r["from_iidx"].astype(int), roads_r["to_iidx"].astype(int))
        ]
        n_nodes_explicit = max((max(u, v) for u, v in pairs_explicit), default=-1) + 1
        return pairs_explicit, n_nodes_explicit

    node_map: Dict[tuple[float, float], int] = {}
    pairs: list[tuple[int, int]] = []

    def _nid(x: float, y: float) -> int:
        key = (round(float(x), ndigits), round(float(y), ndigits))
        if key not in node_map:
            node_map[key] = len(node_map)
        return node_map[key]

    for geom in roads_r.geometry:
        coords = list(geom.coords)
        if len(coords) < 2:
            pairs.append((-1, -1))
            continue
        u = _nid(coords[0][0], coords[0][1])
        v = _nid(coords[-1][0], coords[-1][1])
        pairs.append((u, v))
    return pairs, len(node_map)


@dataclass(frozen=True)
class RoadBackbone:
    """The physical road nodes and edges governed by one expert solution."""

    required_nodes: tuple[int, ...]
    interior_junctions: tuple[int, ...]
    boundary_portals: tuple[int, ...]
    candidate_edges: tuple[int, ...]
    incident: dict[int, list[int]]
    components: tuple[tuple[int, ...], ...]
    component_edges: tuple[tuple[int, ...], ...]
    forced_dead_end_count: int


def _road_backbone_structure(
    intersections: gpd.GeoDataFrame,
    endpoint_pairs: list[tuple[int, int]],
    n_nodes: int,
) -> RoadBackbone:
    """Return the road contract, including every source-connected component.

    Boundary portals represent road stubs that cross the tile boundary.  They
    must be connected to the selected network, but are not counted as interior
    dead ends because clipping a continuing road at a tile boundary is not a
    cul-de-sac.
    """
    endpoint_nodes = {
        node
        for left, right in endpoint_pairs
        for node in (left, right)
        if 0 <= node < n_nodes
    }
    if "kind" in intersections.columns:
        junction_rows = intersections[intersections["kind"].astype(str) == "junction"]
        if "junction_idx" in junction_rows.columns:
            interior_junctions = sorted(
                {int(value) for value in junction_rows["junction_idx"]}
            )
        else:
            interior_junctions = sorted(int(index) for index in junction_rows.index)
        portal_rows = intersections[
            intersections["kind"].astype(str) == "boundary_portal"
        ]
        if "junction_idx" in portal_rows.columns:
            boundary_portals = sorted(
                {int(value) for value in portal_rows["junction_idx"]}
            )
        else:
            boundary_portals = sorted(int(index) for index in portal_rows.index)
    else:
        interior_junctions = sorted(endpoint_nodes)
        boundary_portals = []
    required = sorted(endpoint_nodes)
    if len(required) < 3:
        raise ValueError("道路骨架至少需要 3 个道路端点")
    if any(node < 0 or node >= n_nodes for node in required):
        raise ValueError("junction_idx 与道路端点编号不一致")
    if not set(interior_junctions) <= endpoint_nodes:
        raise ValueError("真实路口与道路端点集合不一致")

    required_set = set(required)
    candidate_edges = [
        edge
        for edge, (left, right) in enumerate(endpoint_pairs)
        if left in required_set and right in required_set and left != right
    ]
    incident = {node: [] for node in required}
    parent = {node: node for node in required}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in candidate_edges:
        left, right = endpoint_pairs[edge]
        incident[left].append(edge)
        incident[right].append(edge)
        union(left, right)

    isolated = [node for node in required if not incident[node]]
    if isolated:
        raise ValueError(
            "道路端点没有可用候选道路，无法构造连通骨架: "
            f"{isolated[:10]}"
        )
    component_nodes_by_root: dict[int, list[int]] = {}
    for node in required:
        component_nodes_by_root.setdefault(find(node), []).append(node)
    components = tuple(
        tuple(sorted(nodes))
        for _root, nodes in sorted(
            component_nodes_by_root.items(), key=lambda item: min(item[1])
        )
    )
    component_edges = tuple(
        tuple(
            edge
            for edge in candidate_edges
            if endpoint_pairs[edge][0] in set(nodes)
            and endpoint_pairs[edge][1] in set(nodes)
        )
        for nodes in components
    )
    if any(len(edges) < len(nodes) - 1 for nodes, edges in zip(components, component_edges)):
        raise ValueError("道路端点分量缺少足以保持内部连通的候选边")
    forced_leaf_count = sum(
        len(incident[node]) == 1 for node in interior_junctions
    )
    return RoadBackbone(
        required_nodes=tuple(required),
        interior_junctions=tuple(interior_junctions),
        boundary_portals=tuple(
            node for node in boundary_portals if node in endpoint_nodes
        ),
        candidate_edges=tuple(candidate_edges),
        incident=incident,
        components=components,
        component_edges=component_edges,
        forced_dead_end_count=forced_leaf_count,
    )


def _traffic_metrics_from_selected(
    roads_r: gpd.GeoDataFrame,
    selected: set[int],
    community_area: float,
    cfg: RoadConfig,
    *,
    dead_end_nodes: tuple[int, ...] | None = None,
) -> Dict[str, float | bool]:
    """按论文原式计算 Traffic；与在线环境共用同一实现。"""
    endpoint_pairs, _ = _endpoints_keyed(roads_r)
    lengths = _road_lengths_m(roads_r)
    return dict(
        road_traffic_metrics(
            endpoint_pairs,
            lengths,
            selected,
            community_area,
            budget=road_budget(len(roads_r), cfg.budget_frac),
            spacing_min_m=cfg.spacing_min_m,
            spacing_max_m=cfg.spacing_max_m,
            dead_end_nodes=dead_end_nodes,
        )
    )


def _max_local_detour_factor(
    endpoint_pairs: list[tuple[int, int]],
    lengths: list[float],
    selected: set[int],
    candidate_edges: tuple[int, ...],
) -> float:
    """Return the largest selected-network detour for an original road edge."""
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for edge in selected:
        left, right = endpoint_pairs[edge]
        length = float(lengths[edge])
        adjacency.setdefault(left, []).append((right, length))
        adjacency.setdefault(right, []).append((left, length))

    def shortest(source: int, target: int) -> float:
        heap: list[tuple[float, int]] = [(0.0, source)]
        distance = {source: 0.0}
        while heap:
            current, node = heapq.heappop(heap)
            if node == target:
                return current
            if current != distance[node]:
                continue
            for neighbor, weight in adjacency.get(node, []):
                proposed = current + weight
                if proposed < distance.get(neighbor, math.inf):
                    distance[neighbor] = proposed
                    heapq.heappush(heap, (proposed, neighbor))
        return math.inf

    ratios = []
    for edge in candidate_edges:
        if edge in selected:
            ratios.append(1.0)
            continue
        direct = max(float(lengths[edge]), 1.0e-9)
        left, right = endpoint_pairs[edge]
        ratios.append(shortest(left, right) / direct)
    return max(ratios, default=1.0)


def solve_road_mip(
    parcels: gpd.GeoDataFrame,
    roads_r: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    cfg: RoadConfig,
    warm_start: Dict[int, float] | None = None,
) -> tuple[Dict[int, float], float, float, float, int]:
    """Optimize exact Traffic subject to a connected, non-tree backbone."""
    model = gp.Model("road_planning_mip")
    model.Params.TimeLimit = cfg.max_runtime_sec
    model.Params.MIPGap = cfg.mip_gap
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.OptimalityTol = 1.0e-9
    model.Params.IntFeasTol = 1.0e-9
    model.Params.NumericFocus = 3
    # The per-edge detour formulation contains many near-parallel flow rows.
    # Keep presolve from incorrectly eliminating a feasible all-road solution
    # under tight geometry coefficients.
    model.Params.Presolve = 0

    R = len(roads_r)
    if R == 0:
        return {}, 0.0, 0.0, 0.0, int(GRB.OPTIMAL)
    budget = road_budget(R, cfg.budget_frac)
    y = model.addVars(range(R), vtype=GRB.BINARY, name="y")
    if warm_start is not None:
        for edge in range(R):
            y[edge].Start = 1.0 if float(warm_start.get(edge, 0.0)) > 0.5 else 0.0

    lengths = _road_lengths_m(roads_r)
    community_area = max(_community_area_m2(parcels), 1e-9)
    is_too_long = [1 if float(lengths[r]) > float(cfg.spacing_max_m) else 0 for r in range(R)]
    is_too_short = [1 if float(lengths[r]) < float(cfg.spacing_min_m) else 0 for r in range(R)]

    endpoint_pairs, n_nodes = _endpoints_keyed(roads_r)
    if n_nodes <= 0:
        raise RuntimeError("road topology has no valid endpoint nodes")
    backbone = _road_backbone_structure(
        intersections, endpoint_pairs, n_nodes
    )
    required = backbone.required_nodes
    candidate_edges = backbone.candidate_edges
    min_selected_by_component = sum(
        min(len(nodes), len(edges))
        for nodes, edges in zip(backbone.components, backbone.component_edges)
    )
    if budget < min_selected_by_component:
        raise ValueError(
            f"预算 {budget} 条边不足以覆盖每个原生道路分量并保留其可用环路所需的至少 "
            f"{min_selected_by_component} 条边；"
            f"请将 budget_frac 提高到至少 {min_selected_by_component / R:.3f}"
        )

    incident: Dict[int, List[int]] = {v: [] for v in range(n_nodes)}
    for r, (u, v) in enumerate(endpoint_pairs):
        if u >= 0:
            incident[u].append(r)
        if v >= 0 and v != u:
            incident[v].append(r)

    candidate_set = set(candidate_edges)
    for edge in range(R):
        if edge not in candidate_set:
            model.addConstr(y[edge] == 0, name=f"non_backbone_edge_{edge}")

    model.addConstr(gp.quicksum(y[r] for r in range(R)) <= budget, name="budget")
    # Each source-connected component is retained as one internally connected
    # road network.  A component that already contains a cycle must keep at
    # least |V| edges; a source tree has no possible cycle, so all |V|-1 of
    # its existing roads are retained rather than silently dropping a portal.
    for component_index, (nodes, edges) in enumerate(
        zip(backbone.components, backbone.component_edges)
    ):
        model.addConstr(
            gp.quicksum(y[edge] for edge in edges) >= min(len(nodes), len(edges)),
            name=f"component_backbone_{component_index}",
        )

    # Every real junction with at least two source roads retains degree >= 2.
    # Portals are external access points and need one selected incident road.
    for node in required:
        source_degree = len(backbone.incident[node])
        minimum_degree = (
            1
            if node in backbone.boundary_portals
            else min(2, source_degree)
        )
        model.addConstr(
            gp.quicksum(y[edge] for edge in backbone.incident[node])
            >= minimum_degree,
            name=f"node_min_degree_{node}",
        )

    # Count only interior junction dead ends. Boundary portals represent roads
    # continuing outside the clipped tile and are deliberately excluded.
    interior_nodes = backbone.interior_junctions
    dead_end = model.addVars(interior_nodes, vtype=GRB.BINARY, name="dead_end")
    zero_degree = model.addVars(interior_nodes, vtype=GRB.BINARY, name="zero_degree")
    multi_degree = model.addVars(interior_nodes, vtype=GRB.BINARY, name="multi_degree")
    for node in interior_nodes:
        degree = gp.quicksum(y[edge] for edge in backbone.incident[node])
        source_degree = len(backbone.incident[node])
        model.addConstr(
            zero_degree[node] + dead_end[node] + multi_degree[node] == 1,
            name=f"degree_class_{node}",
        )
        model.addConstr(
            degree >= dead_end[node] + 2 * multi_degree[node],
            name=f"degree_class_lb_{node}",
        )
        model.addConstr(
            degree <= dead_end[node] + source_degree * multi_degree[node],
            name=f"degree_class_ub_{node}",
        )
    # One commodity flow per *source* component proves that every original
    # road component remains internally connected.  Some OSM tiles contain an
    # isolated through-road clipped by the tile boundary; no existing-edge
    # upgrade decision can join it to the main component, so treating it as a
    # fake zero-length connection would corrupt both topology and reward.
    flow_cap = float(len(required) - 1)
    flow_fwd = model.addVars(range(R), lb=0.0, ub=flow_cap, name="flow_fwd")
    flow_bwd = model.addVars(range(R), lb=0.0, ub=flow_cap, name="flow_bwd")
    inflow: Dict[int, list] = {node: [] for node in range(n_nodes)}
    outflow: Dict[int, list] = {node: [] for node in range(n_nodes)}
    for edge, (left, right) in enumerate(endpoint_pairs):
        if edge not in candidate_set or left < 0 or right < 0 or left == right:
            model.addConstr(flow_fwd[edge] == 0, name=f"no_flow_fwd_{edge}")
            model.addConstr(flow_bwd[edge] == 0, name=f"no_flow_bwd_{edge}")
            continue
        model.addConstr(flow_fwd[edge] <= flow_cap * y[edge], name=f"flow_cap_fwd_{edge}")
        model.addConstr(flow_bwd[edge] <= flow_cap * y[edge], name=f"flow_cap_bwd_{edge}")
        outflow[left].append(flow_fwd[edge])
        inflow[right].append(flow_fwd[edge])
        outflow[right].append(flow_bwd[edge])
        inflow[left].append(flow_bwd[edge])
    component_root: dict[int, int] = {}
    component_size: dict[int, int] = {}
    for nodes in backbone.components:
        root = nodes[0]
        for node in nodes:
            component_root[node] = root
            component_size[node] = len(nodes)
    for node in range(n_nodes):
        net_inflow = gp.quicksum(inflow[node]) - gp.quicksum(outflow[node])
        if node not in component_root:
            demand = 0
        elif node == component_root[node]:
            demand = -(component_size[node] - 1)
        else:
            demand = 1
        model.addConstr(net_inflow == demand, name=f"backbone_flow_{node}")

    # Route one local-access commodity for each original road through the
    # selected graph.  Instead of forcing every edge to a tiny stretch (which
    # degenerates into selecting all existing roads), lexicographically
    # minimize maximum and then mean detour under the fixed construction budget.
    detour_fwd = model.addVars(candidate_edges, candidate_edges, lb=0.0, ub=1.0, name="detour_fwd")
    detour_bwd = model.addVars(candidate_edges, candidate_edges, lb=0.0, ub=1.0, name="detour_bwd")
    max_detour_ratio = model.addVar(lb=1.0, name="max_detour_ratio")
    total_detour_ratio = gp.LinExpr()
    for source_edge in candidate_edges:
        source, target = endpoint_pairs[source_edge]
        for path_edge in candidate_edges:
            model.addConstr(
                detour_fwd[source_edge, path_edge] <= y[path_edge],
                name=f"detour_cap_fwd_{source_edge}_{path_edge}",
            )
            model.addConstr(
                detour_bwd[source_edge, path_edge] <= y[path_edge],
                name=f"detour_cap_bwd_{source_edge}_{path_edge}",
            )
        for node in required:
            inflow_detour = gp.quicksum(
                detour_fwd[source_edge, edge]
                for edge in backbone.incident[node]
                if endpoint_pairs[edge][1] == node
            ) + gp.quicksum(
                detour_bwd[source_edge, edge]
                for edge in backbone.incident[node]
                if endpoint_pairs[edge][0] == node
            )
            outflow_detour = gp.quicksum(
                detour_fwd[source_edge, edge]
                for edge in backbone.incident[node]
                if endpoint_pairs[edge][0] == node
            ) + gp.quicksum(
                detour_bwd[source_edge, edge]
                for edge in backbone.incident[node]
                if endpoint_pairs[edge][1] == node
            )
            balance = -1.0 if node == source else 1.0 if node == target else 0.0
            model.addConstr(
                inflow_detour - outflow_detour == balance,
                name=f"detour_flow_{source_edge}_{node}",
            )
        path_length = gp.quicksum(
            float(lengths[path_edge])
            * (detour_fwd[source_edge, path_edge] + detour_bwd[source_edge, path_edge])
            for path_edge in candidate_edges
        )
        model.addConstr(
            path_length <= max_detour_ratio * float(lengths[source_edge]),
            name=f"detour_max_ratio_{source_edge}",
        )
        total_detour_ratio += path_length / max(float(lengths[source_edge]), 1.0e-9)

    dead_choice = model.addVars(
        range(len(interior_nodes) + 1), vtype=GRB.BINARY, name="dead_count"
    )
    model.addConstr(
        gp.quicksum(dead_choice[k] for k in range(len(interior_nodes) + 1)) == 1
    )
    model.addConstr(
        gp.quicksum(k * dead_choice[k] for k in range(len(interior_nodes) + 1))
        == gp.quicksum(dead_end[node] for node in interior_nodes)
    )
    connectivity_dead_end = gp.quicksum(
        (1.0 / (k + 1.0)) * dead_choice[k]
        for k in range(len(interior_nodes) + 1)
    )
    # Source components cannot be joined by an upgrade-only action.  Their
    # count is therefore a topology constant and is honestly retained in the
    # paper's NCC term rather than pretending those components are connected.
    connectivity = (1.0 / len(backbone.components)) + connectivity_dead_end

    max_long = int(sum(is_too_long))
    long_choice = model.addVars(range(max_long + 1), vtype=GRB.BINARY, name="long_count")
    model.addConstr(gp.quicksum(long_choice[k] for k in range(max_long + 1)) == 1)
    model.addConstr(
        gp.quicksum(k * long_choice[k] for k in range(max_long + 1))
        == gp.quicksum(is_too_long[r] * y[r] for r in range(R))
    )

    max_short = int(sum(is_too_short))
    short_choice = model.addVars(range(max_short + 1), vtype=GRB.BINARY, name="short_count")
    model.addConstr(gp.quicksum(short_choice[k] for k in range(max_short + 1)) == 1)
    model.addConstr(
        gp.quicksum(k * short_choice[k] for k in range(max_short + 1))
        == gp.quicksum(is_too_short[r] * y[r] for r in range(R))
    )
    spacing = (
        gp.quicksum((1.0 / (k + 1.0)) * long_choice[k] for k in range(max_long + 1))
        + gp.quicksum(
            (1.0 / (k + 1.0)) * short_choice[k] for k in range(max_short + 1)
        )
    )
    density = gp.quicksum(
        (1000.0 * float(lengths[edge]) / community_area) * y[edge]
        for edge in range(R)
    )
    traffic = (density + connectivity + spacing) / 3.0
    # Expert supervision and PPO optimize the same terminal reward. Tiny
    # coefficients provide deterministic detour tie-breaking without allowing
    # a lexicographic secondary solve to degrade Traffic by the primary MIP gap.
    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(
        traffic - 1.0e-8 * max_detour_ratio - 1.0e-10 * total_detour_ratio
    )
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        conflicts = [
            constraint.ConstrName
            for constraint in model.getConstrs()
            if constraint.IISConstr
        ]
        raise RuntimeError(
            "Gurobi road model is infeasible; IIS="
            f"{conflicts[:20]}"
        )
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or model.SolCount < 1:
        raise RuntimeError(
            f"Gurobi failed, status={model.Status} "
            f"(R={R}, nodes={n_nodes}, budget={budget})"
        )

    y_val = {r: float(y[r].X) for r in range(R)}
    realized_gap = float(model.MIPGap)
    return (
        y_val,
        float(traffic.getValue()),
        realized_gap,
        float(model.Runtime),
        int(model.Status),
    )


def build_road_trajectory(
    roads_r: gpd.GeoDataFrame,
    y_val: Dict[int, float],
    final_reward: float,
    metrics: Dict[str, float | bool],
    community_area: float,
    cfg: RoadConfig,
) -> dict:
    """Record every binary road decision and emit reward only at termination."""
    R = len(roads_r)
    steps: List[dict] = []
    selected_count = 0
    for r in range(R):
        upgraded = bool(y_val.get(r, 0.0) > 0.5)
        selected_count += int(upgraded)
        steps.append(
            {
                "t": r + 1,
                "state": {
                    "decided_roads_count": r + 1,
                    "upgraded_roads_count": selected_count,
                },
                "action": {"road_row_index": r, "upgraded": upgraded},
                "reward": 0.0,
            }
        )

    if steps:
        steps[-1]["reward"] = float(final_reward)

    rtg = 0.0
    for step in reversed(steps):
        rtg += float(step["reward"])
        step["return_to_go"] = rtg

    return {
        "task_type": "road_planning",
        "objective": "Traffic = (Tdensity + Tconnectivity + Tspacing) / 3",
        "constraints": (
            "span all real junctions and boundary portals; every source road component is "
            "internally connected (and cyclic when its source graph permits it), preserves "
            "interior degree >= 2 where possible, and minimizes maximum then mean local "
            "detour under the road budget"
        ),
        "reward_timing": "terminal_only",
        "metrics": metrics,
        "num_steps": len(steps),
        "final_reward": float(final_reward),
        "steps": steps,
    }


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/topology/synth/0000/topology.gpkg"),
        help="拓扑 GeoPackage 或目录",
    )
    parser.add_argument("--solution-csv", type=Path, default=None, help="路段升级 CSV")
    parser.add_argument("--trajectory-json", type=Path, default=None, help="轨迹 JSON")
    parser.add_argument(
        "--warm-start-csv",
        type=Path,
        default=None,
        help="可选的已有 road_plan_solution.csv，用作 Gurobi MIP start",
    )
    parser.add_argument(
        "--budget-frac",
        type=float,
        default=DEFAULT_ROAD_BUDGET_FRAC,
        help="道路预算上限 = ceil(frac * R)；默认 0.85，保留非平凡选边空间",
    )
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=1.0e-4)


def run(args: argparse.Namespace) -> None:
    cfg = RoadConfig(
        budget_frac=float(args.budget_frac),
        max_runtime_sec=int(args.time_limit),
        mip_gap=float(args.mip_gap),
    )

    topo_path = resolve_topology_path(args.input)
    parcels, roads, intersections = load_topology(topo_path)
    parcels = _ensure_metric_crs(parcels)
    roads = _ensure_metric_crs(roads)
    out_dir = topo_path.parent
    if args.solution_csv is None:
        args.solution_csv = out_dir / "road_plan_solution.csv"
    if args.trajectory_json is None:
        args.trajectory_json = out_dir / "road_plan_trajectory.json"

    roads_r, adj = build_parcel_road_adjacency(parcels, roads)
    R = len(roads_r)
    warm_start = None
    warm_start_path = getattr(args, "warm_start_csv", None)
    if warm_start_path is not None and Path(warm_start_path).is_file():
        existing = pd.read_csv(warm_start_path)
        if {"road_row_index", "upgraded"} <= set(existing.columns):
            warm_start = {
                int(row.road_row_index): float(row.upgraded)
                for row in existing.itertuples(index=False)
            }

    if R == 0:
        y_val, obj_proxy, gap, runtime_sec, solver_status = (
            {},
            0.0,
            0.0,
            0.0,
            int(GRB.OPTIMAL),
        )
    else:
        y_val, obj_proxy, gap, runtime_sec, solver_status = solve_road_mip(
            parcels, roads_r, intersections, cfg, warm_start=warm_start
        )

    selected = {r for r, v in y_val.items() if v > 0.5}
    community_area = _community_area_m2(parcels)
    endpoint_pairs, n_nodes = _endpoints_keyed(roads_r)
    backbone = _road_backbone_structure(
        intersections, endpoint_pairs, n_nodes
    )
    metrics = _traffic_metrics_from_selected(
        roads_r,
        selected,
        community_area,
        cfg,
        dead_end_nodes=backbone.interior_junctions,
    )
    covered_nodes = sum(
        any(edge in selected for edge in backbone.incident[node])
        for node in backbone.required_nodes
    )
    covered_junctions = sum(
        any(edge in selected for edge in backbone.incident[node])
        for node in backbone.interior_junctions
    )
    covered_portals = sum(
        any(edge in selected for edge in backbone.incident[node])
        for node in backbone.boundary_portals
    )
    source_component_count = len(backbone.components)
    backbone_connected = (
        covered_nodes == len(backbone.required_nodes)
        and int(metrics["NCC"]) == source_component_count
        and selected <= set(backbone.candidate_edges)
    )
    max_detour = _max_local_detour_factor(
        endpoint_pairs, _road_lengths_m(roads_r), selected, backbone.candidate_edges
    )
    if not bool(metrics["feasible"]):
        raise RuntimeError("Gurobi returned an empty or over-budget road plan")
    if not backbone_connected:
        raise RuntimeError(
            "Gurobi returned a road plan that does not span every endpoint "
            "or internally connect every source road component"
        )
    if int(metrics["dead_ends"]) != backbone.forced_dead_end_count:
        raise RuntimeError(
            "Gurobi did not attain the topology-implied minimum dead-end count: "
            f"actual={metrics['dead_ends']}, forced={backbone.forced_dead_end_count}"
        )
    if not math.isclose(
        float(metrics["Traffic"]), float(obj_proxy), rel_tol=1.0e-8, abs_tol=1.0e-8
    ):
        raise RuntimeError(
            "road reward/model objective mismatch: "
            f"reward={metrics['Traffic']:.12f}, objective={obj_proxy:.12f}"
        )
    metrics["solver_objective"] = float(obj_proxy)
    metrics["mip_gap"] = float(gap)
    metrics["solver_runtime_sec"] = float(runtime_sec)
    metrics["solver_status"] = float(solver_status)
    metrics["budget_frac"] = float(cfg.budget_frac)
    metrics["density_unit_km_per_km2"] = True
    metrics["required_road_nodes"] = float(len(backbone.required_nodes))
    metrics["covered_road_nodes"] = float(covered_nodes)
    metrics["road_node_coverage"] = float(covered_nodes / len(backbone.required_nodes))
    metrics["required_junctions"] = float(len(backbone.interior_junctions))
    metrics["covered_junctions"] = float(covered_junctions)
    metrics["junction_coverage"] = float(
        covered_junctions / max(len(backbone.interior_junctions), 1)
    )
    metrics["required_boundary_portals"] = float(len(backbone.boundary_portals))
    metrics["covered_boundary_portals"] = float(covered_portals)
    metrics["boundary_portal_coverage"] = float(
        covered_portals / max(len(backbone.boundary_portals), 1)
    )
    metrics["candidate_backbone_roads"] = float(len(backbone.candidate_edges))
    metrics["forced_dead_ends"] = float(backbone.forced_dead_end_count)
    metrics["source_road_components"] = float(source_component_count)
    metrics["connected_within_source_components"] = bool(backbone_connected)
    metrics["junctions_connected"] = bool(
        backbone_connected and source_component_count == 1
    )
    metrics["boundary_portals_connected"] = bool(
        covered_portals == len(backbone.boundary_portals)
    )
    metrics["observed_max_detour_factor"] = float(max_detour)
    cyclic_components = [
        (nodes, edges)
        for nodes, edges in zip(backbone.components, backbone.component_edges)
        if len(edges) >= len(nodes)
    ]
    metrics["non_tree_backbone"] = bool(
        cyclic_components
        and all(
            sum(edge in selected for edge in edges) >= len(nodes)
            for nodes, edges in cyclic_components
        )
    )

    # Do not replace a previously valid expert with partial output: both files
    # are written only after the objective/evaluator equality assertion passes.
    out_dir = args.solution_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    source_ids = (
        roads_r["source_id"].astype(str).tolist() if "source_id" in roads_r.columns else None
    )
    rows = []
    for r in range(R):
        row = {"road_row_index": r, "upgraded": int(y_val.get(r, 0.0) > 0.5)}
        if source_ids is not None:
            row["source_id"] = source_ids[r]
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.solution_csv, index=False)

    traj = build_road_trajectory(
        roads_r=roads_r,
        y_val=y_val,
        final_reward=float(metrics["Traffic"]),
        metrics=metrics,
        community_area=community_area,
        cfg=cfg,
    )
    args.trajectory_json.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[Done] roads={R}, solver_objective={obj_proxy:.6f}, "
        f"Traffic={metrics['Traffic']:.6f}, mip_gap={gap:.6f}"
    )
    print(f"[Done] solution -> {args.solution_csv}")
    print(f"[Done] trajectory -> {args.trajectory_json}")


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "road-plan",
        help="道路规划 MIP：按论文原式优化密度、连通性与道路间距",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(p)
    p.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(description="道路规划 MIP")
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
