"""Compile the frozen traffic contract and generate deterministic SUMO demand.

The topology cleaner already resolves directed edges, per-lane connections and
canonical traffic-light programs in ``traffic_signals.json``.  This module only
serializes that contract; it never asks netconvert to infer turns or phases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from upmodel.sumo_tls import compiled_tls_link_orders

DEFAULT_VEHICLE_COUNT = 2000
DEFAULT_DEPART_END_S = 300.0
BOUNDARY_OD_FRACTION = 0.5
MIN_PREFERRED_ROUTE_EDGES = 3
DEPART_SPEED_FACTOR_RANGE = (0.05, 0.25)
PASSENGER_LENGTH_M = 5.0
PASSENGER_MIN_GAP_M = 2.5
PASSENGER_DECEL_MPS2 = 4.5
SAFE_STOP_SPEED_MARGIN = 0.9


@dataclass(frozen=True)
class PlannedRoute:
    edges: tuple[str, ...]
    od_class: str
    coverage_target_road_id: str | None = None


def _sumo_home() -> Path:
    import sumo  # noqa: WPS433

    if getattr(sumo, "SUMO_HOME", None):
        return Path(sumo.SUMO_HOME)
    if "SUMO_HOME" in os.environ:
        return Path(os.environ["SUMO_HOME"])
    raise RuntimeError("找不到 SUMO_HOME，请确认已安装 eclipse-sumo。")


def _traffic_contract(tile_dir: Path) -> dict[str, Any]:
    path = tile_dir / "traffic_signals.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"缺少 traffic_signals.json: {path}") from None
    network = value.get("sumo_network")
    controllers = value.get("controllers")
    if not isinstance(network, dict) or not isinstance(controllers, list):
        raise TypeError(f"{path}: 缺少清洗后的 sumo_network/controllers 契约")
    required = ("nodes", "directed_edges", "connections")
    missing = [key for key in required if not isinstance(network.get(key), list)]
    if missing:
        raise ValueError(f"{path}: sumo_network 缺少字段 {missing}")
    if not network["nodes"] or not network["directed_edges"] or not controllers:
        raise ValueError(f"{path}: SUMO 节点、道路或信号控制器为空")
    return value


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _plain_xml(
    contract: dict[str, Any],
    sumo_dir: Path,
    *,
    speed: float | None,
    lanes: int | None,
) -> tuple[Path, Path, Path, Path]:
    network = contract["sumo_network"]

    nodes_root = ET.Element("nodes")
    for node in network["nodes"]:
        attrs = {
            "id": str(node["id"]),
            "x": str(float(node["x_m"])),
            "y": str(float(node["y_m"])),
            "type": str(node.get("type", "priority")),
        }
        if node.get("tl_id"):
            attrs["tl"] = str(node["tl_id"])
        if node.get("fringe") in {"outer", "inner"}:
            attrs["fringe"] = str(node["fringe"])
        ET.SubElement(nodes_root, "node", attrs)

    edges_root = ET.Element("edges")
    for edge in network["directed_edges"]:
        shape = " ".join(
            f"{float(point[0]):.3f},{float(point[1]):.3f}"
            for point in edge.get("shape", [])
        )
        attrs = {
            "id": str(edge["id"]),
            "from": str(edge["from_node_id"]),
            "to": str(edge["to_node_id"]),
            "numLanes": str(
                max(1, int(lanes if lanes is not None else edge["num_lanes"]))
            ),
            "speed": f"{float(speed if speed is not None else edge['speed_limit_mps']):.4f}",
            "priority": str(int(edge.get("priority", 1))),
        }
        if shape:
            attrs["shape"] = shape
        allowed = edge.get("allow_vehicle_classes", [])
        if allowed:
            attrs["allow"] = " ".join(str(value) for value in allowed)
        ET.SubElement(edges_root, "edge", attrs)

    connections_root = ET.Element("connections")
    for connection in network["connections"]:
        attrs = {
            "from": str(connection["from_edge_id"]),
            "to": str(connection["to_edge_id"]),
            "fromLane": str(int(connection["from_lane_index"])),
            "toLane": str(int(connection["to_lane_index"])),
        }
        if connection.get("sumo_tl_id") is not None:
            attrs["tl"] = str(connection["sumo_tl_id"])
            attrs["linkIndex"] = str(int(connection["link_index"]))
        ET.SubElement(connections_root, "connection", attrs)

    tls_root = ET.Element("tlLogics")
    for controller in contract["controllers"]:
        bootstrap = controller["program"]["bootstrap_tl_logic"]
        link_count = len(controller["controlled_links"])
        logic = ET.SubElement(
            tls_root,
            "tlLogic",
            {
                "id": str(controller["sumo_tl_id"]),
                "type": str(bootstrap.get("type", "static")),
                "programID": str(bootstrap.get("program_id", "rl_bootstrap")),
                "offset": str(float(bootstrap.get("offset_s", 0))),
            },
        )
        ET.SubElement(
            logic,
            "phase",
            {
                "duration": "1.0",
                "state": "r" * link_count,
                "name": "bootstrap_all_red",
            },
        )
        for link_index in range(link_count):
            green_state = ["r"] * link_count
            green_state[link_index] = "G"
            ET.SubElement(
                logic,
                "phase",
                {
                    "duration": "15.0",
                    "state": "".join(green_state),
                    "name": f"bootstrap_single_green_{link_index}",
                },
            )
            yellow_state = ["r"] * link_count
            yellow_state[link_index] = "y"
            ET.SubElement(
                logic,
                "phase",
                {
                    "duration": "3.0",
                    "state": "".join(yellow_state),
                    "name": f"bootstrap_single_yellow_{link_index}",
                },
            )

    paths = (
        sumo_dir / "network.nod.xml",
        sumo_dir / "network.edg.xml",
        sumo_dir / "network.con.xml",
        sumo_dir / "network.tll.xml",
    )
    for root, path in zip(
        (nodes_root, edges_root, connections_root, tls_root), paths, strict=True
    ):
        _write_xml(root, path)
    return paths


def _validate_compiled_net(contract: dict[str, Any], net_path: Path) -> None:
    root = ET.parse(net_path).getroot()
    actual_edges = {
        str(item.attrib["id"])
        for item in root.findall("edge")
        if not str(item.attrib["id"]).startswith(":")
    }
    expected_edges = {
        str(item["id"]) for item in contract["sumo_network"]["directed_edges"]
    }
    if actual_edges != expected_edges:
        raise RuntimeError(
            f"netconvert 道路集合不一致: missing={sorted(expected_edges - actual_edges)[:5]}, "
            f"extra={sorted(actual_edges - expected_edges)[:5]}"
        )
    actual_tls = {str(item.attrib["id"]) for item in root.findall("tlLogic")}
    expected_tls = {str(item["sumo_tl_id"]) for item in contract["controllers"]}
    if actual_tls != expected_tls:
        raise RuntimeError(
            f"netconvert 信号灯集合不一致: missing={sorted(expected_tls - actual_tls)[:5]}, "
            f"extra={sorted(actual_tls - expected_tls)[:5]}"
        )
    orders = compiled_tls_link_orders(contract, net_path)
    for logic in root.findall("tlLogic"):
        tls_id = str(logic.attrib["id"])
        if tls_id not in orders:
            continue
        states = [str(phase.attrib.get("state", "")) for phase in logic.findall("phase")]
        link_count = len(orders[tls_id].links)
        if (
            not states
            or states[0] != "r" * link_count
            or any(
                len(state) != link_count or set(state) - {"r", "G", "y"}
                for state in states
            )
            or {
                index
                for state in states
                for index, character in enumerate(state)
                if character == "G"
            }
            != set(range(link_count))
        ):
            raise RuntimeError(
                f"{tls_id}: compiled bootstrap 必须先全红且逐 link 提供单独绿/黄灯，"
                f"实际={states[:3]}；请用 --force 重新编译"
            )


def build_sumo_net(
    tile_dir: Path,
    *,
    speed: float | None = None,
    lanes: int | None = None,
    force: bool = False,
) -> Path:
    """Serialize the cleaned lane/connection/controller contract into SUMO."""
    tile_dir = Path(tile_dir)
    contract = _traffic_contract(tile_dir)
    sumo_dir = tile_dir / "sumo"
    sumo_dir.mkdir(parents=True, exist_ok=True)
    net_path = sumo_dir / "net.xml"
    if net_path.is_file() and not force:
        _validate_compiled_net(contract, net_path)
        return net_path

    node_path, edge_path, connection_path, tls_path = _plain_xml(
        contract, sumo_dir, speed=speed, lanes=lanes
    )
    home = _sumo_home()
    os.environ["SUMO_HOME"] = str(home)
    import sumolib  # noqa: WPS433

    command = [
        sumolib.checkBinary("netconvert"),
        "--node-files",
        str(node_path),
        "--edge-files",
        str(edge_path),
        "--connection-files",
        str(connection_path),
        "--tllogic-files",
        str(tls_path),
        "--output-file",
        str(net_path),
        "--no-turnarounds",
        "true",
        "--junctions.internal-link-detail",
        "5",
        "--tls.discard-simple",
        "false",
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    _validate_compiled_net(contract, net_path)
    print(
        f"[sumo-demand] network nodes={len(contract['sumo_network']['nodes'])} "
        f"directed_edges={len(contract['sumo_network']['directed_edges'])} "
        f"connections={len(contract['sumo_network']['connections'])} "
        f"controllers={len(contract['controllers'])}"
    )
    return net_path


def _edge_ids(path: Any) -> tuple[str, ...] | None:
    if path is None:
        return None
    return tuple(str(edge.getID()) for edge in path)


def _shortest_edge_route(net: Any, start: Any, end: Any) -> tuple[str, ...] | None:
    path, _ = net.getShortestPath(start, end, vClass="passenger")
    route = _edge_ids(path)
    if route is None or len(route) != len(set(route)):
        return None
    return route


def _route_is_valid(net: Any, route: tuple[str, ...]) -> bool:
    """Match SUMO's edge/connection passenger-routing semantics."""
    if not route or len(route) != len(set(route)):
        return False
    try:
        edges = [net.getEdge(edge_id) for edge_id in route]
    except KeyError:
        return False
    if any(not edge.allows("passenger") for edge in edges):
        return False
    for current, following in pairwise(edges):
        connections = current.getOutgoing().get(following, ())
        if not any(
            connection.getFromLane().allows("passenger")
            and connection.getToLane().allows("passenger")
            for connection in connections
        ):
            return False
    return True


def _eligible_edges(contract: dict[str, Any], net: Any) -> list[Any]:
    edges = []
    for row in contract["sumo_network"]["directed_edges"]:
        edge = net.getEdge(str(row["id"]))
        if edge.allows("passenger"):
            edges.append(edge)
    if not edges:
        raise RuntimeError("SUMO 路网没有允许 passenger 的 directed edge")
    return sorted(edges, key=lambda edge: edge.getID())


def _is_boundary_pair(
    route: tuple[str, ...], boundary_entries: set[str], boundary_exits: set[str]
) -> bool:
    return route[0] in boundary_entries and route[-1] in boundary_exits


def _passenger_successors(edge: Any) -> list[Any]:
    return sorted(
        (
            following
            for following, connections in edge.getOutgoing().items()
            if following.allows("passenger")
            and any(
                connection.getFromLane().allows("passenger")
                and connection.getToLane().allows("passenger")
                for connection in connections
            )
        ),
        key=lambda following: following.getID(),
    )


def _passenger_predecessors(edge: Any) -> list[Any]:
    return sorted(
        (
            preceding
            for preceding in edge.getIncoming()
            if edge in _passenger_successors(preceding)
        ),
        key=lambda preceding: preceding.getID(),
    )


def _three_edge_routes_through_target(
    net: Any,
    target: Any,
    boundary_entries: set[str],
    boundary_exits: set[str],
) -> list[tuple[str, ...]]:
    """Exhaust all simple three-edge passenger paths containing ``target``."""
    candidates: set[tuple[str, ...]] = set()
    predecessors = _passenger_predecessors(target)
    successors = _passenger_successors(target)
    for second in successors:
        for third in _passenger_successors(second):
            candidates.add(
                (str(target.getID()), str(second.getID()), str(third.getID()))
            )
    for first in predecessors:
        for third in successors:
            candidates.add(
                (str(first.getID()), str(target.getID()), str(third.getID()))
            )
    for second in predecessors:
        for first in _passenger_predecessors(second):
            candidates.add(
                (str(first.getID()), str(second.getID()), str(target.getID()))
            )
    return [
        route
        for route in sorted(candidates)
        if _route_is_valid(net, route)
        and not _is_boundary_pair(route, boundary_entries, boundary_exits)
    ]


def _boundary_route_candidates(
    net: Any,
    boundary_entries: list[Any],
    boundary_exits: list[Any],
) -> list[tuple[str, ...]]:
    routes: list[tuple[str, ...]] = []
    for origin in boundary_entries:
        for destination in boundary_exits:
            route = _shortest_edge_route(net, origin, destination)
            if route is not None and _route_is_valid(net, route):
                routes.append(route)
    return routes


def _targeted_arbitrary_route(
    net: Any,
    targets: list[Any],
    eligible_edges: list[Any],
    boundary_entries: set[str],
    boundary_exits: set[str],
    rng: np.random.Generator,
) -> tuple[str, ...] | None:
    """Find a legal arbitrary OD route that contains one of ``targets``."""
    fallback: tuple[str, ...] | None = None
    target_order = rng.permutation(len(targets))
    edge_order = rng.permutation(len(eligible_edges))
    for target_index in target_order:
        target = targets[int(target_index)]
        # Starting or ending at the target avoids an artificial detour while
        # still preferring a substantive route through several junctions.
        for edge_index in edge_order:
            other = eligible_edges[int(edge_index)]
            for start, end in ((target, other), (other, target)):
                route = _shortest_edge_route(net, start, end)
                if route is None or not _route_is_valid(net, route):
                    continue
                if _is_boundary_pair(route, boundary_entries, boundary_exits):
                    continue
                if len(route) >= MIN_PREFERRED_ROUTE_EDGES:
                    return route
                if fallback is None or len(route) > len(fallback):
                    fallback = route

        # This complete lane-connection graph check covers target at the
        # beginning, middle, or end of a simple three-edge route.
        three_edge_routes = _three_edge_routes_through_target(
            net, target, boundary_entries, boundary_exits
        )
        if three_edge_routes:
            return three_edge_routes[int(rng.integers(len(three_edge_routes)))]
    return fallback


def _random_arbitrary_route(
    net: Any,
    eligible_edges: list[Any],
    boundary_entries: set[str],
    boundary_exits: set[str],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    fallback: tuple[str, ...] | None = None
    attempts = min(512, max(64, len(eligible_edges) * 2))
    for _ in range(attempts):
        start = eligible_edges[int(rng.integers(len(eligible_edges)))]
        end = eligible_edges[int(rng.integers(len(eligible_edges)))]
        route = _shortest_edge_route(net, start, end)
        if route is None or not _route_is_valid(net, route):
            continue
        if _is_boundary_pair(route, boundary_entries, boundary_exits):
            continue
        if len(route) >= MIN_PREFERRED_ROUTE_EDGES:
            return route
        if fallback is None or len(route) > len(fallback):
            fallback = route
    # Exhaustive fallback makes the >=3-edge preference deterministic even on
    # a network with many disconnected components.
    for start in eligible_edges:
        for end in eligible_edges:
            route = _shortest_edge_route(net, start, end)
            if route is None or not _route_is_valid(net, route):
                continue
            if _is_boundary_pair(route, boundary_entries, boundary_exits):
                continue
            if len(route) >= MIN_PREFERRED_ROUTE_EDGES:
                return route
            if fallback is None or len(route) > len(fallback):
                fallback = route
    if fallback is not None:
        return fallback
    raise RuntimeError("无法从全网 directed edges 找到合法的任意 OD 路线")


def _network_supports_long_route(net: Any, eligible_edges: list[Any]) -> bool:
    for first in eligible_edges:
        for second in first.getOutgoing():
            if not second.allows("passenger") or second is first:
                continue
            for third, connections in second.getOutgoing().items():
                if third in {first, second} or not third.allows("passenger"):
                    continue
                if any(
                    connection.getFromLane().allows("passenger")
                    and connection.getToLane().allows("passenger")
                    for connection in connections
                ):
                    return True
    return False


def _plan_routes(
    contract: dict[str, Any],
    net: Any,
    rng: np.random.Generator,
    vehicle_count: int,
) -> list[PlannedRoute]:
    network = contract["sumo_network"]
    eligible_edges = _eligible_edges(contract, net)
    eligible_by_id = {str(edge.getID()): edge for edge in eligible_edges}
    entry_ids = {str(value) for value in network.get("boundary_entry_edge_ids", [])}
    exit_ids = {str(value) for value in network.get("boundary_exit_edge_ids", [])}
    boundary_entries = [
        eligible_by_id[edge_id] for edge_id in sorted(entry_ids & eligible_by_id.keys())
    ]
    boundary_exits = [
        eligible_by_id[edge_id] for edge_id in sorted(exit_ids & eligible_by_id.keys())
    ]

    boundary_count = int(vehicle_count * BOUNDARY_OD_FRACTION)
    arbitrary_count = vehicle_count - boundary_count
    candidates = _boundary_route_candidates(net, boundary_entries, boundary_exits)
    if boundary_count and not candidates:
        raise RuntimeError(
            "无法生成 boundary OD：所有 boundary entry→exit 候选均不可达"
        )
    edge_to_road = {
        str(row["id"]): str(row["topology_road_id"])
        for row in network["directed_edges"]
    }
    required_roads = set(edge_to_road.values())
    preferred = [
        route for route in candidates if len(route) >= MIN_PREFERRED_ROUTE_EDGES
    ]
    boundary_pool = preferred or candidates
    plans: list[PlannedRoute] = []
    boundary_uncovered = set(required_roads)
    while len(plans) < boundary_count and boundary_uncovered:
        coverage_by_route = [
            {edge_to_road[edge_id] for edge_id in route if edge_id in edge_to_road}
            & boundary_uncovered
            for route in boundary_pool
        ]
        best_count = max((len(roads) for roads in coverage_by_route), default=0)
        if not best_count:
            break
        best_indices = [
            index
            for index, roads in enumerate(coverage_by_route)
            if len(roads) == best_count
        ]
        chosen = best_indices[int(rng.integers(len(best_indices)))]
        plans.append(PlannedRoute(boundary_pool[chosen], "boundary"))
        boundary_uncovered -= coverage_by_route[chosen]
    while len(plans) < boundary_count:
        plans.append(
            PlannedRoute(
                boundary_pool[int(rng.integers(len(boundary_pool)))], "boundary"
            )
        )

    targets_by_road: dict[str, list[Any]] = {}
    for edge_id, road_id in edge_to_road.items():
        edge = eligible_by_id.get(edge_id)
        if edge is not None:
            targets_by_road.setdefault(road_id, []).append(edge)
    roads_without_passenger_edge = required_roads - targets_by_road.keys()
    if roads_without_passenger_edge:
        raise RuntimeError(
            "存在无法由 passenger OD 覆盖的 topology road: "
            f"{sorted(roads_without_passenger_edge)[:10]}"
        )

    covered_roads = {
        edge_to_road[edge_id]
        for plan in plans
        for edge_id in plan.edges
        if edge_id in edge_to_road
    }
    uncovered = required_roads - covered_roads
    while uncovered:
        if sum(plan.od_class == "arbitrary" for plan in plans) >= arbitrary_count:
            raise RuntimeError(
                f"{vehicle_count} 辆车无法在精确 boundary 配额下覆盖全部道路；"
                f"仍缺 {len(uncovered)} 条: {sorted(uncovered)[:10]}"
            )
        road_id = sorted(uncovered)[int(rng.integers(len(uncovered)))]
        route = _targeted_arbitrary_route(
            net,
            targets_by_road[road_id],
            eligible_edges,
            entry_ids,
            exit_ids,
            rng,
        )
        if route is None:
            raise RuntimeError(f"topology road {road_id} 无法由任意合法 OD 路线覆盖")
        plans.append(PlannedRoute(route, "arbitrary", coverage_target_road_id=road_id))
        covered_roads.update(
            edge_to_road[edge_id] for edge_id in route if edge_id in edge_to_road
        )
        uncovered = required_roads - covered_roads

    used_arbitrary = sum(plan.od_class == "arbitrary" for plan in plans)
    for _ in range(arbitrary_count - used_arbitrary):
        plans.append(
            PlannedRoute(
                _random_arbitrary_route(net, eligible_edges, entry_ids, exit_ids, rng),
                "arbitrary",
            )
        )
    plans = [plans[int(index)] for index in rng.permutation(len(plans))]
    return plans


def _validate_plans(
    contract: dict[str, Any],
    net: Any,
    plans: list[PlannedRoute],
) -> dict[str, Any]:
    network = contract["sumo_network"]
    entry_ids = {str(value) for value in network.get("boundary_entry_edge_ids", [])}
    exit_ids = {str(value) for value in network.get("boundary_exit_edge_ids", [])}
    invalid = [
        index
        for index, plan in enumerate(plans)
        if not _route_is_valid(net, plan.edges)
    ]
    if invalid:
        raise RuntimeError(f"生成了 SUMO 非法路线，vehicle indexes={invalid[:10]}")
    for plan in plans:
        observed_boundary = _is_boundary_pair(plan.edges, entry_ids, exit_ids)
        if observed_boundary != (plan.od_class == "boundary"):
            raise RuntimeError(
                f"OD 类别与实际端点不一致: class={plan.od_class}, "
                f"origin={plan.edges[0]}, destination={plan.edges[-1]}"
            )

    edge_to_road = {
        str(row["id"]): str(row["topology_road_id"])
        for row in network["directed_edges"]
    }
    planned_edges = {
        edge_id for plan in plans for edge_id in plan.edges if edge_id in edge_to_road
    }
    planned_roads = {edge_to_road[edge_id] for edge_id in planned_edges}
    required_roads = set(edge_to_road.values())
    if planned_roads != required_roads:
        raise RuntimeError(
            f"OD 道路覆盖不完整: missing={sorted(required_roads - planned_roads)[:10]}"
        )

    route_lengths = [len(plan.edges) for plan in plans]
    eligible_edges = _eligible_edges(contract, net)
    supports_long_route = _network_supports_long_route(net, eligible_edges)
    long_route_count = sum(
        length >= MIN_PREFERRED_ROUTE_EDGES for length in route_lengths
    )
    if supports_long_route and not long_route_count:
        raise RuntimeError("路网支持 >=3 edge 路线，但规划结果仍全部为短路线")
    unexplained_short_routes = [
        plan
        for plan in plans
        if len(plan.edges) < MIN_PREFERRED_ROUTE_EDGES
        and plan.od_class == "arbitrary"
        and plan.coverage_target_road_id is None
    ]
    if supports_long_route and unexplained_short_routes:
        raise RuntimeError("任意 OD 填充路线未优先采用路网中可用的 >=3 edge 路线")
    unavoidable_short_road_ids = sorted(
        {
            plan.coverage_target_road_id
            for plan in plans
            if len(plan.edges) < MIN_PREFERRED_ROUTE_EDGES
            and plan.coverage_target_road_id is not None
        }
    )
    targets_by_road: dict[str, list[Any]] = {}
    for edge_id, road_id in edge_to_road.items():
        targets_by_road.setdefault(road_id, []).append(net.getEdge(edge_id))
    incorrectly_marked_unavoidable = [
        road_id
        for road_id in unavoidable_short_road_ids
        if any(
            _three_edge_routes_through_target(net, target, entry_ids, exit_ids)
            for target in targets_by_road[road_id]
        )
    ]
    if incorrectly_marked_unavoidable:
        raise RuntimeError(
            "短路线例外的 lane-connection 可达性证明失败: "
            f"{incorrectly_marked_unavoidable[:10]}"
        )
    short_boundary_count = sum(
        len(plan.edges) < MIN_PREFERRED_ROUTE_EDGES and plan.od_class == "boundary"
        for plan in plans
    )
    short_arbitrary_filler_count = sum(
        len(plan.edges) < MIN_PREFERRED_ROUTE_EDGES
        and plan.od_class == "arbitrary"
        and plan.coverage_target_road_id is None
        for plan in plans
    )
    return {
        "planned_edges": planned_edges,
        "planned_roads": planned_roads,
        "required_roads": required_roads,
        "route_lengths": route_lengths,
        "supports_long_route": supports_long_route,
        "long_route_count": long_route_count,
        "unavoidable_short_road_ids": unavoidable_short_road_ids,
        "short_boundary_count": short_boundary_count,
        "short_arbitrary_filler_count": short_arbitrary_filler_count,
    }


def _validate_routes_with_sumo(net_path: Path, routes_path: Path) -> None:
    import sumolib  # noqa: WPS433

    command = [
        sumolib.checkBinary("sumo"),
        "--net-file",
        str(net_path),
        "--route-files",
        str(routes_path),
        "--begin",
        "0",
        "--end",
        "0",
        "--route-steps",
        "0",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(
            "SUMO 拒绝生成的 routes.xml: "
            + ("\n".join(detail[-10:]) if detail else f"exit={result.returncode}")
        )


def generate_demand(
    tile_dir: Path,
    net_path: Path,
    *,
    seed: int = 0,
    horizon: float | None = None,
    period: float | None = None,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    depart_end: float = DEFAULT_DEPART_END_S,
    force: bool = False,
) -> Path:
    """Write exactly ``vehicle_count`` routable vehicles with all-road coverage."""
    tile_dir = Path(tile_dir)
    routes_path = net_path.parent / "routes.xml"
    manifest_path = net_path.parent / "demand_manifest.json"
    if routes_path.is_file() and manifest_path.is_file() and not force:
        return routes_path
    if vehicle_count <= 0:
        raise ValueError("vehicle_count 必须为正数")

    contract = _traffic_contract(tile_dir)
    home = _sumo_home()
    os.environ["SUMO_HOME"] = str(home)
    import sumolib  # noqa: WPS433

    net = sumolib.net.readNet(str(net_path), withInternal=False)
    rng = np.random.default_rng(int(seed))
    plans = _plan_routes(contract, net, rng, vehicle_count)
    validation = _validate_plans(contract, net, plans)

    departures = np.linspace(0.0, float(depart_end), vehicle_count, endpoint=False)
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        {
            "id": "passenger",
            "vClass": "passenger",
            "length": str(PASSENGER_LENGTH_M),
            "minGap": str(PASSENGER_MIN_GAP_M),
            "accel": "2.6",
            "decel": str(PASSENGER_DECEL_MPS2),
            "sigma": "0.5",
        },
    )
    speed_factors = rng.uniform(
        DEPART_SPEED_FACTOR_RANGE[0],
        DEPART_SPEED_FACTOR_RANGE[1],
        size=vehicle_count,
    )
    depart_speeds: list[float] = []
    actual_speed_factors: list[float] = []
    for index, (departure, plan, speed_factor) in enumerate(
        zip(departures, plans, speed_factors, strict=True)
    ):
        start_edge = net.getEdge(plan.edges[0])
        start_speed_limit = float(start_edge.getSpeed())
        stopping_distance = max(
            float(start_edge.getLength())
            - PASSENGER_LENGTH_M
            - PASSENGER_MIN_GAP_M,
            0.0,
        )
        safe_stop_speed = SAFE_STOP_SPEED_MARGIN * math.sqrt(
            2.0 * PASSENGER_DECEL_MPS2 * stopping_distance
        )
        depart_speed = float(
            f"{min(start_speed_limit * float(speed_factor), safe_stop_speed):.4f}"
        )
        depart_speed = min(depart_speed, start_speed_limit)
        depart_speeds.append(depart_speed)
        actual_speed_factors.append(depart_speed / start_speed_limit)
        vehicle = ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"veh{index:04d}",
                "type": "passenger",
                "depart": f"{float(departure):.2f}",
                "departLane": "best",
                "departSpeed": f"{depart_speed:.4f}",
            },
        )
        ET.SubElement(vehicle, "route", {"edges": " ".join(plan.edges)})
    if vehicle_count > 1 and len(set(depart_speeds)) < 2:
        raise RuntimeError("seed 控制的逐车 departSpeed 未产生速度分布")

    temporary_routes_path = routes_path.with_name(f".{routes_path.name}.tmp")
    _write_xml(root, temporary_routes_path)
    try:
        _validate_routes_with_sumo(net_path, temporary_routes_path)
        temporary_routes_path.replace(routes_path)
    finally:
        if temporary_routes_path.exists():
            temporary_routes_path.unlink()

    planned_edges = validation["planned_edges"]
    planned_roads = validation["planned_roads"]
    required_roads = validation["required_roads"]
    route_lengths = validation["route_lengths"]
    boundary_count = sum(plan.od_class == "boundary" for plan in plans)
    arbitrary_count = vehicle_count - boundary_count
    manifest = {
        "schema_version": 2,
        "seed": int(seed),
        "vehicle_count": int(vehicle_count),
        "depart_begin_s": 0.0,
        "depart_end_s": float(depart_end),
        "route_generation": "sumo_shortest_path_od_with_targeted_road_coverage",
        "boundary_od_target_fraction": BOUNDARY_OD_FRACTION,
        "boundary_od_vehicle_count": boundary_count,
        "boundary_od_actual_fraction": boundary_count / vehicle_count,
        "arbitrary_od_vehicle_count": arbitrary_count,
        "arbitrary_od_actual_fraction": arbitrary_count / vehicle_count,
        "required_topology_road_count": len(required_roads),
        "planned_covered_topology_road_count": len(planned_roads),
        "planned_road_coverage": len(planned_roads) / max(len(required_roads), 1),
        "planned_directed_edge_count": len(planned_edges),
        "route_validity": "sumolib_passenger_connections_and_sumo_load",
        "sumo_route_validation_passed": True,
        "route_length_edges": {
            "min": min(route_lengths),
            "median": float(np.median(route_lengths)),
            "max": max(route_lengths),
        },
        "two_edge_route_count": sum(length == 2 for length in route_lengths),
        "two_edge_route_fraction": sum(length == 2 for length in route_lengths)
        / vehicle_count,
        "three_or_more_edge_route_count": validation["long_route_count"],
        "three_or_more_edge_route_fraction": validation["long_route_count"]
        / vehicle_count,
        "network_supports_three_or_more_edge_route": validation["supports_long_route"],
        "short_route_exception": (
            None
            if not validation["unavoidable_short_road_ids"]
            and not validation["short_boundary_count"]
            and not validation["short_arbitrary_filler_count"]
            else {
                "reason": (
                    "exhaustive passenger lane-connection search found no "
                    "legal simple three-edge route for the required OD class "
                    "or coverage target"
                ),
                "unavoidable_coverage_topology_road_ids": validation[
                    "unavoidable_short_road_ids"
                ],
                "short_boundary_od_vehicle_count": validation["short_boundary_count"],
                "short_arbitrary_filler_vehicle_count": validation[
                    "short_arbitrary_filler_count"
                ],
            }
        ),
        "depart_speed_strategy": {
            "type": "seeded_per_vehicle_numeric",
            "factor_distribution": "uniform",
            "factor_min_inclusive": DEPART_SPEED_FACTOR_RANGE[0],
            "factor_max_exclusive": DEPART_SPEED_FACTOR_RANGE[1],
            "capped_by_start_edge_speed_limit": True,
            "capped_by_safe_stop_speed": True,
            "safe_stop_speed_formula": (
                "0.9 * sqrt(2 * decel * "
                "max(edge_length - vehicle_length - min_gap, 0))"
            ),
        },
        "depart_speed_mps": {
            "min": min(depart_speeds),
            "median": float(np.median(depart_speeds)),
            "max": max(depart_speeds),
            "unique_count": len(set(depart_speeds)),
        },
        "depart_speed_factor": {
            "min": min(actual_speed_factors),
            "median": float(np.median(actual_speed_factors)),
            "max": max(actual_speed_factors),
        },
        "legacy_horizon_argument": horizon,
        "legacy_period_argument": period,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[sumo-demand] vehicles={vehicle_count} depart=0..{depart_end:g}s "
        f"boundary_od={boundary_count}/{vehicle_count} "
        f"route_edges={min(route_lengths)}/{float(np.median(route_lengths)):g}/"
        f"{max(route_lengths)} two_edge={sum(length == 2 for length in route_lengths) / vehicle_count:.1%} "
        f"road_coverage={len(planned_roads)}/{len(required_roads)} "
        f"depart_speed={min(depart_speeds):.2f}..{max(depart_speeds):.2f}m/s"
    )
    return routes_path


def run(args: argparse.Namespace) -> None:
    tile_dir = Path(args.input)
    if tile_dir.suffix.lower() in {".json", ".gpkg"}:
        tile_dir = tile_dir.parent
    net_path = build_sumo_net(
        tile_dir, speed=args.speed, lanes=args.lanes, force=bool(args.force)
    )
    generate_demand(
        tile_dir,
        net_path,
        seed=int(args.seed),
        vehicle_count=int(args.vehicle_count),
        depart_end=float(args.depart_end),
        force=bool(args.force),
    )


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="tile 目录或其 topology.json 路径")
    parser.add_argument(
        "--speed", type=float, default=None, help="调试用统一限速覆盖值 m/s"
    )
    parser.add_argument(
        "--lanes", type=int, default=None, help="调试用统一车道数覆盖值"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vehicle-count", type=int, default=DEFAULT_VEHICLE_COUNT)
    parser.add_argument("--depart-end", type=float, default=DEFAULT_DEPART_END_S)
    parser.add_argument("--force", action="store_true", help="强制覆盖现有网络与需求")


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sumo-demand",
        help="序列化清洗后的 SUMO 契约并生成覆盖全部道路的 OD",
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO 路网与 OD 生成")
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
