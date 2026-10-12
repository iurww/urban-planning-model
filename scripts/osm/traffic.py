"""SUMO lane, connection, detector, and signal-control contract."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations, permutations
from typing import Any

import networkx as nx

DIRECTIONS = ("N", "E", "S", "W")
DIRECTION_INDEX = {value: index for index, value in enumerate(DIRECTIONS)}
SUMO_VEHICLE_CLASSES = ["passenger", "taxi", "bus", "delivery"]
ROAD_PRIORITY = {
    "motorway": 13,
    "trunk": 12,
    "primary": 11,
    "secondary": 10,
    "tertiary": 9,
    "unclassified": 8,
    "residential": 7,
}
SIGNAL_TIMING = {
    "simulation_step_s": 1,
    "decision_interval_s": 15,
    "minimum_green_s": 10,
    "maximum_green_s": 60,
    "yellow_s": 3,
    "all_red_s": 0,
}
DEFAULT_PHASE_ACTIONS = [
    {"phase": 0, "name": "EW_straight", "movements": ["E_W", "W_E"]},
    {"phase": 1, "name": "NS_straight", "movements": ["N_S", "S_N"]},
    {"phase": 2, "name": "EW_left", "movements": ["E_S", "W_N"]},
    {"phase": 3, "name": "NS_left", "movements": ["N_E", "S_W"]},
    {"phase": 4, "name": "W_combined", "movements": ["W_E", "W_N"]},
    {"phase": 5, "name": "E_combined", "movements": ["E_W", "E_S"]},
    {"phase": 6, "name": "N_combined", "movements": ["N_S", "N_E"]},
    {"phase": 7, "name": "S_combined", "movements": ["S_N", "S_W"]},
]
COMBINED_ACTION_BY_ORIGIN = {"W": 4, "E": 5, "N": 6, "S": 7}
SYNTHETIC_LANE_POLICY = "abstract-road-default-v1"
MIN_SYNTHETIC_LANES = 2
MAX_SYNTHETIC_LANES = 3


def default_lane_connections(lane_count: int, speed_kph: float) -> list[dict[str, Any]]:
    """Return the fixed per-direction lane template for an abstract road.

    Lane indices follow SUMO's rightmost-zero convention.  The returned list is
    therefore ordered right-to-left even though the policy is described as
    left/middle/right in the data contract.
    """
    count = max(MIN_SYNTHETIC_LANES, min(MAX_SYNTHETIC_LANES, int(lane_count)))
    if count == 2:
        rows = [
            (0, "right", ["straight", "right"]),
            (1, "left", ["left"]),
        ]
    else:
        rows = [
            (0, "right", ["straight", "right"]),
            (1, "middle", ["straight"]),
            (2, "left", ["left"]),
        ]
    return [
        {
            "lane_index": index,
            "lateral_position": position,
            "allowed_turn_types": turns,
            "speed_kph": round(float(speed_kph), 3),
        }
        for index, position, turns in rows
    ]


def apply_synthetic_lane_defaults(roads: list[dict[str, Any]]) -> dict[str, int]:
    """Normalize abstract roads to the deterministic two/three-lane contract.

    ``total_lanes`` is retained as the compatibility field used by the model,
    but now has per-direction semantics.  The original aggregate OSM-derived
    value is kept in ``source_total_lanes`` for provenance.
    """
    counts = {"roads": 0, "two_lane": 0, "three_lane": 0, "clamped_low": 0, "clamped_high": 0}
    for road in roads:
        raw_value = road.get("source_total_lanes", road.get("total_lanes", 2))
        try:
            source_count = max(int(raw_value), 1)
        except (TypeError, ValueError):
            source_count = 2
        lane_count = max(MIN_SYNTHETIC_LANES, min(MAX_SYNTHETIC_LANES, source_count))
        speed_kph = float(road.get("speed_kph", 50.0) or 50.0)
        road.setdefault("source_total_lanes", source_count)
        road.setdefault(
            "source_lane_data_inferred", bool(road.get("lane_data_inferred", True))
        )
        road["total_lanes"] = lane_count
        road["lanes_per_direction"] = lane_count
        road["lane_count_semantics"] = "per_direction_on_bidirectional_abstract_road"
        road["lane_index_order"] = "rightmost_zero"
        road["lane_connections"] = default_lane_connections(lane_count, speed_kph)
        road["lane_assignment_policy"] = SYNTHETIC_LANE_POLICY
        road["lane_data_inferred"] = True
        counts["roads"] += 1
        counts["two_lane" if lane_count == 2 else "three_lane"] += 1
        counts["clamped_low"] += int(source_count < MIN_SYNTHETIC_LANES)
        counts["clamped_high"] += int(source_count > MAX_SYNTHETIC_LANES)
    return counts


def default_signalized_intersection_ids(
    intersections: list[dict[str, Any]],
) -> list[str]:
    """Every real abstract road intersection is signalized by default."""
    return [
        str(item["id"])
        for item in intersections
        if item.get("kind") == "junction"
    ]


def build_default_traffic_signals(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    *,
    tile_id: str,
) -> dict[str, Any]:
    """Build a complete all-junction signal contract without OSM controllers."""
    graph = nx.Graph()
    graph.add_nodes_from(str(item["id"]) for item in intersections)
    for road in roads:
        graph.add_edge(
            str(road["from_intersection_id"]),
            str(road["to_intersection_id"]),
            weight=float(road.get("length_m", 1.0)),
        )
    grouped = {
        intersection_id: []
        for intersection_id in default_signalized_intersection_ids(intersections)
    }
    return assemble_sumo_ready_traffic(
        source={"controls": [], "phase_actions": DEFAULT_PHASE_ACTIONS},
        grouped=grouped,
        dropped=[],
        roads=roads,
        intersections=intersections,
        graph=graph,
        tile_id=tile_id,
    )


def assemble_sumo_ready_traffic(
    source: dict[str, Any],
    grouped: dict[str, list[tuple[dict[str, Any], float]]],
    dropped: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    graph: nx.Graph,
    tile_id: str,
) -> dict[str, Any]:
    """Build the complete static contract needed by netconvert and a TraCI RL env."""

    source_phase_actions = source.get("phase_actions") or DEFAULT_PHASE_ACTIONS
    controller_ids = {
        intersection_id: _sumo_id("tls", intersection_id)
        for intersection_id in sorted(grouped)
    }
    network = _build_directed_network(roads, intersections, controller_ids)
    edge_by_id = {row["id"]: row for row in network["directed_edges"]}
    lane_by_id = {row["id"]: row for row in network["lanes"]}
    connections_by_intersection: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for connection in network["connections"]:
        connections_by_intersection[connection["via_intersection_id"]].append(connection)

    intersections_by_id = {row["id"]: row for row in intersections}
    controllers: list[dict[str, Any]] = []
    detectors: list[dict[str, Any]] = []
    for intersection_id, mappings in sorted(grouped.items()):
        intersection = intersections_by_id[intersection_id]
        controller_id = f"signal_controller/{intersection_id.rsplit('/', 1)[-1]}"
        sumo_tls_id = controller_ids[intersection_id]
        controlled_connections = sorted(
            connections_by_intersection[intersection_id], key=_connection_sort_key
        )
        for link_index, connection in enumerate(controlled_connections):
            connection["controller_id"] = controller_id
            connection["sumo_tl_id"] = sumo_tls_id
            connection["link_index"] = link_index

        slots = _build_approach_slots(
            intersection_id,
            network["directed_edges"],
            network["lanes"],
        )
        movements = _build_controller_movements(
            intersection_id,
            controlled_connections,
            edge_by_id,
            lane_by_id,
            source_phase_actions,
        )
        action_program = _build_action_program(
            source_phase_actions, controlled_connections
        )
        controlled_links = [
            {
                "link_index": connection["link_index"],
                "connection_id": connection["id"],
                "from_edge_id": connection["from_edge_id"],
                "to_edge_id": connection["to_edge_id"],
                "from_lane_id": connection["from_lane_id"],
                "to_lane_id": connection["to_lane_id"],
                "from_lane_index": connection["from_lane_index"],
                "to_lane_index": connection["to_lane_index"],
                "movement_key": connection["movement_key"],
                "turn_type": connection["turn_type"],
            }
            for connection in controlled_connections
        ]
        incoming_lane_ids = sorted(
            {
                connection["from_lane_id"]
                for connection in controlled_connections
            }
        )
        outgoing_lane_ids = sorted(
            {
                connection["to_lane_id"]
                for connection in controlled_connections
            }
        )
        controller_detectors = [
            _make_lane_detector(controller_id, lane_by_id[lane_id])
            for lane_id in incoming_lane_ids
        ]
        detectors.extend(controller_detectors)
        directions = {key for key, value in slots.items() if value["exists"]}
        head_ids = sorted(
            {
                head_id
                for control, _ in mappings
                for head_id in control.get("signal_head_ids", [])
            }
        )
        maximum_mapping_distance = (
            round(max(distance for _, distance in mappings), 3)
            if mappings
            else None
        )
        controllers.append(
            {
                "id": controller_id,
                "intersection_id": intersection_id,
                "sumo_tl_id": sumo_tls_id,
                "source_intersection_ids": sorted(
                    control["intersection_id"] for control, _ in mappings
                ),
                "source_signal_head_ids": head_ids,
                "source_signal_head_count": len(head_ids),
                "maximum_source_mapping_distance_m": maximum_mapping_distance,
                "controller_source": (
                    "mapped_osm_signal" if mappings else "abstract_default"
                ),
                "signal_group_generation": "regenerated one SUMO linkIndex per lane-to-lane connection; source heads are provenance only",
                "approach_slots": slots,
                "available_phase_mask": action_program["available_action_mask"],
                "movements": movements,
                "controlled_links": controlled_links,
                "program": action_program,
                "observation_lane_ids": incoming_lane_ids,
                "pressure_outgoing_lane_ids": outgoing_lane_ids,
                "detector_ids": [row["id"] for row in controller_detectors],
                "neighbor_controller_ids": [],
                "neighbor_controllers": [],
                "static_features": {
                    "road_degree": intersection["degree"],
                    "cardinal_approach_count": len(directions),
                    "incoming_edge_count": sum(
                        len(value["incoming_edge_ids"]) for value in slots.values()
                    ),
                    "total_incoming_lanes": len(incoming_lane_ids),
                    "controlled_link_count": len(controlled_links),
                },
            }
        )

    _attach_controller_neighbors(controllers, graph)
    controller_by_id = {row["id"]: row for row in controllers}
    for lane in network["lanes"]:
        lane["controlled_movement_ids"] = sorted(
            set(lane.pop("_controlled_movement_ids", []))
        )

    phase_actions = _normalize_phase_actions(source_phase_actions)
    observation_schema = _observation_schema()
    reward_schema = _reward_schema()
    demand_contract = _build_demand_contract(network)
    payload: dict[str, Any] = {
        "schema_version": "abstract-traffic-signals-2.0",
        "readiness_level": "sumo_plain_xml_and_traci_rl_static_input",
        "metadata": {
            "tile_id": tile_id,
            "coordinate_system": "local_metric; node and edge geometry is copied from frozen topology.json",
            "driving_side": "right",
            "mapping_principle": "OSM signal heads are mapped as evidence to regenerated intersections; lanes, connections and signal groups are regenerated for the abstract topology",
            "default_signal_policy": "every kind=junction abstract intersection is signalized",
            "synthetic_lane_policy": SYNTHETIC_LANE_POLICY,
            "topology_geometry_modified": False,
            "dynamic_traffic_values_included": False,
            "inference_policy": {
                "road_direction": "every abstract road is bidirectional because merged source carriageways do not preserve reliable direction",
                "directional_lane_split": "lanes_per_direction is copied to both directions of the synthetic bidirectional abstract road",
                "lane_indexing": "SUMO convention: lane 0 is rightmost, increasing toward the median/left",
                "lane_function": "two lanes: left=left, right=straight/right; three lanes: left=left, middle=straight, right=straight/right",
                "lane_length": "equal to regenerated abstract road segment length",
                "lane_speed": "equal to topology road speed; retain speed_inferred provenance",
                "u_turns": "not generated",
                "right_turn_signal": "permissive lower-case g only while a valid vehicle green action is active; red during yellow and all-red",
            },
        },
        "simulation_contract": {
            "simulator": "SUMO",
            "network_build_input": "sumo_network can be serialized to .nod.xml, .edg.xml, .con.xml and .tll.xml, then compiled with netconvert",
            "control_api": "TraCI or libsumo",
            "traffic_light_method": "setRedYellowGreenState",
            "step_length_s": SIGNAL_TIMING["simulation_step_s"],
            "right_hand_traffic": True,
            "internal_links": True,
            "od_routes_required_at_runtime": True,
            "od_routes_included": False,
            "recommended_vehicle_class": "passenger",
            "time_to_teleport_s": 600,
        },
        "phase_actions": phase_actions,
        "signalized_intersection_ids": [
            row["intersection_id"] for row in controllers
        ],
        "rl_control_schema": {
            "action_space": "Discrete(8) with per-controller available_phase_mask",
            "action_ids": [row["action_id"] for row in phase_actions],
            "invalid_action_policy": "mask before sampling; if received, keep the current action and emit invalid_action=true",
            "transition_execution": "fixed 15 s environment step: if the action changes, apply 3 s current-action yellow then 12 s target green; if unchanged, hold the current green/red state for all 15 s",
            "minimum_green_enforcement": "an action may change only after at least 10 accumulated seconds of current green",
            "maximum_green_enforcement": "after 60 accumulated seconds the next sampled action must differ when another action is available",
            "timing": SIGNAL_TIMING,
            "multi_agent": {
                "agent_id": "controller.id",
                "neighbor_field": "neighbor_controller_ids",
                "parameter_sharing_compatible": True,
            },
        },
        "observation_schema": observation_schema,
        "reward_schema": reward_schema,
        "od_demand_contract": demand_contract,
        "sumo_network": network,
        "detectors": detectors,
        "controllers": controllers,
        "mapping_report": {
            "source_controller_count": len(source.get("controls", [])),
            "mapped_source_controller_count": sum(len(value) for value in grouped.values()),
            "abstract_controller_count": len(controllers),
            "mapped_abstract_controller_count": sum(
                bool(value) for value in grouped.values()
            ),
            "defaulted_abstract_controller_count": sum(
                not value for value in grouped.values()
            ),
            "dropped_source_controller_count": len(dropped),
            "dropped_source_controllers": dropped,
            "maximum_attachment_distance_m": 220,
        },
    }
    payload["validation"] = _validate_payload(payload, controller_by_id)
    if not payload["validation"]["passed"]:
        raise RuntimeError(
            "SUMO traffic-signal payload validation failed: "
            + "; ".join(payload["validation"]["errors"][:20])
        )
    return payload


def _build_directed_network(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    controller_ids: dict[str, str],
) -> dict[str, Any]:
    nodes = []
    for intersection in sorted(intersections, key=lambda row: row["id"]):
        nodes.append(
            {
                "id": _sumo_id("node", intersection["id"]),
                "topology_intersection_id": intersection["id"],
                "x_m": intersection["point"][0],
                "y_m": intersection["point"][1],
                "type": (
                    "traffic_light"
                    if intersection["id"] in controller_ids
                    else "priority"
                ),
                "tl_id": controller_ids.get(intersection["id"]),
                "fringe": "outer" if intersection["kind"] == "boundary_portal" else "default",
            }
        )

    directed_edges: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    for road in sorted(roads, key=lambda row: row["id"]):
        if "lanes_per_direction" in road:
            forward_count = reverse_count = int(road["lanes_per_direction"])
        else:
            total_lanes = max(2, int(road["total_lanes"]))
            forward_count = max(1, math.ceil(total_lanes / 2))
            reverse_count = max(1, math.floor(total_lanes / 2))
        configured_lanes = {
            int(item["lane_index"]): item
            for item in road.get(
                "lane_connections",
                default_lane_connections(
                    max(forward_count, reverse_count),
                    float(road.get("speed_kph", 50.0)),
                ),
            )
        }
        directions = (
            (
                "forward",
                road["from_intersection_id"],
                road["to_intersection_id"],
                road["geometry"],
                forward_count,
            ),
            (
                "reverse",
                road["to_intersection_id"],
                road["from_intersection_id"],
                list(reversed(road["geometry"])),
                reverse_count,
            ),
        )
        for direction, from_id, to_id, geometry, lane_count in directions:
            edge_id = _sumo_id(
                "edge", f"{road['id']}_{'fwd' if direction == 'forward' else 'rev'}"
            )
            lane_ids = [f"{edge_id}_{index}" for index in range(lane_count)]
            directed_edges.append(
                {
                    "id": edge_id,
                    "topology_road_id": road["id"],
                    "direction": direction,
                    "from_node_id": _sumo_id("node", from_id),
                    "to_node_id": _sumo_id("node", to_id),
                    "from_intersection_id": from_id,
                    "to_intersection_id": to_id,
                    "shape": geometry,
                    "length_m": road["length_m"],
                    "num_lanes": lane_count,
                    "lane_ids": lane_ids,
                    "speed_limit_kph": road["speed_kph"],
                    "speed_limit_mps": round(road["speed_kph"] / 3.6, 4),
                    "priority": ROAD_PRIORITY.get(road["highway"], 6),
                    "road_class": road["highway"],
                    "name": road["name"],
                    "allow_vehicle_classes": SUMO_VEHICLE_CLASSES,
                    "direction_inferred": True,
                    "directional_lane_count_inferred": True,
                }
            )
            for index, lane_id in enumerate(lane_ids):
                configured = configured_lanes.get(index, {})
                default_turns = list(configured.get("allowed_turn_types", []))
                lanes.append(
                    {
                        "id": lane_id,
                        "edge_id": edge_id,
                        "topology_road_id": road["id"],
                        "direction": direction,
                        "index": index,
                        "index_order": "rightmost_zero",
                        "from_intersection_id": from_id,
                        "to_intersection_id": to_id,
                        "length_m": road["length_m"],
                        "speed_limit_kph": road["speed_kph"],
                        "speed_limit_mps": round(road["speed_kph"] / 3.6, 4),
                        "width_m": 3.2,
                        "allow_vehicle_classes": SUMO_VEHICLE_CLASSES,
                        "function": None,
                        "lateral_position": configured.get("lateral_position"),
                        "default_allowed_turn_types": default_turns,
                        "allowed_turn_types": [],
                        "allowed_connection_ids": [],
                        "lane_count_inferred": True,
                        "lane_function_inferred": True,
                        "length_inferred_from_road": True,
                        "speed_inferred": bool(road["speed_inferred"]),
                        "_controlled_movement_ids": [],
                    }
                )

    lane_by_id = {row["id"]: row for row in lanes}
    incoming: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in directed_edges:
        incoming[edge["to_intersection_id"]].append(edge)
        outgoing[edge["from_intersection_id"]].append(edge)

    approach_slot_by_edge_id: dict[str, str] = {}
    approach_slot_by_road: dict[tuple[str, str], str] = {}
    for intersection_id, edges in incoming.items():
        assignments = _assign_approach_slots(edges)
        approach_slot_by_edge_id.update(assignments)
        approach_slot_by_road.update(
            {
                (intersection_id, edge["topology_road_id"]): assignments[edge["id"]]
                for edge in edges
            }
        )

    connections: list[dict[str, Any]] = []
    intersection_kind = {
        str(row["id"]): str(row.get("kind", "")) for row in intersections
    }
    for intersection_id in sorted({row["id"] for row in intersections}):
        origin_slots = {
            approach_slot_by_edge_id[edge["id"]]
            for edge in incoming[intersection_id]
        }
        standard_cross = (
            intersection_kind.get(intersection_id) == "junction"
            and len(incoming[intersection_id]) == 4
            and len(origin_slots) == 4
        )
        for from_edge in sorted(incoming[intersection_id], key=lambda row: row["id"]):
            origin_slot = approach_slot_by_edge_id[from_edge["id"]]
            candidates = []
            for to_edge in sorted(outgoing[intersection_id], key=lambda row: row["id"]):
                if from_edge["topology_road_id"] == to_edge["topology_road_id"]:
                    continue
                destination_slot = approach_slot_by_road[
                    (intersection_id, to_edge["topology_road_id"])
                ]
                turn = (
                    _classify_cardinal_turn(
                        origin_slot,
                        destination_slot,
                    )
                    if standard_cross
                    else _classify_edge_turn(from_edge, to_edge)
                )
                if turn is None:
                    continue
                candidates.append((to_edge, turn))
            available_turns = {turn for _, turn in candidates}
            if intersection_kind.get(intersection_id) == "junction":
                permissions = []
                for lane_id in from_edge["lane_ids"]:
                    default_turns = set(
                        lane_by_id[lane_id]["default_allowed_turn_types"]
                    )
                    allowed_turns = default_turns & available_turns
                    if available_turns and not allowed_turns:
                        allowed_turns = _replacement_turns_for_unserved_lane(
                            default_turns,
                            available_turns,
                        )
                    permissions.append(allowed_turns)
            else:
                permissions = [
                    ({"straight"} if "straight" in available_turns else set(available_turns))
                    for _ in from_edge["lane_ids"]
                ]
            for index, allowed_turns in enumerate(permissions):
                lane = lane_by_id[from_edge["lane_ids"][index]]
                lane["allowed_turn_types"] = sorted(
                    allowed_turns, key=("right", "straight", "left").index
                )
                lane["function"] = (
                    _lane_function(
                        allowed_turns
                        or set(lane["default_allowed_turn_types"])
                    )
                    if available_turns
                    else None
                )
            for to_edge, turn in candidates:
                destination_slot = approach_slot_by_road[
                    (intersection_id, to_edge["topology_road_id"])
                ]
                for from_lane_index, allowed_turns in enumerate(permissions):
                    if turn not in allowed_turns:
                        continue
                    to_lane_index = _target_lane_index(
                        turn,
                        from_lane_index,
                        from_edge["num_lanes"],
                        to_edge["num_lanes"],
                    )
                    connection_id = _sumo_id(
                        "conn",
                        f"{intersection_id}_{from_edge['id']}_{from_lane_index}_{to_edge['id']}_{to_lane_index}",
                    )
                    connection = {
                        "id": connection_id,
                        "via_intersection_id": intersection_id,
                        "from_edge_id": from_edge["id"],
                        "to_edge_id": to_edge["id"],
                        "from_lane_id": from_edge["lane_ids"][from_lane_index],
                        "to_lane_id": to_edge["lane_ids"][to_lane_index],
                        "from_lane_index": from_lane_index,
                        "to_lane_index": to_lane_index,
                        "origin_slot": origin_slot,
                        "destination_slot": destination_slot,
                        "movement_key": f"{origin_slot}_{destination_slot}",
                        "turn_type": turn,
                        "u_turn": False,
                        "controller_id": None,
                        "sumo_tl_id": None,
                        "link_index": None,
                    }
                    connections.append(connection)
                    lane_by_id[connection["from_lane_id"]][
                        "allowed_connection_ids"
                    ].append(connection_id)
        if intersection_kind.get(intersection_id) == "junction":
            _cover_outgoing_edge_lanes(
                intersection_id,
                outgoing[intersection_id],
                connections,
                lane_by_id,
            )

    # Boundary exit lanes intentionally have no outgoing connection.
    for lane in lanes:
        if lane["function"] is None:
            lane["function"] = "network_exit"
            lane["allowed_turn_types"] = []

    boundary_ids = {
        row["id"] for row in intersections if row["kind"] == "boundary_portal"
    }
    entry_edges = sorted(
        edge["id"]
        for edge in directed_edges
        if edge["from_intersection_id"] in boundary_ids
        and edge["to_intersection_id"] not in boundary_ids
    )
    exit_edges = sorted(
        edge["id"]
        for edge in directed_edges
        if edge["to_intersection_id"] in boundary_ids
        and edge["from_intersection_id"] not in boundary_ids
    )
    return {
        "schema_version": "sumo-directed-network-1.0",
        "coordinate_system": "local_metric",
        "lane_index_convention": "rightmost lane is index 0",
        "plain_xml_files": {
            "nodes": "network.nod.xml",
            "edges": "network.edg.xml",
            "connections": "network.con.xml",
            "traffic_lights": "network.tll.xml",
            "compiled_network": "network.net.xml",
        },
        "netconvert": {
            "required_inputs": [
                "network.nod.xml",
                "network.edg.xml",
                "network.con.xml",
                "network.tll.xml",
            ],
            "recommended_options": [
                "--no-turnarounds=true",
                "--junctions.internal-link-detail=5",
                "--tls.discard-simple=false",
            ],
        },
        "nodes": nodes,
        "directed_edges": directed_edges,
        "lanes": lanes,
        "connections": connections,
        "boundary_entry_edge_ids": entry_edges,
        "boundary_exit_edge_ids": exit_edges,
    }


def _build_approach_slots(
    intersection_id: str,
    directed_edges: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lane_by_id = {row["id"]: row for row in lanes}
    slots: dict[str, dict[str, Any]] = {
        direction: {
            "exists": False,
            "topology_road_ids": [],
            "incoming_edge_ids": [],
            "incoming_lane_ids": [],
            "incoming_lanes": 0,
            "approaches": [],
        }
        for direction in DIRECTIONS
    }
    incoming_edges = sorted(
        (
            edge
            for edge in directed_edges
            if edge["to_intersection_id"] == intersection_id
        ),
        key=lambda row: row["id"],
    )
    slot_by_edge_id = _assign_approach_slots(incoming_edges)
    for edge in incoming_edges:
        direction = slot_by_edge_id[edge["id"]]
        center = edge["shape"][-1]
        outside = edge["shape"][-2]
        bearing = math.degrees(
            math.atan2(outside[0] - center[0], outside[1] - center[1])
        ) % 360
        lane_ids = edge["lane_ids"]
        approach = {
            "topology_road_id": edge["topology_road_id"],
            "incoming_edge_id": edge["id"],
            "bearing_deg": round(bearing, 2),
            "incoming_lane_ids": lane_ids,
            "incoming_lanes": len(lane_ids),
            "lane_functions": [lane_by_id[lane_id]["function"] for lane_id in lane_ids],
            "length_m": edge["length_m"],
            "speed_limit_kph": edge["speed_limit_kph"],
            "speed_limit_mps": edge["speed_limit_mps"],
            "road_class": edge["road_class"],
            "lane_attributes_inferred": True,
        }
        slot = slots[direction]
        slot["exists"] = True
        slot["topology_road_ids"].append(edge["topology_road_id"])
        slot["incoming_edge_ids"].append(edge["id"])
        slot["incoming_lane_ids"].extend(lane_ids)
        slot["incoming_lanes"] += len(lane_ids)
        slot["approaches"].append(approach)
    return slots


def _build_controller_movements(
    intersection_id: str,
    connections: list[dict[str, Any]],
    edge_by_id: dict[str, dict[str, Any]],
    lane_by_id: dict[str, dict[str, Any]],
    phase_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phase_lookup: defaultdict[str, list[int]] = defaultdict(list)
    for action in phase_actions:
        action_id = int(action.get("action_id", action.get("phase", 0)))
        for key in action.get("movements", []):
            phase_lookup[key].append(action_id)
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for connection in connections:
        grouped[connection["movement_key"]].append(connection)
    movements = []
    suffix = intersection_id.rsplit("/", 1)[-1]
    for key, rows in sorted(grouped.items()):
        origin, destination = key.split("_", 1)
        turn_types = {row["turn_type"] for row in rows}
        turn_type = next(iter(turn_types)) if len(turn_types) == 1 else "mixed"
        protected_action_ids = sorted(phase_lookup.get(key, []))
        if not protected_action_ids and turn_type != "right":
            protected_action_ids = [COMBINED_ACTION_BY_ORIGIN[origin]]
        movement_id = f"movement/{suffix}/{key}"
        incoming_lane_ids = sorted({row["from_lane_id"] for row in rows})
        for lane_id in incoming_lane_ids:
            lane_by_id[lane_id]["_controlled_movement_ids"].append(movement_id)
        movements.append(
            {
                "id": movement_id,
                "slot_key": key,
                "origin_slot": origin,
                "destination_slot": destination,
                "turn_type": turn_type,
                "incoming_topology_road_ids": sorted(
                    {
                        edge_by_id[row["from_edge_id"]]["topology_road_id"]
                        for row in rows
                    }
                ),
                "outgoing_topology_road_ids": sorted(
                    {
                        edge_by_id[row["to_edge_id"]]["topology_road_id"]
                        for row in rows
                    }
                ),
                "incoming_edge_ids": sorted({row["from_edge_id"] for row in rows}),
                "outgoing_edge_ids": sorted({row["to_edge_id"] for row in rows}),
                "incoming_lane_ids": incoming_lane_ids,
                "outgoing_lane_ids": sorted({row["to_lane_id"] for row in rows}),
                "connection_ids": [row["id"] for row in rows],
                "link_indices": sorted(row["link_index"] for row in rows),
                "protected_action_ids": protected_action_ids,
                "permissive": turn_type == "right" and not protected_action_ids,
            }
        )
    return movements


def _build_action_program(
    phase_actions: list[dict[str, Any]], connections: list[dict[str, Any]]
) -> dict[str, Any]:
    link_count = len(connections)
    actions = []
    available_mask = []
    right_indices = {
        row["link_index"] for row in connections if row["turn_type"] == "right"
    }
    connection_keys = defaultdict(set)
    for connection in connections:
        connection_keys[connection["movement_key"]].add(connection["link_index"])
    phase_movement_keys = {
        key for phase in phase_actions for key in phase.get("movements", [])
    }
    for phase in phase_actions:
        action_id = int(phase.get("action_id", phase.get("phase", 0)))
        protected_indices = sorted(
            {
                index
                for key in phase.get("movements", [])
                for index in connection_keys.get(key, set())
            }
            | {
                row["link_index"]
                for row in connections
                if row["movement_key"] not in phase_movement_keys
                and row["turn_type"] != "right"
                and COMBINED_ACTION_BY_ORIGIN.get(row["origin_slot"]) == action_id
            }
        )
        available = bool(protected_indices)
        available_mask.append(available)
        if available:
            state = ["r"] * link_count
            for index in right_indices:
                state[index] = "g"
            for index in protected_indices:
                state[index] = "G"
            green_state = "".join(state)
            yellow_state = "".join("y" if char in {"G", "g"} else "r" for char in state)
        else:
            green_state = None
            yellow_state = None
        actions.append(
            {
                "action_id": action_id,
                "name": phase.get("name", f"action_{action_id}"),
                "available": available,
                "movement_keys": phase.get("movements", []),
                "protected_link_indices": protected_indices,
                "permissive_right_link_indices": sorted(right_indices) if available else [],
                "green_state": green_state,
                "yellow_state": yellow_state,
                "minimum_green_s": SIGNAL_TIMING["minimum_green_s"],
                "maximum_green_s": SIGNAL_TIMING["maximum_green_s"],
            }
        )
    valid_actions = [row for row in actions if row["available"]]
    initial = valid_actions[0]
    return {
        "sumo_program_id": "rl_bootstrap",
        "control_mode": "traci_external_state",
        "state_character_order": "link_index_0_to_n_minus_1",
        "state_length": link_count,
        "state_alphabet": {
            "G": "protected green",
            "g": "permissive green; must yield to foes",
            "y": "yellow",
            "r": "red",
        },
        "available_action_mask": available_mask,
        "initial_action_id": initial["action_id"],
        "all_red_state": "r" * link_count,
        "actions": actions,
        "bootstrap_tl_logic": {
            "type": "static",
            "program_id": "rl_bootstrap",
            "offset_s": 0,
            "phases": [
                {
                    "duration_s": 1000,
                    "state": initial["green_state"],
                    "name": f"bootstrap_{initial['name']}",
                }
            ],
        },
    }


def _make_lane_detector(controller_id: str, lane: dict[str, Any]) -> dict[str, Any]:
    usable_length = max(1.0, lane["length_m"] - 1.0)
    detector_length = min(100.0, usable_length)
    pos = max(0.0, lane["length_m"] - detector_length - 1.0)
    return {
        "id": _sumo_id("e2", f"{controller_id}_{lane['id']}"),
        "type": "laneAreaDetector",
        "controller_id": controller_id,
        "lane_id": lane["id"],
        "position_m": round(pos, 3),
        "length_m": round(detector_length, 3),
        "period_s": SIGNAL_TIMING["decision_interval_s"],
        "output_file": "lane_area_detector_output.xml",
        "optional_for_training": True,
        "note": "Core observation fields can be read directly with traci.lane; this detector is supplied for interval diagnostics.",
    }


def _attach_controller_neighbors(
    controllers: list[dict[str, Any]], graph: nx.Graph
) -> None:
    by_intersection = {row["intersection_id"]: row for row in controllers}
    signalized = set(by_intersection)
    for intersection_id, controller in by_intersection.items():
        lengths = nx.single_source_dijkstra_path_length(
            graph, intersection_id, weight="weight"
        )
        nearest = sorted(
            (distance, target)
            for target, distance in lengths.items()
            if target in signalized and target != intersection_id
        )[:4]
        neighbors = [
            {
                "controller_id": by_intersection[target]["id"],
                "intersection_id": target,
                "network_distance_m": round(distance, 3),
            }
            for distance, target in nearest
        ]
        controller["neighbor_controllers"] = neighbors
        controller["neighbor_controller_ids"] = [
            row["controller_id"] for row in neighbors
        ]


def _normalize_phase_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "action_id": int(row.get("action_id", row.get("phase", 0))),
            "name": row["name"],
            "movement_keys": list(row.get("movements", [])),
        }
        for row in rows
    ]


def _observation_schema() -> dict[str, Any]:
    return {
        "runtime_only": True,
        "api": "TraCI lane subscriptions; no physical detector file is required",
        "decision_interval_s": SIGNAL_TIMING["decision_interval_s"],
        "cardinal_slot_order": list(DIRECTIONS),
        "per_lane_features": [
            {
                "name": "vehicle_count",
                "dtype": "float32",
                "unit": "vehicles",
                "traci_method": "lane.getLastStepVehicleNumber",
            },
            {
                "name": "queue_count",
                "dtype": "float32",
                "unit": "vehicles_below_0.1_mps",
                "traci_method": "lane.getLastStepHaltingNumber",
            },
            {
                "name": "occupancy",
                "dtype": "float32",
                "unit": "percent",
                "traci_method": "lane.getLastStepOccupancy",
            },
            {
                "name": "mean_speed",
                "dtype": "float32",
                "unit": "m/s",
                "traci_method": "lane.getLastStepMeanSpeed",
            },
            {
                "name": "waiting_time",
                "dtype": "float32",
                "unit": "vehicle_seconds",
                "traci_method": "lane.getWaitingTime",
            },
        ],
        "fixed_controller_tensor": {
            "shape": [4, 6],
            "axis_0": ["N", "E", "S", "W"],
            "axis_1": [
                "exists_mask",
                "lane_count",
                "vehicle_count_sum",
                "queue_count_sum",
                "occupancy_lane_mean",
                "speed_ratio_lane_mean",
            ],
            "speed_ratio": "mean_speed_mps / max(speed_limit_mps, 0.1)",
            "missing_approach_policy": "all zeros including exists_mask",
        },
        "controller_features": [
            {"name": "current_action_one_hot", "shape": [8]},
            {"name": "available_action_mask", "shape": [8]},
            {"name": "green_elapsed_ratio", "formula": "min(green_elapsed_s / 60, 1)"},
            {"name": "is_transition", "range": [0, 1]},
        ],
        "pressure_definition": "sum vehicle_count on observation_lane_ids minus sum vehicle_count on pressure_outgoing_lane_ids",
    }


def _reward_schema() -> dict[str, Any]:
    return {
        "runtime_only": True,
        "scope": "one scalar per controller per environment decision",
        "default_formula": "-(1.0*queue_per_lane + 0.01*waiting_seconds_per_lane + 0.25*absolute_pressure_per_lane + 0.1*phase_change)",
        "terms": [
            {
                "name": "queue_per_lane",
                "weight": -1.0,
                "formula": "sum incoming lane.getLastStepHaltingNumber / max(incoming_lane_count, 1)",
            },
            {
                "name": "waiting_seconds_per_lane",
                "weight": -0.01,
                "formula": "sum incoming lane.getWaitingTime / max(incoming_lane_count, 1)",
            },
            {
                "name": "absolute_pressure_per_lane",
                "weight": -0.25,
                "formula": "abs(sum incoming vehicle_count - sum outgoing vehicle_count) / max(incoming_lane_count, 1)",
            },
            {
                "name": "phase_change",
                "weight": -0.1,
                "formula": "1 if selected_action_id differs from previous_action_id else 0",
            },
        ],
        "team_reward_options": {
            "independent": "use local controller reward",
            "cooperative": "mean local reward over controller plus neighbor_controller_ids",
        },
    }


def _build_demand_contract(network: dict[str, Any]) -> dict[str, Any]:
    return {
        "included_in_this_file": False,
        "next_stage_input": "OD records or SUMO trips/routes",
        "origin_id_type": "sumo directed edge id",
        "destination_id_type": "sumo directed edge id",
        "boundary_origin_edge_ids": network["boundary_entry_edge_ids"],
        "boundary_destination_edge_ids": network["boundary_exit_edge_ids"],
        "record_schema": {
            "origin_edge_id": "string; must reference sumo_network.directed_edges.id",
            "destination_edge_id": "string; must reference sumo_network.directed_edges.id",
            "begin_s": "number >= 0",
            "end_s": "number > begin_s",
            "vehicles_per_hour": "number >= 0",
            "vehicle_type_id": "string; default passenger",
        },
        "route_generation_requirements": [
            "origin and destination must differ",
            "a directed route must exist",
            "generated routes must use only allowed lane vehicle classes",
        ],
    }


def _validate_payload(
    payload: dict[str, Any], controller_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    network = payload["sumo_network"]
    node_ids = {row["id"] for row in network["nodes"]}
    edge_ids = {row["id"] for row in network["directed_edges"]}
    lane_ids = {row["id"] for row in network["lanes"]}
    connection_ids = {row["id"] for row in network["connections"]}
    errors: list[str] = []
    if len(node_ids) != len(network["nodes"]):
        errors.append("duplicate SUMO node ids")
    if len(edge_ids) != len(network["directed_edges"]):
        errors.append("duplicate SUMO edge ids")
    if len(lane_ids) != len(network["lanes"]):
        errors.append("duplicate SUMO lane ids")
    if len(connection_ids) != len(network["connections"]):
        errors.append("duplicate SUMO connection ids")
    for edge in network["directed_edges"]:
        if edge["from_node_id"] not in node_ids or edge["to_node_id"] not in node_ids:
            errors.append(f"edge references missing node: {edge['id']}")
        if any(lane_id not in lane_ids for lane_id in edge["lane_ids"]):
            errors.append(f"edge references missing lane: {edge['id']}")
        if edge["num_lanes"] != len(edge["lane_ids"]):
            errors.append(f"edge lane count mismatch: {edge['id']}")
    for lane in network["lanes"]:
        if lane["edge_id"] not in edge_ids:
            errors.append(f"lane references missing edge: {lane['id']}")
        if lane["length_m"] <= 0 or lane["speed_limit_mps"] <= 0:
            errors.append(f"lane has non-positive length/speed: {lane['id']}")
        if not lane["function"]:
            errors.append(f"lane function missing: {lane['id']}")
        if any(value not in connection_ids for value in lane["allowed_connection_ids"]):
            errors.append(f"lane references missing connection: {lane['id']}")
    for connection in network["connections"]:
        if connection["from_edge_id"] not in edge_ids or connection["to_edge_id"] not in edge_ids:
            errors.append(f"connection references missing edge: {connection['id']}")
        if connection["from_lane_id"] not in lane_ids or connection["to_lane_id"] not in lane_ids:
            errors.append(f"connection references missing lane: {connection['id']}")
        if connection["turn_type"] not in {"left", "straight", "right"}:
            errors.append(f"invalid connection turn type: {connection['id']}")
    for controller in payload["controllers"]:
        links = controller["controlled_links"]
        expected = list(range(len(links)))
        actual = [row["link_index"] for row in links]
        if actual != expected:
            errors.append(f"non-contiguous link indices: {controller['id']}")
        if not any(controller["available_phase_mask"]):
            errors.append(f"controller has no valid action: {controller['id']}")
        if len(controller["available_phase_mask"]) != 8:
            errors.append(f"controller phase mask is not length 8: {controller['id']}")
        program = controller["program"]
        if len(program["all_red_state"]) != len(links):
            errors.append(f"all-red state length mismatch: {controller['id']}")
        protected: set[int] = set()
        for action in program["actions"]:
            if action["available"]:
                if len(action["green_state"]) != len(links):
                    errors.append(f"green state length mismatch: {controller['id']}")
                if len(action["yellow_state"]) != len(links):
                    errors.append(f"yellow state length mismatch: {controller['id']}")
                protected.update(action["protected_link_indices"])
        for link in links:
            if link["turn_type"] != "right" and link["link_index"] not in protected:
                errors.append(
                    f"non-right controlled link has no protected action: {controller['id']}:{link['link_index']}"
                )
        if any(value not in connection_ids for value in (row["connection_id"] for row in links)):
            errors.append(f"controller references missing connection: {controller['id']}")
    controller_ids = set(controller_by_id)
    for controller in payload["controllers"]:
        if any(value not in controller_ids for value in controller["neighbor_controller_ids"]):
            errors.append(f"controller references missing neighbor: {controller['id']}")
    detector_ids = {row["id"] for row in payload["detectors"]}
    for detector in payload["detectors"]:
        if detector["lane_id"] not in lane_ids:
            errors.append(f"detector references missing lane: {detector['id']}")
        if detector["controller_id"] not in controller_ids:
            errors.append(f"detector references missing controller: {detector['id']}")
    for controller in payload["controllers"]:
        if any(value not in detector_ids for value in controller["detector_ids"]):
            errors.append(f"controller references missing detector: {controller['id']}")

    route_graph = nx.DiGraph()
    route_graph.add_nodes_from(edge_ids)
    route_graph.add_edges_from(
        (row["from_edge_id"], row["to_edge_id"])
        for row in network["connections"]
    )
    origins = network["boundary_entry_edge_ids"]
    destinations = network["boundary_exit_edge_ids"]
    unreachable_od_pairs = 0
    for origin in origins:
        reachable = nx.descendants(route_graph, origin) | {origin}
        unreachable_od_pairs += sum(
            destination not in reachable for destination in destinations
        )
    connection_from_lanes = {
        row["from_lane_id"] for row in network["connections"]
    }
    connection_to_lanes = {
        row["to_lane_id"] for row in network["connections"]
    }
    boundary_intersection_ids = {
        str(row["topology_intersection_id"])
        for row in network["nodes"]
        if row.get("fringe") == "outer"
    }
    boundary_origin_edge_ids = {
        str(row["id"])
        for row in network["directed_edges"]
        if str(row["from_intersection_id"]) in boundary_intersection_ids
    }
    non_exit_lanes_without_connection = sum(
        row["function"] != "network_exit" and row["id"] not in connection_from_lanes
        for row in network["lanes"]
    )
    non_entry_lanes_without_connection = sum(
        row["edge_id"] not in boundary_origin_edge_ids
        and row["id"] not in connection_to_lanes
        for row in network["lanes"]
    )
    if non_exit_lanes_without_connection:
        errors.append(
            "non-exit lanes without downstream connection: "
            f"{non_exit_lanes_without_connection}"
        )
    if non_entry_lanes_without_connection:
        errors.append(
            "non-entry lanes without upstream connection: "
            f"{non_entry_lanes_without_connection}"
        )
    # Reachability is a demand-generation concern, not a structural defect of
    # the static contract: OD sampling must skip non-routable pairs at runtime.
    # Arbitrary batch tiles legitimately contain dead-end boundary branches, so
    # unreachable OD pairs remain a warning instead of a fatal error.
    warnings: list[str] = []
    if unreachable_od_pairs:
        warnings.append(f"unreachable boundary OD pairs: {unreachable_od_pairs}")
    signal_connections = [
        row for row in network["connections"] if row["controller_id"] is not None
    ]
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "node_count": len(node_ids),
            "directed_edge_count": len(edge_ids),
            "lane_count": len(lane_ids),
            "lane_connection_count": len(connection_ids),
            "signalized_connection_count": len(signal_connections),
            "controller_count": len(payload["controllers"]),
            "detector_count": len(payload["detectors"]),
            "all_lanes_have_length_speed_and_function": all(
                row["length_m"] > 0
                and row["speed_limit_mps"] > 0
                and bool(row["function"])
                for row in network["lanes"]
            ),
            "all_controlled_links_have_link_index": all(
                row["link_index"] is not None and row["sumo_tl_id"]
                for row in signal_connections
            ),
            "all_phase_state_lengths_match_link_count": not any(
                "state length mismatch" in error for error in errors
            ),
            "all_non_right_links_have_protected_action": not any(
                "has no protected action" in error for error in errors
            ),
            "boundary_origin_count": len(origins),
            "boundary_destination_count": len(destinations),
            "unreachable_boundary_od_pair_count": unreachable_od_pairs,
            "all_boundary_od_pairs_are_routable": unreachable_od_pairs == 0,
            "non_exit_lanes_without_downstream_connection": non_exit_lanes_without_connection,
            "non_entry_lanes_without_upstream_connection": non_entry_lanes_without_connection,
        },
    }


def _replacement_turns_for_unserved_lane(
    default_turns: set[str],
    available_turns: set[str],
) -> set[str]:
    """Give a lane a useful role when its template movement is absent."""
    if default_turns == {"left"}:
        preferred = ("straight",)
    elif default_turns == {"right"}:
        preferred = ("straight", "right")
    elif default_turns == {"straight"}:
        preferred = ("left",)
    elif default_turns == {"straight", "right"}:
        preferred = ("straight", "right")
    else:
        preferred = ("straight", "left", "right")

    replacement = set(preferred) & available_turns
    if replacement:
        return replacement
    for turn in ("straight", "left", "right"):
        if turn in available_turns:
            return {turn}
    return set()


def _cover_outgoing_edge_lanes(
    intersection_id: str,
    outgoing_edges: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    lane_by_id: dict[str, dict[str, Any]],
) -> None:
    """Add the minimum legal fan-out needed to cover every receiving lane."""
    turn_priority = {"straight": 0, "right": 1, "left": 2}
    for to_edge in sorted(outgoing_edges, key=lambda row: row["id"]):
        rows = [
            row
            for row in connections
            if row["via_intersection_id"] == intersection_id
            and row["to_edge_id"] == to_edge["id"]
        ]
        if not rows:
            raise RuntimeError(
                f"{intersection_id}: 出口道路 {to_edge['id']} 没有任何合法上游 connection"
            )

        covered = {int(row["to_lane_index"]) for row in rows}
        missing = sorted(set(range(int(to_edge["num_lanes"]))) - covered)
        for to_lane_index in missing:
            source = min(
                rows,
                key=lambda row: (
                    abs(int(row["to_lane_index"]) - to_lane_index),
                    turn_priority[str(row["turn_type"])],
                    str(row["from_edge_id"]),
                    int(row["from_lane_index"]),
                ),
            )
            connection_id = _sumo_id(
                "conn",
                f"{intersection_id}_{source['from_edge_id']}_"
                f"{source['from_lane_index']}_{to_edge['id']}_{to_lane_index}",
            )
            connection = {
                **source,
                "id": connection_id,
                "to_lane_id": to_edge["lane_ids"][to_lane_index],
                "to_lane_index": to_lane_index,
            }
            connections.append(connection)
            rows.append(connection)
            lane_by_id[connection["from_lane_id"]][
                "allowed_connection_ids"
            ].append(connection_id)


def _lane_function(turns: set[str]) -> str:
    lookup = {
        frozenset(): "network_exit",
        frozenset({"right"}): "right_only",
        frozenset({"straight"}): "through_only",
        frozenset({"left"}): "left_only",
        frozenset({"right", "straight"}): "through_right",
        frozenset({"left", "straight"}): "left_through",
        frozenset({"left", "right"}): "left_right",
        frozenset({"left", "straight", "right"}): "all_movements",
    }
    return lookup[frozenset(turns)]


def _target_lane_index(
    turn: str, from_index: int, from_count: int, to_count: int
) -> int:
    if turn == "right":
        return 0
    if turn == "left":
        return to_count - 1
    if from_count <= 1 or to_count <= 1:
        return 0
    return min(to_count - 1, round(from_index * (to_count - 1) / (from_count - 1)))


def _classify_edge_turn(
    incoming: dict[str, Any], outgoing: dict[str, Any]
) -> str | None:
    upstream, center = incoming["shape"][-2], incoming["shape"][-1]
    downstream = outgoing["shape"][1]
    incoming_vector = (center[0] - upstream[0], center[1] - upstream[1])
    outgoing_vector = (downstream[0] - center[0], downstream[1] - center[1])
    cross = incoming_vector[0] * outgoing_vector[1] - incoming_vector[1] * outgoing_vector[0]
    dot = incoming_vector[0] * outgoing_vector[0] + incoming_vector[1] * outgoing_vector[1]
    angle = math.degrees(math.atan2(cross, dot))
    if abs(angle) >= 150:
        return None
    if abs(angle) <= 45:
        return "straight"
    return "left" if angle > 0 else "right"


def _classify_cardinal_turn(origin: str, destination: str) -> str | None:
    delta = (DIRECTION_INDEX[destination] - DIRECTION_INDEX[origin]) % 4
    return {1: "left", 2: "straight", 3: "right"}.get(delta)


def _edge_origin_slot(edge: dict[str, Any]) -> str:
    return _cardinal_direction(_edge_origin_bearing(edge))


def _edge_origin_bearing(edge: dict[str, Any]) -> float:
    center = edge["shape"][-1]
    upstream = edge["shape"][-2]
    return math.degrees(
        math.atan2(upstream[0] - center[0], upstream[1] - center[1])
    ) % 360


def _assign_approach_slots(
    incoming_edges: list[dict[str, Any]],
) -> dict[str, str]:
    """Assign incoming approaches to deterministic cardinal slots.

    Bearings are clockwise from north.  Degree four minimizes total circular
    angular distance over every one-to-one N/E/S/W assignment.  Degree three
    first chooses the pair with the largest minor angle (closest to opposing),
    maps that pair to the better-fitting opposing axis, then maps the branch to
    the nearer remaining perpendicular slot.  Exact ties use edge-id order,
    followed by ``DIRECTIONS`` order.  Other degrees keep independent nearest
    cardinal classification.
    """
    edges = sorted(incoming_edges, key=lambda row: row["id"])
    bearings = {edge["id"]: _edge_origin_bearing(edge) for edge in edges}
    if len(edges) == 4:
        best = min(
            (
                sum(
                    _angular_distance(bearings[edge["id"]], direction)
                    for edge, direction in zip(edges, assignment)
                ),
                tuple(DIRECTION_INDEX[direction] for direction in assignment),
                assignment,
            )
            for assignment in permutations(DIRECTIONS)
        )
        return {
            edge["id"]: direction
            for edge, direction in zip(edges, best[2])
        }
    if len(edges) == 3:
        main_pair = min(
            combinations(edges, 2),
            key=lambda pair: (
                -_bearing_separation(
                    bearings[pair[0]["id"]], bearings[pair[1]["id"]]
                ),
                pair[0]["id"],
                pair[1]["id"],
            ),
        )
        first, second = main_pair
        opposing_assignments = (("N", "S"), ("S", "N"), ("E", "W"), ("W", "E"))
        main_directions = min(
            opposing_assignments,
            key=lambda assignment: (
                _angular_distance(bearings[first["id"]], assignment[0])
                + _angular_distance(bearings[second["id"]], assignment[1]),
                DIRECTION_INDEX[assignment[0]],
                DIRECTION_INDEX[assignment[1]],
            ),
        )
        main_ids = {edge["id"] for edge in main_pair}
        branch = next(edge for edge in edges if edge["id"] not in main_ids)
        branch_candidates = tuple(
            direction for direction in DIRECTIONS if direction not in main_directions
        )
        branch_direction = min(
            branch_candidates,
            key=lambda direction: (
                _angular_distance(bearings[branch["id"]], direction),
                DIRECTION_INDEX[direction],
            ),
        )
        return {
            first["id"]: main_directions[0],
            second["id"]: main_directions[1],
            branch["id"]: branch_direction,
        }
    return {edge["id"]: _edge_origin_slot(edge) for edge in edges}


def _bearing_separation(left: float, right: float) -> float:
    difference = abs(left - right) % 360
    return min(difference, 360 - difference)


def _angular_distance(bearing: float, direction: str) -> float:
    target = DIRECTION_INDEX[direction] * 90.0
    return _bearing_separation(bearing, target)


def _cardinal_direction(bearing: float) -> str:
    return DIRECTIONS[int((bearing + 45) // 90) % 4]


def _connection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        DIRECTION_INDEX[row["origin_slot"]],
        row["from_edge_id"],
        row["from_lane_index"],
        DIRECTION_INDEX[row["destination_slot"]],
        row["to_edge_id"],
        row["to_lane_index"],
    )


def _sumo_id(prefix: str, raw_id: str) -> str:
    safe = raw_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    return f"{prefix}_{safe}"
