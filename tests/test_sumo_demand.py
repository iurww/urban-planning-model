import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib

from scripts.sumo_demand import (
    PASSENGER_DECEL_MPS2,
    PASSENGER_LENGTH_M,
    PASSENGER_MIN_GAP_M,
    SAFE_STOP_SPEED_MARGIN,
    _plain_xml,
    _route_is_valid,
    _three_edge_routes_through_target,
    generate_demand,
)


def _write_line_network(tile_dir: Path, *, include_isolated_road: bool = False) -> Path:
    sumo_dir = tile_dir / "sumo"
    sumo_dir.mkdir(parents=True)

    nodes = ET.Element("nodes")
    for index in range(6):
        ET.SubElement(
            nodes,
            "node",
            {
                "id": f"n{index}",
                "x": str(index * 100),
                "y": "0",
                "type": "priority",
            },
        )
    if include_isolated_road:
        for node_id, x in (("isolated_a", 0), ("isolated_b", 100)):
            ET.SubElement(
                nodes,
                "node",
                {
                    "id": node_id,
                    "x": str(x),
                    "y": "200",
                    "type": "priority",
                },
            )
    edges = ET.Element("edges")
    directed_edges = []
    for index in range(5):
        for suffix, start, end in (
            ("fwd", index, index + 1),
            ("rev", index + 1, index),
        ):
            edge_id = f"e{index}_{suffix}"
            ET.SubElement(
                edges,
                "edge",
                {
                    "id": edge_id,
                    "from": f"n{start}",
                    "to": f"n{end}",
                    "numLanes": "1",
                    "speed": "13.89",
                },
            )
            directed_edges.append(
                {
                    "id": edge_id,
                    "topology_road_id": f"road_{index}",
                }
            )
    if include_isolated_road:
        for edge_id, start, end in (
            ("isolated_fwd", "isolated_a", "isolated_b"),
            ("isolated_rev", "isolated_b", "isolated_a"),
        ):
            ET.SubElement(
                edges,
                "edge",
                {
                    "id": edge_id,
                    "from": start,
                    "to": end,
                    "numLanes": "1",
                    "speed": "8.33",
                },
            )
            directed_edges.append(
                {
                    "id": edge_id,
                    "topology_road_id": "isolated_road",
                }
            )
    node_path = sumo_dir / "test.nod.xml"
    edge_path = sumo_dir / "test.edg.xml"
    ET.ElementTree(nodes).write(node_path, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(edges).write(edge_path, encoding="utf-8", xml_declaration=True)
    net_path = sumo_dir / "net.xml"
    subprocess.run(
        [
            sumolib.checkBinary("netconvert"),
            "--node-files",
            str(node_path),
            "--edge-files",
            str(edge_path),
            "--output-file",
            str(net_path),
            "--no-turnarounds",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    contract = {
        "sumo_network": {
            "nodes": [{"id": f"n{index}"} for index in range(6)]
            + (
                [{"id": "isolated_a"}, {"id": "isolated_b"}]
                if include_isolated_road
                else []
            ),
            "directed_edges": directed_edges,
            "connections": [{"from_edge_id": "e0_fwd", "to_edge_id": "e1_fwd"}],
            "boundary_entry_edge_ids": ["e0_fwd", "e4_rev"],
            "boundary_exit_edge_ids": ["e4_fwd", "e0_rev"],
        },
        "controllers": [{}],
    }
    (tile_dir / "traffic_signals.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return net_path


def _vehicles(routes_path: Path) -> list[ET.Element]:
    return ET.parse(routes_path).getroot().findall("vehicle")


def test_plain_xml_preserves_boundary_fringe(tmp_path: Path) -> None:
    node_path, _, _, _ = _plain_xml(
        {
            "sumo_network": {
                "nodes": [
                    {
                        "id": "boundary",
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "type": "priority",
                        "fringe": "outer",
                    },
                    {
                        "id": "interior",
                        "x_m": 1.0,
                        "y_m": 0.0,
                        "type": "priority",
                        "fringe": "default",
                    },
                ],
                "directed_edges": [],
                "connections": [],
            },
            "controllers": [],
        },
        tmp_path,
        speed=None,
        lanes=None,
    )

    nodes = {
        row.attrib["id"]: row.attrib
        for row in ET.parse(node_path).getroot().findall("node")
    }
    assert nodes["boundary"]["fringe"] == "outer"
    assert "fringe" not in nodes["interior"]


def test_demand_has_exact_boundary_od_legal_routes_and_full_coverage(
    tmp_path: Path,
) -> None:
    net_path = _write_line_network(tmp_path)
    routes_path = generate_demand(
        tmp_path, net_path, seed=31, vehicle_count=20, force=True
    )
    vehicles = _vehicles(routes_path)
    routes = [
        tuple(vehicle.find("route").attrib["edges"].split()) for vehicle in vehicles
    ]
    boundary_entries = {"e0_fwd", "e4_rev"}
    boundary_exits = {"e4_fwd", "e0_rev"}

    assert (
        sum(
            route[0] in boundary_entries and route[-1] in boundary_exits
            for route in routes
        )
        == 10
    )
    assert any(
        route[0] not in boundary_entries or route[-1] not in boundary_exits
        for route in routes
    )

    net = sumolib.net.readNet(str(net_path), withInternal=False)
    assert all(_route_is_valid(net, route) for route in routes)
    assert all(len(route) == len(set(route)) for route in routes)
    assert {edge_id.split("_")[0] for route in routes for edge_id in route} == {
        f"e{index}" for index in range(5)
    }
    assert any(len(route) >= 3 for route in routes)
    assert not all(len(route) == 2 for route in routes)

    manifest = json.loads(
        (tmp_path / "sumo" / "demand_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["boundary_od_vehicle_count"] == 10
    assert manifest["boundary_od_actual_fraction"] == 0.5
    assert manifest["arbitrary_od_vehicle_count"] == 10
    assert manifest["planned_road_coverage"] == 1.0
    assert manifest["sumo_route_validation_passed"] is True
    assert manifest["route_length_edges"]["min"] >= 3
    assert manifest["two_edge_route_fraction"] == 0.0


def test_depart_speeds_are_seeded_varied_and_safe_for_start_edge(
    tmp_path: Path,
) -> None:
    first_tile = tmp_path / "first"
    first_net = _write_line_network(first_tile)
    first_routes = generate_demand(
        first_tile, first_net, seed=47, vehicle_count=20, force=True
    )

    second_tile = tmp_path / "second"
    second_tile.mkdir()
    shutil.copytree(first_tile / "sumo", second_tile / "sumo")
    shutil.copy(first_tile / "traffic_signals.json", second_tile)
    second_routes = generate_demand(
        second_tile,
        second_tile / "sumo" / "net.xml",
        seed=47,
        vehicle_count=20,
        force=True,
    )
    assert first_routes.read_bytes() == second_routes.read_bytes()

    vehicles = _vehicles(first_routes)
    speeds = [float(vehicle.attrib["departSpeed"]) for vehicle in vehicles]
    assert len(set(speeds)) > 1
    assert all(vehicle.attrib["departSpeed"] != "max" for vehicle in vehicles)
    net = sumolib.net.readNet(str(first_net), withInternal=False)
    for vehicle, speed in zip(vehicles, speeds, strict=True):
        start_edge = vehicle.find("route").attrib["edges"].split()[0]
        edge = net.getEdge(start_edge)
        stopping_distance = max(
            edge.getLength() - PASSENGER_LENGTH_M - PASSENGER_MIN_GAP_M,
            0.0,
        )
        safe_stop_speed = SAFE_STOP_SPEED_MARGIN * math.sqrt(
            2.0 * PASSENGER_DECEL_MPS2 * stopping_distance
        )
        assert 0 < speed <= edge.getSpeed()
        assert speed <= safe_stop_speed + 1e-4

    manifest = json.loads(
        (first_tile / "sumo" / "demand_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["depart_speed_strategy"]["type"] == "seeded_per_vehicle_numeric"
    assert manifest["depart_speed_strategy"]["capped_by_safe_stop_speed"] is True
    assert manifest["depart_speed_mps"]["unique_count"] == len(set(speeds))


def test_short_coverage_route_requires_exhaustive_lane_graph_exception(
    tmp_path: Path,
) -> None:
    net_path = _write_line_network(tmp_path, include_isolated_road=True)
    generate_demand(tmp_path, net_path, seed=53, vehicle_count=20, force=True)

    net = sumolib.net.readNet(str(net_path), withInternal=False)
    assert (
        _three_edge_routes_through_target(
            net,
            net.getEdge("isolated_fwd"),
            {"e0_fwd", "e4_rev"},
            {"e4_fwd", "e0_rev"},
        )
        == []
    )
    manifest = json.loads(
        (tmp_path / "sumo" / "demand_manifest.json").read_text(encoding="utf-8")
    )
    exception = manifest["short_route_exception"]
    assert exception["unavoidable_coverage_topology_road_ids"] == ["isolated_road"]
    assert "exhaustive passenger lane-connection search" in exception["reason"]
