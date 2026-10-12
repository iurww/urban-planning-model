from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point

from scripts.osm.reconstruction import attach_adjacency, collapse_geometry_breaks
from scripts.osm.traffic import (
    apply_synthetic_lane_defaults,
    build_default_traffic_signals,
)
from scripts.topology_json import load_signalized_ids
from upmodel.data.topology import load_urban_graph


class SyntheticTrafficDefaultsTest(unittest.TestCase):
    def _build_signal_for_bearings(
        self,
        bearings: list[float],
        lane_counts: list[int] | None = None,
    ) -> dict[str, Any]:
        if lane_counts is not None and len(lane_counts) != len(bearings):
            raise ValueError("lane_counts must match bearings")
        center = {
            "id": "intersection/center",
            "kind": "junction",
            "degree": len(bearings),
            "point": [0.0, 0.0],
        }
        portals = []
        roads = []
        for index, bearing in enumerate(bearings):
            radians = math.radians(bearing)
            point = [
                round(100.0 * math.sin(radians), 6),
                round(100.0 * math.cos(radians), 6),
            ]
            portal = {
                "id": f"intersection/p{index}",
                "kind": "boundary_portal",
                "degree": 1,
                "point": point,
            }
            portals.append(portal)
            roads.append(
                {
                    "id": f"road/r{index}",
                    "from_intersection_id": center["id"],
                    "to_intersection_id": portal["id"],
                    "geometry": [center["point"], point],
                    "length_m": 100.0,
                    "name": None,
                    "highway": "residential",
                    "total_lanes": (
                        lane_counts[index] if lane_counts is not None else 2
                    ),
                    "speed_kph": 40.0,
                    "speed_inferred": True,
                }
            )
        apply_synthetic_lane_defaults(roads)
        return build_default_traffic_signals(
            roads, [center, *portals], tile_id="skewed_junction"
        )

    def test_geometry_break_is_folded_into_a_polyline(self) -> None:
        intersections = [
            {
                "id": "intersection/entry",
                "kind": "boundary_portal",
                "degree": 1,
                "point": [0.0, 0.0],
                "_metric": Point(0.0, 0.0),
            },
            {
                "id": "intersection/bend",
                "kind": "geometry_break",
                "degree": 2,
                "point": [50.0, 0.0],
                "_metric": Point(50.0, 0.0),
            },
            {
                "id": "intersection/crossing",
                "kind": "junction",
                "degree": 1,
                "point": [80.0, 40.0],
                "_metric": Point(80.0, 40.0),
            },
        ]
        roads = []
        for index, (start, end, geometry) in enumerate(
            [
                ("intersection/entry", "intersection/bend", [[0.0, 0.0], [50.0, 0.0]]),
                ("intersection/bend", "intersection/crossing", [[50.0, 0.0], [80.0, 40.0]]),
            ]
        ):
            roads.append(
                {
                    "id": f"road/r{index}",
                    "from_intersection_id": start,
                    "to_intersection_id": end,
                    "geometry": geometry,
                    "length_m": LineString(geometry).length,
                    "total_lanes": 2,
                    "speed_kph": 40.0,
                    "speed_inferred": True,
                    "highway": "residential",
                    "name": None,
                    "source_road_ids": [],
                    "source_osm_way_ids": [],
                    "source_corridor_ids": [],
                    "_metric": LineString(geometry),
                }
            )

        merged, retained, graph, stats = collapse_geometry_breaks(
            roads, intersections
        )

        self.assertEqual(len(merged), 1)
        self.assertIn(
            merged[0]["geometry"],
            (
                [[0.0, 0.0], [50.0, 0.0], [80.0, 40.0]],
                [[80.0, 40.0], [50.0, 0.0], [0.0, 0.0]],
            ),
        )
        self.assertEqual({row["kind"] for row in retained}, {"junction", "boundary_portal"})
        self.assertEqual(stats["geometry_breaks_removed"], 1)
        self.assertEqual(stats["polyline_roads"], 1)
        self.assertTrue(graph.has_edge("intersection/entry", "intersection/crossing"))

    def test_parallel_polylines_are_preserved_and_repair_is_idempotent(self) -> None:
        intersections = [
            {
                "id": "intersection/a",
                "kind": "junction",
                "degree": 2,
                "point": [0.0, 0.0],
                "_metric": Point(0.0, 0.0),
            },
            {
                "id": "intersection/b",
                "kind": "junction",
                "degree": 2,
                "point": [100.0, 0.0],
                "_metric": Point(100.0, 0.0),
            },
        ]
        roads = []
        for index, geometry in enumerate(
            (
                [[0.0, 0.0], [50.0, 10.0], [100.0, 0.0]],
                [[0.0, 0.0], [50.0, -10.0], [100.0, 0.0]],
            )
        ):
            roads.append(
                {
                    "id": f"road/r{index}",
                    "from_intersection_id": "intersection/a",
                    "to_intersection_id": "intersection/b",
                    "geometry": geometry,
                    "length_m": LineString(geometry).length,
                    "_metric": LineString(geometry),
                }
            )

        retained_roads, retained_intersections, graph, stats = collapse_geometry_breaks(
            roads, intersections
        )
        attach_adjacency(retained_roads, retained_intersections, [], graph)

        self.assertEqual([row["id"] for row in retained_roads], ["road/r0", "road/r1"])
        self.assertEqual(stats["geometry_breaks_removed"], 0)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            retained_intersections[0]["road_ids"], ["road/r0", "road/r1"]
        )

    def test_lane_counts_are_clamped_and_templates_are_deterministic(self) -> None:
        roads = [
            {"id": "r1", "total_lanes": 1, "speed_kph": 30.0},
            {"id": "r2", "total_lanes": 2, "speed_kph": 40.0},
            {"id": "r3", "total_lanes": 3, "speed_kph": 50.0},
            {"id": "r8", "total_lanes": 8, "speed_kph": 60.0},
        ]

        stats = apply_synthetic_lane_defaults(roads)

        self.assertEqual([road["total_lanes"] for road in roads], [2, 2, 3, 3])
        self.assertEqual([road["source_total_lanes"] for road in roads], [1, 2, 3, 8])
        self.assertEqual(stats["clamped_low"], 1)
        self.assertEqual(stats["clamped_high"], 1)
        self.assertEqual(
            [row["allowed_turn_types"] for row in roads[0]["lane_connections"]],
            [["straight", "right"], ["left"]],
        )
        self.assertEqual(
            [row["allowed_turn_types"] for row in roads[2]["lane_connections"]],
            [["straight", "right"], ["straight"], ["left"]],
        )

    def test_three_way_reassigns_unserved_lanes_and_covers_receiving_lanes(
        self,
    ) -> None:
        traffic = self._build_signal_for_bearings(
            [0.0, 90.0, 180.0],
            lane_counts=[3, 3, 3],
        )
        network = traffic["sumo_network"]
        incoming = [
            lane
            for lane in network["lanes"]
            if lane["to_intersection_id"] == "intersection/center"
        ]
        outgoing = [
            lane
            for lane in network["lanes"]
            if lane["from_intersection_id"] == "intersection/center"
        ]
        from_lanes = {
            row["from_lane_id"] for row in network["connections"]
        }
        to_lanes = {
            row["to_lane_id"] for row in network["connections"]
        }

        self.assertTrue(all(lane["id"] in from_lanes for lane in incoming))
        self.assertTrue(all(lane["id"] in to_lanes for lane in outgoing))
        reassigned = {
            (
                tuple(lane["default_allowed_turn_types"]),
                tuple(lane["allowed_turn_types"]),
            )
            for lane in incoming
        }
        self.assertIn((("left",), ("straight",)), reassigned)
        self.assertIn((("straight",), ("left",)), reassigned)
        self.assertEqual(
            traffic["validation"]["checks"][
                "non_exit_lanes_without_downstream_connection"
            ],
            0,
        )
        self.assertEqual(
            traffic["validation"]["checks"][
                "non_entry_lanes_without_upstream_connection"
            ],
            0,
        )

    def test_mismatched_cross_uses_minimal_fanout_to_cover_target_lanes(
        self,
    ) -> None:
        traffic = self._build_signal_for_bearings(
            [0.0, 90.0, 180.0, 270.0],
            lane_counts=[3, 2, 3, 3],
        )
        target_edge_id = "edge_road_r3_fwd"
        rows = [
            row
            for row in traffic["sumo_network"]["connections"]
            if row["via_intersection_id"] == "intersection/center"
            and row["to_edge_id"] == target_edge_id
        ]
        targets_by_source: dict[str, set[int]] = {}
        for row in rows:
            targets_by_source.setdefault(row["from_lane_id"], set()).add(
                int(row["to_lane_index"])
            )

        self.assertEqual(
            {int(row["to_lane_index"]) for row in rows},
            {0, 1, 2},
        )
        self.assertTrue(
            all(len(targets) < 3 for targets in targets_by_source.values())
        )

    def test_standard_cross_has_all_junction_signal_and_eight_actions(self) -> None:
        center = {"id": "intersection/center", "kind": "junction", "degree": 4, "point": [0.0, 0.0]}
        portals = [
            {"id": "intersection/n", "kind": "boundary_portal", "degree": 1, "point": [0.0, 100.0]},
            {"id": "intersection/e", "kind": "boundary_portal", "degree": 1, "point": [100.0, 0.0]},
            {"id": "intersection/s", "kind": "boundary_portal", "degree": 1, "point": [0.0, -100.0]},
            {"id": "intersection/w", "kind": "boundary_portal", "degree": 1, "point": [-100.0, 0.0]},
        ]
        roads = []
        for index, portal in enumerate(portals):
            roads.append(
                {
                    "id": f"road/r{index}",
                    "from_intersection_id": center["id"],
                    "to_intersection_id": portal["id"],
                    "geometry": [center["point"], portal["point"]],
                    "length_m": 100.0,
                    "name": None,
                    "highway": "residential",
                    "total_lanes": 2,
                    "speed_kph": 40.0,
                    "speed_inferred": True,
                }
            )
        apply_synthetic_lane_defaults(roads)

        traffic = build_default_traffic_signals(
            roads,
            [center, *portals],
            tile_id="test_cross",
        )

        self.assertEqual(traffic["signalized_intersection_ids"], [center["id"]])
        self.assertEqual(len(traffic["controllers"]), 1)
        controller = traffic["controllers"][0]
        self.assertEqual(controller["controller_source"], "abstract_default")
        self.assertEqual(controller["available_phase_mask"], [True] * 8)
        self.assertEqual(len(controller["program"]["actions"]), 8)
        network = traffic["sumo_network"]
        self.assertTrue(all(edge["num_lanes"] == 2 for edge in network["directed_edges"]))
        incoming = [
            lane
            for lane in network["lanes"]
            if lane["to_intersection_id"] == center["id"]
        ]
        self.assertEqual(len(incoming), 8)
        self.assertEqual(
            {tuple(lane["default_allowed_turn_types"]) for lane in incoming},
            {("straight", "right"), ("left",)},
        )
        self.assertEqual(
            traffic["simulation_contract"]["time_to_teleport_s"],
            600,
        )
        self.assertIn(
            "if unchanged, hold the current green/red state for all 15 s",
            traffic["rl_control_schema"]["transition_execution"],
        )
        self.assertTrue(traffic["validation"]["passed"])

    def test_skewed_four_way_uses_unique_optimal_cardinal_slots(self) -> None:
        # Bearings 10 and 40 degrees both independently round to N.
        traffic = self._build_signal_for_bearings([10.0, 40.0, 170.0, 260.0])

        controller = traffic["controllers"][0]
        slots = controller["approach_slots"]
        self.assertEqual(
            {
                road_id: direction
                for direction, slot in slots.items()
                for road_id in slot["topology_road_ids"]
            },
            {
                "road/r0": "N",
                "road/r1": "E",
                "road/r2": "S",
                "road/r3": "W",
            },
        )
        self.assertTrue(all(len(slot["approaches"]) == 1 for slot in slots.values()))
        slot_by_edge = {
            edge_id: direction
            for direction, slot in slots.items()
            for edge_id in slot["incoming_edge_ids"]
        }
        for connection in traffic["sumo_network"]["connections"]:
            if connection["via_intersection_id"] == "intersection/center":
                self.assertEqual(
                    connection["origin_slot"],
                    slot_by_edge[connection["from_edge_id"]],
                )

    def test_skewed_three_way_keeps_opposing_pair_and_branch_slots_unique(self) -> None:
        # Bearings 5 and 40 degrees both independently round to N; 5 and 190
        # form the most nearly opposing pair.
        traffic = self._build_signal_for_bearings([5.0, 40.0, 190.0])

        slots = traffic["controllers"][0]["approach_slots"]
        slot_by_road = {
            road_id: direction
            for direction, slot in slots.items()
            for road_id in slot["topology_road_ids"]
        }
        self.assertEqual(
            slot_by_road,
            {"road/r0": "N", "road/r1": "E", "road/r2": "S"},
        )
        self.assertEqual(len(set(slot_by_road.values())), 3)
        self.assertEqual(
            sum(slot["exists"] for slot in slots.values()),
            3,
        )

    def test_v2_controller_ids_are_read_by_compact_sumo_loader(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "traffic_signals.json").write_text(
                json.dumps(
                    {
                        "schema_version": "abstract-traffic-signals-2.0",
                        "controllers": [
                            {"intersection_id": "intersection/i0001"},
                            {"intersection_id": "intersection/i0002"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_signalized_ids(root),
                ["intersection/i0001", "intersection/i0002"],
            )

    def test_encoder_excludes_boundary_and_legacy_geometry_support_points(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            topology = {
                "schema_version": "abstract-urban-topology-1.1",
                "metadata": {"tile_id": "entity_filter", "tile_size_m": 100.0},
                "parcels": [],
                "roads": [
                    {
                        "id": "road/r0",
                        "from_intersection_id": "intersection/entry",
                        "to_intersection_id": "intersection/crossing",
                        "geometry": [[0.0, 0.0], [50.0, 20.0], [100.0, 50.0]],
                        "length_m": 120.0,
                        "total_lanes": 2,
                        "speed_kph": 40.0,
                        "highway": "residential",
                    }
                ],
                "intersections": [
                    {
                        "id": "intersection/entry",
                        "kind": "boundary_portal",
                        "degree": 1,
                        "point": [0.0, 0.0],
                    },
                    {
                        "id": "intersection/legacy_bend",
                        "kind": "geometry_break",
                        "degree": 2,
                        "point": [50.0, 20.0],
                    },
                    {
                        "id": "intersection/crossing",
                        "kind": "junction",
                        "degree": 3,
                        "point": [100.0, 50.0],
                    },
                ],
            }
            (root / "topology.json").write_text(json.dumps(topology), encoding="utf-8")

            graph = load_urban_graph(root)

            self.assertEqual(graph.road_entities.numel(), 1)
            self.assertEqual(graph.intersection_entities.numel(), 1)
            self.assertEqual(graph.num_entities, 2)
            self.assertEqual(graph.entity_types.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
