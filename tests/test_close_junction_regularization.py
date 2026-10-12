from __future__ import annotations

import math
import unittest
from typing import Any

from shapely.geometry import LineString, Point, box

from scripts.osm.common import TileSpec
from scripts.osm.reconstruction import regularize_close_junctions


class CloseJunctionRegularizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tile = TileSpec(
            tile_id="test_4km_c0_r0",
            city="Test",
            crs="EPSG:32650",
            size_m=4000.0,
            column=0,
            row=0,
            bounds_m=(500_000.0, 4_000_000.0, 504_000.0, 4_004_000.0),
            polygon_wgs84=box(0.0, 0.0, 1.0, 1.0),
        )

    @staticmethod
    def _intersection(
        intersection_id: str,
        point: tuple[float, float],
        *,
        kind: str = "junction",
    ) -> dict[str, Any]:
        return {
            "id": intersection_id,
            "kind": kind,
            "degree": 0,
            "point": list(point),
            "_metric": Point(point),
        }

    @staticmethod
    def _road(
        road_id: str,
        start: dict[str, Any],
        end: dict[str, Any],
    ) -> dict[str, Any]:
        geometry = [list(start["point"]), list(end["point"])]
        return {
            "id": road_id,
            "from_intersection_id": start["id"],
            "to_intersection_id": end["id"],
            "geometry": geometry,
            "length_m": LineString(geometry).length,
            "_metric": LineString(geometry),
        }

    def test_close_three_way_pair_becomes_one_four_way_junction(self) -> None:
        left = self._intersection("intersection/a", (100.0, 100.0))
        right = self._intersection("intersection/b", (130.0, 100.0))
        west = self._intersection(
            "intersection/w", (0.0, 100.0), kind="boundary_portal"
        )
        north = self._intersection(
            "intersection/n", (100.0, 200.0), kind="boundary_portal"
        )
        east = self._intersection(
            "intersection/e", (230.0, 100.0), kind="boundary_portal"
        )
        south = self._intersection(
            "intersection/s", (130.0, 0.0), kind="boundary_portal"
        )
        roads = [
            self._road("road/center", left, right),
            self._road("road/west", west, left),
            self._road("road/north", north, left),
            self._road("road/east", right, east),
            self._road("road/south", right, south),
        ]

        roads, intersections, graph, stats = regularize_close_junctions(
            roads,
            [left, right, west, north, east, south],
            self.tile,
        )

        self.assertEqual(stats["merged_pair_count"], 1)
        self.assertEqual(stats["stretched_edge_count"], 0)
        self.assertEqual(
            stats["intersection_aliases"],
            {"intersection/b": "intersection/a"},
        )
        self.assertNotIn("road/center", {road["id"] for road in roads})
        self.assertNotIn(
            "intersection/b", {intersection["id"] for intersection in intersections}
        )
        self.assertEqual(graph.degree("intersection/a"), 4)
        merged = next(
            row for row in intersections if row["id"] == "intersection/a"
        )
        self.assertEqual(merged["point"], [115.0, 100.0])
        self.assertEqual(merged["degree"], 4)
        self.assertEqual(
            (merged["_metric"].x, merged["_metric"].y),
            (500_115.0, 4_000_100.0),
        )
        positions = {row["id"]: row["point"] for row in intersections}
        for road in roads:
            self.assertEqual(
                road["geometry"][0],
                positions[road["from_intersection_id"]],
            )
            self.assertEqual(
                road["geometry"][-1],
                positions[road["to_intersection_id"]],
            )

    def test_stretch_cascades_to_newly_shortened_neighbor(self) -> None:
        left = self._intersection("intersection/a", (100.0, 100.0))
        middle = self._intersection("intersection/b", (170.0, 100.0))
        right = self._intersection("intersection/c", (260.0, 100.0))
        roads = [
            self._road("road/ab", left, middle),
            self._road("road/bc", middle, right),
        ]

        roads, intersections, graph, stats = regularize_close_junctions(
            roads,
            [left, middle, right],
            self.tile,
        )

        self.assertEqual(stats["merged_pair_count"], 0)
        self.assertEqual(stats["stretched_edge_count"], 2)
        self.assertEqual(graph.number_of_edges(), 2)
        positions = {row["id"]: row["point"] for row in intersections}
        for road in roads:
            start = positions[road["from_intersection_id"]]
            end = positions[road["to_intersection_id"]]
            center_distance = math.hypot(
                float(end[0]) - float(start[0]),
                float(end[1]) - float(start[1]),
            )
            self.assertGreaterEqual(center_distance, 100.0)
            self.assertEqual(road["geometry"][0], start)
            self.assertEqual(road["geometry"][-1], end)
            self.assertAlmostEqual(road["length_m"], center_distance, places=3)


if __name__ == "__main__":
    unittest.main()
