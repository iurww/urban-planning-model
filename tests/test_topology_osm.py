from __future__ import annotations

import argparse
import unittest

from scripts.topology_osm import add_subcommand, build_metric_tile_spec
from scripts.osm import TileSpec, extract_osm_evidence, reconstruct_topology


class OsmTopologyLayoutTest(unittest.TestCase):
    def test_core_pipeline_lives_in_osm_package(self) -> None:
        self.assertEqual(TileSpec.__module__, "scripts.osm.common")
        self.assertEqual(extract_osm_evidence.__module__, "scripts.osm.extraction")
        self.assertEqual(reconstruct_topology.__module__, "scripts.osm.reconstruction")

    def test_only_one_osm_topology_subcommand_is_registered(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_subcommand(subparsers)
        self.assertEqual(set(subparsers.choices), {"batch-osm"})

    def test_projected_square_size_is_parameterized(self) -> None:
        tile = build_metric_tile_spec(
            crs="EPSG:32650",
            min_x=428_000.0,
            min_y=4_412_000.0,
            size_m=2_500.0,
            tile_id="test_tile",
            city="Test",
            column=0,
            row=0,
        )
        min_x, min_y, max_x, max_y = tile.bounds_m
        self.assertEqual(max_x - min_x, 2_500.0)
        self.assertEqual(max_y - min_y, 2_500.0)


if __name__ == "__main__":
    unittest.main()
