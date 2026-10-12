from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from upmodel.data import (
    canonical_road_fingerprint,
    group_aware_splits,
    topology_fingerprint,
    topology_group_key,
)
from upmodel.state import STATIC_FEATURE_DIM, UrbanGraph
from upmodel.tasks import MAX_ACTIONS


class GroupAwareSplitTest(unittest.TestCase):
    def test_fingerprint_ignores_identity_but_covers_policy_inputs(self) -> None:
        original = _graph("original", marker=1.0)
        duplicate = replace(original, tile_id="duplicate", tile_dir=Path("duplicate"))
        changed = replace(
            duplicate,
            features=duplicate.features.clone(),
            tile_id="changed",
        )
        changed.features[0, 0] += 1.0

        self.assertEqual(topology_fingerprint(original), topology_fingerprint(duplicate))
        self.assertNotEqual(topology_fingerprint(original), topology_fingerprint(changed))

    def test_group_key_prefers_spatial_footprint(self) -> None:
        original = _graph("original", marker=1.0)
        metadata = {
            "source_crs": "EPSG:32650",
            "local_origin_in_source_crs": [444000.0, 4428000.0],
            "tile_width_m": 4000.0,
            "tile_height_m": 4000.0,
        }
        original.raw_topology["metadata"] = metadata
        changed_policy = replace(
            original,
            tile_id="changed-policy",
            features=original.features + 1.0,
            raw_topology={"metadata": dict(metadata)},
        )
        self.assertNotEqual(
            topology_fingerprint(original), topology_fingerprint(changed_policy)
        )
        self.assertEqual(
            topology_group_key(original), topology_group_key(changed_policy)
        )

    def test_canonical_road_key_ignores_ids_order_and_direction(self) -> None:
        left = replace(
            _graph("left", marker=1.0),
            raw_topology=_road_topology(
                intersection_ids=("a", "b", "c"), reverse=False
            ),
        )
        right = replace(
            _graph("right", marker=2.0),
            raw_topology=_road_topology(
                intersection_ids=("x", "y", "z"), reverse=True
            ),
        )

        self.assertEqual(
            canonical_road_fingerprint(left),
            canonical_road_fingerprint(right),
        )

    def test_only_test_equivalents_are_excluded_from_training(self) -> None:
        shared_train = _graph("shared-train", marker=1.0)
        shared_test = replace(shared_train, tile_id="shared-test")
        pair_left = _graph("pair-left", marker=2.0)
        pair_right = replace(pair_left, tile_id="pair-right")
        singles = [_graph(f"single-{index}", marker=10.0 + index) for index in range(5)]
        final_test = [shared_test, _graph("test-only", marker=99.0)]

        splits = group_aware_splits(
            [shared_train, pair_left, pair_right, *singles],
            final_test,
            validation_topologies=3,
            seed=7,
        )

        self.assertEqual(
            [graph.tile_id for graph in splits.excluded_test_duplicates],
            ["shared-train"],
        )
        self.assertEqual(len(splits.training), 7)
        training_ids = {graph.tile_id for graph in splits.training}
        self.assertIn(pair_left.tile_id, training_ids)
        self.assertIn(pair_right.tile_id, training_ids)
        group_keys = {
            name: {topology_group_key(graph) for graph in graphs}
            for name, graphs in (
                ("training", splits.training),
                ("test", splits.test),
            )
        }
        self.assertFalse(group_keys["training"] & group_keys["test"])

    def test_all_clean_training_rows_are_kept_regardless_of_seed(self) -> None:
        pool = [_graph(f"train-{index}", marker=float(index)) for index in range(8)]
        test = [_graph("test", marker=100.0)]
        first = group_aware_splits(
            pool, test, validation_topologies=3, seed=42
        )
        second = group_aware_splits(
            list(reversed(pool)), test, validation_topologies=3, seed=42
        )
        self.assertEqual(
            {graph.tile_id for graph in first.training},
            {graph.tile_id for graph in second.training},
        )
        self.assertEqual(len(first.training), len(pool))

    def test_duplicate_final_test_is_rejected(self) -> None:
        duplicate = _graph("test-a", marker=50.0)
        with self.assertRaisesRegex(ValueError, "final test contains duplicate"):
            group_aware_splits(
                [_graph("train-a", marker=1.0), _graph("train-b", marker=2.0)],
                [duplicate, replace(duplicate, tile_id="test-b")],
                validation_topologies=1,
                seed=0,
            )


def _graph(tile_id: str, *, marker: float) -> UrbanGraph:
    features = torch.zeros((3, STATIC_FEATURE_DIM), dtype=torch.float32)
    features[0, 0] = marker
    return UrbanGraph(
        tile_id=tile_id,
        tile_dir=Path(tile_id),
        features=features,
        entity_types=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        parcel_entities=torch.tensor([0], dtype=torch.long),
        road_entities=torch.tensor([1], dtype=torch.long),
        intersection_entities=torch.tensor([2], dtype=torch.long),
        signal_entities=torch.tensor([2], dtype=torch.long),
        signal_action_mask=torch.ones((1, MAX_ACTIONS), dtype=torch.bool),
        raw_topology={},
        signal_meta=None,
    )


def _road_topology(
    *, intersection_ids: tuple[str, str, str], reverse: bool
) -> dict:
    first, second, third = intersection_ids
    intersections = [
        {"id": first, "point": [0.0, 0.0], "kind": "portal"},
        {"id": second, "point": [10.0, 0.0], "kind": "junction"},
        {"id": third, "point": [10.0, 10.0], "kind": "portal"},
    ]
    roads = [
        {
            "id": "road-1",
            "from_intersection_id": first,
            "to_intersection_id": second,
            "geometry": [[0.0, 0.0], [10.0, 0.0]],
            "length_m": 10.0,
            "highway": "primary",
            "total_lanes": 2,
            "speed_kph": 50.0,
        },
        {
            "id": "road-2",
            "from_intersection_id": second,
            "to_intersection_id": third,
            "geometry": [[10.0, 0.0], [10.0, 10.0]],
            "length_m": 10.0,
            "highway": "secondary",
            "total_lanes": 3,
            "speed_kph": 40.0,
        },
    ]
    if reverse:
        intersections.reverse()
        roads.reverse()
        for road in roads:
            road["from_intersection_id"], road["to_intersection_id"] = (
                road["to_intersection_id"],
                road["from_intersection_id"],
            )
            road["geometry"] = list(reversed(road["geometry"]))
    return {"intersections": intersections, "roads": roads}


if __name__ == "__main__":
    unittest.main()
