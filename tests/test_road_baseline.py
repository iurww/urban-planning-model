from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from upmodel.state import STATIC_FEATURE_DIM, UrbanGraph
from upmodel.tasks import MAX_ACTIONS, Task
from upmodel.training.road_baseline import (
    ROAD_BASELINE_VERSION,
    RoadBaselineSample,
    build_road_baseline_manifest,
    load_road_baseline_manifest,
    road_baseline_manifest_sha256,
    road_baseline_seed,
    rollout_uniform_legal_road,
    write_road_baseline_manifest,
)


class _FakeRoadEnv:
    def __init__(self, masks: list[list[bool]], terminal: dict) -> None:
        self.masks = [torch.tensor(mask, dtype=torch.bool) for mask in masks]
        self.terminal = terminal
        self.actions: list[int] = []
        self.position = 0

    def reset(self):
        self.actions = []
        self.position = 0
        return SimpleNamespace(action_mask=self.masks[0])

    def step(self, action: int):
        if not bool(self.masks[self.position][action]):
            raise AssertionError(f"illegal action selected: {action}")
        self.actions.append(action)
        self.position += 1
        done = self.position == len(self.masks)
        if done:
            return None, float(self.terminal["reward"]), True, self.terminal["metrics"]
        return SimpleNamespace(action_mask=self.masks[self.position]), 0.0, False, {}


class RoadBaselineTest(unittest.TestCase):
    def test_seed_uses_stable_domain_separated_sha256(self) -> None:
        digest = hashlib.sha256(
            ROAD_BASELINE_VERSION.encode("ascii")
            + b"\0"
            + b"footprint:abc"
            + b"\0"
            + b"7"
        ).digest()
        expected = int.from_bytes(digest[:8], "big")

        self.assertEqual(road_baseline_seed("footprint:abc", 7), expected)
        self.assertEqual(
            road_baseline_seed("footprint:abc", 7),
            road_baseline_seed("footprint:abc", 7),
        )
        self.assertNotEqual(
            road_baseline_seed("footprint:abc", 7),
            road_baseline_seed("footprint:abc", 8),
        )

    def test_rollout_supplies_explicit_schedule_and_uses_only_legal_actions(self) -> None:
        graph = _road_graph("explicit", road_count=20)
        captured: dict[str, object] = {}
        terminal = {
            "reward": 1.5,
            "metrics": {
                "Traffic": 1.5,
                "Tdensity": 0.5,
                "Tconnectivity": 2.0,
                "Tspacing": 2.0,
            },
        }

        def factory(graph_arg, task, *, what_schedule):
            captured["graph"] = graph_arg
            captured["task"] = task
            captured["schedule"] = list(what_schedule)
            return _FakeRoadEnv(
                [
                    [False, True, False, True],
                    [True, False, False, False],
                    [False, False, True, False],
                ],
                terminal,
            )

        result = rollout_uniform_legal_road(graph, seed=123, env_factory=factory)

        self.assertIs(captured["graph"], graph)
        self.assertEqual(captured["task"], Task.ROAD)
        self.assertEqual(captured["schedule"], [0, 0, 0])
        self.assertEqual(result.steps, 3)
        self.assertFalse(result.stuck)

    def test_default_environment_does_not_read_expert_actions(self) -> None:
        graph = _road_graph("no-expert", road_count=7)
        with patch(
            "upmodel.envs.static.load_static_expert_plan",
            side_effect=AssertionError("expert actions must not be read"),
        ):
            result = rollout_uniform_legal_road(graph, seed=4)

        self.assertEqual(result.steps, 1)
        self.assertFalse(result.stuck)
        self.assertGreater(result.traffic, 0.0)

    def test_rollout_retains_raw_reward_and_each_component(self) -> None:
        graph = _road_graph("raw", road_count=7)
        terminal = {
            "reward": 0.0,
            "metrics": {
                "Traffic": 1.75,
                "Tdensity": 0.25,
                "Tconnectivity": 2.0,
                "Tspacing": 1.5,
                "stuck_before_budget": True,
            },
        }

        def factory(*args, **kwargs):
            return _FakeRoadEnv([[True, False]], terminal)

        sample = rollout_uniform_legal_road(graph, seed=9, env_factory=factory)

        self.assertEqual(sample.reward, 0.0)
        self.assertEqual(sample.traffic, 1.75)
        self.assertEqual(sample.density, 0.25)
        self.assertEqual(sample.connectivity, 2.0)
        self.assertEqual(sample.spacing, 1.5)
        self.assertTrue(sample.stuck)

    def test_duplicate_group_is_sampled_once_via_canonical_member(self) -> None:
        duplicate_a = _road_graph("duplicate-a", road_count=7)
        duplicate_b = replace(
            duplicate_a,
            tile_id="duplicate-b",
            tile_dir=Path("duplicate-b"),
        )
        validation = _road_graph("validation", road_count=7, offset=100.0)
        calls: list[tuple[str, int]] = []

        def rollout(graph, *, seed):
            calls.append((graph.tile_id, seed))
            return RoadBaselineSample(
                reward=1.0,
                traffic=1.25,
                density=0.25,
                connectivity=2.0,
                spacing=1.5,
                stuck=False,
                steps=1,
                expected_steps=1,
            )

        manifest = build_road_baseline_manifest(
            {
                "training": [duplicate_b, duplicate_a],
                "validation": [validation],
            },
            samples_per_group=3,
            rollout_fn=rollout,
            expert_fn=_fake_expert,
        )

        self.assertEqual(len(manifest["groups"]), 2)
        duplicate_rows = [
            row for row in manifest["groups"] if row["member_count"] == 2
        ]
        self.assertEqual(len(duplicate_rows), 1)
        self.assertEqual(duplicate_rows[0]["canonical_tile_id"], "duplicate-a")
        self.assertEqual(duplicate_rows[0]["uniform"]["reward"]["mean"], 1.0)
        self.assertEqual(
            duplicate_rows[0]["uniform"]["components"]["Traffic"]["mean"],
            1.25,
        )
        self.assertEqual(
            duplicate_rows[0]["uniform"]["components"]["Tdensity"]["mean"],
            0.25,
        )
        self.assertEqual(
            duplicate_rows[0]["gap"]["expert_minus_uniform_mean"], 9.0
        )
        self.assertEqual(
            sum(tile_id == "duplicate-a" for tile_id, _ in calls), 3
        )
        self.assertEqual(
            sum(tile_id == "duplicate-b" for tile_id, _ in calls), 0
        )
        self.assertEqual(len(calls), 6)

    def test_loader_rejects_aggregate_tampering(self) -> None:
        manifest = build_road_baseline_manifest(
            {
                "training": [_road_graph("training", road_count=7)],
                "validation": [
                    _road_graph("validation", road_count=7, offset=100.0)
                ],
            },
            samples_per_group=2,
            rollout_fn=_fake_rollout,
            expert_fn=_fake_expert,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "road-baseline.json"
            write_road_baseline_manifest(path, manifest)
            loaded = load_road_baseline_manifest(path)
            self.assertEqual(
                loaded["aggregate_sha256"], manifest["aggregate_sha256"]
            )

            loaded["groups"][0]["uniform"]["reward"]["mean"] += 0.5
            path.write_text(json.dumps(loaded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aggregate SHA-256 mismatch"):
                load_road_baseline_manifest(path)

    def test_loader_rejects_inconsistent_split_even_with_recomputed_sha(self) -> None:
        manifest = build_road_baseline_manifest(
            {
                "training": [_road_graph("training", road_count=7)],
                "validation": [
                    _road_graph("validation", road_count=7, offset=100.0)
                ],
            },
            samples_per_group=1,
            rollout_fn=_fake_rollout,
            expert_fn=_fake_expert,
        )
        moved_group = manifest["split_ownership"]["training"].pop()
        manifest["split_ownership"]["validation"].append(moved_group)
        manifest["aggregate_sha256"] = road_baseline_manifest_sha256(manifest)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "road-baseline.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ownership.*inconsistent"):
                load_road_baseline_manifest(path)

    def test_group_cannot_be_owned_by_both_splits(self) -> None:
        training = _road_graph("training", road_count=7)
        validation = replace(
            training,
            tile_id="validation",
            tile_dir=Path("validation"),
        )
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            build_road_baseline_manifest(
                {"training": [training], "validation": [validation]},
                samples_per_group=1,
                rollout_fn=_fake_rollout,
                expert_fn=_fake_expert,
            )

def _fake_rollout(graph, *, seed) -> RoadBaselineSample:
    value = float(seed % 11)
    return RoadBaselineSample(
        reward=value,
        traffic=value + 0.1,
        density=value + 0.2,
        connectivity=value + 0.3,
        spacing=value + 0.4,
        stuck=False,
        steps=1,
        expected_steps=1,
    )


def _fake_expert(graph, task):
    if task != Task.ROAD:
        raise AssertionError("only Road expert reward is expected")
    return 10.0, {
        "Traffic": 10.0,
        "Tdensity": 8.0,
        "Tconnectivity": 1.0,
        "Tspacing": 1.0,
    }


def _road_graph(
    tile_id: str,
    *,
    road_count: int,
    offset: float = 0.0,
) -> UrbanGraph:
    intersections = [
        {
            "id": f"i-{index}",
            "point": [offset + float(index), float(index % 2)],
            "kind": "junction" if index == 0 else "portal",
        }
        for index in range(road_count)
    ]
    roads = []
    for index in range(road_count):
        left = intersections[index]
        right = intersections[(index + 1) % road_count]
        roads.append(
            {
                "id": f"r-{index}",
                "from_intersection_id": left["id"],
                "to_intersection_id": right["id"],
                "geometry": [left["point"], right["point"]],
                "length_m": 200.0 + index,
                "highway": "residential",
                "total_lanes": 2,
                "speed_kph": 30.0,
            }
        )
    topology = {
        "metadata": {},
        "intersections": intersections,
        "roads": roads,
        "parcels": [
            {
                "id": "parcel",
                "area_m2": 10_000.0,
                "geometry": [
                    [
                        [offset - 1.0, -1.0],
                        [offset + road_count + 1.0, -1.0],
                        [offset + road_count + 1.0, 2.0],
                        [offset - 1.0, 2.0],
                        [offset - 1.0, -1.0],
                    ]
                ],
            }
        ],
    }
    entity_count = road_count + 1
    return UrbanGraph(
        tile_id=tile_id,
        tile_dir=Path(tile_id),
        features=torch.zeros(
            (entity_count, STATIC_FEATURE_DIM), dtype=torch.float32
        ),
        entity_types=torch.zeros(entity_count, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        parcel_entities=torch.tensor([0], dtype=torch.long),
        road_entities=torch.arange(1, entity_count, dtype=torch.long),
        intersection_entities=torch.empty(0, dtype=torch.long),
        signal_entities=torch.empty(0, dtype=torch.long),
        signal_action_mask=torch.empty(
            (0, MAX_ACTIONS), dtype=torch.bool
        ),
        raw_topology=topology,
        signal_meta=None,
    )


if __name__ == "__main__":
    unittest.main()
