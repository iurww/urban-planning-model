"""Leakage-safe training/test ownership without a validation split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch

from upmodel.state import UrbanGraph


POLICY_FINGERPRINT_VERSION = "policy-input-v1"
ROAD_FINGERPRINT_VERSION = "canonical-road-v1"
GROUP_KEY_VERSION = "spatial-footprint-road-policy-v1"


@dataclass(frozen=True)
class GraphSplits:
    training: list[UrbanGraph]
    test: list[UrbanGraph]
    excluded_test_duplicates: list[UrbanGraph]

    @property
    def quarantined_training(self) -> list[UrbanGraph]:
        """Deprecated name for rows excluded as duplicates of test data."""
        return self.excluded_test_duplicates

    def manifest(self) -> dict:
        return {
            "schema_version": 2,
            "group_key_version": GROUP_KEY_VERSION,
            "policy_fingerprint_version": POLICY_FINGERPRINT_VERSION,
            "road_fingerprint_version": ROAD_FINGERPRINT_VERSION,
            "counts": {
                "training": len(self.training),
                "test": len(self.test),
                "excluded_test_duplicates": len(
                    self.excluded_test_duplicates
                ),
            },
            "training": _manifest_rows(self.training),
            "test": _manifest_rows(self.test),
            "excluded_test_duplicates": _manifest_rows(
                self.excluded_test_duplicates
            ),
        }


def topology_fingerprint(graph: UrbanGraph) -> str:
    """Hash the complete, ordered policy input while ignoring tile identity."""
    digest = hashlib.sha256(POLICY_FINGERPRINT_VERSION.encode("ascii"))
    for name, tensor in (
        ("features", graph.features),
        ("entity_types", graph.entity_types),
        ("edge_index", graph.edge_index),
        ("signal_entities", graph.signal_entities),
        ("signal_action_mask", graph.signal_action_mask),
    ):
        value = tensor.detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        digest.update(json.dumps(descriptor, sort_keys=True).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def topology_group_key(graph: UrbanGraph) -> str:
    """Prefer physical footprint, then canonical road graph, then policy input."""
    metadata = graph.raw_topology.get("metadata", {})
    origin = metadata.get("local_origin_in_source_crs")
    source_crs = str(metadata.get("source_crs", "")).strip()
    width = metadata.get("tile_width_m", metadata.get("tile_size_m"))
    height = metadata.get("tile_height_m", metadata.get("tile_size_m"))
    if (
        source_crs
        and isinstance(origin, list)
        and len(origin) >= 2
        and width is not None
        and height is not None
    ):
        footprint = {
            "source_crs": source_crs,
            "origin": [float(origin[0]), float(origin[1])],
            "width_m": float(width),
            "height_m": float(height),
        }
        encoded = json.dumps(
            footprint, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return "footprint:" + hashlib.sha256(encoded).hexdigest()
    road_fingerprint = canonical_road_fingerprint(graph)
    if road_fingerprint is not None:
        return "road:" + road_fingerprint
    return "policy:" + topology_fingerprint(graph)


def canonical_road_fingerprint(graph: UrbanGraph) -> str | None:
    """Hash a road network independent of entity IDs, order, and direction."""
    topology = graph.raw_topology
    intersections = topology.get("intersections")
    roads = topology.get("roads")
    if not isinstance(intersections, list) or not isinstance(roads, list) or not roads:
        return None
    points: dict[str, tuple[int, int, str]] = {}
    for item in intersections:
        point = item.get("point", [0.0, 0.0])
        points[str(item.get("id", ""))] = (
            _millimeters(point[0]),
            _millimeters(point[1]),
            str(item.get("kind", "junction")),
        )
    road_rows = []
    for item in roads:
        left_id = str(item.get("from_intersection_id", ""))
        right_id = str(item.get("to_intersection_id", ""))
        if left_id not in points or right_id not in points:
            return None
        endpoints = sorted((points[left_id], points[right_id]))
        raw_geometry = item.get("geometry", [])
        geometry = tuple(
            (_millimeters(point[0]), _millimeters(point[1]))
            for point in raw_geometry
        )
        reverse_geometry = tuple(reversed(geometry))
        normalized_geometry = min(geometry, reverse_geometry)
        road_rows.append(
            {
                "endpoints": endpoints,
                "geometry_mm": normalized_geometry,
                "length_mm": _millimeters(item.get("length_m", 0.0)),
                "highway": str(item.get("highway", "")),
                "total_lanes": float(item.get("total_lanes", 0.0)),
                "speed_kph": float(item.get("speed_kph", 0.0)),
            }
        )
    payload = {
        "version": ROAD_FINGERPRINT_VERSION,
        "intersections": sorted(points.values()),
        "roads": sorted(
            road_rows,
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def group_aware_splits(
    training_pool: Iterable[UrbanGraph],
    final_test: Iterable[UrbanGraph],
    *,
    validation_topologies: int | None = None,
    seed: int | None = None,
) -> GraphSplits:
    """Use all non-test training rows and exclude only test-equivalent rows.

    The two optional keyword arguments are accepted for compatibility with old
    callers. They no longer affect ownership because no validation set is made.
    """
    del validation_topologies, seed
    pool = list(training_pool)
    test = list(final_test)
    if not pool or not test:
        raise ValueError("training/test ownership requires nonempty data sets")
    _require_unique_tile_ids(pool, split="training pool")
    _require_unique_tile_ids(test, split="final test")
    overlap = {graph.tile_id for graph in pool} & {graph.tile_id for graph in test}
    if overlap:
        raise ValueError(f"train/test tile id overlap: {sorted(overlap)[:10]}")

    test_groups = _fingerprint_groups(test)
    duplicate_test = [
        [graph.tile_id for graph in group]
        for group in test_groups.values()
        if len(group) > 1
    ]
    if duplicate_test:
        raise ValueError(
            "final test contains duplicate policy topologies: "
            f"{duplicate_test[:3]}"
        )

    test_group_keys = set(test_groups)
    excluded_test_duplicates = [
        graph for graph in pool if topology_group_key(graph) in test_group_keys
    ]
    training = [
        graph for graph in pool if topology_group_key(graph) not in test_group_keys
    ]
    if not training:
        raise ValueError(
            "all training rows duplicate a test geography or equivalent topology"
        )
    return GraphSplits(
        training=training,
        test=test,
        excluded_test_duplicates=excluded_test_duplicates,
    )


def _fingerprint_groups(
    graphs: Iterable[UrbanGraph],
) -> dict[str, list[UrbanGraph]]:
    groups: dict[str, list[UrbanGraph]] = defaultdict(list)
    for graph in graphs:
        groups[topology_group_key(graph)].append(graph)
    return dict(groups)


def _require_unique_tile_ids(graphs: list[UrbanGraph], *, split: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for graph in graphs:
        if graph.tile_id in seen:
            duplicates.add(graph.tile_id)
        seen.add(graph.tile_id)
    if duplicates:
        raise ValueError(f"{split} contains duplicate tile ids: {sorted(duplicates)[:10]}")


def _manifest_rows(graphs: list[UrbanGraph]) -> list[dict[str, str]]:
    return [
        {
            "tile_id": graph.tile_id,
            "group_key": topology_group_key(graph),
            "policy_fingerprint": topology_fingerprint(graph),
            "road_fingerprint": canonical_road_fingerprint(graph) or "",
        }
        for graph in graphs
    ]


def _millimeters(value) -> int:
    return int(round(float(value) * 1000.0))
