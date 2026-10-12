"""Load the JSON topology into a framework-independent heterogeneous graph."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from upmodel.state import STATIC_FEATURE_DIM, UrbanGraph
from upmodel.tasks import MAX_ACTIONS


SCHEMA_PREFIX = "abstract-urban-topology"
HIGHWAYS = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified")
INTERSECTION_KINDS = ("junction",)


def discover_tile_dirs(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    # Training selections are one-level tile directories and may themselves
    # be directory symlinks.  pathlib.rglob does not descend into directory
    # symlinks and would also pick up intermediate evidence/topology.json.
    # Prefer canonical one-level layouts; only fall back to abstract topology
    # directories when the root has no prepared/training-ready tiles.
    topology_paths = sorted(root.glob("*/topology.json"))
    if not topology_paths and (root / "topology.json").is_file():
        topology_paths = [root / "topology.json"]
    if not topology_paths:
        topology_paths = sorted(root.glob("*/abstract/topology.json"))
    paths = [path.parent for path in topology_paths]
    if not paths:
        raise ValueError(f"no topology.json found below {root}")
    return paths


def load_graphs(
    data_root: str | Path, *, show_progress: bool = False
) -> list[UrbanGraph]:
    tile_dirs = discover_tile_dirs(data_root)
    return [
        load_urban_graph(tile_dir)
        for tile_dir in tqdm(
            tile_dirs,
            desc="加载拓扑",
            unit="个",
            disable=not show_progress,
        )
    ]


def load_urban_graph(tile_dir: str | Path) -> UrbanGraph:
    tile_dir = Path(tile_dir)
    topology = _read_json(tile_dir / "topology.json")
    schema = str(topology.get("schema_version", ""))
    if not schema.startswith(SCHEMA_PREFIX):
        raise ValueError(f"{tile_dir}: unsupported topology schema {schema!r}")
    for field in ("parcels", "roads", "intersections"):
        if not isinstance(topology.get(field), list):
            raise ValueError(f"{tile_dir}: topology.{field} must be a list")

    parcels = topology["parcels"]
    roads = topology["roads"]
    topology_intersections = topology["intersections"]
    # Boundary portals are needed to compile the road network, but they are
    # not road crossings.  Geometry breaks are accepted only for legacy files
    # and likewise never become encoder entities.
    intersections = [
        item for item in topology_intersections if item.get("kind") == "junction"
    ]
    p_map = _unique_index(parcels, "parcel", tile_dir)
    r_map = _unique_index(roads, "road", tile_dir)
    _unique_index(topology_intersections, "topology support point", tile_dir)
    i_map = _unique_index(intersections, "intersection", tile_dir)
    p_offset = 0
    r_offset = len(parcels)
    i_offset = r_offset + len(roads)

    tile_size = float(topology.get("metadata", {}).get("tile_size_m", 0.0) or 0.0)
    if tile_size <= 0.0:
        coordinates = [item.get("point", [0.0, 0.0]) for item in topology_intersections]
        tile_size = max(
            [float(value) for point in coordinates for value in point[:2]] + [1.0]
        )

    signal_meta_path = tile_dir / "signal_meta.json"
    signal_meta = _read_json(signal_meta_path) if signal_meta_path.is_file() else None
    signal_source_ids = {
        str(item.get("source_id"))
        for item in ((signal_meta or {}).get("junctions", []))
    }
    features = []
    entity_types = []
    for item in parcels:
        features.append(_parcel_features(item, tile_size))
        entity_types.append(0)
    for item in roads:
        features.append(_road_features(item, tile_size))
        entity_types.append(1)
    for item in intersections:
        features.append(
            _intersection_features(item, tile_size, str(item["id"]) in signal_source_ids)
        )
        entity_types.append(2)

    edges: set[tuple[int, int]] = set()

    def connect(left: int, right: int) -> None:
        if left == right:
            return
        edges.add((left, right))
        edges.add((right, left))

    for p_index, item in enumerate(parcels):
        for target in item.get("neighbor_parcel_ids", []):
            connect(p_offset + p_index, p_offset + _lookup(p_map, target, tile_dir))
        for target in item.get("boundary_road_ids", []):
            connect(p_offset + p_index, r_offset + _lookup(r_map, target, tile_dir))
        for target in item.get("corner_intersection_ids", []):
            if str(target) in i_map:
                connect(p_offset + p_index, i_offset + i_map[str(target)])
    for r_index, item in enumerate(roads):
        for endpoint in (item["from_intersection_id"], item["to_intersection_id"]):
            if str(endpoint) in i_map:
                connect(r_offset + r_index, i_offset + i_map[str(endpoint)])
        for target in item.get("adjacent_road_ids", []):
            connect(r_offset + r_index, r_offset + _lookup(r_map, target, tile_dir))
        for target in item.get("parcel_ids", []):
            connect(r_offset + r_index, p_offset + _lookup(p_map, target, tile_dir))
    sorted_edges = sorted(edges)
    if sorted_edges:
        edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    signal_entities, signal_mask = _signal_layout(signal_meta, i_map, i_offset, tile_dir)
    return UrbanGraph(
        tile_id=str(topology.get("metadata", {}).get("tile_id", tile_dir.name)),
        tile_dir=tile_dir,
        features=torch.tensor(features, dtype=torch.float32),
        entity_types=torch.tensor(entity_types, dtype=torch.long),
        edge_index=edge_index,
        parcel_entities=torch.arange(p_offset, r_offset, dtype=torch.long),
        road_entities=torch.arange(r_offset, i_offset, dtype=torch.long),
        intersection_entities=torch.arange(i_offset, i_offset + len(intersections), dtype=torch.long),
        signal_entities=signal_entities,
        signal_action_mask=signal_mask,
        raw_topology=topology,
        signal_meta=signal_meta,
    )


def _signal_layout(
    signal_meta: dict[str, Any] | None,
    intersection_map: dict[str, int],
    offset: int,
    tile_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    if signal_meta is None:
        return torch.empty(0, dtype=torch.long), torch.empty((0, MAX_ACTIONS), dtype=torch.bool)
    entities: list[int] = []
    masks: list[list[bool]] = []
    for junction in signal_meta.get("junctions", []):
        source_id = str(junction.get("source_id"))
        if source_id not in intersection_map:
            raise ValueError(f"{tile_dir}: signal junction {source_id!r} is absent from topology")
        raw_mask = junction.get("available_phase_mask")
        if raw_mask is not None:
            mask = [bool(value) for value in raw_mask]
            if len(mask) != MAX_ACTIONS or not any(mask):
                raise ValueError(
                    f"{tile_dir}: signal junction {source_id!r} has invalid "
                    f"canonical action mask {mask}"
                )
        else:
            # Legacy expert data stored only a compact list of raw SUMO green
            # phases. New data must preserve sparse canonical masks (for
            # example a T junction can legally use actions 0,2,3,4,5,6).
            num_green = len(junction.get("green_phases", []))
            if not 1 <= num_green <= MAX_ACTIONS:
                raise ValueError(
                    f"{tile_dir}: signal junction {source_id!r} has {num_green} green phases; "
                    f"supported range is 1..{MAX_ACTIONS}"
                )
            mask = [index < num_green for index in range(MAX_ACTIONS)]
        entities.append(offset + intersection_map[source_id])
        masks.append(mask)
    return torch.tensor(entities, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _parcel_features(item: dict[str, Any], tile_size: float) -> list[float]:
    feature = [0.0] * STATIC_FEATURE_DIM
    x, y = _polygon_centroid(item.get("geometry", []))
    feature[0:2] = [x / tile_size, y / tile_size]
    area = max(float(item.get("area_m2", 0.0)), 0.0)
    feature[2] = math.log1p(area) / max(math.log1p(tile_size * tile_size), 1.0)
    feature[3] = float(item.get("perimeter_m", 0.0)) / max(4.0 * tile_size, 1.0)
    feature[4] = float(item.get("min_rotated_width_m", 0.0)) / tile_size
    feature[5] = math.log1p(max(float(item.get("rotated_aspect_ratio", 0.0)), 0.0)) / 4.0
    feature[6] = float(item.get("compactness", 0.0))
    feature[7] = float(item.get("convexity_ratio", 0.0))
    feature[8] = float(item.get("rectangularity_ratio", 0.0))
    feature[9] = min(float(item.get("corner_count", 0.0)) / 12.0, 1.0)
    feature[10] = min(float(item.get("boundary_vertex_count", 0.0)) / 20.0, 1.0)
    feature[11] = float(item.get("min_interior_angle_deg", 0.0)) / 180.0
    feature[12] = float(item.get("max_interior_angle_deg", 0.0)) / 360.0
    feature[13] = min(len(item.get("neighbor_parcel_ids", [])) / 8.0, 1.0)
    feature[14] = min(len(item.get("boundary_road_ids", [])) / 12.0, 1.0)
    feature[15] = min(len(item.get("corner_intersection_ids", [])) / 12.0, 1.0)
    feature[16] = 1.0
    land_use = str(item.get("land_use", ""))
    source_categories = ("residential", "commercial", "industrial", "public_service", "green_open_space")
    if land_use in source_categories:
        feature[20 + source_categories.index(land_use)] = 1.0
    return feature


def _road_features(item: dict[str, Any], tile_size: float) -> list[float]:
    feature = [0.0] * STATIC_FEATURE_DIM
    geometry = item.get("geometry", [[0.0, 0.0], [0.0, 0.0]])
    first, last = geometry[0], geometry[-1]
    feature[0:2] = [((float(first[0]) + float(last[0])) * 0.5) / tile_size,
                    ((float(first[1]) + float(last[1])) * 0.5) / tile_size]
    length = max(float(item.get("length_m", 0.0)), 0.0)
    feature[2] = length / tile_size
    feature[3] = math.log1p(length) / max(math.log1p(tile_size), 1.0)
    feature[4] = min(float(item.get("total_lanes", 0.0)) / 8.0, 1.0)
    feature[5] = min(float(item.get("speed_kph", 0.0)) / 130.0, 1.0)
    feature[6] = min(len(item.get("adjacent_road_ids", [])) / 10.0, 1.0)
    feature[7] = min(len(item.get("parcel_ids", [])) / 2.0, 1.0)
    feature[8] = math.log1p(max(float(item.get("source_carriageway_count", 0.0)), 0.0)) / 5.0
    feature[17] = 1.0
    highway = str(item.get("highway", ""))
    if highway in HIGHWAYS:
        feature[20 + HIGHWAYS.index(highway)] = 1.0
    return feature


def _intersection_features(item: dict[str, Any], tile_size: float, signalized: bool) -> list[float]:
    feature = [0.0] * STATIC_FEATURE_DIM
    point = item.get("point", [0.0, 0.0])
    feature[0:2] = [float(point[0]) / tile_size, float(point[1]) / tile_size]
    feature[2] = min(float(item.get("degree", 0.0)) / 8.0, 1.0)
    feature[3] = min(len(item.get("road_ids", [])) / 8.0, 1.0)
    feature[4] = min(len(item.get("parcel_ids", [])) / 8.0, 1.0)
    feature[5] = float(signalized)
    feature[18] = 1.0
    kind = str(item.get("kind", ""))
    if kind in INTERSECTION_KINDS:
        feature[20 + INTERSECTION_KINDS.index(kind)] = 1.0
    return feature


def _polygon_centroid(rings: Iterable[Iterable[Iterable[float]]]) -> tuple[float, float]:
    rings = list(rings)
    if not rings:
        return 0.0, 0.0
    points = [list(point) for point in list(rings[0])]
    if len(points) < 3:
        return 0.0, 0.0
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for left, right in zip(points, points[1:] + points[:1]):
        cross = float(left[0]) * float(right[1]) - float(right[0]) * float(left[1])
        twice_area += cross
        x_sum += (float(left[0]) + float(right[0])) * cross
        y_sum += (float(left[1]) + float(right[1])) * cross
    if abs(twice_area) < 1.0e-12:
        return float(points[0][0]), float(points[0][1])
    return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)


def _unique_index(items: list[dict[str, Any]], label: str, tile_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, item in enumerate(items):
        entity_id = str(item.get("id", ""))
        if not entity_id or entity_id in result:
            raise ValueError(f"{tile_dir}: invalid or duplicate {label} id {entity_id!r}")
        result[entity_id] = index
    return result


def _lookup(index: dict[str, int], entity_id: Any, tile_dir: Path) -> int:
    key = str(entity_id)
    try:
        return index[key]
    except KeyError as exc:
        raise ValueError(f"{tile_dir}: dangling topology reference {key!r}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required data file is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value
