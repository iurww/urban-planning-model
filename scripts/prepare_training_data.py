#!/usr/bin/env python3
"""Filter candidate topologies and synchronize SFT/PPO/test selection links.

Candidates stay in the directory produced by ``batch-osm``.  This command
never copies or rewrites topology/evidence files: it writes one selection CSV,
one split report, and relative directory symlinks for the selected SFT/PPO/test
tiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GENERATOR = "scripts.prepare_training_data"
TOPOLOGY_SCHEMA_PREFIX = "abstract-urban-topology"
SELECTION_CSV_FIELDS = ("tile_id", "city", "topology_path", "split")
VALID_SPLITS = ("sft", "ppo", "test")
SELECTION_HISTORY_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _structural_pass(topology: dict[str, Any], *, max_complex_parcels: int) -> bool:
    validation = topology.get("metadata", {}).get("validation", {})
    if not isinstance(validation, dict):
        return True
    # v7's aggregate flag also rejects a whole tile when only one parcel has
    # more than 12 corners.  That is useful for cartographic regularity but too
    # strict for dataset selection.  Keep all graph/topology invariants strict
    # while allowing a small, explicitly bounded number of complex parcels.
    roads_are_valid = bool(
        validation.get(
            "all_roads_are_valid_lines_or_polylines",
            validation.get("all_roads_are_two_point_straight_segments", True),
        )
    )
    true_checks = ("all_road_endpoints_reference_intersections",)
    zero_checks = (
        "geometry_breaks_stored_as_intersections",
        "interior_geometric_crossings",
        "interior_degree_one_intersections",
        "redundant_collinear_degree_two_intersections",
        "invalid_adjacency_references",
        "invalid_parcels",
        "parcels_with_holes",
        "roads_without_adjacent_parcel",
        "roads_with_more_than_two_adjacent_parcels",
        "parcels_area_below_10000m2",
        "parcels_width_below_80m",
        "parcels_aspect_above_12",
        "triangle_parcels",
        "irregular_trapezoid_parcels",
    )
    return bool(
        roads_are_valid
        and all(bool(validation.get(key, True)) for key in true_checks)
        and all(float(validation.get(key, 0)) == 0.0 for key in zero_checks)
        and float(validation.get("parcel_coverage_ratio", 1.0)) >= 0.99999
        and float(validation.get("parcel_overlap_area_m2", 0.0)) <= 0.01
        and int(validation.get("parcels_corners_above_12", 0)) <= max_complex_parcels
    )


def _tile_dimensions_m(topology: dict[str, Any]) -> tuple[float, float]:
    """Return actual local-coordinate extents, including legacy v7 files."""
    xs: list[float] = []
    ys: list[float] = []

    def add(point: Any) -> None:
        if (
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
        ):
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    for item in topology.get("intersections", []):
        add(item.get("point"))
    for item in topology.get("roads", []):
        for point in item.get("geometry", []):
            add(point)
    for item in topology.get("parcels", []):
        for ring in item.get("geometry", []):
            for point in ring:
                add(point)
    if xs and ys:
        return max(xs) - min(xs), max(ys) - min(ys)
    metadata = topology.get("metadata", {})
    width = metadata.get("tile_width_m")
    height = metadata.get("tile_height_m")
    if width is not None and height is not None:
        return float(width), float(height)
    return 0.0, 0.0


def _is_requested_square(topology: dict[str, Any], args: argparse.Namespace) -> bool:
    width, height = _tile_dimensions_m(topology)
    expected = float(args.tile_size_m)
    tolerance = float(args.tile_size_tolerance_m)
    return (
        math.isclose(width, expected, abs_tol=tolerance)
        and math.isclose(height, expected, abs_tol=tolerance)
        and math.isclose(width, height, abs_tol=tolerance)
    )


def _read_selection_rows(selection_csv: Path) -> list[dict[str, str]]:
    if not selection_csv.is_file():
        raise FileNotFoundError(f"selection CSV not found: {selection_csv}")
    with selection_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not ({"tile_id", "tile"} & set(reader.fieldnames)):
            raise ValueError(f"{selection_csv}: CSV must contain tile_id")
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, 2):
            tile_id = str(row.get("tile_id") or row.get("tile") or "").strip()
            if not tile_id:
                continue
            if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z_.-]*", tile_id):
                raise ValueError(
                    f"{selection_csv}:{row_number}: invalid tile_id {tile_id!r}"
                )
            if tile_id not in seen:
                seen.add(tile_id)
                split = str(row.get("split") or "sft").strip().lower()
                if split not in VALID_SPLITS:
                    raise ValueError(
                        f"{selection_csv}:{row_number}: split must be one of "
                        f"{', '.join(VALID_SPLITS)}, "
                        f"got {split!r}"
                    )
                result.append(
                    {
                        "tile_id": tile_id,
                        "city": str(row.get("city") or "").strip(),
                        "topology_path": str(row.get("topology_path") or "").strip(),
                        "split": split,
                    }
                )
    return result


def _canonical_candidate_topologies(data_root: Path) -> dict[str, Path]:
    """Return one canonical topology per one-level candidate tile directory."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"selection data root not found: {data_root}")
    result: dict[str, Path] = {}
    for tile_root in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not tile_root.is_dir():
            continue
        abstract = tile_root / "abstract" / "topology.json"
        root_topology = tile_root / "topology.json"
        topology_path = abstract if abstract.is_file() else root_topology
        if topology_path.is_file():
            result[tile_root.name] = topology_path.resolve()
    return result


def _candidate_complexity(
    topology_path: Path,
    topology: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, int], list[str]]:
    """Return candidate counts and every failed pre-solve filter condition."""
    parcels = len(topology.get("parcels", []))
    roads = len(topology.get("roads", []))
    intersections = len(topology.get("intersections", []))
    junctions = sum(
        item.get("kind") == "junction" and int(item.get("degree", 0)) >= 3
        for item in topology.get("intersections", [])
    )
    counts = {
        "parcels": parcels,
        "roads": roads,
        "intersections": intersections,
        "junctions": junctions,
    }
    reasons: list[str] = []
    if not str(topology.get("schema_version", "")).startswith(TOPOLOGY_SCHEMA_PREFIX):
        reasons.append("topology_schema")
    if not _is_requested_square(topology, args):
        reasons.append("tile_size")
    if not _structural_pass(
        topology, max_complex_parcels=int(args.max_complex_parcels)
    ):
        reasons.append("structure")
    bounds = (
        ("parcels", int(args.min_parcels), int(args.max_parcels)),
        ("roads", int(args.min_roads), int(args.max_roads)),
        ("intersections", int(args.min_intersections), int(args.max_intersections)),
        ("junctions", int(args.min_junctions), int(args.max_junctions)),
    )
    for name, lower, upper in bounds:
        if not lower <= counts[name] <= upper:
            reasons.append(name)

    signals_path = topology_path.with_name("traffic_signals.json")
    if not signals_path.is_file():
        reasons.append("traffic_signals_missing")
    else:
        try:
            signals = _read_json(signals_path)
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("traffic_signals_invalid")
        else:
            topology_ids = {
                str(item.get("id")) for item in topology.get("intersections", [])
            }
            controllers = signals.get("controllers")
            if not (
                str(signals.get("schema_version", "")).startswith(
                    "abstract-traffic-signals-2"
                )
                and isinstance(signals.get("sumo_network"), dict)
                and isinstance(controllers, list)
                and any(
                    isinstance(controller, dict)
                    and str(controller.get("intersection_id")) in topology_ids
                    for controller in controllers
                )
            ):
                reasons.append("traffic_signals_contract")
    return counts, reasons


def _selection_city(tile_id: str, topology: dict[str, Any]) -> str:
    metadata = topology.get("metadata", {})
    provenance = metadata.get("dataset_provenance", {}) if isinstance(metadata, dict) else {}
    city = provenance.get("city") if isinstance(provenance, dict) else None
    if not city and isinstance(metadata, dict):
        city = metadata.get("city")
    if city:
        return str(city)
    slug = tile_id.split("_", 1)[0]
    return slug.replace("-", " ").title() or "Unknown"


def _selection_row(
    tile_id: str,
    topology_path: Path,
    selection_csv: Path,
    *,
    split: str = "sft",
) -> dict[str, str]:
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split {split!r}")
    topology = _read_json(topology_path)
    return {
        "tile_id": tile_id,
        "city": _selection_city(tile_id, topology),
        "topology_path": os.path.relpath(topology_path, selection_csv.parent.resolve()),
        "split": split,
    }


def _write_selection_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _selection_history_path(selection_csv: Path) -> Path:
    return selection_csv.with_name(f"{selection_csv.stem}.history.json")


def _read_selection_history(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = _read_json(path)
    raw_ids = payload.get("ever_selected_tile_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError(f"{path}: ever_selected_tile_ids must be a list")
    return {str(tile_id) for tile_id in raw_ids if str(tile_id).strip()}


def _write_selection_history(
    path: Path,
    tile_ids: set[str],
    *,
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SELECTION_HISTORY_SCHEMA_VERSION,
        "generator": GENERATOR,
        "data_root": str(data_root),
        "ever_selected_tile_ids": sorted(tile_ids),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _land_quota_feasible(
    topology: dict[str, Any], *, time_limit_s: float
) -> tuple[bool, str]:
    """Solve the exact six-label area-quota feasibility problem.

    This is deliberately smaller than the land expert MIQP: it proves that at
    least one integral parcel assignment satisfies every lower/upper quota,
    without spending time on service/ecology optimization.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    parcels = topology.get("parcels", [])
    areas = np.asarray([float(row.get("area_m2", 0.0)) for row in parcels])
    if len(areas) == 0 or not bool(np.isfinite(areas).all()) or bool((areas <= 0).any()):
        return False, "invalid parcel areas"
    lower = np.asarray([0.08, 0.10, 0.20, 0.08, 0.08, 0.08])
    upper = np.asarray([0.30, 0.30, 0.50, 0.20, 0.20, 0.30])
    labels = len(lower)
    variables = len(areas) * labels
    matrix = lil_matrix((len(areas) + labels, variables), dtype=float)
    lb = np.empty(len(areas) + labels, dtype=float)
    ub = np.empty(len(areas) + labels, dtype=float)
    for parcel_index in range(len(areas)):
        start = parcel_index * labels
        matrix[parcel_index, start : start + labels] = 1.0
        lb[parcel_index] = ub[parcel_index] = 1.0
    total = float(areas.sum())
    for label in range(labels):
        row = len(areas) + label
        matrix[row, label::labels] = areas
        lb[row] = lower[label] * total
        ub[row] = upper[label] * total
    result = milp(
        c=np.zeros(variables, dtype=float),
        integrality=np.ones(variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(matrix.tocsr(), lb, ub),
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    if bool(result.success):
        return True, str(result.message)
    return False, str(result.message)


def _topology_match_features(topology: dict[str, Any]) -> dict[str, float]:
    parcels = topology.get("parcels", [])
    roads = topology.get("roads", [])
    junctions = [
        row for row in topology.get("intersections", []) if row.get("kind") == "junction"
    ]
    degrees = [float(row.get("degree", 0.0)) for row in junctions]
    compactness = [float(row.get("compactness", 0.0)) for row in parcels]
    lengths = [float(row.get("length_m", 0.0)) for row in roads]
    return {
        "parcels": float(len(parcels)),
        "roads": float(len(roads)),
        "junctions": float(len(junctions)),
        "entities": float(len(parcels) + len(roads) + len(junctions)),
        "mean_junction_degree": statistics.fmean(degrees) if degrees else 0.0,
        "degree4_fraction": (
            sum(value == 4.0 for value in degrees) / len(degrees) if degrees else 0.0
        ),
        "mean_parcel_compactness": (
            statistics.fmean(compactness) if compactness else 0.0
        ),
        "mean_road_length_m": statistics.fmean(lengths) if lengths else 0.0,
        "polyline_road_fraction": (
            sum(len(row.get("geometry", [])) > 2 for row in roads) / len(roads)
            if roads
            else 0.0
        ),
    }


def _spatial_footprint_key(topology: dict[str, Any]) -> str | None:
    """Return a stable physical-tile key independent of city and tile aliases."""
    metadata = topology.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    source_crs = str(metadata.get("source_crs", "")).strip()
    origin = metadata.get("local_origin_in_source_crs")
    width = metadata.get("tile_width_m", metadata.get("tile_size_m"))
    height = metadata.get("tile_height_m", metadata.get("tile_size_m"))
    if (
        not source_crs
        or not isinstance(origin, (list, tuple))
        or len(origin) < 2
        or width is None
        or height is None
    ):
        return None
    try:
        dimensions = (
            float(origin[0]),
            float(origin[1]),
            float(width),
            float(height),
        )
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in dimensions):
        return None
    payload = {
        "source_crs": source_crs,
        "origin": list(dimensions[:2]),
        "width_m": dimensions[2],
        "height_m": dimensions[3],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "footprint:" + hashlib.sha256(encoded).hexdigest()


def _matched_train_test_rows(
    eligible: list[dict[str, Any]], *, train_count: int, test_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create morphology-matched splits while keeping footprints atomic.

    Singleton footprints supply test medoids across node-count strata. Training
    candidates are selected as whole footprint groups, with each member consuming
    one requested slot. This keeps topology/expert variants in one split.
    """
    import numpy as np

    if test_count < 1 or train_count < 1:
        raise ValueError("train_count and test_count must be positive")
    if train_count % test_count:
        raise ValueError("train_count must be divisible by test_count for matched groups")
    peers = train_count // test_count
    required = train_count + test_count
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} land-feasible topologies for {required} slots")
    ordered_rows = sorted(eligible, key=lambda row: str(row["tile_id"]))
    feature_names = list(ordered_rows[0]["features"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in ordered_rows],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized = (matrix - mean) / scale

    indices_by_footprint: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered_rows):
        footprint_key = str(row.get("footprint_key") or f"tile:{row['tile_id']}")
        indices_by_footprint[footprint_key].append(index)
    atomic_groups = [
        {
            "footprint_key": footprint_key,
            "indices": tuple(indices),
            "tile_ids": tuple(ordered_rows[index]["tile_id"] for index in indices),
        }
        for footprint_key, indices in indices_by_footprint.items()
    ]
    atomic_groups.sort(key=lambda group: group["tile_ids"])
    group_by_row_index = {
        index: group
        for group in atomic_groups
        for index in group["indices"]
    }
    singleton_indices = {
        group["indices"][0] for group in atomic_groups if len(group["indices"]) == 1
    }
    if len(singleton_indices) < test_count:
        raise ValueError(
            f"only {len(singleton_indices)} singleton footprint groups can supply "
            f"test_count={test_count}; test footprints must be singleton"
        )

    order = sorted(
        range(len(ordered_rows)),
        key=lambda index: (
            ordered_rows[index]["features"]["entities"],
            ordered_rows[index]["tile_id"],
        ),
    )
    chunks = np.array_split(np.asarray(order, dtype=int), test_count)
    test_indices: list[int] = []
    selected_test_indices: set[int] = set()
    for raw_chunk in chunks:
        chunk = [int(value) for value in raw_chunk]
        center = normalized[chunk].mean(axis=0)
        local_candidates = [
            index
            for index in chunk
            if index in singleton_indices and index not in selected_test_indices
        ]
        candidates = local_candidates or sorted(singleton_indices - selected_test_indices)
        test_index = min(
            candidates,
            key=lambda index: (
                float(np.linalg.norm(normalized[index] - center)),
                ordered_rows[index]["tile_id"],
            ),
        )
        test_indices.append(test_index)
        selected_test_indices.add(test_index)

    train_candidates: list[dict[str, Any]] = []
    test_group_keys = {
        group_by_row_index[index]["footprint_key"] for index in test_indices
    }
    for group in atomic_groups:
        if group["footprint_key"] in test_group_keys:
            continue
        test_costs = []
        for test_index in test_indices:
            cost = sum(
                float(np.linalg.norm(normalized[index] - normalized[test_index]))
                + (
                    0.20
                    if ordered_rows[index]["city"] != ordered_rows[test_index]["city"]
                    else 0.0
                )
                for index in group["indices"]
            )
            test_costs.append(cost)
        nearest_test = min(
            range(test_count),
            key=lambda index: (test_costs[index], ordered_rows[test_indices[index]]["tile_id"]),
        )
        train_candidates.append(
            {
                **group,
                "size": len(group["indices"]),
                "test_costs": tuple(test_costs),
                "nearest_test": nearest_test,
                "selection_cost": test_costs[nearest_test],
            }
        )

    # Exact-capacity 0/1 knapsack. Only one best deterministic selection is
    # retained per member count, so footprint groups can never be split.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for candidate_index, candidate in enumerate(train_candidates):
        weight = int(candidate["size"])
        if weight > train_count:
            continue
        for count, (score, chosen) in sorted(states.items(), reverse=True):
            new_count = count + weight
            if new_count > train_count:
                continue
            proposal = (score + float(candidate["selection_cost"]), chosen + (candidate_index,))
            current = states.get(new_count)
            if current is None or proposal[0] < current[0] - 1.0e-12 or (
                abs(proposal[0] - current[0]) <= 1.0e-12 and proposal[1] < current[1]
            ):
                states[new_count] = proposal
    if train_count not in states:
        size_counts: dict[int, int] = {}
        for candidate in train_candidates:
            size = int(candidate["size"])
            size_counts[size] = size_counts.get(size, 0) + 1
        nearest_counts = sorted(states, key=lambda count: (abs(count - train_count), count))[:8]
        raise ValueError(
            f"cannot fill train_count={train_count} atomically after reserving "
            f"test_count={test_count} singleton footprints; footprint group sizes="
            f"{dict(sorted(size_counts.items()))}; nearest reachable counts="
            f"{sorted(nearest_counts)}"
        )

    selected_candidate_indices = states[train_count][1]
    selected_train_groups = [train_candidates[index] for index in selected_candidate_indices]

    # Keep report groups close to the former one-test/N-peer contract. Largest
    # footprint groups are placed first so size-two expert variants consume two
    # peer slots together. If uncommon larger sizes prevent balanced bins, the
    # nearest-test assignment still truthfully reports the selected split.
    assignments: list[list[dict[str, Any]]] = [[] for _ in test_indices]
    remaining_capacity = [peers for _ in test_indices]
    balanced = True

    def assignment_priority(group: dict[str, Any]) -> tuple[Any, ...]:
        costs = sorted(float(value) for value in group["test_costs"])
        regret = costs[1] - costs[0] if len(costs) > 1 else math.inf
        return (-int(group["size"]), -regret, group["tile_ids"])

    assignment_order = sorted(
        selected_train_groups,
        key=assignment_priority,
    )
    for group in assignment_order:
        feasible_tests = [
            index
            for index in range(test_count)
            if remaining_capacity[index] >= int(group["size"])
        ]
        if not feasible_tests:
            balanced = False
            break
        assigned_test = min(
            feasible_tests,
            key=lambda index: (
                group["test_costs"][index],
                -remaining_capacity[index],
                ordered_rows[test_indices[index]]["tile_id"],
            ),
        )
        assignments[assigned_test].append(group)
        remaining_capacity[assigned_test] -= int(group["size"])
    if not balanced or any(remaining_capacity):
        assignments = [[] for _ in test_indices]
        for group in selected_train_groups:
            assignments[int(group["nearest_test"])].append(group)
        balanced = False

    selected_rows: list[dict[str, Any]] = [
        {**ordered_rows[index], "split": "test"} for index in test_indices
    ]
    for group in selected_train_groups:
        selected_rows.extend(
            {**ordered_rows[index], "split": "train"} for index in group["indices"]
        )

    groups: list[dict[str, Any]] = []
    for group_index, test_index in enumerate(test_indices):
        assigned_indices = [
            index
            for train_group in assignments[group_index]
            for index in train_group["indices"]
        ]
        test_row = ordered_rows[test_index]
        groups.append(
            {
                "group": group_index,
                "test": test_row["tile_id"],
                "test_footprint_key": group_by_row_index[test_index]["footprint_key"],
                "train": [ordered_rows[index]["tile_id"] for index in assigned_indices],
                "train_footprint_keys": [
                    train_group["footprint_key"] for train_group in assignments[group_index]
                ],
                "test_entities": test_row["features"]["entities"],
                "train_entities": [
                    ordered_rows[index]["features"]["entities"] for index in assigned_indices
                ],
            }
        )

    train_rows = [row for row in selected_rows if row["split"] == "train"]
    test_rows = [row for row in selected_rows if row["split"] == "test"]
    if len(train_rows) != train_count or len(test_rows) != test_count:
        raise AssertionError("matched split produced incorrect counts")
    split_by_footprint: dict[str, set[str]] = defaultdict(set)
    selected_ids_by_footprint: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        footprint_key = str(row.get("footprint_key") or f"tile:{row['tile_id']}")
        split_by_footprint[footprint_key].add(row["split"])
        selected_ids_by_footprint[footprint_key].add(row["tile_id"])
    if any(len(splits) != 1 for splits in split_by_footprint.values()):
        raise AssertionError("matched split divided an atomic footprint group")
    for group in atomic_groups:
        selected_ids = selected_ids_by_footprint.get(group["footprint_key"], set())
        if selected_ids and selected_ids != set(group["tile_ids"]):
            raise AssertionError("matched split partially selected an atomic footprint group")
    balance: dict[str, Any] = {}
    for name in feature_names:
        train_values = [float(row["features"][name]) for row in train_rows]
        test_values = [float(row["features"][name]) for row in test_rows]
        pooled = statistics.pstdev(train_values + test_values) or 1.0
        balance[name] = {
            "train_mean": statistics.fmean(train_values),
            "test_mean": statistics.fmean(test_values),
            "standardized_mean_difference": (
                statistics.fmean(test_values) - statistics.fmean(train_values)
            )
            / pooled,
        }
    return sorted(selected_rows, key=lambda row: (row["split"], row["tile_id"])), {
        "policy": (
            "node-count quantile singleton test medoids; atomic footprint train "
            "groups matched by morphology"
        ),
        "peer_capacity_balanced": balanced,
        "feature_balance": balance,
        "matched_groups": groups,
    }


def prepare_training_selection(args: argparse.Namespace) -> None:
    """Append one deterministic automatic selection without removing existing rows."""
    candidate_root, selection_csv, _, _, _ = _dataset_layout_paths(args)
    data_root = candidate_root.resolve()
    canonical = _canonical_candidate_topologies(data_root)
    current_rows = _read_selection_rows(selection_csv) if selection_csv.is_file() else []
    current_ids = [row["tile_id"] for row in current_rows]
    current_set = set(current_ids)
    missing = [tile_id for tile_id in current_ids if tile_id not in canonical]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"selection CSV contains {len(missing)} tiles missing below {data_root}: "
            f"{examples}; existing selections were not changed"
        )
    eligible: list[dict[str, Any]] = []
    complexity_rejected: list[dict[str, Any]] = []
    land_rejected: list[dict[str, str]] = []
    excluded = set(args.exclude_tile or [])
    print(
        f"[prepare-training-data] filtering {len(canonical)} candidate topologies "
        f"below {data_root}"
    )
    for index, (tile_id, topology_path) in enumerate(canonical.items(), 1):
        if tile_id in excluded:
            complexity_rejected.append(
                {"tile_id": tile_id, "reasons": ["excluded"], "counts": {}}
            )
            continue
        topology = _read_json(topology_path)
        counts, reasons = _candidate_complexity(topology_path, topology, args)
        if reasons:
            complexity_rejected.append(
                {"tile_id": tile_id, "reasons": reasons, "counts": counts}
            )
            continue
        feasible, reason = _land_quota_feasible(
            topology, time_limit_s=float(args.land_feasibility_time_limit)
        )
        if not feasible:
            land_rejected.append({"tile_id": tile_id, "reason": reason})
            continue
        eligible.append(
            {
                "tile_id": tile_id,
                "city": _selection_city(tile_id, topology),
                "topology_path": topology_path,
                "footprint_key": _spatial_footprint_key(topology) or f"tile:{tile_id}",
                "features": _topology_match_features(topology),
            }
        )
        if index % 50 == 0:
            print(
                f"[prepare-training-data] checked={index}/{len(canonical)} "
                f"eligible={len(eligible)}",
                flush=True,
            )
    selected, matching = _matched_train_test_rows(
        eligible,
        train_count=int(args.train_count),
        test_count=int(args.test_count),
    )
    # Automatic additions enter SFT by default. PPO membership is an explicit
    # selection decision made through the CSV/UI; this preserves the
    # append-only behavior without silently moving existing rows.
    selected = [
        {**row, "split": "sft" if row["split"] == "train" else row["split"]}
        for row in selected
    ]
    rejection_counts: dict[str, int] = {}
    for row in complexity_rejected:
        for reason in row["reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    generated_rows = [
        _selection_row(
            row["tile_id"],
            row["topology_path"],
            selection_csv,
            split=row["split"],
        )
        for row in selected
    ]
    added_rows = [
        row
        for row in generated_rows
        if row["tile_id"] not in current_set
    ]
    csv_rows = current_rows + added_rows
    final_split_counts = {
        split: sum(row["split"] == split for row in csv_rows)
        for split in VALID_SPLITS
    }
    report = {
        "schema_version": 1,
        "generator": GENERATOR,
        "data_root": str(data_root),
        "candidate_count": len(canonical),
        "train_count": int(args.train_count),
        "test_count": int(args.test_count),
        "complexity_bounds": {
            "tile_size_m": float(args.tile_size_m),
            "tile_size_tolerance_m": float(args.tile_size_tolerance_m),
            "parcels": [int(args.min_parcels), int(args.max_parcels)],
            "roads": [int(args.min_roads), int(args.max_roads)],
            "intersections": [
                int(args.min_intersections),
                int(args.max_intersections),
            ],
            "junctions": [int(args.min_junctions), int(args.max_junctions)],
            "max_complex_parcels": int(args.max_complex_parcels),
        },
        "complexity_eligible_count": len(eligible) + len(land_rejected),
        "complexity_rejection_counts": dict(sorted(rejection_counts.items())),
        "complexity_rejected": complexity_rejected,
        "land_feasible_candidates": len(eligible),
        "land_infeasible_rejected": land_rejected,
        "selection_update": {
            "policy": "append-only; preserve existing CSV rows and splits",
            "existing_count": len(current_rows),
            "generated_count": len(generated_rows),
            "added_count": len(added_rows),
            "final_count": len(csv_rows),
            "final_split_counts": final_split_counts,
        },
        **matching,
    }
    report_path = selection_csv.with_name(f"{selection_csv.stem}.split_report.json")
    if not args.dry_run:
        _write_selection_csv(selection_csv, csv_rows)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[prepare-training-data] sft={args.train_count} test={args.test_count} "
        f"complexity_rejected={len(complexity_rejected)} "
        f"land_feasible={len(eligible)} land_rejected={len(land_rejected)} "
        f"existing={len(current_rows)} added={len(added_rows)} "
        f"total={len(csv_rows)} "
        f"report={report_path}"
    )
    prepare_selection_links(args, rows=csv_rows)


def _selected_candidate_dir(data_root: Path, tile_id: str) -> Path:
    tile_root = data_root / tile_id
    abstract = tile_root / "abstract"
    if (abstract / "topology.json").is_file():
        return abstract.resolve()
    if (tile_root / "topology.json").is_file():
        return tile_root.resolve()
    raise FileNotFoundError(
        f"selected tile has no canonical topology.json: {tile_id} below {data_root}"
    )


def prepare_selection_links(
    args: argparse.Namespace,
    *,
    tile_ids: list[str] | None = None,
    rows: list[dict[str, str]] | None = None,
) -> None:
    candidate_root, selection_csv, sft_root, ppo_root, test_root = (
        _dataset_layout_paths(args)
    )
    data_root = candidate_root.resolve()
    if rows is None:
        if tile_ids is None:
            rows = _read_selection_rows(selection_csv)
        else:
            rows = [
                {"tile_id": tile_id, "split": "sft", "city": "", "topology_path": ""}
                for tile_id in tile_ids
            ]
    grouped = {
        split: [row["tile_id"] for row in rows if row.get("split", "sft") == split]
        for split in VALID_SPLITS
    }
    roots = {
        "sft": sft_root,
        "ppo": ppo_root,
        "test": test_root,
    }
    manifests = []
    for split in VALID_SPLITS:
        manifests.append(
            _sync_selection_links(
                args,
                split=split,
                tile_ids=grouped[split],
                data_root=data_root,
                output_root=roots[split],
                selection_csv=selection_csv,
            )
        )
    print(
        "[selection-link] "
        + " ".join(f"{item['split']}={item['count']}" for item in manifests)
    )


def _sync_selection_links(
    args: argparse.Namespace,
    *,
    split: str,
    tile_ids: list[str],
    data_root: Path,
    output_root: Path,
    selection_csv: Path,
) -> dict[str, Any]:
    sources = {
        tile_id: _selected_candidate_dir(data_root, tile_id) for tile_id in tile_ids
    }
    selected = set(tile_ids)
    created = 0
    unchanged = 0
    updated = 0
    removed = 0

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_dir() and not args.keep_unselected_links:
        for path in sorted(output_root.iterdir()):
            if path.name in selected or not path.is_symlink():
                continue
            print(f"[selection-link] remove {path}")
            if not args.dry_run:
                path.unlink()
            removed += 1

    manifest_rows: list[dict[str, str]] = []
    for index, tile_id in enumerate(tile_ids, 1):
        source = sources[tile_id]
        destination = output_root / tile_id
        action = "create"
        if destination.is_symlink():
            if destination.resolve() == source:
                action = "unchanged"
                unchanged += 1
            else:
                action = "update"
                updated += 1
                if not args.dry_run:
                    destination.unlink()
        elif destination.exists():
            raise FileExistsError(
                f"refusing to replace non-symlink training path: {destination}"
            )
        if action != "unchanged":
            relative_target = Path(os.path.relpath(source, destination.parent.resolve()))
            if not args.dry_run:
                destination.symlink_to(relative_target, target_is_directory=True)
            if action == "create":
                created += 1
        print(f"[{index}/{len(tile_ids)}] {action.upper()} {destination} -> {source}")
        manifest_rows.append(
            {
                "tile_id": tile_id,
                "split": split,
                "link": str(destination),
                "source": str(source),
            }
        )

    manifest = {
        "schema_version": 1,
        "split": split,
        "selection_csv": str(selection_csv.resolve()),
        "data_root": str(data_root),
        "link_output_root": str(output_root.resolve()),
        "count": len(tile_ids),
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "removed": removed,
        "tiles": manifest_rows,
    }
    manifest_path = output_root / "selection_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"[selection-link:{split}] count={len(tile_ids)} created={created} updated={updated} "
        f"unchanged={unchanged} removed={removed} manifest={manifest_path}"
    )
    return manifest


def run(args: argparse.Namespace) -> None:
    if getattr(args, "from_csv", None) is not None:
        prepare_selection_links(args)
        return
    prepare_training_selection(args)


def _dataset_layout_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve sibling artifacts from the candidate directory's dataset root."""
    candidate_root = Path(args.data_root)
    dataset_root = candidate_root.parent
    raw_selection = getattr(args, "from_csv", None)
    if raw_selection is None:
        raw_selection = getattr(args, "selection_csv", None)
    raw_sft = getattr(
        args,
        "sft_link_output_root",
        getattr(args, "link_output_root", None),
    )
    raw_ppo = getattr(args, "ppo_link_output_root", None)
    raw_test = getattr(args, "test_link_output_root", None)
    return (
        candidate_root,
        Path(raw_selection) if raw_selection is not None else dataset_root / "topology_selection.csv",
        Path(raw_sft) if raw_sft is not None else dataset_root / "sft",
        Path(raw_ppo) if raw_ppo is not None else dataset_root / "ppo",
        Path(raw_test) if raw_test is not None else dataset_root / "test",
    )


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/final"),
        help="batch-osm 生成的候选根目录；候选不会被复制或改写",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help="带 sft/ppo/test split 的选择 CSV；默认 <data-root>/../topology_selection.csv",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "仅按指定 CSV 同步 SFT/PPO/Test 符号链接和 manifest；"
            "跳过候选筛选、Land 可行性检查、自动划分及 CSV/report 写入"
        ),
    )
    parser.add_argument(
        "--link-output-root",
        "--sft-link-output-root",
        dest="sft_link_output_root",
        type=Path,
        default=argparse.SUPPRESS,
        help="SFT 符号链接输出目录；默认 <data-root>/../sft",
    )
    parser.add_argument(
        "--ppo-link-output-root",
        type=Path,
        default=argparse.SUPPRESS,
        help="PPO 符号链接输出目录；默认 <data-root>/../ppo",
    )
    parser.add_argument(
        "--test-link-output-root",
        type=Path,
        default=argparse.SUPPRESS,
        help="test 符号链接输出目录；默认 <data-root>/../test",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=80,
        help="每轮自动增加到 SFT 的数量；已有 CSV 行不计入且不会删除",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=20,
        help="每轮自动选择的 test 数；已有 CSV 行不计入且不会删除",
    )
    parser.add_argument(
        "--land-feasibility-time-limit",
        type=float,
        default=5.0,
        help="每个候选的精确 land 配额可行性 MILP 时间上限（秒）",
    )
    parser.add_argument(
        "--keep-unselected-links",
        action="store_true",
        help="不移除 link output root 中本来就不在 CSV 内的旧符号链接",
    )
    parser.add_argument(
        "--tile-size-m",
        type=float,
        default=4000.0,
        help="required width and height of every topology in local metric coordinates",
    )
    parser.add_argument(
        "--tile-size-tolerance-m",
        type=float,
        default=0.01,
        help="absolute tolerance for both topology dimensions",
    )
    parser.add_argument("--min-parcels", type=int, default=30)
    parser.add_argument("--max-parcels", type=int, default=110)
    parser.add_argument("--min-roads", type=int, default=30)
    parser.add_argument("--max-roads", type=int, default=150)
    parser.add_argument("--min-intersections", type=int, default=20)
    parser.add_argument("--max-intersections", type=int, default=110)
    parser.add_argument("--min-junctions", type=int, default=4)
    parser.add_argument("--max-junctions", type=int, default=65)
    parser.add_argument(
        "--max-complex-parcels",
        type=int,
        default=2,
        help="maximum parcels with more than 12 corners in one topology",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exclude-tile",
        action="append",
        default=[],
        metavar="TILE_ID",
        help="排除已知下游失败 tile；可重复",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter candidates, create a matched SFT/test CSV, and sync SFT/PPO/test links",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    return parser


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-training-data",
        help="筛选中等复杂度真实拓扑并准备训练数据目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
