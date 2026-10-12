"""Abstract road, parcel, adjacency, and signal reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import polygonize, transform, unary_union
from shapely.strtree import STRtree

from scripts.osm.common import TileSpec, iter_lines, iter_polygons, normalize_name, write_json
from scripts.osm.traffic import (
    apply_synthetic_lane_defaults,
    assemble_sumo_ready_traffic,
)

def shape_metrics(geometry: Polygon) -> dict[str, float | int]:
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    sides = [Point(coordinates[index]).distance(Point(coordinates[index + 1])) for index in range(4)]
    width = min(sides) if sides else 0.0
    length = max(sides) if sides else 0.0
    perimeter = geometry.length
    return {
        "area_m2": geometry.area,
        "perimeter_m": perimeter,
        "min_rotated_width_m": width,
        "rotated_aspect_ratio": length / max(width, 1e-9),
        "compactness": 4 * math.pi * geometry.area / max(perimeter * perimeter, 1e-9),
        "hole_count": len(geometry.interiors),
        "exterior_vertex_count": len(geometry.exterior.coords),
    }

CLASS_RANK = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
}
ABSTRACT_CLASSES = set(CLASS_RANK)
PARALLEL_ANGLE_DEG = 16.0
PARALLEL_OVERLAP_RATIO = 0.70
CLOSE_JUNCTION_MERGE_DISTANCE_M = 50.0
CLOSE_JUNCTION_STRETCH_TRIGGER_DISTANCE_M = 80.0
CLOSE_JUNCTION_STRETCH_TARGET_DISTANCE_M = 100.0
CLOSE_JUNCTION_POSITION_TOLERANCE_M = 1e-6
CLOSE_JUNCTION_SERIALIZATION_CLEARANCE_M = 0.005
CLOSE_JUNCTION_MAX_STRETCH_ITERATIONS = 2_000


@dataclass
class SourceCorridor:
    corridor_id: str
    u: str
    v: str
    geometry: LineString
    name: str
    highway: str
    lanes: int
    lane_inferred: bool
    speed_kph: float
    speed_inferred: bool
    carriageway_count: int
    source_road_ids: list[str] = field(default_factory=list)
    source_osm_way_ids: list[int] = field(default_factory=list)
    merged_parallel_ids: list[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return CLASS_RANK.get(self.highway.removesuffix("_link"), 9)


def reconstruct_topology(base_output: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    roads_payload = read_json(base_output / "roads.geojson")
    intersections_payload = read_json(base_output / "intersections.geojson")
    landuse_payload = read_json(base_output / "landuse_source.geojson")
    source_traffic = read_json(base_output / "traffic_control.json")
    metadata = roads_payload["metadata"]
    tile = TileSpec.from_dict(metadata["tile"])
    crs = metadata["metric_crs"]
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    source_graph, source_stats = build_source_graph(
        roads_payload, intersections_payload, forward, tile
    )
    corridors = collapse_to_corridors(source_graph)
    corridors, parallel_stats = suppress_parallel_corridors(corridors)
    abstract_lines, reconstruction_stats = reconstruct_straight_planar_graph(
        [corridor.geometry for corridor in corridors], tile
    )
    abstract_lines, face_regularization = regularize_bad_faces(
        abstract_lines, corridors, tile
    )
    reconstruction_stats["segments_after_face_regularization"] = len(abstract_lines)
    abstract_roads, intersections, graph, geometry_stats = materialize_abstract_graph(
        abstract_lines, corridors, tile
    )
    abstract_roads, intersections, graph, close_junction_stats = (
        regularize_close_junctions(abstract_roads, intersections, tile)
    )
    lane_stats = apply_synthetic_lane_defaults(abstract_roads)
    landuse_sources = project_landuse(landuse_payload, forward, tile)
    parcels, parcel_stats = build_abstract_parcels(
        abstract_roads, intersections, landuse_sources, tile
    )
    attach_adjacency(abstract_roads, intersections, parcels, graph)

    validation = validate_topology(abstract_roads, intersections, parcels, tile)
    tile_width_m = float(tile.bounds_m[2] - tile.bounds_m[0])
    tile_height_m = float(tile.bounds_m[3] - tile.bounds_m[1])
    topology = {
        "schema_version": "abstract-urban-topology-1.1",
        "metadata": {
            "tile_id": tile.tile_id,
            "coordinate_system": "local_metric",
            "unit": "meter",
            "local_origin_in_source_crs": [tile.bounds_m[0], tile.bounds_m[1]],
            "source_crs": crs,
            "tile_size_m": tile.size_m,
            "tile_width_m": tile_width_m,
            "tile_height_m": tile_height_m,
            "tile_area_m2": tile_width_m * tile_height_m,
            "is_exact_square": math.isclose(tile_width_m, tile_height_m, abs_tol=1e-6),
            "construction": {
                "principle": "OSM is reference only; geometry is newly reconstructed",
                "intersection": "only a generated road crossing/branch with degree other than two; boundary portals remain support points and are not model intersection entities",
                "road": "one generated straight or polyline corridor between two junction/boundary support points; degree-two bends are geometry vertices, not intersections",
                "parcel": "one polygonized face enclosed by generated road segments or tile boundary",
                "road_hierarchy": {
                    "primary_skeleton": "all non-link motorway, trunk, primary and secondary corridors plus named tertiary corridors",
                    "secondary_subdivision": "named unclassified/residential street paths anchored to the primary skeleton at both ends",
                    "excluded_local_detail": "all unnamed local roads, one-sided named local branches and living-street roads",
                },
                "source_geometry_preserved": False,
            },
            "source_stats": source_stats,
            "parallel_corridor_processing": parallel_stats,
            "reconstruction_stats": reconstruction_stats,
            "face_regularization": face_regularization,
            "road_geometry_normalization": geometry_stats,
            "close_junction_regularization": close_junction_stats,
            "parcel_stats": parcel_stats,
            "synthetic_lane_defaults": lane_stats,
            "validation": validation,
        },
        "intersections": [public_record(row) for row in intersections],
        "roads": [public_record(row) for row in abstract_roads],
        "parcels": [public_record(row) for row in parcels],
    }
    write_json(output_dir / "topology.json", topology, compact=True)

    traffic = build_traffic_signals(
        source_traffic,
        intersections_payload,
        forward,
        abstract_roads,
        intersections,
        graph,
        tile,
    )
    traffic["metadata"]["topology_file"] = "topology.json"
    traffic["metadata"]["topology_sha256"] = file_sha256(output_dir / "topology.json")
    write_json(output_dir / "traffic_signals.json", traffic, compact=True)
    return {
        "output_files": ["topology.json", "traffic_signals.json"],
        "counts": {
            "intersections": len(intersections),
            "roads": len(abstract_roads),
            "parcels": len(parcels),
            "signal_controllers": len(traffic["controllers"]),
        },
        "validation": validation,
        "source_stats": source_stats,
        "parallel_stats": parallel_stats,
        "reconstruction_stats": reconstruction_stats,
        "face_regularization": face_regularization,
        "close_junction_regularization": close_junction_stats,
        "parcel_stats": parcel_stats,
        "signal_mapping": traffic["mapping_report"],
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_graph(
    roads_payload: dict[str, Any],
    intersections_payload: dict[str, Any],
    forward: Transformer,
    tile: TileSpec,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    graph = nx.MultiGraph()
    primary_endpoint_ids: set[str] = set()
    for feature in roads_payload["features"]:
        properties = feature["properties"]
        highway = properties.get("highway", "")
        base_class = highway.removesuffix("_link")
        name = properties.get("name", "").strip()
        selected_as_primary = (
            not properties.get("is_link")
            and not highway.endswith("_link")
            and (
                base_class in {"motorway", "trunk", "primary", "secondary"}
                or base_class == "tertiary" and bool(name)
            )
        )
        if selected_as_primary:
            primary_endpoint_ids.update(
                value
                for value in (
                    properties.get("from_intersection"),
                    properties.get("to_intersection"),
                )
                if value
            )
    # Local streets are promoted only as named, two-sided connectors between
    # points already anchored to the primary skeleton. Shortest paths are
    # calculated per street name, so a route cannot jump between unrelated
    # community streets merely because their OSM nodes touch.
    local_graphs: defaultdict[str, nx.Graph] = defaultdict(nx.Graph)
    for feature in roads_payload["features"]:
        properties = feature["properties"]
        highway = properties.get("highway", "")
        base_class = highway.removesuffix("_link")
        name = normalize_name({"name": properties.get("name", "")})
        u = properties.get("from_intersection")
        v = properties.get("to_intersection")
        if (
            base_class not in {"unclassified", "residential"}
            or properties.get("is_link")
            or not name
            or not u
            or not v
            or u == v
        ):
            continue
        local_graphs[name].add_edge(
            u,
            v,
            feature_id=feature["id"],
            weight=float(properties.get("logical_length_m") or 1),
        )
    promoted_local_feature_ids: set[str] = set()
    for local_graph in local_graphs.values():
        for component_nodes in nx.connected_components(local_graph):
            terminals = sorted(set(component_nodes) & primary_endpoint_ids)
            if len(terminals) < 2:
                continue
            component = local_graph.subgraph(component_nodes)
            for left, right in combinations(terminals, 2):
                path = nx.shortest_path(component, left, right, weight="weight")
                for u, v in zip(path, path[1:]):
                    promoted_local_feature_ids.add(component[u][v]["feature_id"])
    intersection_points: dict[str, Point] = {}
    for feature in intersections_payload["features"]:
        point = transform(forward.transform, shape(feature["geometry"]))
        intersection_points[feature["id"]] = point
        graph.add_node(feature["id"], point=point, source_intersection_id=feature["id"])

    selected_classes: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    terminal_count = 0
    for feature in roads_payload["features"]:
        properties = feature["properties"]
        highway = properties.get("highway", "")
        base_class = highway.removesuffix("_link")
        name = properties.get("name", "").strip()
        if properties.get("is_link") or highway.endswith("_link"):
            excluded["link_or_ramp_absorbed_as_attribute"] += 1
            continue
        if base_class not in ABSTRACT_CLASSES:
            excluded[f"non_structural:{base_class or 'missing'}"] += 1
            continue
        if base_class == "tertiary" and not name:
            excluded["unnamed_local:tertiary"] += 1
            continue
        if base_class in {"unclassified", "residential"}:
            if not name:
                excluded[f"unnamed_local:{base_class}"] += 1
                continue
            if feature["id"] not in promoted_local_feature_ids:
                excluded[f"named_local_not_two_sided:{base_class}"] += 1
                continue
        geometry = transform(forward.transform, shape(feature["geometry"])).intersection(
            tile.metric_polygon
        )
        parts = [line for line in iter_lines(geometry) if line.length >= 10]
        if not parts:
            excluded["empty_after_clip"] += 1
            continue
        representative = max(parts, key=lambda line: line.length)
        start_point = Point(representative.coords[0])
        end_point = Point(representative.coords[-1])
        source_u = properties.get("from_intersection")
        source_v = properties.get("to_intersection")
        u = source_u if source_u in intersection_points else None
        v = source_v if source_v in intersection_points else None
        if u and v:
            pass
        elif u:
            known = intersection_points[u]
            terminal_point = start_point if start_point.distance(known) > end_point.distance(known) else end_point
            v = f"terminal/{terminal_count:04d}"
            terminal_count += 1
            graph.add_node(v, point=terminal_point, source_intersection_id=None)
        elif v:
            known = intersection_points[v]
            terminal_point = start_point if start_point.distance(known) > end_point.distance(known) else end_point
            u = f"terminal/{terminal_count:04d}"
            terminal_count += 1
            graph.add_node(u, point=terminal_point, source_intersection_id=None)
        else:
            u = f"terminal/{terminal_count:04d}"
            terminal_count += 1
            v = f"terminal/{terminal_count:04d}"
            terminal_count += 1
            graph.add_node(u, point=start_point, source_intersection_id=None)
            graph.add_node(v, point=end_point, source_intersection_id=None)
        if u == v:
            excluded["self_loop"] += 1
            continue
        selected_classes[base_class] += 1
        graph.add_edge(
            u,
            v,
            key=feature["id"],
            uid=feature["id"],
            source_road_id=feature["id"],
            name=name,
            highway=base_class,
            length_m=float(properties.get("logical_length_m") or representative.length),
            lanes=int(properties.get("total_lanes") or 1),
            lane_inferred=bool(properties.get("lane_data_inferred", True)),
            speed_kph=float(properties.get("speed_kph") or 30),
            speed_inferred=bool(properties.get("speed_inferred", True)),
            carriageway_count=int(properties.get("carriageway_count") or 1),
            source_osm_way_ids=list(properties.get("member_osm_way_ids", [])),
        )
    # Remove source intersections unused after selecting the abstract hierarchy.
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    return graph, {
        "input_v4_roads": len(roads_payload["features"]),
        "selected_source_edges": graph.number_of_edges(),
        "selected_source_nodes": graph.number_of_nodes(),
        "selected_classes": dict(sorted(selected_classes.items())),
        "excluded": dict(sorted(excluded.items())),
        "artificial_boundary_terminals": terminal_count,
    }


def collapse_to_corridors(graph: nx.MultiGraph) -> list[SourceCorridor]:
    terminals = {
        node_id
        for node_id in graph.nodes
        if graph.degree(node_id) != 2
        or str(node_id).startswith("terminal/")
        or not is_semantic_continuation(graph, node_id)
    }
    used: set[str] = set()
    corridors: list[SourceCorridor] = []

    def incident(node_id: str) -> list[tuple[str, str, str, dict[str, Any]]]:
        return list(graph.edges(node_id, keys=True, data=True))

    for start in sorted(terminals):
        for first in incident(start):
            if first[3]["uid"] in used:
                continue
            current = start
            edge = first
            members: list[dict[str, Any]] = []
            while True:
                left, right, _, data = edge
                used.add(data["uid"])
                members.append(data)
                next_node = right if current == left else left
                if next_node in terminals:
                    break
                choices = [item for item in incident(next_node) if item[3]["uid"] not in used]
                if len(choices) != 1:
                    break
                current = next_node
                edge = choices[0]
            if start == next_node:
                continue
            start_point = graph.nodes[start]["point"]
            end_point = graph.nodes[next_node]["point"]
            geometry = LineString([start_point.coords[0], end_point.coords[0]])
            if geometry.length < 15:
                continue
            name_lengths: Counter[str] = Counter()
            for member in members:
                if member["name"]:
                    name_lengths[member["name"]] += member["length_m"]
            name = name_lengths.most_common(1)[0][0] if name_lengths else ""
            highway = min(
                (member["highway"] for member in members),
                key=lambda value: CLASS_RANK.get(value, 9),
            )
            corridors.append(
                SourceCorridor(
                    corridor_id=f"source-corridor/{len(corridors):04d}",
                    u=start,
                    v=next_node,
                    geometry=geometry,
                    name=name,
                    highway=highway,
                    lanes=max(member["lanes"] for member in members),
                    lane_inferred=all(member["lane_inferred"] for member in members),
                    speed_kph=max(member["speed_kph"] for member in members),
                    speed_inferred=all(member["speed_inferred"] for member in members),
                    carriageway_count=max(member["carriageway_count"] for member in members),
                    source_road_ids=sorted({member["source_road_id"] for member in members}),
                    source_osm_way_ids=sorted(
                        {
                            way_id
                            for member in members
                            for way_id in member["source_osm_way_ids"]
                        }
                    ),
                )
            )

    # Exact same-endpoint edges are different carriageways of one abstract road.
    grouped: defaultdict[tuple[str, str], list[SourceCorridor]] = defaultdict(list)
    for corridor in corridors:
        grouped[tuple(sorted((corridor.u, corridor.v)))].append(corridor)
    deduplicated: list[SourceCorridor] = []
    for members in grouped.values():
        primary = min(members, key=lambda item: (item.rank, -item.geometry.length))
        for other in members:
            if other is primary:
                continue
            merge_corridor_attributes(primary, other)
        primary.corridor_id = f"source-corridor/{len(deduplicated):04d}"
        deduplicated.append(primary)
    return deduplicated


def is_semantic_continuation(graph: nx.MultiGraph, node_id: str) -> bool:
    """Whether a degree-two source node can safely disappear into one road.

    A node is removable only when both incident source edges describe the same
    street and are almost straight through the node. This prevents two named
    streets, or a sharp bend between them, from being replaced by a fabricated
    diagonal chord.
    """
    edges = list(graph.edges(node_id, keys=True, data=True))
    if len(edges) != 2:
        return False
    left, right = edges
    left_other = left[1] if left[0] == node_id else left[0]
    right_other = right[1] if right[0] == node_id else right[0]
    if left_other == right_other:
        return False
    center = graph.nodes[node_id]["point"]
    left_point = graph.nodes[left_other]["point"]
    right_point = graph.nodes[right_other]["point"]
    left_vector = (left_point.x - center.x, left_point.y - center.y)
    right_vector = (right_point.x - center.x, right_point.y - center.y)
    norm = math.hypot(*left_vector) * math.hypot(*right_vector)
    if norm <= 1e-9:
        return False
    angle = math.degrees(
        math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    (left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1])
                    / norm,
                ),
            )
        )
    )
    if abs(180.0 - angle) > 25.0:
        return False
    left_data, right_data = left[3], right[3]
    left_name = normalize_name({"name": left_data.get("name", "")})
    right_name = normalize_name({"name": right_data.get("name", "")})
    shared_way = bool(
        set(left_data.get("source_osm_way_ids", []))
        & set(right_data.get("source_osm_way_ids", []))
    )
    if left_name or right_name:
        return bool(left_name and left_name == right_name) or shared_way
    return left_data.get("highway") == right_data.get("highway") or shared_way


def suppress_parallel_corridors(
    corridors: list[SourceCorridor],
) -> tuple[list[SourceCorridor], dict[str, Any]]:
    kept: list[SourceCorridor] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in sorted(corridors, key=lambda item: (item.rank, -item.geometry.length)):
        match: SourceCorridor | None = None
        match_evidence: tuple[float, float, float] | None = None
        for existing in kept:
            angle = line_angle_difference(candidate.geometry, existing.geometry)
            if angle > PARALLEL_ANGLE_DEG:
                continue
            threshold = parallel_distance_threshold(candidate, existing)
            if candidate.geometry.distance(existing.geometry) > threshold:
                continue
            overlap = candidate.geometry.intersection(
                existing.geometry.buffer(threshold, cap_style="flat")
            ).length / max(candidate.geometry.length, 1e-9)
            if overlap < PARALLEL_OVERLAP_RATIO:
                continue
            evidence = (overlap, -candidate.geometry.distance(existing.geometry), -existing.rank)
            if match_evidence is None or evidence > match_evidence:
                match = existing
                match_evidence = evidence
        if match is None:
            kept.append(candidate)
            continue
        overlap = match_evidence[0] if match_evidence else 0.0
        distance = candidate.geometry.distance(match.geometry)
        suppressed.append(
            {
                "source_corridor_id": candidate.corridor_id,
                "kept_corridor_id": match.corridor_id,
                "source_name": candidate.name,
                "kept_name": match.name,
                "distance_m": round(distance, 3),
                "overlap_ratio": round(overlap, 4),
            }
        )
        merge_corridor_attributes(match, candidate)
    for index, corridor in enumerate(kept):
        corridor.corridor_id = f"corridor/{index:04d}"
    return kept, {
        "input_corridors": len(corridors),
        "retained_corridors": len(kept),
        "suppressed_parallel_corridors": len(suppressed),
        "angle_threshold_deg": PARALLEL_ANGLE_DEG,
        "overlap_threshold": PARALLEL_OVERLAP_RATIO,
        "examples": suppressed[:20],
    }


def merge_corridor_attributes(target: SourceCorridor, source: SourceCorridor) -> None:
    if source.rank < target.rank:
        target.highway = source.highway
    if not target.name and source.name:
        target.name = source.name
    target.lanes = max(target.lanes, source.lanes)
    target.lane_inferred = target.lane_inferred and source.lane_inferred
    target.speed_kph = max(target.speed_kph, source.speed_kph)
    target.speed_inferred = target.speed_inferred and source.speed_inferred
    target.carriageway_count += source.carriageway_count
    target.source_road_ids = sorted(set(target.source_road_ids + source.source_road_ids))
    target.source_osm_way_ids = sorted(
        set(target.source_osm_way_ids + source.source_osm_way_ids)
    )
    target.merged_parallel_ids.append(source.corridor_id)


def parallel_distance_threshold(left: SourceCorridor, right: SourceCorridor) -> float:
    left_name = normalize_name({"name": left.name})
    right_name = normalize_name({"name": right.name})
    if left_name and left_name == right_name:
        return 75.0
    if min(left.rank, right.rank) <= 1:
        return 115.0
    if "辅路" in left.name or "辅路" in right.name:
        return 100.0
    if min(left.rank, right.rank) == 2:
        return 55.0
    return 28.0


def line_angle(line: LineString) -> float:
    start, end = line.coords[0], line.coords[-1]
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180


def line_angle_difference(left: LineString, right: LineString) -> float:
    difference = abs(line_angle(left) - line_angle(right))
    return min(difference, 180 - difference)


def reconstruct_straight_planar_graph(
    source_lines: list[LineString], tile: TileSpec
) -> tuple[list[LineString], dict[str, Any]]:
    current = [line.intersection(tile.metric_polygon) for line in source_lines]
    current = [part for geometry in current for part in iter_lines(geometry) if part.length >= 15]
    noded_parts = [
        LineString([part.coords[0], part.coords[-1]])
        for part in iter_lines(unary_union(current))
        if part.length >= 5
    ]
    final_graph, coordinates = geometry_graph(noded_parts)
    final_lines = maximal_collinear_chords(final_graph, coordinates)
    fallback = count_interior_crossings(final_lines) > 0
    if fallback:
        final_lines = noded_parts
    final_lines = deduplicate_lines(final_lines)
    before_pruning = len(final_lines)
    final_lines, pruning_iterations = prune_interior_dangling_branches(
        final_lines, tile
    )
    after_pruning = len(final_lines)
    pruned_graph, pruned_coordinates = geometry_graph(final_lines)
    collapsed_after_pruning = maximal_collinear_chords(
        pruned_graph, pruned_coordinates
    )
    post_pruning_collapse_used = bool(
        collapsed_after_pruning
        and count_interior_crossings(collapsed_after_pruning) == 0
    )
    if post_pruning_collapse_used:
        final_lines = collapsed_after_pruning
    return final_lines, {
        "input_corridor_lines": len(source_lines),
        "output_straight_segments": len(final_lines),
        "noded_segments_before_collinear_collapse": len(noded_parts),
        "collapse_policy": "only nearly-collinear degree-two nodes; never arbitrary endpoint chording",
        "noded_fallback_used": fallback,
        "remaining_interior_crossings": count_interior_crossings(final_lines),
        "interior_dangling_segments_removed": before_pruning - after_pruning,
        "degree_two_segments_collapsed_after_pruning": after_pruning - len(final_lines),
        "dangling_pruning_iterations": pruning_iterations,
        "degree_two_nodes_collapsed_after_pruning": post_pruning_collapse_used,
    }


def point_key(coordinate: tuple[float, float]) -> tuple[float, float]:
    return (round(coordinate[0], 3), round(coordinate[1], 3))


def geometry_graph(
    lines: list[LineString],
) -> tuple[nx.Graph, dict[tuple[float, float], tuple[float, float]]]:
    graph = nx.Graph()
    coordinates: dict[tuple[float, float], tuple[float, float]] = {}
    for line in lines:
        start = point_key(line.coords[0])
        end = point_key(line.coords[-1])
        if start == end:
            continue
        coordinates[start] = tuple(line.coords[0])
        coordinates[end] = tuple(line.coords[-1])
        length = Point(line.coords[0]).distance(Point(line.coords[-1]))
        if graph.has_edge(start, end):
            continue
        graph.add_edge(start, end, length=length)
    return graph, coordinates


def maximal_collinear_chords(
    graph: nx.Graph,
    coordinates: dict[tuple[float, float], tuple[float, float]],
    tolerance_deg: float = 8.0,
) -> list[LineString]:
    """Collapse only redundant, nearly-collinear degree-two geometry nodes."""

    def is_collinear(node: tuple[float, float]) -> bool:
        if graph.degree(node) != 2:
            return False
        left, right = list(graph.neighbors(node))
        center_xy = coordinates[node]
        left_xy = coordinates[left]
        right_xy = coordinates[right]
        left_vector = (left_xy[0] - center_xy[0], left_xy[1] - center_xy[1])
        right_vector = (right_xy[0] - center_xy[0], right_xy[1] - center_xy[1])
        norm = math.hypot(*left_vector) * math.hypot(*right_vector)
        if norm <= 1e-9:
            return False
        angle = math.degrees(
            math.acos(
                max(
                    -1.0,
                    min(
                        1.0,
                        (left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1])
                        / norm,
                    ),
                )
            )
        )
        return abs(180.0 - angle) <= tolerance_deg

    terminals = {node for node in graph.nodes if not is_collinear(node)}
    used: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    chords: list[LineString] = []

    def edge_key(left: tuple[float, float], right: tuple[float, float]) -> tuple[Any, Any]:
        return tuple(sorted((left, right)))

    for start in sorted(terminals):
        for first_neighbor in sorted(graph.neighbors(start)):
            if edge_key(start, first_neighbor) in used:
                continue
            previous, current = start, first_neighbor
            used.add(edge_key(previous, current))
            while current not in terminals:
                choices = [neighbor for neighbor in graph.neighbors(current) if neighbor != previous]
                if len(choices) != 1:
                    break
                next_node = choices[0]
                used.add(edge_key(current, next_node))
                previous, current = current, next_node
            if start != current:
                line = LineString([coordinates[start], coordinates[current]])
                if line.length >= 5:
                    chords.append(line)
    return deduplicate_lines(chords)


def deduplicate_lines(lines: list[LineString]) -> list[LineString]:
    result: dict[tuple[Any, Any], LineString] = {}
    for line in lines:
        start, end = point_key(line.coords[0]), point_key(line.coords[-1])
        key = tuple(sorted((start, end)))
        if key not in result or line.length > result[key].length:
            result[key] = line
    return list(result.values())


def count_interior_crossings(lines: list[LineString]) -> int:
    if not lines:
        return 0
    tree = STRtree(lines)
    count = 0
    for index, line in enumerate(lines):
        endpoints = {point_key(line.coords[0]), point_key(line.coords[-1])}
        for value in tree.query(line):
            other_index = int(value)
            if other_index <= index:
                continue
            intersection = line.intersection(lines[other_index])
            if intersection.geom_type == "Point":
                other_endpoints = {
                    point_key(lines[other_index].coords[0]),
                    point_key(lines[other_index].coords[-1]),
                }
                key = point_key(intersection.coords[0])
                if key not in endpoints or key not in other_endpoints:
                    count += 1
            elif not intersection.is_empty and intersection.length > 0.01:
                count += 1
    return count


def prune_interior_dangling_branches(
    lines: list[LineString], tile: TileSpec
) -> tuple[list[LineString], int]:
    """Remove branches ending inside the tile that cannot bound any parcel.

    Boundary portals remain legal degree-one nodes. Interior degree-one nodes
    are source-detail artefacts in this abstraction: they neither represent a
    road-to-road intersection nor participate in a polygonized face.
    """
    current = list(lines)
    iterations = 0
    while current:
        graph, coordinates = geometry_graph(current)
        interior_leaves = {
            node
            for node in graph.nodes
            if graph.degree(node) == 1
            and Point(coordinates[node]).distance(tile.metric_polygon.boundary) > 1.0
        }
        if not interior_leaves:
            break
        current = [
            line
            for line in current
            if point_key(line.coords[0]) not in interior_leaves
            and point_key(line.coords[-1]) not in interior_leaves
        ]
        iterations += 1
    return deduplicate_lines(current), iterations


def normalize_line_endpoints(
    lines: list[LineString], tile: TileSpec
) -> list[LineString]:
    """Quantize endpoints and snap boundary portals exactly onto the tile."""
    min_x, min_y, max_x, max_y = tile.bounds_m

    def normalized(coordinate: tuple[float, float]) -> tuple[float, float]:
        x, y = round(coordinate[0], 3), round(coordinate[1], 3)
        if abs(x - min_x) <= 1.0:
            x = min_x
        elif abs(x - max_x) <= 1.0:
            x = max_x
        if abs(y - min_y) <= 1.0:
            y = min_y
        elif abs(y - max_y) <= 1.0:
            y = max_y
        return x, y

    normalized_lines = []
    for line in lines:
        start, end = normalized(line.coords[0]), normalized(line.coords[-1])
        if start != end:
            normalized_lines.append(LineString([start, end]))
    return deduplicate_lines(normalized_lines)


def regularize_bad_faces(
    lines: list[LineString], corridors: list[SourceCorridor], tile: TileSpec
) -> tuple[list[LineString], dict[str, Any]]:
    """Remove abstract edges that create sliver or undersized training faces.

    This is graph editing, not polygon repair: each rejected face is merged by
    deleting one generated road edge, after which intersections and straight
    segments are reconstructed again.
    """
    current = normalize_line_endpoints(lines, tile)
    removed: list[dict[str, Any]] = []
    for iteration in range(80):
        faces = [
            polygon
            for polygon in polygonize(
                unary_union([tile.metric_polygon.boundary, *current])
            )
            if polygon.area > 1
            and tile.metric_polygon.covers(polygon.representative_point())
        ]
        violations: list[tuple[float, Polygon, dict[str, float | int], list[str]]] = []
        for face in faces:
            metrics = shape_metrics(face)
            structure = face_structure_metrics(face)
            reasons = []
            if metrics["area_m2"] < 10_000:
                reasons.append("area_below_10000m2")
            if metrics["min_rotated_width_m"] < 80:
                reasons.append("width_below_80m")
            if metrics["rotated_aspect_ratio"] > 12:
                reasons.append("aspect_above_12")
            if metrics["hole_count"] > 0:
                reasons.append("face_has_holes")
            if structure["corner_count"] == 3:
                reasons.append("triangle_face")
            if (
                structure["corner_count"] == 4
                and structure["quadrilateral_parallel_error_deg"] > 15
            ):
                reasons.append("irregular_trapezoid_face")
            if not reasons:
                continue
            severity = min(float(metrics["area_m2"]) / 10_000, 1.0)
            severity += min(float(metrics["min_rotated_width_m"]) / 80, 1.0)
            severity += min(12 / max(float(metrics["rotated_aspect_ratio"]), 1e-9), 1.0)
            if "triangle_face" in reasons or "face_has_holes" in reasons:
                severity -= 10.0
            elif "irregular_trapezoid_face" in reasons:
                severity -= 5.0
            violations.append((severity, face, metrics, reasons))
        if not violations:
            break
        _, face, metrics, reasons = min(violations, key=lambda item: item[0])
        candidates: list[tuple[int, float, float, int, LineString, dict[str, Any]]] = []
        for line_index, line in enumerate(current):
            shared = face.boundary.intersection(line).length
            if shared <= 0.5:
                continue
            evidence = match_corridor_evidence(line, corridors)
            rank = CLASS_RANK.get(evidence["highway"], 9)
            candidates.append(
                (
                    rank,
                    shared / max(line.length, 1e-9),
                    -line.length,
                    line_index,
                    line,
                    evidence,
                )
            )
        if not candidates:
            break
        _, _, _, remove_index, removed_line, evidence = max(
            candidates, key=lambda item: item[:3]
        )
        removed.append(
            {
                "iteration": iteration + 1,
                "reason": reasons,
                "bad_face_area_m2": round(float(metrics["area_m2"]), 3),
                "bad_face_min_width_m": round(float(metrics["min_rotated_width_m"]), 3),
                "bad_face_aspect_ratio": round(float(metrics["rotated_aspect_ratio"]), 4),
                "removed_generated_segment_length_m": round(removed_line.length, 3),
                "mapped_highway": evidence["highway"],
                "mapped_name": evidence["name"],
                "source_corridor_ids": evidence["source_corridor_ids"],
            }
        )
        current.pop(remove_index)
        current, _ = prune_interior_dangling_branches(current, tile)
        graph, coordinates = geometry_graph(current)
        collapsed = maximal_collinear_chords(graph, coordinates)
        if collapsed and count_interior_crossings(collapsed) == 0:
            current = collapsed
        current = normalize_line_endpoints(current, tile)

    final_faces = [
        polygon
        for polygon in polygonize(
            unary_union([tile.metric_polygon.boundary, *current])
        )
        if polygon.area > 1
        and tile.metric_polygon.covers(polygon.representative_point())
    ]
    final_metrics = [shape_metrics(face) for face in final_faces]
    final_structures = [face_structure_metrics(face) for face in final_faces]
    return current, {
        "principle": "bad faces are merged by deleting a generated road edge, never by editing parcel geometry independently",
        "coordinate_quantization_m": 0.001,
        "boundary_snap_tolerance_m": 1.0,
        "removed_generated_segments": len(removed),
        "removal_audit": removed,
        "final_face_count": len(final_faces),
        "final_area_below_10000m2": sum(item["area_m2"] < 10_000 for item in final_metrics),
        "final_width_below_80m": sum(item["min_rotated_width_m"] < 80 for item in final_metrics),
        "final_aspect_above_12": sum(item["rotated_aspect_ratio"] > 12 for item in final_metrics),
        "final_triangle_faces": sum(item["corner_count"] == 3 for item in final_structures),
        "final_irregular_trapezoid_faces": sum(
            item["corner_count"] == 4
            and item["quadrilateral_parallel_error_deg"] > 15
            for item in final_structures
        ),
        "final_faces_with_holes": sum(item["hole_count"] > 0 for item in final_metrics),
        "final_strongly_concave_faces": sum(
            item["convexity_ratio"] < 0.8 for item in final_structures
        ),
        "final_faces_above_8_corners": sum(
            item["corner_count"] > 8 for item in final_structures
        ),
    }


def face_structure_metrics(geometry: Polygon) -> dict[str, float | int]:
    """Describe face corners after removing collinear boundary nodes."""
    simplified = geometry.simplify(0.1, preserve_topology=True)
    coordinates = list(simplified.exterior.coords)[:-1]
    corner_count = len(coordinates)
    if corner_count < 3:
        return {
            "corner_count": corner_count,
            "min_interior_angle_deg": 0.0,
            "max_interior_angle_deg": 0.0,
            "quadrilateral_parallel_error_deg": 180.0,
            "convexity_ratio": 0.0,
            "rectangularity_ratio": 0.0,
        }
    orientations: list[float] = []
    angles: list[float] = []
    for index, coordinate in enumerate(coordinates):
        next_coordinate = coordinates[(index + 1) % corner_count]
        orientations.append(
            math.degrees(
                math.atan2(
                    next_coordinate[1] - coordinate[1],
                    next_coordinate[0] - coordinate[0],
                )
            )
            % 180
        )
        previous = coordinates[index - 1]
        left = (previous[0] - coordinate[0], previous[1] - coordinate[1])
        right = (
            next_coordinate[0] - coordinate[0],
            next_coordinate[1] - coordinate[1],
        )
        norm = math.hypot(*left) * math.hypot(*right)
        cosine = 1.0 if norm <= 1e-9 else (
            left[0] * right[0] + left[1] * right[1]
        ) / norm
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    parallel_error = 0.0
    if corner_count == 4:
        parallel_error = max(
            orientation_difference(orientations[0], orientations[2]),
            orientation_difference(orientations[1], orientations[3]),
        )
    return {
        "corner_count": corner_count,
        "min_interior_angle_deg": min(angles),
        "max_interior_angle_deg": max(angles),
        "quadrilateral_parallel_error_deg": parallel_error,
        "convexity_ratio": geometry.area / max(geometry.convex_hull.area, 1e-9),
        "rectangularity_ratio": geometry.area
        / max(geometry.minimum_rotated_rectangle.area, 1e-9),
    }


def orientation_difference(left: float, right: float) -> float:
    difference = abs(left - right) % 180
    return min(difference, 180 - difference)


def materialize_abstract_graph(
    lines: list[LineString], corridors: list[SourceCorridor], tile: TileSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], nx.Graph, dict[str, int]]:
    """Create records and fold degree-two geometry nodes into road polylines."""
    min_x, min_y, _, _ = tile.bounds_m
    graph, coordinates = geometry_graph(lines)
    node_ids = {
        key: f"intersection/i{index:04d}"
        for index, key in enumerate(sorted(graph.nodes, key=lambda item: (item[1], item[0])))
    }
    intersections: list[dict[str, Any]] = []
    for key in sorted(graph.nodes, key=lambda item: (item[1], item[0])):
        metric = Point(coordinates[key])
        on_boundary = metric.distance(tile.metric_polygon.boundary) <= 1.0
        degree = int(graph.degree(key))
        kind = (
            "boundary_portal"
            if on_boundary
            else "geometry_break"
            if degree == 2
            else "junction"
        )
        intersections.append(
            {
                "id": node_ids[key],
                "point": [round(metric.x - min_x, 3), round(metric.y - min_y, 3)],
                "kind": kind,
                "degree": degree,
                "_metric": metric,
            }
        )

    road_rows: list[dict[str, Any]] = []
    ordered_lines = sorted(
        lines,
        key=lambda line: (
            min(point_key(line.coords[0]), point_key(line.coords[-1])),
            max(point_key(line.coords[0]), point_key(line.coords[-1])),
        ),
    )
    abstract_graph = nx.Graph()
    abstract_graph.add_nodes_from(row["id"] for row in intersections)
    for index, line in enumerate(ordered_lines):
        start_key = point_key(line.coords[0])
        end_key = point_key(line.coords[-1])
        if start_key not in node_ids or end_key not in node_ids or start_key == end_key:
            continue
        evidence = match_corridor_evidence(line, corridors)
        road_id = f"road/r{index:04d}"
        start_id, end_id = node_ids[start_key], node_ids[end_key]
        row = {
            "id": road_id,
            "from_intersection_id": start_id,
            "to_intersection_id": end_id,
            "geometry": [
                [round(line.coords[0][0] - min_x, 3), round(line.coords[0][1] - min_y, 3)],
                [round(line.coords[-1][0] - min_x, 3), round(line.coords[-1][1] - min_y, 3)],
            ],
            "length_m": round(line.length, 3),
            "name": evidence["name"],
            "highway": evidence["highway"],
            "total_lanes": evidence["lanes"],
            "lane_data_inferred": evidence["lane_inferred"],
            "speed_kph": evidence["speed_kph"],
            "speed_inferred": evidence["speed_inferred"],
            "source_carriageway_count": evidence["carriageway_count"],
            "source_road_ids": evidence["source_road_ids"],
            "source_osm_way_ids": evidence["source_osm_way_ids"],
            "source_corridor_ids": evidence["source_corridor_ids"],
            "geometry_regenerated": True,
            "_metric": line,
        }
        road_rows.append(row)
        abstract_graph.add_edge(start_id, end_id, road_id=road_id, weight=line.length)
    road_rows, intersections, abstract_graph, geometry_stats = collapse_geometry_breaks(
        road_rows, intersections
    )
    return road_rows, intersections, abstract_graph, geometry_stats


def collapse_geometry_breaks(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], nx.Graph, dict[str, int]]:
    """Merge maximal road chains through ``geometry_break`` support nodes.

    A bend with graph degree two is road geometry, not a road intersection.  A
    boundary portal is deliberately retained as a support point so SUMO still
    has explicit network entry/exit nodes, but it is not a model intersection
    entity.
    """
    by_intersection = {str(row["id"]): row for row in intersections}
    by_road = {str(row["id"]): row for row in roads}
    graph = nx.MultiGraph()
    graph.add_nodes_from(by_intersection)
    for road in roads:
        start = str(road["from_intersection_id"])
        end = str(road["to_intersection_id"])
        if start not in by_intersection or end not in by_intersection:
            raise ValueError(f"road {road.get('id')!r} references a missing endpoint")
        graph.add_edge(
            start,
            end,
            key=str(road["id"]),
            road_id=str(road["id"]),
            weight=float(road.get("length_m", 1.0)),
        )

    collapsible = {
        node_id
        for node_id, row in by_intersection.items()
        if row.get("kind") == "geometry_break" and graph.degree(node_id) == 2
    }
    if not collapsible:
        for intersection in intersections:
            intersection["degree"] = int(graph.degree(str(intersection["id"])))
        return roads, intersections, graph, {
            "roads_before": len(roads),
            "roads_after": len(roads),
            "geometry_breaks_removed": 0,
            "polyline_roads": sum(len(row.get("geometry", [])) > 2 for row in roads),
        }

    def unused_incident(node_id: str) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for left, right, _, data in graph.edges(node_id, keys=True, data=True):
            road_id = str(data["road_id"])
            if road_id in used_roads:
                continue
            result.append((str(right if left == node_id else left), road_id))
        return sorted(result)

    terminals = set(graph) - collapsible
    used_roads: set[str] = set()
    chains: list[tuple[str, str, list[tuple[dict[str, Any], str, str]]]] = []

    for start in sorted(terminals):
        for first_neighbor, first_road_id in unused_incident(start):
            if first_road_id in used_roads:
                continue
            current = start
            following = first_neighbor
            next_road_id = first_road_id
            chain: list[tuple[dict[str, Any], str, str]] = []
            while True:
                road_id = next_road_id
                if road_id in used_roads:
                    raise ValueError(f"road chain revisits {road_id!r}")
                used_roads.add(road_id)
                chain.append((by_road[road_id], current, following))
                if following not in collapsible:
                    break
                candidates = unused_incident(following)
                if len(candidates) != 1:
                    raise ValueError(
                        f"geometry break {following!r} does not continue through exactly one road"
                    )
                current, (following, next_road_id) = following, candidates[0]
            if start == following:
                raise ValueError(f"closed road chain at {start!r} needs an explicit junction")
            chains.append((start, following, chain))

    missing = set(by_road) - used_roads
    if missing:
        raise ValueError(
            "road components made only of geometry breaks cannot be represented as corridors: "
            + ", ".join(sorted(missing)[:10])
        )

    chains.sort(key=lambda item: (item[0], item[1], [row[0]["id"] for row in item[2]]))
    merged_roads: list[dict[str, Any]] = []
    for index, (start, end, chain) in enumerate(chains):
        representative = max(
            (entry[0] for entry in chain),
            key=lambda row: (float(row.get("length_m", 0.0)), str(row.get("id", ""))),
        )
        merged = dict(representative)
        local_coordinates: list[list[float]] = []
        metric_coordinates: list[tuple[float, float]] = []
        source_segment_ids: set[str] = set()
        for row, oriented_start, _ in chain:
            forward = str(row["from_intersection_id"]) == oriented_start
            local = [list(point) for point in row.get("geometry", [])]
            metric_line = row.get("_metric")
            metric = (
                [tuple(point) for point in metric_line.coords]
                if metric_line is not None
                else [tuple(point) for point in local]
            )
            if not forward:
                local.reverse()
                metric.reverse()
            local_coordinates.extend(local if not local_coordinates else local[1:])
            metric_coordinates.extend(metric if not metric_coordinates else metric[1:])
            source_segment_ids.update(
                str(value) for value in row.get("source_segment_ids", [row["id"]])
            )

        merged.update(
            {
                "id": f"road/r{index:04d}",
                "from_intersection_id": start,
                "to_intersection_id": end,
                "geometry": local_coordinates,
                "length_m": round(float(LineString(metric_coordinates).length), 3),
                "source_segment_ids": sorted(source_segment_ids),
                "source_segment_count": len(source_segment_ids),
                "geometry_regenerated": True,
                "_metric": LineString(metric_coordinates),
            }
        )
        for field in (
            "intersection_ids",
            "adjacent_road_ids",
            "parcel_ids",
            "lane_connections",
            "lane_assignment_policy",
        ):
            merged.pop(field, None)
        for field in (
            "source_road_ids",
            "source_osm_way_ids",
            "source_corridor_ids",
        ):
            merged[field] = sorted(
                {
                    value
                    for row, _, _ in chain
                    for value in row.get(field, [])
                }
            )
        merged["source_carriageway_count"] = max(
            int(row.get("source_carriageway_count", 0) or 0)
            for row, _, _ in chain
        )
        merged_roads.append(merged)

    retained_intersections = [
        row for row in intersections if str(row["id"]) not in collapsible
    ]
    merged_graph = nx.MultiGraph()
    merged_graph.add_nodes_from(str(row["id"]) for row in retained_intersections)
    for road in merged_roads:
        merged_graph.add_edge(
            road["from_intersection_id"],
            road["to_intersection_id"],
            key=road["id"],
            road_id=road["id"],
            weight=float(road["length_m"]),
        )
    for intersection in retained_intersections:
        intersection["degree"] = int(merged_graph.degree(str(intersection["id"])))
        for field in ("road_ids", "neighbor_intersection_ids", "parcel_ids"):
            intersection.pop(field, None)
    return merged_roads, retained_intersections, merged_graph, {
        "roads_before": len(roads),
        "roads_after": len(merged_roads),
        "geometry_breaks_removed": len(collapsible),
        "polyline_roads": sum(len(row["geometry"]) > 2 for row in merged_roads),
    }


def _local_point_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(
        float(right[0]) - float(left[0]),
        float(right[1]) - float(left[1]),
    )


def _abstract_topology_graph(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_nodes_from(str(row["id"]) for row in intersections)
    for road in roads:
        graph.add_edge(
            str(road["from_intersection_id"]),
            str(road["to_intersection_id"]),
            key=str(road["id"]),
            road_id=str(road["id"]),
            weight=float(road.get("length_m", 1.0)),
        )
    return graph


def _set_local_road_endpoints(
    roads: list[dict[str, Any]],
    positions: dict[str, list[float]],
) -> None:
    for road in roads:
        road_id = str(road["id"])
        start_id = str(road["from_intersection_id"])
        end_id = str(road["to_intersection_id"])
        geometry = [list(map(float, point)) for point in road.get("geometry", [])]
        if len(geometry) < 2:
            raise ValueError(f"{road_id}: geometry must contain at least two points")
        geometry[0] = list(positions[start_id])
        geometry[-1] = list(positions[end_id])
        line = LineString(geometry)
        if (
            line.is_empty
            or not line.is_valid
            or line.length <= CLOSE_JUNCTION_POSITION_TOLERANCE_M
        ):
            raise ValueError(f"{road_id}: invalid geometry after moving junctions")
        road["geometry"] = [
            [round(float(x), 3), round(float(y), 3)] for x, y in line.coords
        ]
        road["length_m"] = round(float(line.length), 3)


def _merge_close_three_way_pairs(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    *,
    threshold_m: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    """Merge disjoint directly connected degree-three pairs, shortest first."""
    graph = _abstract_topology_graph(roads, intersections)
    by_id = {str(row["id"]): row for row in intersections}
    candidates: list[tuple[float, str, str, str]] = []
    for road in roads:
        left_id = str(road["from_intersection_id"])
        right_id = str(road["to_intersection_id"])
        left = by_id[left_id]
        right = by_id[right_id]
        if left.get("kind") != "junction" or right.get("kind") != "junction":
            continue
        distance_m = _local_point_distance(left["point"], right["point"])
        if (
            distance_m < threshold_m
            and graph.degree(left_id) == 3
            and graph.degree(right_id) == 3
        ):
            candidates.append((distance_m, str(road["id"]), left_id, right_id))

    used: set[str] = set()
    removed_intersections: set[str] = set()
    removed_roads: set[str] = set()
    aliases: dict[str, str] = {}
    merges: list[dict[str, Any]] = []
    for distance_m, road_id, left_id, right_id in sorted(candidates):
        if left_id in used or right_id in used:
            continue
        graph = _abstract_topology_graph(
            [row for row in roads if str(row["id"]) not in removed_roads],
            [
                row
                for row in intersections
                if str(row["id"]) not in removed_intersections
            ],
        )
        if (
            left_id not in graph
            or right_id not in graph
            or graph.degree(left_id) != 3
            or graph.degree(right_id) != 3
        ):
            continue
        connecting_ids = sorted(
            str(data["road_id"])
            for _, neighbor, _, data in graph.edges(
                left_id, keys=True, data=True
            )
            if str(neighbor) == right_id
        )
        if connecting_ids != [road_id]:
            raise RuntimeError(
                f"{left_id} and {right_id}: expected unique connecting road "
                f"{road_id}, got {connecting_ids}"
            )
        external_road_ids = {
            str(data["road_id"])
            for node_id in (left_id, right_id)
            for _, neighbor, _, data in graph.edges(
                node_id, keys=True, data=True
            )
            if str(neighbor) not in {left_id, right_id}
        }
        if len(external_road_ids) != 4:
            raise RuntimeError(
                f"{left_id} and {right_id}: merging degree-three junctions "
                f"must leave four external roads, got "
                f"{sorted(external_road_ids)}"
            )

        keep_id, drop_id = sorted((left_id, right_id))
        keep = by_id[keep_id]
        drop = by_id[drop_id]
        keep_point = list(map(float, keep["point"]))
        drop_point = list(map(float, drop["point"]))
        keep["point"] = [
            (keep_point[0] + drop_point[0]) / 2.0,
            (keep_point[1] + drop_point[1]) / 2.0,
        ]
        keep["kind"] = "junction"
        keep["degree"] = 4
        aliases[drop_id] = keep_id
        removed_intersections.add(drop_id)
        removed_roads.add(road_id)
        used.update((left_id, right_id))
        merges.append(
            {
                "kept_intersection_id": keep_id,
                "removed_intersection_id": drop_id,
                "removed_road_id": road_id,
                "original_distance_m": round(distance_m, 3),
                "new_point": [round(value, 3) for value in keep["point"]],
            }
        )

    retained_roads = []
    for road in roads:
        road_id = str(road["id"])
        if road_id in removed_roads:
            continue
        road["from_intersection_id"] = aliases.get(
            str(road["from_intersection_id"]),
            str(road["from_intersection_id"]),
        )
        road["to_intersection_id"] = aliases.get(
            str(road["to_intersection_id"]),
            str(road["to_intersection_id"]),
        )
        if road["from_intersection_id"] == road["to_intersection_id"]:
            raise RuntimeError(f"{road_id}: merge produced an unexpected self-loop")
        retained_roads.append(road)
    retained_intersections = [
        row
        for row in intersections
        if str(row["id"]) not in removed_intersections
    ]
    return retained_roads, retained_intersections, merges, aliases


def _stretch_close_junction_pairs(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    *,
    trigger_distance_m: float,
    target_distance_m: float,
    tile_width_m: float,
    tile_height_m: float,
) -> list[dict[str, Any]]:
    """Move close junctions until selected center distances reach the target."""
    constraint_distance_m = (
        target_distance_m + CLOSE_JUNCTION_SERIALIZATION_CLEARANCE_M
    )
    by_id = {str(row["id"]): row for row in intersections}
    positions = {
        intersection_id: list(map(float, row["point"]))
        for intersection_id, row in by_id.items()
    }
    junction_edges = [
        (
            str(road["id"]),
            str(road["from_intersection_id"]),
            str(road["to_intersection_id"]),
        )
        for road in roads
        if by_id[str(road["from_intersection_id"])].get("kind") == "junction"
        and by_id[str(road["to_intersection_id"])].get("kind") == "junction"
    ]
    original_positions = {
        intersection_id: list(point)
        for intersection_id, point in positions.items()
    }
    original_distances = {
        road_id: _local_point_distance(positions[left_id], positions[right_id])
        for road_id, left_id, right_id in junction_edges
    }
    selected_road_ids = {
        road_id
        for road_id, distance_m in original_distances.items()
        if distance_m < trigger_distance_m
    }

    for _ in range(CLOSE_JUNCTION_MAX_STRETCH_ITERATIONS):
        violations = []
        for road_id, left_id, right_id in junction_edges:
            if road_id not in selected_road_ids:
                continue
            distance_m = _local_point_distance(
                positions[left_id], positions[right_id]
            )
            if (
                distance_m
                < constraint_distance_m
                - CLOSE_JUNCTION_POSITION_TOLERANCE_M
            ):
                violations.append((distance_m, road_id, left_id, right_id))
        if not violations:
            break
        for _, _, left_id, right_id in sorted(violations):
            left = positions[left_id]
            right = positions[right_id]
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            distance_m = math.hypot(dx, dy)
            if distance_m <= CLOSE_JUNCTION_POSITION_TOLERANCE_M:
                raise RuntimeError(
                    f"{left_id} and {right_id}: coincident junctions have no "
                    "stretch direction"
                )
            if (
                distance_m
                >= constraint_distance_m
                - CLOSE_JUNCTION_POSITION_TOLERANCE_M
            ):
                continue
            correction_m = (constraint_distance_m - distance_m) / 2.0
            unit_x = dx / distance_m
            unit_y = dy / distance_m
            positions[left_id] = [
                left[0] - unit_x * correction_m,
                left[1] - unit_y * correction_m,
            ]
            positions[right_id] = [
                right[0] + unit_x * correction_m,
                right[1] + unit_y * correction_m,
            ]
        selected_road_ids.update(
            road_id
            for road_id, left_id, right_id in junction_edges
            if _local_point_distance(positions[left_id], positions[right_id])
            < trigger_distance_m - CLOSE_JUNCTION_POSITION_TOLERANCE_M
        )
    else:
        raise RuntimeError(
            "close-junction stretch constraints did not converge within "
            f"{CLOSE_JUNCTION_MAX_STRETCH_ITERATIONS} iterations"
        )

    violations = [
        (road_id, _local_point_distance(positions[left_id], positions[right_id]))
        for road_id, left_id, right_id in junction_edges
        if road_id in selected_road_ids
        and _local_point_distance(positions[left_id], positions[right_id])
        < constraint_distance_m - 1e-3
    ]
    if violations:
        raise RuntimeError(
            f"junction roads remain below the {target_distance_m:g} m target: "
            f"{violations[:10]}"
        )

    for intersection_id, point in positions.items():
        if not (
            -CLOSE_JUNCTION_POSITION_TOLERANCE_M
            <= point[0]
            <= tile_width_m + CLOSE_JUNCTION_POSITION_TOLERANCE_M
            and -CLOSE_JUNCTION_POSITION_TOLERANCE_M
            <= point[1]
            <= tile_height_m + CLOSE_JUNCTION_POSITION_TOLERANCE_M
        ):
            raise RuntimeError(
                f"{intersection_id}: stretched point ({point[0]:.3f}, "
                f"{point[1]:.3f}) leaves the tile"
            )
        by_id[intersection_id]["point"] = [
            round(point[0], 3),
            round(point[1], 3),
        ]
    positions = {
        intersection_id: list(map(float, row["point"]))
        for intersection_id, row in by_id.items()
    }
    rounded_violations = [
        (road_id, _local_point_distance(positions[left_id], positions[right_id]))
        for road_id, left_id, right_id in junction_edges
        if road_id in selected_road_ids
        and _local_point_distance(positions[left_id], positions[right_id])
        < target_distance_m - 1e-3
    ]
    if rounded_violations:
        raise RuntimeError(
            f"serialized junction roads remain below the "
            f"{target_distance_m:g} m target: {rounded_violations[:10]}"
        )
    _set_local_road_endpoints(roads, positions)

    changes = []
    for road_id, left_id, right_id in junction_edges:
        if road_id not in selected_road_ids:
            continue
        changes.append(
            {
                "road_id": road_id,
                "from_intersection_id": left_id,
                "to_intersection_id": right_id,
                "original_distance_m": round(original_distances[road_id], 3),
                "final_distance_m": round(
                    _local_point_distance(
                        positions[left_id], positions[right_id]
                    ),
                    3,
                ),
                "from_displacement_m": round(
                    _local_point_distance(
                        original_positions[left_id], positions[left_id]
                    ),
                    3,
                ),
                "to_displacement_m": round(
                    _local_point_distance(
                        original_positions[right_id], positions[right_id]
                    ),
                    3,
                ),
            }
        )
    return sorted(
        changes,
        key=lambda row: (row["original_distance_m"], row["road_id"]),
    )


def regularize_close_junctions(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    tile: TileSpec,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    nx.MultiGraph,
    dict[str, Any],
]:
    """Merge close 3+3 junctions, then stretch every other close pair."""
    roads, intersections, merges, aliases = _merge_close_three_way_pairs(
        roads,
        intersections,
        threshold_m=CLOSE_JUNCTION_MERGE_DISTANCE_M,
    )
    tile_width_m = float(tile.bounds_m[2] - tile.bounds_m[0])
    tile_height_m = float(tile.bounds_m[3] - tile.bounds_m[1])
    stretches = _stretch_close_junction_pairs(
        roads,
        intersections,
        trigger_distance_m=CLOSE_JUNCTION_STRETCH_TRIGGER_DISTANCE_M,
        target_distance_m=CLOSE_JUNCTION_STRETCH_TARGET_DISTANCE_M,
        tile_width_m=tile_width_m,
        tile_height_m=tile_height_m,
    )

    min_x, min_y, _, _ = tile.bounds_m
    graph = _abstract_topology_graph(roads, intersections)
    for intersection in intersections:
        x, y = map(float, intersection["point"])
        intersection["_metric"] = Point(x + min_x, y + min_y)
        intersection["degree"] = int(graph.degree(str(intersection["id"])))
    for road in roads:
        road["_metric"] = LineString(
            [
                (float(point[0]) + min_x, float(point[1]) + min_y)
                for point in road["geometry"]
            ]
        )
        road["length_m"] = round(float(road["_metric"].length), 3)

    stats = {
        "merge_rule": (
            "directly connected degree-three pairs with center distance "
            f"< {CLOSE_JUNCTION_MERGE_DISTANCE_M:g} m; shortest-first "
            "disjoint matching"
        ),
        "stretch_rule": (
            "remaining directly connected junction pairs with center "
            f"distance < {CLOSE_JUNCTION_STRETCH_TRIGGER_DISTANCE_M:g} m "
            f"are displaced along their connecting axis to "
            f"{CLOSE_JUNCTION_STRETCH_TARGET_DISTANCE_M:g} m"
        ),
        "merge_distance_m": CLOSE_JUNCTION_MERGE_DISTANCE_M,
        "stretch_trigger_distance_m": (
            CLOSE_JUNCTION_STRETCH_TRIGGER_DISTANCE_M
        ),
        "stretch_target_distance_m": (
            CLOSE_JUNCTION_STRETCH_TARGET_DISTANCE_M
        ),
        "merged_pair_count": len(merges),
        "stretched_edge_count": len(stretches),
        "intersection_aliases": dict(sorted(aliases.items())),
        "merges": merges,
        "stretches": stretches,
    }
    return roads, intersections, graph, stats


def match_corridor_evidence(
    line: LineString, corridors: list[SourceCorridor]
) -> dict[str, Any]:
    scored: list[tuple[float, SourceCorridor]] = []
    for corridor in corridors:
        angle_difference = line_angle_difference(line, corridor.geometry)
        if angle_difference > 28:
            continue
        distance = line.distance(corridor.geometry)
        if distance > 180:
            continue
        overlap = line.intersection(corridor.geometry.buffer(140, cap_style="flat")).length
        if overlap < min(30.0, line.length * 0.12):
            continue
        score = overlap / max(line.length, 1.0) - distance / 700 - corridor.rank * 0.015
        scored.append((score, corridor))
    if not scored:
        nearest = min(corridors, key=lambda item: line.distance(item.geometry))
        scored = [(0.0, nearest)]
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]
    supporting = [item for score, item in scored if score >= scored[0][0] - 0.18][:6]
    name_lengths: Counter[str] = Counter()
    for item in supporting:
        if item.name:
            name_lengths[item.name] += item.geometry.length
    return {
        "name": name_lengths.most_common(1)[0][0] if name_lengths else best.name,
        "highway": min((item.highway for item in supporting), key=lambda value: CLASS_RANK.get(value, 9)),
        "lanes": max(item.lanes for item in supporting),
        "lane_inferred": all(item.lane_inferred for item in supporting),
        "speed_kph": max(item.speed_kph for item in supporting),
        "speed_inferred": all(item.speed_inferred for item in supporting),
        "carriageway_count": max(item.carriageway_count for item in supporting),
        "source_road_ids": sorted({source_id for item in supporting for source_id in item.source_road_ids}),
        "source_osm_way_ids": sorted({way_id for item in supporting for way_id in item.source_osm_way_ids}),
        "source_corridor_ids": sorted(item.corridor_id for item in supporting),
    }


def project_landuse(
    payload: dict[str, Any], forward: Transformer, tile: TileSpec
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        metric = transform(forward.transform, shape(feature["geometry"])).intersection(
            tile.metric_polygon
        )
        for polygon in iter_polygons(metric):
            if polygon.area < 10:
                continue
            rows.append(
                {
                    "id": feature["id"],
                    "land_use": feature.get("properties", {}).get("land_use", "unclassified"),
                    "metric": polygon,
                }
            )
    return rows


def build_abstract_parcels(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    landuse_sources: list[dict[str, Any]],
    tile: TileSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Polygonize faces; road width is never subtracted from parcel interiors."""
    linework = unary_union(
        [tile.metric_polygon.boundary, *(row["_metric"] for row in roads)]
    )
    faces = [
        polygon
        for polygon in polygonize(linework)
        if polygon.area > 1 and tile.metric_polygon.covers(polygon.representative_point())
    ]
    faces.sort(key=lambda polygon: (polygon.centroid.y, polygon.centroid.x))
    min_x, min_y, _, _ = tile.bounds_m
    parcels: list[dict[str, Any]] = []
    for index, polygon in enumerate(faces):
        overlaps: list[tuple[float, dict[str, Any]]] = []
        for source in landuse_sources:
            if not polygon.intersects(source["metric"]):
                continue
            area = polygon.intersection(source["metric"]).area
            if area > 1:
                overlaps.append((area, source))
        overlaps.sort(key=lambda item: item[0], reverse=True)
        dominant = overlaps[0][1] if overlaps else None
        metrics = shape_metrics(polygon)
        structure = face_structure_metrics(polygon)
        output_polygon = polygon.simplify(0.1, preserve_topology=True)
        ring = [
            [round(x - min_x, 3), round(y - min_y, 3)]
            for x, y in output_polygon.exterior.coords
        ]
        parcels.append(
            {
                "id": f"parcel/p{index:04d}",
                "geometry": [ring],
                "land_use": dominant["land_use"] if dominant else "unclassified",
                "source_landuse_ids": sorted({source["id"] for _, source in overlaps}),
                "area_m2": round(float(metrics["area_m2"]), 3),
                "perimeter_m": round(float(metrics["perimeter_m"]), 3),
                "min_rotated_width_m": round(float(metrics["min_rotated_width_m"]), 3),
                "rotated_aspect_ratio": round(float(metrics["rotated_aspect_ratio"]), 4),
                "compactness": round(float(metrics["compactness"]), 5),
                "hole_count": int(metrics["hole_count"]),
                "boundary_vertex_count": len(output_polygon.exterior.coords),
                "noded_boundary_vertex_count": int(metrics["exterior_vertex_count"]),
                "corner_count": int(structure["corner_count"]),
                "min_interior_angle_deg": round(float(structure["min_interior_angle_deg"]), 3),
                "max_interior_angle_deg": round(float(structure["max_interior_angle_deg"]), 3),
                "quadrilateral_parallel_error_deg": round(
                    float(structure["quadrilateral_parallel_error_deg"]), 3
                ),
                "convexity_ratio": round(float(structure["convexity_ratio"]), 5),
                "rectangularity_ratio": round(float(structure["rectangularity_ratio"]), 5),
                "_metric": polygon,
            }
        )
    coverage = unary_union(faces).area / tile.metric_polygon.area if faces else 0.0
    return parcels, {
        "face_count": len(parcels),
        "coverage_ratio": round(coverage, 8),
        "road_buffer_subtracted": False,
        "holes_created_by_roads": False,
        "area_below_10000m2": sum(row["area_m2"] < 10_000 for row in parcels),
        "width_below_80m": sum(row["min_rotated_width_m"] < 80 for row in parcels),
        "aspect_above_12": sum(row["rotated_aspect_ratio"] > 12 for row in parcels),
        "triangle_faces": sum(row["corner_count"] == 3 for row in parcels),
        "irregular_trapezoid_faces": sum(
            row["corner_count"] == 4
            and row["quadrilateral_parallel_error_deg"] > 15
            for row in parcels
        ),
        "strongly_concave_faces": sum(row["convexity_ratio"] < 0.8 for row in parcels),
        "faces_above_8_corners": sum(row["corner_count"] > 8 for row in parcels),
    }


def attach_adjacency(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    graph: nx.Graph,
) -> None:
    roads_by_id = {row["id"]: row for row in roads}
    intersections_by_id = {row["id"]: row for row in intersections}
    for intersection in intersections:
        node_id = intersection["id"]
        if graph.is_multigraph():
            road_ids = sorted(
                str(data["road_id"])
                for _, _, _, data in graph.edges(node_id, keys=True, data=True)
            )
        else:
            road_ids = sorted(
                str(data["road_id"])
                for _, _, data in graph.edges(node_id, data=True)
            )
        intersection["road_ids"] = road_ids
        intersection["neighbor_intersection_ids"] = sorted(graph.neighbors(node_id))
        intersection["parcel_ids"] = []
    for road in roads:
        endpoints = [road["from_intersection_id"], road["to_intersection_id"]]
        road["intersection_ids"] = endpoints
        adjacent = set()
        for endpoint in endpoints:
            adjacent.update(intersections_by_id[endpoint]["road_ids"])
        adjacent.discard(road["id"])
        road["adjacent_road_ids"] = sorted(adjacent)
        road["parcel_ids"] = []

    for parcel in parcels:
        polygon = parcel["_metric"]
        boundary_roads = [
            road["id"]
            for road in roads
            if polygon.boundary.intersection(road["_metric"]).length > 0.5
        ]
        corners = [
            intersection["id"]
            for intersection in intersections
            if polygon.boundary.distance(intersection["_metric"]) <= 0.05
        ]
        parcel["boundary_road_ids"] = sorted(boundary_roads)
        parcel["corner_intersection_ids"] = sorted(corners)
        parcel["neighbor_parcel_ids"] = []
        for road_id in boundary_roads:
            roads_by_id[road_id]["parcel_ids"].append(parcel["id"])
        for intersection_id in corners:
            intersections_by_id[intersection_id]["parcel_ids"].append(parcel["id"])
    for left_index, left in enumerate(parcels):
        for right in parcels[left_index + 1 :]:
            if left["_metric"].boundary.intersection(right["_metric"].boundary).length <= 0.5:
                continue
            left["neighbor_parcel_ids"].append(right["id"])
            right["neighbor_parcel_ids"].append(left["id"])
    for road in roads:
        road["parcel_ids"] = sorted(set(road["parcel_ids"]))
    for intersection in intersections:
        intersection["parcel_ids"] = sorted(set(intersection["parcel_ids"]))
    for parcel in parcels:
        parcel["neighbor_parcel_ids"] = sorted(set(parcel["neighbor_parcel_ids"]))


def validate_topology(
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    tile: TileSpec,
) -> dict[str, Any]:
    intersection_ids = {row["id"] for row in intersections}
    road_ids = {row["id"] for row in roads}
    parcel_ids = {row["id"] for row in parcels}
    pair_counts: Counter[tuple[str, str]] = Counter(
        tuple(sorted((row["from_intersection_id"], row["to_intersection_id"])))
        for row in roads
    )
    invalid_adjacency = 0
    for row in roads:
        invalid_adjacency += sum(value not in intersection_ids for value in row["intersection_ids"])
        invalid_adjacency += sum(value not in road_ids for value in row["adjacent_road_ids"])
        invalid_adjacency += sum(value not in parcel_ids for value in row["parcel_ids"])
    for row in intersections:
        invalid_adjacency += sum(value not in road_ids for value in row["road_ids"])
        invalid_adjacency += sum(value not in parcel_ids for value in row["parcel_ids"])
        invalid_adjacency += sum(value not in intersection_ids for value in row["neighbor_intersection_ids"])
    for row in parcels:
        invalid_adjacency += sum(value not in road_ids for value in row["boundary_road_ids"])
        invalid_adjacency += sum(value not in parcel_ids for value in row["neighbor_parcel_ids"])
        invalid_adjacency += sum(value not in intersection_ids for value in row["corner_intersection_ids"])
    parcel_union = unary_union([row["_metric"] for row in parcels]) if parcels else Polygon()
    coverage_ratio = parcel_union.area / tile.metric_polygon.area
    overlap_area = sum(row["_metric"].area for row in parcels) - parcel_union.area
    crossings = count_interior_crossings([row["_metric"] for row in roads])
    roads_by_id = {row["id"]: row for row in roads}
    intersections_by_id = {row["id"]: row for row in intersections}
    redundant_degree_two = 0
    for intersection in intersections:
        if intersection["kind"] == "boundary_portal" or intersection["degree"] != 2:
            continue
        vectors = []
        for road_id in intersection["road_ids"]:
            road = roads_by_id[road_id]
            other_id = (
                road["to_intersection_id"]
                if road["from_intersection_id"] == intersection["id"]
                else road["from_intersection_id"]
            )
            other = intersections_by_id[other_id]["_metric"]
            center = intersection["_metric"]
            vectors.append((other.x - center.x, other.y - center.y))
        if len(vectors) == 2:
            norm = math.hypot(*vectors[0]) * math.hypot(*vectors[1])
            cosine = 1.0 if norm <= 1e-9 else (
                vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]
            ) / norm
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if abs(180.0 - angle) <= 8.0:
                redundant_degree_two += 1
    checks = {
        "all_roads_are_two_point_straight_segments": all(len(row["geometry"]) == 2 for row in roads),
        "all_roads_are_valid_lines_or_polylines": all(
            len(row["geometry"]) >= 2 and row["_metric"].length >= 5 for row in roads
        ),
        "geometry_breaks_stored_as_intersections": sum(
            row["kind"] == "geometry_break" for row in intersections
        ),
        "all_road_endpoints_reference_intersections": all(
            row["from_intersection_id"] in intersection_ids
            and row["to_intersection_id"] in intersection_ids
            for row in roads
        ),
        "duplicate_road_endpoint_pairs": sum(value - 1 for value in pair_counts.values() if value > 1),
        "interior_geometric_crossings": crossings,
        "interior_degree_one_intersections": sum(
            row["kind"] != "boundary_portal" and row["degree"] == 1 for row in intersections
        ),
        "interior_degree_two_intersections": sum(
            row["kind"] != "boundary_portal" and row["degree"] == 2 for row in intersections
        ),
        "redundant_collinear_degree_two_intersections": redundant_degree_two,
        "invalid_adjacency_references": invalid_adjacency,
        "parcel_coverage_ratio": round(coverage_ratio, 8),
        "parcel_overlap_area_m2": round(overlap_area, 5),
        "invalid_parcels": sum(not row["_metric"].is_valid for row in parcels),
        "parcels_with_holes": sum(row["hole_count"] > 0 for row in parcels),
        "roads_without_adjacent_parcel": sum(len(row["parcel_ids"]) == 0 for row in roads),
        "roads_with_more_than_two_adjacent_parcels": sum(len(row["parcel_ids"]) > 2 for row in roads),
        "parcels_area_below_10000m2": sum(row["area_m2"] < 10_000 for row in parcels),
        "parcels_width_below_80m": sum(row["min_rotated_width_m"] < 80 for row in parcels),
        "parcels_aspect_above_12": sum(row["rotated_aspect_ratio"] > 12 for row in parcels),
        "parcels_corners_above_12": sum(row["corner_count"] > 12 for row in parcels),
        "triangle_parcels": sum(row["corner_count"] == 3 for row in parcels),
        "irregular_trapezoid_parcels": sum(
            row["corner_count"] == 4
            and row["quadrilateral_parallel_error_deg"] > 15
            for row in parcels
        ),
        "strongly_concave_parcels": sum(row["convexity_ratio"] < 0.8 for row in parcels),
        "parcels_above_8_corners": sum(row["corner_count"] > 8 for row in parcels),
    }
    checks["structural_checks_passed"] = bool(
        checks["all_roads_are_valid_lines_or_polylines"]
        and checks["geometry_breaks_stored_as_intersections"] == 0
        and checks["all_road_endpoints_reference_intersections"]
        and checks["interior_geometric_crossings"] == 0
        and checks["interior_degree_one_intersections"] == 0
        and checks["redundant_collinear_degree_two_intersections"] == 0
        and checks["invalid_adjacency_references"] == 0
        and checks["parcel_coverage_ratio"] >= 0.99999
        and checks["parcel_overlap_area_m2"] <= 0.01
        and checks["invalid_parcels"] == 0
        and checks["parcels_with_holes"] == 0
        and checks["roads_without_adjacent_parcel"] == 0
        and checks["roads_with_more_than_two_adjacent_parcels"] == 0
        and checks["parcels_area_below_10000m2"] == 0
        and checks["parcels_width_below_80m"] == 0
        and checks["parcels_aspect_above_12"] == 0
        and checks["parcels_corners_above_12"] == 0
        and checks["triangle_parcels"] == 0
        and checks["irregular_trapezoid_parcels"] == 0
    )
    return checks


def build_traffic_signals(
    source: dict[str, Any],
    source_intersections: dict[str, Any],
    forward: Transformer,
    roads: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    graph: nx.Graph,
    tile: TileSpec,
) -> dict[str, Any]:
    source_points = {
        feature["id"]: transform(forward.transform, shape(feature["geometry"]))
        for feature in source_intersections.get("features", [])
    }
    candidates = [row for row in intersections if row["kind"] == "junction"]
    grouped: defaultdict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    dropped: list[dict[str, Any]] = []
    for control in source.get("controls", []):
        point = source_points.get(control["intersection_id"])
        if point is None or not candidates:
            dropped.append({"source_intersection_id": control["intersection_id"], "reason": "missing_source_or_target"})
            continue
        target = min(candidates, key=lambda row: point.distance(row["_metric"]))
        distance = point.distance(target["_metric"])
        if distance > 220:
            dropped.append(
                {
                    "source_intersection_id": control["intersection_id"],
                    "reason": "nearest_abstract_intersection_over_220m",
                    "distance_m": round(distance, 3),
                }
            )
            continue
        grouped[target["id"]].append((control, distance))

    # The abstract graph is synthetic, so OSM signal heads are provenance only.
    # Every real road intersection receives a default controller even when no
    # source signal can be mapped to it.
    for candidate in candidates:
        grouped.setdefault(candidate["id"], [])

    return assemble_sumo_ready_traffic(
        source=source,
        grouped=grouped,
        dropped=dropped,
        roads=roads,
        intersections=intersections,
        graph=graph,
        tile_id=tile.tile_id,
    )


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}
