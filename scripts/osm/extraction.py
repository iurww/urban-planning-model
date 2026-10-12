"""OSM evidence extraction, geometry cleaning, and junction refinement."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from scripts.osm.common import (
    Projection,
    ROAD_LEVEL,
    TileSpec,
    classify_landuse,
    extract_pbf,
    geojson_feature,
    infer_road_width,
    iter_lines,
    iter_polygons,
    normalize_name,
    parse_lanes,
    parse_speed_kph,
    relation_counts,
    utc_now,
    write_geojson,
    write_json,
    write_jsonl,
)

DIRECTIONS = ("N", "E", "S", "W")

LAND_COLORS = {
    "residential": "#edc9af",
    "commercial": "#e79ac3",
    "industrial": "#b9a4d0",
    "public_service": "#f5d76e",
    "green_open_space": "#9bcf8d",
    "water": "#8fc9e8",
    "agriculture": "#c8dda0",
    "other": "#d7d7d7",
}


SIGNAL_ATTACH_M = 85.0
MIN_PARCEL_AREA_M2 = 600.0
PARCEL_ROAD_CLEARANCE_M = 3.0
PARCEL_ACROSS_ROAD_M = 42.0


class UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_raw_graph(raw_roads: Iterable[Any], projection: Projection, tile: TileSpec) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    working_extent = tile.metric_polygon.buffer(120)
    for raw in raw_roads:
        for segment_index, (left, right) in enumerate(zip(raw.node_ids, raw.node_ids[1:])):
            left_coord = raw.coordinates[segment_index]
            right_coord = raw.coordinates[segment_index + 1]
            left_point = projection.project(Point(left_coord))
            right_point = projection.project(Point(right_coord))
            if left_point.equals(right_point):
                continue
            line = LineString([left_point, right_point])
            if not line.intersects(working_extent):
                continue
            graph.add_node(left, point=left_point, wgs84=left_coord)
            graph.add_node(right, point=right_point, wgs84=right_coord)
            graph.add_edge(
                left,
                right,
                key=f"{raw.osm_id}:{segment_index}",
                uid=f"{raw.osm_id}:{segment_index}",
                way_id=raw.osm_id,
                segment_index=segment_index,
                geometry=line,
                length_m=line.length,
                tags=raw.tags,
            )
    return graph


def node_level_signature(graph: nx.MultiGraph, node_id: int) -> set[int]:
    signatures: set[int] = set()
    for _, _, data in graph.edges(node_id, data=True):
        tags = data["tags"]
        try:
            layer = int(tags.get("layer", "0"))
        except ValueError:
            layer = 0
        if tags.get("bridge") == "yes" and "layer" not in tags:
            layer = 1
        if tags.get("tunnel") == "yes" and "layer" not in tags:
            layer = -1
        signatures.add(layer)
    return signatures or {0}


def collapse_paths(
    graph: nx.MultiGraph,
    junctions: list[dict[str, Any]],
    raw_node_to_junction: dict[int, str],
    tile: TileSpec,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    junction_centers = {row["id"]: row["metric"] for row in junctions}
    collapsed = nx.MultiGraph()
    internal_edges = 0
    for left, right, key, data in graph.edges(keys=True, data=True):
        collapsed_left = raw_node_to_junction.get(left, f"raw/{left}")
        collapsed_right = raw_node_to_junction.get(right, f"raw/{right}")
        if collapsed_left == collapsed_right:
            internal_edges += 1
            continue
        geometry = data["geometry"]
        if Point(geometry.coords[0]).distance(graph.nodes[left]["point"]) > 0.01:
            geometry = LineString(list(reversed(geometry.coords)))
        attributes = {
            **data,
            "geometry": geometry,
            "original_left": left,
            "original_right": right,
            "added_left": collapsed_left,
            "added_right": collapsed_right,
        }
        collapsed.add_edge(collapsed_left, collapsed_right, key=key, **attributes)

    terminals = {
        node_id
        for node_id in collapsed.nodes
        if node_id.startswith("intersection/") or collapsed.degree(node_id) != 2
    }
    used: set[str] = set()
    paths: list[dict[str, Any]] = []
    discarded_internal_paths = 0

    def trace(start: str, first: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any]:
        current = start
        edge_tuple = first
        coordinates: list[tuple[float, float]] = []
        edge_data: list[dict[str, Any]] = []
        while True:
            left, right, _, data = edge_tuple
            used.add(data["uid"])
            next_node = right if current == left else left
            segment_coords = oriented_edge_coordinates(data, current, left, right)
            if coordinates and coordinates[-1] != segment_coords[0]:
                coordinates.append(segment_coords[0])
            coordinates.extend(segment_coords[1:] if coordinates else segment_coords)
            edge_data.append(data)
            if next_node in terminals:
                break
            choices = [item for item in incident_edges(collapsed, next_node) if item[3]["uid"] not in used]
            if len(choices) != 1:
                break
            current = next_node
            edge_tuple = choices[0]
        if start in junction_centers:
            coordinates[0] = tuple(junction_centers[start].coords[0])
        if next_node in junction_centers:
            coordinates[-1] = tuple(junction_centers[next_node].coords[0])
        return aggregate_path(start, next_node, coordinates, edge_data)

    for terminal in sorted(terminals):
        for item in incident_edges(collapsed, terminal):
            if item[3]["uid"] in used:
                continue
            path = trace(terminal, item)
            if path["start"] == path["end"] and path["length_m"] < 140:
                discarded_internal_paths += 1
                continue
            paths.append(path)

    # Closed loops without a terminal (rare roundabouts/service loops).
    for left, right, key, data in collapsed.edges(keys=True, data=True):
        if data["uid"] in used:
            continue
        path = trace(left, (left, right, key, data))
        if path["length_m"] >= 30:
            paths.append(path)

    clipped_paths: list[dict[str, Any]] = []
    for path in paths:
        clipped = path["metric"].intersection(tile.metric_polygon)
        parts = sorted(iter_lines(clipped), key=lambda line: line.length, reverse=True)
        if not parts or parts[0].length < 5:
            continue
        path = {**path, "metric": parts[0], "length_m": parts[0].length}
        clipped_paths.append(path)
    return clipped_paths, {
        "raw_graph_nodes": graph.number_of_nodes(),
        "raw_graph_edges": graph.number_of_edges(),
        "collapsed_internal_edges": internal_edges,
        "discarded_internal_paths": discarded_internal_paths,
        "traced_paths": len(clipped_paths),
    }


def incident_edges(graph: nx.MultiGraph, node_id: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    return [(left, right, key, data) for left, right, key, data in graph.edges(node_id, keys=True, data=True)]


def oriented_edge_coordinates(
    data: dict[str, Any], current: str, collapsed_left: str, collapsed_right: str
) -> list[tuple[float, float]]:
    coordinates = list(data["geometry"].coords)
    if current == data["added_left"]:
        return coordinates
    if current == data["added_right"]:
        return list(reversed(coordinates))
    raise KeyError(f"{current} is not an endpoint of {data['uid']}")


def aggregate_path(
    start: str,
    end: str,
    coordinates: list[tuple[float, float]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    # Remove consecutive duplicate coordinates introduced by collapsed nodes.
    clean_coordinates = [coordinates[0]]
    for coordinate in coordinates[1:]:
        if coordinate != clean_coordinates[-1]:
            clean_coordinates.append(coordinate)
    geometry = LineString(clean_coordinates).simplify(0.25, preserve_topology=True)
    dominant = max(edges, key=lambda item: item["length_m"])
    return {
        "start": start,
        "end": end,
        "metric": geometry,
        "length_m": geometry.length,
        "tags": dominant["tags"],
        "member_way_ids": sorted({item["way_id"] for item in edges}),
        "member_edge_count": len(edges),
        "highways": sorted({item["tags"].get("highway", "") for item in edges}),
    }


def merge_corridors(
    paths: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    projection: Projection,
    tile: TileSpec,
) -> list[dict[str, Any]]:
    junction_ids = {row["id"] for row in junctions}
    groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        endpoint_key = tuple(sorted((path["start"], path["end"])))
        name = normalize_name(path["tags"])
        highway = path["tags"].get("highway", "residential")
        signature = f"name:{name}" if name else f"class:{highway}"
        groups[(*endpoint_key, signature)].append(path)

    roads: list[dict[str, Any]] = []
    for group_key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        parts = [member["metric"] for member in members]
        geometry_metric = parts[0] if len(parts) == 1 else MultiLineString(parts)
        all_way_ids = sorted({way_id for member in members for way_id in member["member_way_ids"]})
        dominant = min(
            members,
            key=lambda member: ROAD_LEVEL.get(member["tags"].get("highway", "service"), 9),
        )
        tags = dominant["tags"]
        lane_details = [parse_lanes(member["tags"]) for member in members]
        total_lanes = sum(detail["total"] for detail in lane_details)
        widths = [infer_road_width(member["tags"]) for member in members]
        total_width = sum(value for value, _ in widths)
        speed_values = [parse_speed_kph(member["tags"]) for member in members]
        speed_kph = max(value for value, _ in speed_values)
        identifier_seed = f"{group_key}|{all_way_ids}".encode()
        digest = hashlib.sha1(identifier_seed).hexdigest()[:10]
        road_id = f"road/r{all_way_ids[0]}-{digest}"
        start, end = members[0]["start"], members[0]["end"]
        from_intersection = start if start in junction_ids else None
        to_intersection = end if end in junction_ids else None
        name_values = sorted({member["tags"].get("name", "") for member in members if member["tags"].get("name")})
        turn_lanes = sorted(
            {
                value
                for member in members
                for key, value in member["tags"].items()
                if key.startswith("turn:lanes") and value
            }
        )
        properties = {
            "name": name_values[0] if len(name_values) == 1 else " / ".join(name_values),
            "ref": tags.get("ref", ""),
            "highway": tags.get("highway", ""),
            "highway_members": sorted({value for member in members for value in member["highways"]}),
            "logical_length_m": round(max(member["length_m"] for member in members), 3),
            "carriageway_length_m": round(sum(member["length_m"] for member in members), 3),
            "width_m": round(total_width, 2),
            "width_inferred": all(inferred for _, inferred in widths),
            "total_lanes": total_lanes,
            "lane_data_inferred": all(detail["inferred"] for detail in lane_details),
            "speed_kph": speed_kph,
            "speed_inferred": all(inferred for _, inferred in speed_values),
            "carriageway_count": len(members),
            "member_osm_way_ids": all_way_ids,
            "member_edge_count": sum(member["member_edge_count"] for member in members),
            "from_intersection": from_intersection,
            "to_intersection": to_intersection,
            "boundary_endpoint_count": int(from_intersection is None) + int(to_intersection is None),
            "oneway_members": sum(parse_lanes(member["tags"])["oneway"] for member in members),
            "turn_lanes": turn_lanes,
            "is_link": tags.get("highway", "").endswith("_link"),
            "source": "collapsed_osm_corridor",
        }
        roads.append(
            {
                "id": road_id,
                "metric": geometry_metric,
                "geometry": projection.unproject(geometry_metric),
                "properties": properties,
                "path_members": members,
            }
        )
    return roads


def build_parcels(
    roads: list[dict[str, Any]],
    raw_areas: Iterable[Any],
    projection: Projection,
    tile: TileSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    buffered_roads = []
    for road in roads:
        half_width = road["properties"]["width_m"] / 2 + PARCEL_ROAD_CLEARANCE_M
        buffered_roads.append(road["metric"].buffer(half_width, cap_style="flat", join_style="round"))
    road_space = unary_union(buffered_roads).intersection(tile.metric_polygon) if buffered_roads else None
    available_land = tile.metric_polygon.difference(road_space) if road_space is not None else tile.metric_polygon

    landuse_sources: list[dict[str, Any]] = []
    for raw in raw_areas:
        projected = projection.project(raw.geometry).intersection(tile.metric_polygon)
        for part_number, polygon in enumerate(iter_polygons(projected)):
            if polygon.area < 50:
                continue
            landuse_sources.append(
                {
                    "id": f"landuse_source/osm/{raw.osm_id}/{part_number}",
                    "metric": polygon,
                    "geometry": projection.unproject(polygon),
                    "properties": {
                        "osm_area_id": raw.osm_id,
                        "land_use": classify_landuse(raw.tags),
                        "area_m2": round(polygon.area, 3),
                        "tags": raw.tags,
                    },
                }
            )
    source_geometries = [row["metric"] for row in landuse_sources]
    source_tree = STRtree(source_geometries) if source_geometries else None

    parcels: list[dict[str, Any]] = []
    discarded_small_area = 0.0
    for polygon in sorted(iter_polygons(available_land), key=lambda item: (-item.area, item.centroid.x, item.centroid.y)):
        if polygon.area < MIN_PARCEL_AREA_M2:
            discarded_small_area += polygon.area
            continue
        overlaps: list[tuple[float, int]] = []
        if source_tree is not None:
            for source_index in source_tree.query(polygon):
                index = int(source_index)
                overlap_area = polygon.intersection(source_geometries[index]).area
                if overlap_area > 0:
                    overlaps.append((overlap_area, index))
        overlaps.sort(reverse=True)
        best_overlap = overlaps[0][0] if overlaps else 0.0
        best_source = landuse_sources[overlaps[0][1]] if overlaps else None
        land_use = best_source["properties"]["land_use"] if best_source and best_overlap / polygon.area >= 0.08 else "unclassified"
        source_ids = [landuse_sources[index]["id"] for area, index in overlaps[:5] if area / polygon.area >= 0.01]
        min_x, min_y, max_x, max_y = polygon.bounds
        compactness = 4 * math.pi * polygon.area / max(polygon.length * polygon.length, 1e-9)
        touches_boundary = polygon.boundary.distance(tile.metric_polygon.boundary) < 0.25
        seed = f"{round(polygon.centroid.x, 2)}:{round(polygon.centroid.y, 2)}:{round(polygon.area, 1)}"
        parcel_id = f"parcel/p{hashlib.sha1(seed.encode()).hexdigest()[:10]}"
        properties = {
            "land_use": land_use,
            "area_m2": round(polygon.area, 3),
            "perimeter_m": round(polygon.length, 3),
            "bbox_width_m": round(max_x - min_x, 3),
            "bbox_height_m": round(max_y - min_y, 3),
            "compactness": round(compactness, 5),
            "centroid_x_m": round(polygon.centroid.x, 3),
            "centroid_y_m": round(polygon.centroid.y, 3),
            "touches_tile_boundary": touches_boundary,
            "landuse_coverage_ratio": round(best_overlap / polygon.area, 4),
            "source_landuse_ids": source_ids,
            "source": "road_space_face",
        }
        parcels.append(
            {
                "id": parcel_id,
                "metric": polygon,
                "geometry": projection.unproject(polygon),
                "properties": properties,
            }
        )
    planning_area = available_land.area
    retained_area = sum(row["metric"].area for row in parcels)
    return parcels, landuse_sources, {
        "tile_area_m2": tile.metric_polygon.area,
        "road_space_area_m2": round(road_space.area if road_space is not None else 0.0, 3),
        "planning_area_m2": round(planning_area, 3),
        "retained_parcel_area_m2": round(retained_area, 3),
        "discarded_small_area_m2": round(discarded_small_area, 3),
        "planning_area_coverage_ratio": round(retained_area / max(planning_area, 1), 5),
        "labeled_parcel_count": sum(row["properties"]["land_use"] != "unclassified" for row in parcels),
        "unclassified_parcel_count": sum(row["properties"]["land_use"] == "unclassified" for row in parcels),
        "landuse_source_count": len(landuse_sources),
    }


PHASE_ACTIONS = [
    {"phase": 0, "name": "EW_straight", "movements": ["E_W", "W_E"]},
    {"phase": 1, "name": "NS_straight", "movements": ["N_S", "S_N"]},
    {"phase": 2, "name": "EW_left", "movements": ["E_S", "W_N"]},
    {"phase": 3, "name": "NS_left", "movements": ["N_E", "S_W"]},
    {"phase": 4, "name": "W_combined", "movements": ["W_E", "W_N"]},
    {"phase": 5, "name": "E_combined", "movements": ["E_W", "E_S"]},
    {"phase": 6, "name": "N_combined", "movements": ["N_S", "N_E"]},
    {"phase": 7, "name": "S_combined", "movements": ["S_N", "S_W"]},
]


def build_signal_controls(
    raw_signals: dict[int, Any],
    junctions: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    raw_graph: nx.MultiGraph,
    projection: Projection,
    tile: TileSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    junction_points = [row["metric"] for row in junctions]
    junction_tree = STRtree(junction_points) if junction_points else None
    signals: list[dict[str, Any]] = []
    signals_by_junction: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_signals.values():
        point = projection.project(Point(raw.coordinate))
        if not tile.metric_polygon.covers(point):
            continue
        attached: str | None = None
        distance_m: float | None = None
        if junction_tree is not None:
            candidate_indices = junction_tree.query(point.buffer(SIGNAL_ATTACH_M))
            if len(candidate_indices):
                index = min((int(value) for value in candidate_indices), key=lambda value: point.distance(junction_points[value]))
                distance_m = point.distance(junction_points[index])
                if distance_m <= SIGNAL_ATTACH_M:
                    attached = junctions[index]["id"]
        kind = "vehicle_signal_head" if attached else "crossing_or_unattached_signal"
        record = {
            "id": f"signal/osm-node/{raw.osm_id}",
            "metric": point,
            "geometry": Point(raw.coordinate),
            "properties": {
                "osm_node_id": raw.osm_id,
                "kind": kind,
                "attached_intersection": attached,
                "attachment_distance_m": round(distance_m, 3) if distance_m is not None else None,
                "on_road_graph": raw.osm_id in raw_graph,
                "tags": raw.tags,
            },
        }
        signals.append(record)
        if attached:
            signals_by_junction[attached].append(record)

    roads_by_junction: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for road in roads:
        for endpoint in (road["properties"]["from_intersection"], road["properties"]["to_intersection"]):
            if endpoint:
                roads_by_junction[endpoint].append(road)
    junction_by_id = {row["id"]: row for row in junctions}
    controls: list[dict[str, Any]] = []
    for junction_id, heads in sorted(signals_by_junction.items()):
        junction = junction_by_id[junction_id]
        approaches = [build_approach(junction, road) for road in roads_by_junction[junction_id]]
        approaches = [approach for approach in approaches if approach is not None]
        slots: dict[str, dict[str, Any]] = {}
        for direction in ("N", "E", "S", "W"):
            matches = [approach for approach in approaches if approach["compass"] == direction]
            slots[direction] = {
                "exists": bool(matches),
                "road_ids": [item["road_id"] for item in matches],
                "incoming_lanes": sum(item["incoming_lanes"] for item in matches),
                "approaches": matches,
            }
        available_directions = {direction for direction, slot in slots.items() if slot["exists"]}
        phase_mask = []
        for phase in PHASE_ACTIONS:
            valid_movement_count = sum(
                movement[0] in available_directions and movement[2] in available_directions
                for movement in phase["movements"]
            )
            phase_mask.append(valid_movement_count > 0)
        control = {
            "intersection_id": junction_id,
            "signal_head_ids": [head["id"] for head in heads],
            "signal_head_count": len(heads),
            "approach_slots": slots,
            "available_phase_mask": phase_mask,
            "neighbor_intersection_ids": [],
            "static_features": {
                "logical_road_degree": len({road["id"] for road in roads_by_junction[junction_id]}),
                "approach_count": len(approaches),
                "cardinal_approach_count": len(available_directions),
                "total_incoming_lanes": sum(item["incoming_lanes"] for item in approaches),
            },
        }
        controls.append(control)
        junction["properties"]["signalized"] = True
        junction["properties"]["signal_head_count"] = len(heads)
    for junction in junctions:
        incident = {road["id"] for road in roads_by_junction[junction["id"]]}
        junction["properties"]["logical_road_degree"] = len(incident)
    attached_count = sum(row["properties"]["attached_intersection"] is not None for row in signals)
    return signals, controls, {
        "raw_signal_count": len(signals),
        "attached_signal_head_count": attached_count,
        "unattached_signal_count": len(signals) - attached_count,
        "signalized_intersection_count": len(controls),
        "attachment_ratio": round(attached_count / max(1, len(signals)), 5),
        "mean_heads_per_signalized_intersection": round(attached_count / max(1, len(controls)), 3),
    }


def build_approach(junction: dict[str, Any], road: dict[str, Any]) -> dict[str, Any] | None:
    center = junction["metric"]
    best: tuple[float, LineString, bool] | None = None
    for line in iter_lines(road["metric"]):
        start_distance = center.distance(Point(line.coords[0]))
        end_distance = center.distance(Point(line.coords[-1]))
        candidate = (min(start_distance, end_distance), line, start_distance <= end_distance)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    _, line, starts_at_junction = best
    sample_distance = min(30.0, max(1.0, line.length * 0.25))
    if starts_at_junction:
        outside = line.interpolate(sample_distance)
    else:
        outside = line.interpolate(max(0.0, line.length - sample_distance))
    dx, dy = outside.x - center.x, outside.y - center.y
    if abs(dx) + abs(dy) < 1e-6:
        return None
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    compass = bearing_to_compass(bearing)
    properties = road["properties"]
    incoming_lanes = max(1, math.ceil(properties["total_lanes"] / 2))
    return {
        "road_id": road["id"],
        "compass": compass,
        "bearing_deg": round(bearing, 2),
        "incoming_lanes": incoming_lanes,
        "total_lanes": properties["total_lanes"],
        "speed_kph": properties["speed_kph"],
        "turn_lanes": properties["turn_lanes"],
        "road_class": properties["highway"],
        "oneway_members": properties["oneway_members"],
    }


def bearing_to_compass(bearing: float) -> str:
    if bearing < 45 or bearing >= 315:
        return "N"
    if bearing < 135:
        return "E"
    if bearing < 225:
        return "S"
    return "W"


def make_edge(
    relation: str,
    source: str,
    target: str,
    properties: dict[str, Any],
    *,
    directed: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"edge/{relation}/{source}/{target}",
        "relation": relation,
        "source": source,
        "target": target,
        "directed": directed,
        "properties": properties,
    }


def build_relations(
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    signal_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        source, target = row["source"], row["target"]
        if not row["directed"] and source > target:
            source, target = target, source
            row = {**row, "source": source, "target": target}
        key = (row["relation"], source, target)
        if key not in edge_map:
            edge_map[key] = row

    roads_by_junction: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for road in roads:
        for junction_id in (road["properties"]["from_intersection"], road["properties"]["to_intersection"]):
            if junction_id:
                roads_by_junction[junction_id].append(road)
                add(make_edge("road_intersection", road["id"], junction_id, {"endpoint": True}))
    for junction_id, incident in roads_by_junction.items():
        unique = {row["id"]: row for row in incident}
        for left, right in combinations(sorted(unique), 2):
            add(make_edge("road_road", left, right, {"via_intersection": junction_id}))

    parcel_road_ids: defaultdict[str, set[str]] = defaultdict(set)
    road_geometries = [row["metric"] for row in roads]
    road_tree = STRtree(road_geometries) if road_geometries else None
    if road_tree:
        for parcel in parcels:
            for road_index in road_tree.query(parcel["metric"].buffer(PARCEL_ACROSS_ROAD_M)):
                road = roads[int(road_index)]
                width = road["properties"]["width_m"]
                distance = parcel["metric"].distance(road["metric"])
                threshold = width / 2 + PARCEL_ROAD_CLEARANCE_M + 3
                if distance <= threshold:
                    facing = parcel["metric"].boundary.intersection(
                        road["metric"].buffer(threshold, cap_style="flat")
                    ).length
                    if facing >= 1:
                        parcel_road_ids[parcel["id"]].add(road["id"])
                        add(
                            make_edge(
                                "parcel_road",
                                parcel["id"],
                                road["id"],
                                {"distance_m": round(distance, 3), "facing_boundary_m": round(facing, 3)},
                            )
                        )

    junction_geometries = [row["metric"] for row in junctions]
    junction_tree = STRtree(junction_geometries) if junction_geometries else None
    if junction_tree:
        for parcel in parcels:
            for junction_index in junction_tree.query(parcel["metric"].buffer(55)):
                junction = junctions[int(junction_index)]
                distance = parcel["metric"].distance(junction["metric"])
                if distance <= 45:
                    add(
                        make_edge(
                            "parcel_intersection",
                            parcel["id"],
                            junction["id"],
                            {"distance_m": round(distance, 3)},
                        )
                    )

    parcel_geometries = [row["metric"] for row in parcels]
    parcel_tree = STRtree(parcel_geometries) if parcel_geometries else None
    if parcel_tree:
        for index, parcel in enumerate(parcels):
            for other_value in parcel_tree.query(parcel["metric"].buffer(PARCEL_ACROSS_ROAD_M)):
                other_index = int(other_value)
                if other_index <= index:
                    continue
                other = parcels[other_index]
                distance = parcel["metric"].distance(other["metric"])
                common_roads = sorted(parcel_road_ids[parcel["id"]].intersection(parcel_road_ids[other["id"]]))
                if distance <= 0.5:
                    mode = "shared_boundary"
                elif distance <= PARCEL_ACROSS_ROAD_M and common_roads:
                    mode = "across_road"
                else:
                    continue
                shared = parcel["metric"].boundary.intersection(other["metric"].boundary.buffer(0.25)).length
                add(
                    make_edge(
                        "parcel_parcel",
                        parcel["id"],
                        other["id"],
                        {
                            "mode": mode,
                            "distance_m": round(distance, 3),
                            "shared_boundary_m": round(shared, 3),
                            "separating_road_ids": common_roads,
                        },
                    )
                )

    for signal in signals:
        junction_id = signal["properties"]["attached_intersection"]
        if junction_id:
            add(
                make_edge(
                    "signal_intersection",
                    signal["id"],
                    junction_id,
                    {"distance_m": signal["properties"]["attachment_distance_m"]},
                )
            )
    return list(edge_map.values())


def add_signal_neighbors(
    edges: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    neighbor_count: int = 4,
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(row["id"] for row in junctions)
    for road in roads:
        left = road["properties"]["from_intersection"]
        right = road["properties"]["to_intersection"]
        if left and right and left != right:
            weight = road["properties"]["logical_length_m"]
            if graph.has_edge(left, right):
                graph[left][right]["weight"] = min(graph[left][right]["weight"], weight)
            else:
                graph.add_edge(left, right, weight=weight)
    signalized = {control["intersection_id"] for control in controls}
    control_by_id = {control["intersection_id"]: control for control in controls}
    for source in sorted(signalized):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        nearest = sorted(
            ((distance, target) for target, distance in lengths.items() if target in signalized and target != source),
            key=lambda item: (item[0], item[1]),
        )[:neighbor_count]
        control_by_id[source]["neighbor_intersection_ids"] = [target for _, target in nearest]
        for rank, (distance, target) in enumerate(nearest, start=1):
            edges.append(
                make_edge(
                    "intersection_neighbor",
                    source,
                    target,
                    {"network_distance_m": round(distance, 3), "rank": rank},
                    directed=True,
                )
            )


def write_outputs(
    output_dir: Path,
    pbf: Path,
    tile: TileSpec,
    projection: Projection,
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    landuse_sources: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    signal_controls: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    pipeline_version: str = "v2_clean",
) -> None:
    metadata = {
        "pipeline_version": pipeline_version,
        "generated_at": utc_now(),
        "tile": tile.as_dict(),
        "metric_crs": projection.crs.to_string(),
        "input": str(pbf),
    }
    layers = {
        "roads.geojson": roads,
        "intersections.geojson": junctions,
        "parcels.geojson": parcels,
        "landuse_source.geojson": landuse_sources,
        "signals.geojson": signals,
    }
    for filename, rows in layers.items():
        write_geojson(
            output_dir / filename,
            [geojson_feature(row["id"], row["geometry"], row["properties"]) for row in rows],
            metadata,
        )
    write_jsonl(output_dir / "edges.jsonl", edges)

    observation_schema = {
        "description": "X-Light-compatible runtime observation placeholders; values come from SUMO, not OSM.",
        "per_approach_features": [
            {"name": "queue_length", "unit": "vehicles", "source": "SUMO lane halting count"},
            {"name": "occupancy", "unit": "ratio", "source": "SUMO lane-area detector"},
            {"name": "flow", "unit": "vehicles_per_interval", "source": "SUMO detector/edge counts"},
            {"name": "stopped_vehicles", "unit": "vehicles", "source": "SUMO speed threshold"},
        ],
        "intersection_features": [
            {"name": "current_phase", "dtype": "int8", "range": [0, 7]},
            {"name": "phase_elapsed_seconds", "dtype": "float32"},
        ],
        "decision_interval_seconds": 15,
        "cardinal_slot_order": ["N", "E", "S", "W"],
        "missing_approach_policy": "zero padding plus exists mask",
    }
    reward_schema = {
        "runtime_only": True,
        "terms": ["queue_length", "wait_time", "delay_time", "pressure"],
        "note": "Weights are experiment configuration; no dynamic traffic values are fabricated from PBF.",
    }
    write_json(
        output_dir / "traffic_control.json",
        {
            **metadata,
            "phase_actions": PHASE_ACTIONS,
            "observation_schema": observation_schema,
            "reward_schema": reward_schema,
            "controls": signal_controls,
        },
    )

    topology_nodes: list[dict[str, Any]] = []
    for node_type, rows in (
        ("road", roads),
        ("intersection", junctions),
        ("parcel", parcels),
        ("signal_head", signals),
    ):
        for row in rows:
            centroid = row["metric"].centroid
            topology_nodes.append(
                {
                    "id": row["id"],
                    "node_type": node_type,
                    "x_m": round(centroid.x, 3),
                    "y_m": round(centroid.y, 3),
                    "properties": row["properties"],
                }
            )
    write_json(
        output_dir / "topology.json",
        {
            **metadata,
            "nodes": topology_nodes,
            "edges": edges,
            "traffic_control_file": "traffic_control.json",
            "geometry_layers": list(layers),
        },
        compact=True,
    )
    write_json(
        output_dir / "schema.json",
        {
            "schema_version": "urban-topology-1.0",
            "coordinate_systems": {
                "geojson": "EPSG:4326",
                "metric_features": tile.crs,
            },
            "node_types": {
                "parcel": "Non-overlapping planning face after subtracting road space; OSM land-use provides a traceable label.",
                "road": "Logical corridor between macro intersections/tile endpoints; may contain multiple OSM carriageways.",
                "intersection": "Cluster of same-grade nearby OSM branching/signal nodes.",
                "signal_head": "Raw static OSM traffic signal attached to a macro intersection when possible.",
            },
            "edge_types": {
                "road_road": "Logical roads meet at a macro intersection.",
                "road_intersection": "Road endpoint is the macro intersection.",
                "parcel_parcel": "Shared boundary or adjacency across a common road.",
                "parcel_road": "Parcel faces a road-space boundary.",
                "parcel_intersection": "Parcel lies within 45 m of a macro intersection.",
                "signal_intersection": "Raw signal head belongs to a macro intersection.",
                "intersection_neighbor": "Directed nearest signalized-neighbor relation by road-network distance.",
            },
            "id_stability": "IDs derive from OSM member IDs or stable geometry hashes within the fixed tile/grid definition.",
            "dynamic_data_boundary": observation_schema,
        },
    )


def build_quality_report(
    pbf: Path,
    tile: TileSpec,
    extracted: Any,
    raw_graph: nx.MultiGraph,
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    cluster_stats: dict[str, Any],
    path_stats: dict[str, Any],
    parcel_stats: dict[str, Any],
    signal_stats: dict[str, Any],
    pipeline_version: str = "v2_clean",
) -> dict[str, Any]:
    incident_road_count: Counter[str] = Counter()
    for edge in edges:
        if edge["relation"] == "road_intersection":
            road_id = edge["source"] if edge["source"].startswith("road/") else edge["target"]
            incident_road_count[road_id] += 1
    graph = nx.Graph()
    graph.add_nodes_from(row["id"] for row in junctions)
    for road in roads:
        left = road["properties"]["from_intersection"]
        right = road["properties"]["to_intersection"]
        if left and right and left != right:
            graph.add_edge(left, right)
    connected_components = list(nx.connected_components(graph))
    parcel_union_area = unary_union([row["metric"] for row in parcels]).area if parcels else 0.0
    parcel_sum_area = sum(row["metric"].area for row in parcels)
    warnings: list[str] = []
    if cluster_stats["oversized_clusters"]:
        warnings.append(
            f"{cluster_stats['oversized_clusters']} junction clusters exceed 100 m diameter and should be inspected visually."
        )
    if signal_stats["attachment_ratio"] < 0.9:
        warnings.append("Less than 90% of raw signal heads attach to a vehicle macro intersection.")
    if parcel_stats["planning_area_coverage_ratio"] < 0.98:
        warnings.append("Small-face filtering leaves more than 2% of planning area uncovered.")
    inferred_lane_roads = sum(row["properties"]["lane_data_inferred"] for row in roads)
    size_key = f"{int(tile.size_m / 1000)}km"
    invariants = {
        f"tile_is_exact_{size_key}_square": {
            "passed": math.isclose(
                tile.metric_polygon.area,
                tile.size_m * tile.size_m,
                abs_tol=1.0e-6,
            ),
            "evidence": {"bounds_m": list(tile.bounds_m), "area_m2": tile.metric_polygon.area},
        },
        "parcel_polygons_do_not_overlap": {
            "passed": abs(parcel_sum_area - parcel_union_area) < 1.0,
            "evidence": {"sum_area_m2": round(parcel_sum_area, 3), "union_area_m2": round(parcel_union_area, 3)},
        },
        "all_geometry_valid": {
            "passed": all(row["metric"].is_valid and not row["metric"].is_empty for row in roads + junctions + parcels),
            "evidence": {
                "invalid_count": sum(not row["metric"].is_valid for row in roads + junctions + parcels),
                "empty_count": sum(row["metric"].is_empty for row in roads + junctions + parcels),
            },
        },
        "signal_controls_have_static_approaches": {
            "passed": all(control["static_features"]["approach_count"] > 0 for control in controls),
            "evidence": {
                "control_count": len(controls),
                "zero_approach_count": sum(control["static_features"]["approach_count"] == 0 for control in controls),
            },
        },
    }
    return {
        "pipeline_version": pipeline_version,
        "generated_at": utc_now(),
        "input": str(pbf),
        "tile": tile.as_dict(),
        "counts": {
            "raw_osm_road_ways": len(extracted.roads),
            "raw_graph_nodes": raw_graph.number_of_nodes(),
            "raw_graph_edges": raw_graph.number_of_edges(),
            "logical_roads": len(roads),
            "macro_intersections": len(junctions),
            "planning_parcels": len(parcels),
            "raw_signal_heads": len(signals),
            "signalized_intersections": len(controls),
            "edges": len(edges),
            "edge_types": relation_counts(edges),
        },
        "metrics": {
            "road_way_to_logical_road_ratio": round(len(roads) / max(1, len(extracted.roads)), 4),
            "multi_carriageway_corridor_count": sum(row["properties"]["carriageway_count"] > 1 for row in roads),
            "link_corridor_count": sum(row["properties"]["is_link"] for row in roads),
            "roads_without_macro_intersection": sum(incident_road_count[row["id"]] == 0 for row in roads),
            "lane_inference_ratio": round(inferred_lane_roads / max(1, len(roads)), 4),
            "junction_graph_component_count": len(connected_components),
            "largest_junction_component": max((len(component) for component in connected_components), default=0),
            "isolated_macro_intersections": sum(graph.degree(row["id"]) == 0 for row in junctions),
            "signal_head_to_agent_ratio": round(len(signals) / max(1, len(controls)), 3),
            "parcel_overlap_area_m2": round(max(0.0, parcel_sum_area - parcel_union_area), 6),
        },
        "junction_clustering": cluster_stats,
        "path_collapse": path_stats,
        "parcel_generation": parcel_stats,
        "signal_processing": signal_stats,
        "invariants": invariants,
        "warnings": warnings,
        "extractor_stats": dict(extracted.stats),
        "dynamic_data_note": "OSM/PBF supplies static topology only. Queue, phase history, occupancy, flow and stopped-vehicle values are populated by SUMO at training/runtime.",
    }


def plot_overview(
    path: Path,
    tile: TileSpec,
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    pipeline_version: str = "v2",
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
    for axis in axes.flat:
        axis.set_xlim(tile.bounds_m[0], tile.bounds_m[2])
        axis.set_ylim(tile.bounds_m[1], tile.bounds_m[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])

    road_lines: list[list[tuple[float, float]]] = []
    road_colors: list[str] = []
    road_widths: list[float] = []
    class_colors = {
        "motorway": "#6d3a9c",
        "trunk": "#8c4fa8",
        "primary": "#d1495b",
        "secondary": "#e78b42",
        "tertiary": "#e3bd48",
        "residential": "#59656f",
        "service": "#a2a9ad",
    }
    for road in roads:
        highway = road["properties"]["highway"].replace("_link", "")
        for line in iter_lines(road["metric"]):
            road_lines.append(list(line.coords))
            road_colors.append(class_colors.get(highway, "#737f87"))
            road_widths.append(max(0.45, 2.5 - ROAD_LEVEL.get(road["properties"]["highway"], 8) * 0.23))
    axes[0, 0].add_collection(LineCollection(road_lines, colors=road_colors, linewidths=road_widths, alpha=0.95))
    axes[0, 0].scatter(
        [row["metric"].x for row in junctions],
        [row["metric"].y for row in junctions],
        s=6,
        color="#1769aa",
        zorder=3,
    )
    axes[0, 0].set_title(f"Logical roads ({len(roads)}) + macro intersections ({len(junctions)})")

    parcel_patches: list[MplPolygon] = []
    parcel_colors: list[str] = []
    for parcel in parcels:
        for polygon in iter_polygons(parcel["metric"]):
            parcel_patches.append(MplPolygon(list(polygon.exterior.coords), closed=True))
            parcel_colors.append(LAND_COLORS.get(parcel["properties"]["land_use"], "#eeeeee"))
    if parcel_patches:
        axes[0, 1].add_collection(
            PatchCollection(parcel_patches, facecolors=parcel_colors, edgecolors="#777", linewidths=0.25, alpha=0.88)
        )
    axes[0, 1].add_collection(LineCollection(road_lines, colors="#2f3539", linewidths=0.55, alpha=0.8))
    axes[0, 1].scatter(
        [row["metric"].x for row in junctions],
        [row["metric"].y for row in junctions],
        s=5,
        color="#1769aa",
        zorder=3,
    )
    axes[0, 1].set_title(f"Non-overlapping planning parcels ({len(parcels)})")

    axes[1, 0].add_collection(LineCollection(road_lines, colors="#9aa2a8", linewidths=0.45, alpha=0.65))
    attached = [row for row in signals if row["properties"]["attached_intersection"]]
    unattached = [row for row in signals if not row["properties"]["attached_intersection"]]
    if attached:
        axes[1, 0].scatter(
            [row["metric"].x for row in attached],
            [row["metric"].y for row in attached],
            s=5,
            color="#e52033",
            alpha=0.7,
            label="attached signal head",
        )
    if unattached:
        axes[1, 0].scatter(
            [row["metric"].x for row in unattached],
            [row["metric"].y for row in unattached],
            s=9,
            marker="x",
            color="#ff9f1c",
            label="unattached/crossing",
        )
    signalized = [row for row in junctions if row["properties"]["signalized"]]
    axes[1, 0].scatter(
        [row["metric"].x for row in signalized],
        [row["metric"].y for row in signalized],
        s=[18 + row["properties"]["signal_head_count"] * 1.5 for row in signalized],
        facecolors="none",
        edgecolors="#961526",
        linewidths=0.9,
        label="signal-control agent",
    )
    axes[1, 0].set_title(f"{len(signals)} signal heads → {len(signalized)} control intersections")
    axes[1, 0].legend(loc="lower right", fontsize=7)

    centers: dict[str, Point] = {}
    for rows in (roads, junctions, parcels, signals):
        centers.update({row["id"]: row["metric"].centroid for row in rows})
    relation_color = {
        "parcel_parcel": "#6b8e23",
        "parcel_road": "#7d5fff",
        "parcel_intersection": "#00a6a6",
        "road_intersection": "#d1495b",
    }
    relation_lines: defaultdict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for edge in edges:
        if edge["relation"] not in relation_color or edge["source"] not in centers or edge["target"] not in centers:
            continue
        source, target = centers[edge["source"]], centers[edge["target"]]
        relation_lines[edge["relation"]].append([(source.x, source.y), (target.x, target.y)])
    for relation, lines in relation_lines.items():
        axes[1, 1].add_collection(
            LineCollection(lines, colors=relation_color[relation], linewidths=0.22, alpha=0.22, label=relation)
        )
    axes[1, 1].scatter(
        [row["metric"].centroid.x for row in parcels],
        [row["metric"].centroid.y for row in parcels],
        s=3,
        color="#6b8e23",
        label="parcel node",
    )
    axes[1, 1].add_collection(LineCollection(road_lines, colors="#535b60", linewidths=0.45, alpha=0.65))
    axes[1, 1].scatter(
        [row["metric"].x for row in junctions],
        [row["metric"].y for row in junctions],
        s=4,
        color="#1769aa",
        label="intersection node",
    )
    axes[1, 1].set_title("Unified cross-topology relations")
    axes[1, 1].legend(loc="lower right", fontsize=6, markerscale=2)
    size_label = f"{tile.size_m / 1000:g} km"
    figure.suptitle(
        f"{pipeline_version} cleaned topology · {tile.tile_id} · "
        f"exact {size_label} × {size_label}",
        fontsize=16,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_signal_detail(
    path: Path,
    tile: TileSpec,
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    junction_by_id = {row["id"]: row for row in junctions}
    control_by_id = {row["intersection_id"]: row for row in controls}
    road_lines = [list(line.coords) for road in roads for line in iter_lines(road["metric"])]
    axes[0].add_collection(LineCollection(road_lines, colors="#9aa2a8", linewidths=0.4, alpha=0.55))
    neighbor_lines = []
    for control in controls:
        source = junction_by_id[control["intersection_id"]]["metric"]
        for target_id in control["neighbor_intersection_ids"]:
            if target_id in junction_by_id:
                target = junction_by_id[target_id]["metric"]
                neighbor_lines.append([(source.x, source.y), (target.x, target.y)])
    axes[0].add_collection(LineCollection(neighbor_lines, colors="#7d5fff", linewidths=0.45, alpha=0.28))
    signalized_junctions = [junction_by_id[control["intersection_id"]] for control in controls]
    if signalized_junctions:
        axes[0].scatter(
            [row["metric"].x for row in signalized_junctions],
            [row["metric"].y for row in signalized_junctions],
            s=[20 + row["properties"]["signal_head_count"] * 1.5 for row in signalized_junctions],
            color="#c9182b",
            alpha=0.75,
        )
    axes[0].set_xlim(tile.bounds_m[0], tile.bounds_m[2])
    axes[0].set_ylim(tile.bounds_m[1], tile.bounds_m[3])
    axes[0].set_title("Signalized agents and four nearest network neighbors")

    detail = max(
        controls,
        key=lambda row: (row["static_features"]["cardinal_approach_count"], row["signal_head_count"]),
        default=None,
    )
    if detail:
        center = junction_by_id[detail["intersection_id"]]["metric"]
        radius = 260
        local_lines = [line for line in road_lines if LineString(line).distance(center) <= radius]
        axes[1].add_collection(LineCollection(local_lines, colors="#6d777d", linewidths=1.0, alpha=0.8))
        direction_colors = {"N": "#1677ff", "E": "#f59e0b", "S": "#22a06b", "W": "#9b59b6"}
        for direction, slot in detail["approach_slots"].items():
            for approach in slot["approaches"]:
                angle = math.radians(approach["bearing_deg"])
                dx, dy = math.sin(angle) * 125, math.cos(angle) * 125
                axes[1].arrow(
                    center.x,
                    center.y,
                    dx,
                    dy,
                    width=2.2,
                    head_width=18,
                    color=direction_colors[direction],
                    length_includes_head=True,
                    alpha=0.85,
                )
                axes[1].text(
                    center.x + dx * 1.08,
                    center.y + dy * 1.08,
                    f"{direction}: {approach['incoming_lanes']}L",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=direction_colors[direction],
                )
        axes[1].scatter([center.x], [center.y], marker="s", s=65, color="#c9182b", zorder=5)
        mask = "".join("1" if value else "0" for value in detail["available_phase_mask"])
        axes[1].text(
            center.x - radius + 10,
            center.y - radius + 10,
            f"{detail['intersection_id']}\nheads={detail['signal_head_count']}  phase_mask={mask}",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbb"},
        )
        axes[1].set_xlim(center.x - radius, center.x + radius)
        axes[1].set_ylim(center.y - radius, center.y + radius)
        axes[1].set_title("Fixed N/E/S/W approach slots for X-Light/SUMO")
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Traffic-control topology · {tile.tile_id}", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)

MAX_CLUSTER_DIAMETER_M = 80.0
BRANCH_MERGE_M = 42.0
SIGNAL_TO_CLUSTER_M = 65.0


def cluster_junctions(
    graph: Any,
    raw_signals: dict[int, Any],
    projection: Projection,
    tile: TileSpec,
) -> tuple[list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    inside = tile.metric_polygon.buffer(10)
    branch_nodes = {
        node_id
        for node_id in graph.nodes
        if inside.covers(graph.nodes[node_id]["point"])
        and len(set(graph.neighbors(node_id))) >= 3
    }
    if not branch_nodes:
        return [], {}, {
            "branch_candidates": 0,
            "signal_candidates": 0,
            "clusters": 0,
            "oversized_clusters": 0,
            "max_cluster_members": 0,
            "max_cluster_diameter_m": 0,
        }
    branch_ids = sorted(branch_nodes)
    branch_points = [graph.nodes[node_id]["point"] for node_id in branch_ids]
    tree = STRtree(branch_points)
    levels = {node_id: node_level_signature(graph, node_id) for node_id in branch_ids}
    union_find = UnionFind(branch_ids)
    group_members: dict[int, set[int]] = {node_id: {node_id} for node_id in branch_ids}
    pairs: list[tuple[float, int, int]] = []
    for index, node_id in enumerate(branch_ids):
        for candidate_value in tree.query(branch_points[index].buffer(BRANCH_MERGE_M)):
            other_index = int(candidate_value)
            if other_index <= index:
                continue
            other_id = branch_ids[other_index]
            distance = branch_points[index].distance(branch_points[other_index])
            if distance <= BRANCH_MERGE_M and levels[node_id].intersection(levels[other_id]):
                pairs.append((distance, node_id, other_id))
    for _, left, right in sorted(pairs):
        root_left, root_right = union_find.find(left), union_find.find(right)
        if root_left == root_right:
            continue
        combined = group_members[root_left] | group_members[root_right]
        points = [graph.nodes[node_id]["point"] for node_id in combined]
        diameter = max((a.distance(b) for a in points for b in points), default=0.0)
        if diameter > MAX_CLUSTER_DIAMETER_M:
            continue
        union_find.union(root_left, root_right)
        new_root = union_find.find(root_left)
        group_members[new_root] = combined
        if root_left != new_root:
            group_members.pop(root_left, None)
        if root_right != new_root:
            group_members.pop(root_right, None)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for node_id in branch_ids:
        groups[union_find.find(node_id)].append(node_id)
    ordered_groups = sorted(groups.values(), key=lambda values: min(values))
    centers = [unary_union([graph.nodes[node_id]["point"] for node_id in members]).centroid for members in ordered_groups]
    center_tree = STRtree(centers)
    signal_members: defaultdict[int, list[int]] = defaultdict(list)
    nearby_signal_count = 0
    for signal_id in raw_signals:
        if signal_id not in graph:
            continue
        signal_point = graph.nodes[signal_id]["point"]
        if not inside.covers(signal_point):
            continue
        candidates = center_tree.query(signal_point.buffer(SIGNAL_TO_CLUSTER_M))
        if not len(candidates):
            continue
        signal_levels = node_level_signature(graph, signal_id)
        compatible = [
            int(index)
            for index in candidates
            if signal_levels.intersection(
                set().union(*(levels[node_id] for node_id in ordered_groups[int(index)]))
            )
        ]
        if not compatible:
            continue
        chosen = min(compatible, key=lambda index: signal_point.distance(centers[index]))
        if signal_point.distance(centers[chosen]) <= SIGNAL_TO_CLUSTER_M:
            signal_members[chosen].append(signal_id)
            nearby_signal_count += 1

    junctions: list[dict[str, Any]] = []
    raw_node_to_junction: dict[int, str] = {}
    for group_index, branches in enumerate(ordered_groups):
        center = centers[group_index]
        if not tile.metric_polygon.buffer(5).covers(center):
            continue
        members = sorted(set(branches) | set(signal_members[group_index]))
        branch_member_points = [graph.nodes[node_id]["point"] for node_id in branches]
        diameter = max((a.distance(b) for a in branch_member_points for b in branch_member_points), default=0.0)
        junction_id = f"intersection/j{min(branches)}"
        incident_way_ids = {
            data["way_id"] for node_id in branches for _, _, data in graph.edges(node_id, data=True)
        }
        properties = {
            "member_osm_node_ids": members,
            "branch_osm_node_ids": branches,
            "member_count": len(members),
            "raw_degree_max": max(len(set(graph.neighbors(node_id))) for node_id in branches),
            "incident_osm_way_count": len(incident_way_ids),
            "cluster_diameter_m": round(diameter, 3),
            "signalized": False,
            "signal_head_count": 0,
            "grade_layers": sorted(set().union(*(levels[node_id] for node_id in branches))),
        }
        junctions.append(
            {
                "id": junction_id,
                "metric": center,
                "geometry": projection.unproject(center),
                "properties": properties,
            }
        )
        # Branches alter road topology. Signal nodes are added only when they
        # are already branch nodes; degree-2 heads must not split roads.
        for node_id in branches:
            raw_node_to_junction[node_id] = junction_id
    return junctions, raw_node_to_junction, {
        "branch_candidates": len(branch_nodes),
        "signal_candidates": nearby_signal_count,
        "clusters": len(junctions),
        "oversized_clusters": 0,
        "max_cluster_members": max((row["properties"]["member_count"] for row in junctions), default=0),
        "max_cluster_diameter_m": round(
            max((row["properties"]["cluster_diameter_m"] for row in junctions), default=0), 3
        ),
        "hard_diameter_cap_m": MAX_CLUSTER_DIAMETER_M,
        "signal_nodes_participate_in_union": False,
    }

def extract_osm_evidence(pbf: Path, tile: TileSpec, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The tile's projected metric bounds are authoritative.  Inferring UTM
    # again from its WGS84 extraction bbox can switch zones for a tile that
    # touches a zone boundary (Hangzhou sits near 120°E).
    projection = Projection.for_crs(tile.crs)
    extracted = extract_pbf(pbf, tile.extraction_bbox, include_service=False)
    graph = build_raw_graph(extracted.roads.values(), projection, tile)
    junctions, raw_node_to_junction, cluster_stats = cluster_junctions(
        graph, extracted.signals, projection, tile
    )
    paths, path_stats = collapse_paths(graph, junctions, raw_node_to_junction, tile)
    roads = merge_corridors(paths, junctions, projection, tile)
    parcels, landuse_sources, parcel_stats = build_parcels(
        roads, extracted.areas.values(), projection, tile
    )
    all_signals, controls, original_signal_stats = build_signal_controls(
        extracted.signals, junctions, roads, graph, projection, tile
    )
    accepted_signals = [row for row in all_signals if row["properties"]["attached_intersection"]]
    rejected_signals = [row for row in all_signals if not row["properties"]["attached_intersection"]]
    add_movement_tables(controls)
    edges = build_relations(roads, junctions, parcels, accepted_signals, controls)
    add_signal_neighbors(edges, roads, junctions, controls)
    write_outputs(
        output_dir,
        pbf,
        tile,
        projection,
        roads,
        junctions,
        parcels,
        landuse_sources,
        accepted_signals,
        controls,
        edges,
        pipeline_version="v4_training",
    )
    audit_metadata = {
        "pipeline_version": "v4_training",
        "generated_at": utc_now(),
        "tile": tile.as_dict(),
        "role": "audit_only_not_part_of_training_topology",
        "reason": "signal node could not attach to a retained vehicle macro intersection",
    }
    write_geojson(
        output_dir / "audit_rejected_signals.geojson",
        [geojson_feature(row["id"], row["geometry"], row["properties"]) for row in rejected_signals],
        audit_metadata,
    )
    write_sumo_seed(output_dir / "sumo_static_seed.json", tile, roads, junctions, controls)
    signal_stats = {
        **original_signal_stats,
        "training_signal_head_count": len(accepted_signals),
        "rejected_signal_count": len(rejected_signals),
        "training_attachment_ratio": 1.0,
    }
    report = build_quality_report(
        pbf,
        tile,
        extracted,
        graph,
        roads,
        junctions,
        parcels,
        accepted_signals,
        controls,
        edges,
        cluster_stats,
        path_stats,
        parcel_stats,
        signal_stats,
        pipeline_version="v4_training",
    )
    report["counts"]["raw_osm_signal_nodes"] = len(all_signals)
    report["counts"]["training_signal_heads"] = len(accepted_signals)
    report["counts"]["rejected_signal_nodes"] = len(rejected_signals)
    report["metrics"]["signal_head_to_agent_ratio"] = round(len(accepted_signals) / max(1, len(controls)), 3)
    report["invariants"]["all_training_signals_are_attached"] = {
        "passed": all(row["properties"]["attached_intersection"] for row in accepted_signals),
        "evidence": {
            "training_signal_count": len(accepted_signals),
            "unattached_training_signal_count": sum(
                not row["properties"]["attached_intersection"] for row in accepted_signals
            ),
        },
    }
    report["invariants"]["all_controls_have_explicit_movements"] = {
        "passed": all(control["movements"] for control in controls),
        "evidence": {
            "control_count": len(controls),
            "zero_movement_count": sum(not control["movements"] for control in controls),
        },
    }
    report["refinements_from_v3"] = [
        "Removed unattached/crossing-only signal nodes from topology.json and signals.geojson.",
        "Retained rejected signal nodes only in audit_rejected_signals.geojson.",
        "Added explicit origin-destination movement tables to every signal controller.",
        "Added sumo_static_seed.json without fabricating dynamic traffic observations.",
    ]
    write_json(output_dir / "quality_report.json", report)
    plot_overview(
        output_dir / "overview.png",
        tile,
        roads,
        junctions,
        parcels,
        accepted_signals,
        edges,
        pipeline_version="v4",
    )
    plot_signal_detail(output_dir / "signal_topology.png", tile, roads, junctions, controls)
    return report


def add_movement_tables(controls: list[dict[str, Any]]) -> None:
    phase_lookup: dict[str, list[int]] = {}
    for phase in PHASE_ACTIONS:
        for movement in phase["movements"]:
            phase_lookup.setdefault(movement, []).append(phase["phase"])
    direction_index = {direction: index for index, direction in enumerate(DIRECTIONS)}
    for control in controls:
        movements: list[dict[str, Any]] = []
        slots = control["approach_slots"]
        for origin in DIRECTIONS:
            if not slots[origin]["exists"]:
                continue
            for destination in DIRECTIONS:
                if origin == destination or not slots[destination]["exists"]:
                    continue
                key = f"{origin}_{destination}"
                delta = (direction_index[destination] - direction_index[origin]) % 4
                turn_type = {1: "left", 2: "straight", 3: "right"}[delta]
                phase_ids = phase_lookup.get(key, [])
                movements.append(
                    {
                        "movement_id": f"{control['intersection_id']}/{key}",
                        "slot_key": key,
                        "origin": origin,
                        "destination": destination,
                        "turn_type": turn_type,
                        "incoming_road_ids": slots[origin]["road_ids"],
                        "outgoing_road_ids": slots[destination]["road_ids"],
                        "protected_phase_ids": phase_ids,
                        "permissive": turn_type == "right" and not phase_ids,
                    }
                )
        control["movements"] = movements


def write_sumo_seed(
    path: Path,
    tile: TileSpec,
    roads: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> None:
    junction_by_id = {row["id"]: row for row in junctions}
    write_json(
        path,
        {
            "schema_version": "sumo-static-seed-1.0",
            "tile": tile.as_dict(),
            "coordinate_system": tile.crs,
            "geometry_file": "roads.geojson",
            "note": "Static conversion seed only. netconvert connection/link indices and live detector values are generated in the SUMO integration stage.",
            "roads": [
                {
                    "road_id": road["id"],
                    "from_intersection": road["properties"]["from_intersection"],
                    "to_intersection": road["properties"]["to_intersection"],
                    "member_osm_way_ids": road["properties"]["member_osm_way_ids"],
                    "total_lanes": road["properties"]["total_lanes"],
                    "lane_data_inferred": road["properties"]["lane_data_inferred"],
                    "speed_kph": road["properties"]["speed_kph"],
                    "speed_inferred": road["properties"]["speed_inferred"],
                    "oneway_members": road["properties"]["oneway_members"],
                    "turn_lanes": road["properties"]["turn_lanes"],
                }
                for road in roads
            ],
            "controllers": [
                {
                    **control,
                    "x_m": round(junction_by_id[control["intersection_id"]]["metric"].x, 3),
                    "y_m": round(junction_by_id[control["intersection_id"]]["metric"].y, 3),
                }
                for control in controls
            ],
            "phase_actions": PHASE_ACTIONS,
        },
    )
