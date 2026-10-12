#!/usr/bin/env python3
# coding: utf-8

"""abstract-urban-topology-1.0（topology.json）加载与 GeoDataFrame 适配。

新拓扑数据格式（data/final/<tile_id>/）：
    topology.json         intersections / roads / parcels 三类实体 + 显式邻接
    traffic_signals.json  signalized_intersection_ids（信号化路口子集）

坐标为本地米制（metadata.coordinate_system = local_metric），无需再投影。

本模块把 JSON 转成与既有 MIP / SUMO 代码兼容的 GeoDataFrame：
- parcels:        parcel_id(0 起行序), source_id, area, adjacent_parcel_ids, geometry
- roads:          道路行序即 road_row_index; source_id, from_iidx, to_iidx,
                  length_m, speed_kph, total_lanes, geometry
- intersections:  junction_idx, source_id, kind, degree, signalized, geometry

实体行序与 topology.json 中列表顺序一致，训练时可据此对齐实体图。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

TOPOLOGY_JSON_NAME = "topology.json"
SIGNALS_JSON_NAME = "traffic_signals.json"
TOPOLOGY_SCHEMA_PREFIX = "abstract-urban-topology"

KIND_BOUNDARY_PORTAL = "boundary_portal"


def load_topology_json_doc(path: Path) -> dict[str, Any]:
    """读取并校验 topology.json 原始文档。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(doc.get("schema_version", ""))
    if not schema.startswith(TOPOLOGY_SCHEMA_PREFIX):
        raise ValueError(f"{path}: 未知 schema_version={schema!r}")
    for key in ("intersections", "roads", "parcels"):
        if key not in doc:
            raise ValueError(f"{path}: 缺少 {key} 字段")
    return doc


def load_signalized_ids(tile_dir: Path) -> list[str]:
    """读取新旧 traffic_signals.json 中的信号化路口 id；缺文件时返回空。"""
    sig_path = Path(tile_dir) / SIGNALS_JSON_NAME
    if not sig_path.is_file():
        return []
    doc = json.loads(sig_path.read_text(encoding="utf-8"))
    result = [str(x) for x in doc.get("signalized_intersection_ids", [])]
    for controller in doc.get("controllers", []):
        if not isinstance(controller, dict) or not controller.get("intersection_id"):
            continue
        intersection_id = str(controller["intersection_id"])
        if intersection_id not in result:
            result.append(intersection_id)
    return result


def build_index_maps(doc: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """实体 id 字符串 -> 列表行序（0 起）。返回 (iid_map, rid_map, pid_map)。"""
    iid = {str(item["id"]): idx for idx, item in enumerate(doc["intersections"])}
    rid = {str(item["id"]): idx for idx, item in enumerate(doc["roads"])}
    pid = {str(item["id"]): idx for idx, item in enumerate(doc["parcels"])}
    return iid, rid, pid


def topology_json_to_gdfs(
    doc: dict[str, Any],
    signalized_ids: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """把 topology.json 文档转成 (parcels, roads, intersections) 三个 GeoDataFrame。"""
    iid_map, rid_map, pid_map = build_index_maps(doc)
    signalized = set(signalized_ids or [])

    # parcels
    p_rows: list[dict[str, Any]] = []
    p_geoms: list[Polygon] = []
    for idx, item in enumerate(doc["parcels"]):
        rings = item["geometry"]
        geom = Polygon(rings[0], holes=rings[1:] if len(rings) > 1 else None)
        neigh = [
            pid_map[nid] for nid in item.get("neighbor_parcel_ids", []) if nid in pid_map
        ]
        p_rows.append(
            {
                "parcel_id": idx,
                "source_id": str(item["id"]),
                "area": float(item.get("area_m2", geom.area)),
                "area_m2": float(item.get("area_m2", geom.area)),
                "source_land_use": item.get("land_use"),
                "adjacent_parcel_ids": ",".join(str(n) for n in neigh),
                "boundary_road_idxs": ",".join(
                    str(rid_map[r]) for r in item.get("boundary_road_ids", []) if r in rid_map
                ),
            }
        )
        p_geoms.append(geom)
    parcels = gpd.GeoDataFrame(p_rows, geometry=p_geoms, crs=None)

    # roads
    r_rows: list[dict[str, Any]] = []
    r_geoms: list[LineString] = []
    for idx, item in enumerate(doc["roads"]):
        geom = LineString(item["geometry"])
        r_rows.append(
            {
                "road_row_index": idx,
                "source_id": str(item["id"]),
                "from_iidx": int(iid_map[item["from_intersection_id"]]),
                "to_iidx": int(iid_map[item["to_intersection_id"]]),
                "length_m": float(item.get("length_m", geom.length)),
                "speed_kph": float(item.get("speed_kph", 50.0)),
                "total_lanes": int(item.get("total_lanes", 2)),
                "highway": item.get("highway"),
                "name": item.get("name"),
            }
        )
        r_geoms.append(geom)
    roads = gpd.GeoDataFrame(r_rows, geometry=r_geoms, crs=None)

    # intersections
    i_rows: list[dict[str, Any]] = []
    i_geoms: list[Point] = []
    for idx, item in enumerate(doc["intersections"]):
        i_rows.append(
            {
                "junction_idx": idx,
                "source_id": str(item["id"]),
                "kind": item.get("kind"),
                "degree": int(item.get("degree", 0)),
                "signalized": bool(str(item["id"]) in signalized),
            }
        )
        i_geoms.append(Point(item["point"]))
    intersections = gpd.GeoDataFrame(i_rows, geometry=i_geoms, crs=None)

    return parcels, roads, intersections


def load_topology_json(
    path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """从 topology.json（或其所在目录）加载三个 GeoDataFrame。"""
    path = Path(path)
    if path.is_dir():
        path = path / TOPOLOGY_JSON_NAME
    doc = load_topology_json_doc(path)
    signalized = load_signalized_ids(path.parent)
    return topology_json_to_gdfs(doc, signalized)
