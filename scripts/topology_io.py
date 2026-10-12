#!/usr/bin/env python3
# coding: utf-8

"""共享拓扑读写、概要 JSON 与统一目录约定。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd

# 统一根目录（相对项目根）
TOPOLOGY_ROOT = Path("data/topology")
REAL_SUBDIR = "real"
SYNTH_SUBDIR = "synth"

SCHEMA_VERSION = 1

# type 与 visualize / MIP 一致
TYPE_SMALL_PARCEL = 0
TYPE_LARGE_PARCEL = 1
TYPE_ROAD = 2
TYPE_JUNCTION = 13


def real_topology_dir(tile_id: str) -> Path:
    return TOPOLOGY_ROOT / REAL_SUBDIR / tile_id


def synth_topology_dir(sample_index: int) -> Path:
    return TOPOLOGY_ROOT / SYNTH_SUBDIR / f"{sample_index:04d}"


def default_topology_gpkg(topology_dir: Path) -> Path:
    return topology_dir / "topology.gpkg"


def resolve_topology_path(path: Path) -> Path:
    """把目录或文件路径解析成具体拓扑文件；目录内优先 topology.json（新格式）。"""
    path = Path(path)
    if path.is_dir():
        json_path = path / "topology.json"
        if json_path.is_file():
            return json_path
        return path / "topology.gpkg"
    return path


def default_summary_path(topology_dir: Path) -> Path:
    return topology_dir / "summary.json"


def load_topology(topo_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """加载拓扑：大地块、道路、交叉口。

    支持两种格式：
    - topology.json（abstract-urban-topology-1.0，data/final 新格式）
    - topology.gpkg（旧 GeoPackage 格式，type 列区分实体）
    """
    topo_path = resolve_topology_path(Path(topo_path))
    if topo_path.suffix.lower() == ".json":
        from scripts.topology_json import load_topology_json

        return load_topology_json(topo_path)

    gdf = gpd.read_file(topo_path)
    if "type" not in gdf.columns:
        raise ValueError("GeoPackage 必须包含 type 列。")

    parcels = gdf[gdf["type"] == TYPE_LARGE_PARCEL].copy()
    roads = gdf[gdf["type"] == TYPE_ROAD].copy()
    intersections = gdf[gdf["type"] == TYPE_JUNCTION].copy()

    if parcels.empty:
        raise ValueError("未找到大地块 (type=1)。")

    parcels = parcels.reset_index(drop=True)
    if "parcel_id" in parcels.columns:
        parcels["parcel_id"] = parcels["parcel_id"].astype(int)
    else:
        parcels["parcel_id"] = parcels.index.astype(int)
    if "area_m2" in parcels.columns and parcels["area_m2"].notna().any():
        parcels["area"] = parcels["area_m2"].astype(float)
    elif parcels.crs is not None and getattr(parcels.crs, "is_geographic", False):
        metric_crs = parcels.estimate_utm_crs()
        parcels["area"] = parcels.to_crs(metric_crs).geometry.area.astype(float)
    else:
        parcels["area"] = parcels.geometry.area.astype(float)

    return parcels, roads, intersections


def build_counts_dict(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    return {
        "parcels_large": int((gdf["type"] == TYPE_LARGE_PARCEL).sum()),
        "parcels_small": int((gdf["type"] == TYPE_SMALL_PARCEL).sum()),
        "roads": int((gdf["type"] == TYPE_ROAD).sum()),
        "junctions": int((gdf["type"] == TYPE_JUNCTION).sum()),
    }


def build_topology_summary(
    gdf: gpd.GeoDataFrame,
    *,
    topology_id: str,
    source: str,
    crs: Optional[Any] = None,
    bbox: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构造与数据源无关的拓扑概要（写入 summary.json）。"""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "topology_id": topology_id,
        "source": source,
        "counts": build_counts_dict(gdf),
    }
    if crs is None and gdf.crs is not None:
        out["crs"] = str(gdf.crs)
    elif crs is not None:
        out["crs"] = str(crs)
    if bbox is not None:
        out["bbox"] = bbox
    if extra:
        out["extra"] = extra
    return out


def write_summary_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
