#!/usr/bin/env python3
# coding: utf-8

"""
Batch synthetic topology generator.

Features:
- Start from an n*n regular grid of parcels.
- Two-phase splitting for realistic hierarchy:
    Phase 1 (main roads): preferentially splits depth-0 grid cells with
    near-central ratios, establishing the primary road network first.
    Phase 2 (subdivisions): normal weighted random selection for finer splits.
- Street coherence: new splits preferentially align with existing road lines,
  creating through-streets that cross multiple blocks.
- Four split operations:
    1) horizontal split
    2) vertical split
    3) cut a small rectangle from one corner
    4) diagonal (oblique) split — produces trapezoidal/irregular parcels
- Aspect-ratio guidance: prefer splitting along the longer axis.
- Track split depth per parcel.
  - horizontal/vertical/diagonal: both children depth = parent_depth + 1
  - corner cut: small corner child depth = parent_depth + 1,
                remaining parent piece keeps parent_depth
- Avoid over-splitting one local area with:
  - max depth
  - min parcel area
  - cooldown after split
  - per-root split balancing (discourage repeatedly splitting one root)

Output format (OSM-like topology style):
- type=1: parcel polygons
- type=2: road line segments (parcel boundaries)
- type=13: intersections (line endpoints)
- columns include: id, type, existence, depth, root_id, geometry
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, split, unary_union

from scripts.topology_io import (
    build_topology_summary,
    default_summary_path,
    default_topology_gpkg,
    write_summary_json,
)


@dataclass
class Parcel:
    pid: int
    geometry: Polygon
    depth: int
    root_id: int
    cooldown: int = 0


def _is_valid_polygon(poly: Polygon, min_area: float) -> bool:
    return poly is not None and not poly.is_empty and poly.area >= min_area and poly.is_valid


def _is_axis_aligned_rectangle(poly: Polygon, eps: float = 1e-7) -> bool:
    if poly.is_empty:
        return False
    coords = list(poly.exterior.coords)
    if len(coords) != 5:
        return False
    minx, miny, maxx, maxy = poly.bounds
    bbox = box(minx, miny, maxx, maxy)
    return abs(poly.area - bbox.area) <= eps


def _split_vertical(poly: Polygon, ratio: float, min_area: float) -> Optional[Tuple[Polygon, Polygon]]:
    minx, miny, maxx, maxy = poly.bounds
    if maxx - minx <= 1e-9:
        return None
    x0 = minx + ratio * (maxx - minx)
    knife = LineString([(x0, miny - 1.0), (x0, maxy + 1.0)])
    parts = [p for p in split(poly, knife).geoms if p.geom_type == "Polygon"]
    if len(parts) != 2:
        return None
    if not all(_is_valid_polygon(p, min_area) for p in parts):
        return None 
    return parts[0], parts[1]


def _split_horizontal(poly: Polygon, ratio: float, min_area: float) -> Optional[Tuple[Polygon, Polygon]]:
    minx, miny, maxx, maxy = poly.bounds
    if maxy - miny <= 1e-9:
        return None
    y0 = miny + ratio * (maxy - miny)
    knife = LineString([(minx - 1.0, y0), (maxx + 1.0, y0)])
    parts = [p for p in split(poly, knife).geoms if p.geom_type == "Polygon"]
    if len(parts) != 2:
        return None
    if not all(_is_valid_polygon(p, min_area) for p in parts):
        return None
    return parts[0], parts[1]


def _corner_cut(poly: Polygon, min_area: float, rng: random.Random, max_tries: int = 8) -> Optional[Tuple[Polygon, Polygon]]:
    # For stable topology quality, corner-cut only rectangular parcels.
    if not _is_axis_aligned_rectangle(poly):
        return None

    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 1e-9 or h <= 1e-9:
        return None

    for _ in range(max_tries):
        wr = rng.uniform(0.20, 0.38)
        hr = rng.uniform(0.20, 0.38)
        cw = w * wr
        ch = h * hr
        corner = rng.choice(["bl", "br", "tr", "tl"])

        if corner == "bl":
            cut_rect = box(minx, miny, minx + cw, miny + ch)
        elif corner == "br":
            cut_rect = box(maxx - cw, miny, maxx, miny + ch)
        elif corner == "tr":
            cut_rect = box(maxx - cw, maxy - ch, maxx, maxy)
        else:
            cut_rect = box(minx, maxy - ch, minx + cw, maxy)

        small = poly.intersection(cut_rect)
        remain = poly.difference(cut_rect)

        if small.geom_type != "Polygon" or remain.geom_type != "Polygon":
            continue
        if not _is_valid_polygon(small, min_area):
            continue
        if not _is_valid_polygon(remain, min_area):
            continue

        return small, remain

    return None


def _split_diagonal(
    poly: Polygon,
    min_area: float,
    rng: random.Random,
    max_tries: int = 10,
) -> Optional[Tuple[Polygon, Polygon]]:
    """Oblique (slightly angled) split that produces trapezoidal parcels.

    The knife is a near-horizontal or near-vertical line with a small tilt,
    mimicking roads that follow rivers, railways, or organic street patterns.
    """
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w < 1e-9 or h < 1e-9:
        return None

    for _ in range(max_tries):
        if rng.random() < 0.5:
            # Near-horizontal: enter from left edge, exit from right at different y.
            y_center = miny + rng.uniform(0.28, 0.72) * h
            # Tilt: 6%-22% height difference between left and right crossing points.
            offset = rng.uniform(0.06, 0.22) * h * rng.choice([-1, 1])
            y_left = float(np.clip(y_center - offset / 2, miny + 0.05 * h, maxy - 0.05 * h))
            y_right = float(np.clip(y_center + offset / 2, miny + 0.05 * h, maxy - 0.05 * h))
            knife = LineString([(minx - 1.0, y_left), (maxx + 1.0, y_right)])
        else:
            # Near-vertical: enter from bottom, exit from top at different x.
            x_center = minx + rng.uniform(0.28, 0.72) * w
            offset = rng.uniform(0.06, 0.22) * w * rng.choice([-1, 1])
            x_bot = float(np.clip(x_center - offset / 2, minx + 0.05 * w, maxx - 0.05 * w))
            x_top = float(np.clip(x_center + offset / 2, minx + 0.05 * w, maxx - 0.05 * w))
            knife = LineString([(x_bot, miny - 1.0), (x_top, maxy + 1.0)])

        try:
            parts = [p for p in split(poly, knife).geoms if p.geom_type == "Polygon"]
        except Exception:
            continue

        if len(parts) != 2:
            continue
        if all(_is_valid_polygon(p, min_area) for p in parts):
            return parts[0], parts[1]

    return None


def _choose_candidate_index(
    parcels: List[Parcel],
    root_split_counts: Dict[int, int],
    rng: random.Random,
    depth_filter: Optional[int] = None,
) -> Optional[int]:
    weights = []
    idxs = []
    for idx, p in enumerate(parcels):
        if p.cooldown > 0:
            continue
        if depth_filter is not None and p.depth != depth_filter:
            continue
        area = float(p.geometry.area)
        # Area promotes realism; depth and root history suppress over-localized recursion.
        w = area / ((p.depth + 1.0) ** 1.8) / ((root_split_counts[p.root_id] + 1.0) ** 1.2)
        if w > 0:
            idxs.append(idx)
            weights.append(w)

    if not idxs:
        return None

    arr = np.array(weights, dtype=float)
    arr = arr / arr.sum()
    return idxs[int(rng.choices(range(len(idxs)), weights=arr, k=1)[0])]


def generate_one(
    n: int,
    total_splits: int,
    map_size: float,
    max_depth: int,
    min_area_ratio: float,
    cooldown_steps: int,
    min_corner_cuts: int,
    seed: int,
    align_prob: float = 0.55,
    phase1_frac: float = 0.25,
    diagonal_weight: float = 0.10,
) -> Tuple[List[Parcel], List[Dict]]:
    """Generate one synthetic parcel map.

    Realism improvements over the naive random approach:
    - Phase 1 (first phase1_frac of steps) only targets depth-0 grid cells with
      near-central ratios, creating the primary road grid before subdivisions.
    - Street coherence: when road_xs/road_ys already contain a line passing through
      the candidate parcel, new splits align to it with probability align_prob,
      creating continuous through-streets.
    - Diagonal splits add organic, irregular parcel shapes.
    - Aspect-ratio guidance biases split direction toward the longer axis.
    """
    rng = random.Random(seed)

    # Initial n*n grid
    cell = map_size / n
    parcels: List[Parcel] = []
    pid = 0
    for r in range(n):
        for c in range(n):
            poly = box(c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)
            parcels.append(Parcel(pid=pid, geometry=poly, depth=0, root_id=pid, cooldown=0))
            pid += 1

    min_area = (cell * cell) * min_area_ratio
    root_split_counts: Dict[int, int] = {p.root_id: 0 for p in parcels}
    op_log: List[Dict] = []

    # Street coherence: track road positions.
    # Seed with interior grid boundaries (the initial road lines).
    road_xs: List[float] = [c * cell for c in range(1, n)]
    road_ys: List[float] = [r * cell for r in range(1, n)]

    # Phase threshold
    n_phase1 = max(0, int(total_splits * phase1_frac))

    # Base op weights (before aspect-ratio adjustment)
    _bh, _bv, _bc, _bd = 0.35, 0.35, 0.20, max(0.0, diagonal_weight)
    _s = _bh + _bv + _bc + _bd
    base_w = [_bh / _s, _bv / _s, _bc / _s, _bd / _s]

    op_choices = ["h", "v", "corner", "diagonal"]
    corner_done = 0
    corner_target = min(max(min_corner_cuts, 0), total_splits)

    for step in range(total_splits):
        for p in parcels:
            if p.cooldown > 0:
                p.cooldown -= 1

        # Phase 1: prefer uncut grid cells so main roads get laid out first.
        is_phase1 = step < n_phase1
        candidate_idx: Optional[int] = None
        if is_phase1:
            candidate_idx = _choose_candidate_index(parcels, root_split_counts, rng, depth_filter=0)
        if candidate_idx is None:
            candidate_idx = _choose_candidate_index(parcels, root_split_counts, rng)
        if candidate_idx is None:
            break

        parent = parcels[candidate_idx]
        if parent.geometry.area < 2.1 * min_area:
            continue

        must_force_corner = (corner_done < corner_target) and (
            (total_splits - step) <= (corner_target - corner_done)
        )

        minx, miny, maxx, maxy = parent.geometry.bounds
        w_dim = maxx - minx
        h_dim = maxy - miny

        if must_force_corner:
            ops = ["corner", "h", "v", "diagonal"]
        else:
            # Aspect-ratio guidance: favor splitting the longer dimension.
            if w_dim > h_dim * 1.4:
                scale = [0.65, 1.45, 1.0, 1.0]   # wider -> prefer vertical
            elif h_dim > w_dim * 1.4:
                scale = [1.45, 0.65, 1.0, 1.0]   # taller -> prefer horizontal
            else:
                scale = [1.0, 1.0, 1.0, 1.0]
            adj = [base_w[i] * scale[i] for i in range(4)]
            s = sum(adj)
            adj = [x / s for x in adj]
            ops = rng.choices(op_choices, weights=adj, k=4)

        success = False
        for op in ops:
            if parent.depth >= max_depth:
                continue

            if op == "h":
                # Street coherence: align with an existing horizontal road if one
                # passes through this parcel.
                guard = 0.08 * h_dim
                valid_roads = [y for y in road_ys if miny + guard < y < maxy - guard]
                if valid_roads and rng.random() < align_prob:
                    y0 = rng.choice(valid_roads)
                    ratio = (y0 - miny) / h_dim
                    is_new_road = False
                else:
                    lo, hi = (0.35, 0.65) if is_phase1 else (0.25, 0.75)
                    ratio = rng.uniform(lo, hi)
                    y0 = miny + ratio * h_dim
                    is_new_road = True

                result = _split_horizontal(parent.geometry, ratio, min_area)
                if result is None:
                    continue
                p1, p2 = result
                child1 = Parcel(pid=pid, geometry=p1, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                child2 = Parcel(pid=pid, geometry=p2, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                op_name = "horizontal"
                if is_new_road:
                    road_ys.append(y0)

            elif op == "v":
                guard = 0.08 * w_dim
                valid_roads = [x for x in road_xs if minx + guard < x < maxx - guard]
                if valid_roads and rng.random() < align_prob:
                    x0 = rng.choice(valid_roads)
                    ratio = (x0 - minx) / w_dim
                    is_new_road = False
                else:
                    lo, hi = (0.35, 0.65) if is_phase1 else (0.25, 0.75)
                    ratio = rng.uniform(lo, hi)
                    x0 = minx + ratio * w_dim
                    is_new_road = True

                result = _split_vertical(parent.geometry, ratio, min_area)
                if result is None:
                    continue
                p1, p2 = result
                child1 = Parcel(pid=pid, geometry=p1, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                child2 = Parcel(pid=pid, geometry=p2, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                op_name = "vertical"
                if is_new_road:
                    road_xs.append(x0)

            elif op == "corner":
                result = _corner_cut(parent.geometry, min_area, rng)
                if result is None:
                    continue
                small, remain = result
                child1 = Parcel(
                    pid=pid,
                    geometry=small,
                    depth=parent.depth + 1,
                    root_id=parent.root_id,
                    cooldown=cooldown_steps,
                )
                pid += 1
                child2 = Parcel(
                    pid=pid,
                    geometry=remain,
                    depth=parent.depth,
                    root_id=parent.root_id,
                    cooldown=cooldown_steps,
                )
                pid += 1
                op_name = "corner_rect"
                corner_done += 1

            else:  # diagonal
                result = _split_diagonal(parent.geometry, min_area, rng)
                if result is None:
                    continue
                p1, p2 = result
                child1 = Parcel(pid=pid, geometry=p1, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                child2 = Parcel(pid=pid, geometry=p2, depth=parent.depth + 1, root_id=parent.root_id, cooldown=cooldown_steps)
                pid += 1
                op_name = "diagonal"

            root_split_counts[parent.root_id] += 1
            removed = parcels.pop(candidate_idx)
            parcels.append(child1)
            parcels.append(child2)

            op_log.append(
                {
                    "step": step,
                    "operation": op_name,
                    "parent_id": removed.pid,
                    "parent_depth": removed.depth,
                    "child_ids": [child1.pid, child2.pid],
                    "child_depths": [child1.depth, child2.depth],
                    "root_id": removed.root_id,
                }
            )
            success = True
            break

        if not success:
            # No valid op for this parent, mild cooldown to reduce futile retries.
            parent.cooldown = max(parent.cooldown, 1)

    return parcels, op_log


def build_topology_gdf(parcels: List[Parcel]) -> gpd.GeoDataFrame:
    parcel_geoms = [p.geometry for p in parcels]

    # Use parcel boundaries as roads, then endpoint set as intersections.
    boundary_union = unary_union([g.boundary for g in parcel_geoms])
    road_lines = []
    if boundary_union.geom_type == "LineString":
        road_lines = [boundary_union]
    elif hasattr(boundary_union, "geoms"):
        road_lines = [g for g in boundary_union.geoms if g.geom_type == "LineString" and g.length > 1e-8]

    points = []
    for line in road_lines:
        coords = list(line.coords)
        if len(coords) >= 2:
            points.append(Point(coords[0]))
            points.append(Point(coords[-1]))

    # Unique endpoints as intersections.
    p_union = unary_union(points) if points else None
    intersections = []
    if p_union is not None:
        if p_union.geom_type == "Point":
            intersections = [p_union]
        elif hasattr(p_union, "geoms"):
            intersections = [g for g in p_union.geoms if g.geom_type == "Point"]

    rows = []
    rid = 0

    for line in road_lines:
        rows.append(
            {
                "id": rid,
                "type": 2,
                "existence": True,
                "depth": None,
                "root_id": None,
                "geometry": line,
            }
        )
        rid += 1

    for pt in intersections:
        rows.append(
            {
                "id": rid,
                "type": 13,
                "existence": True,
                "depth": None,
                "root_id": None,
                "geometry": pt,
            }
        )
        rid += 1

    for p in parcels:
        rows.append(
            {
                "id": rid,
                "type": 1,
                "existence": True,
                "depth": int(p.depth),
                "root_id": int(p.root_id),
                "geometry": p.geometry,
            }
        )
        rid += 1

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32650")

    # Build adjacency columns compatible with scripts/visualize.py:
    # - adjacent_road_ids
    # - adjacent_node_ids
    # For non-parcel rows keep empty string.
    roads_gdf = gdf[gdf["type"] == 2]
    nodes_gdf = gdf[gdf["type"] == 13]
    parcels_gdf = gdf[gdf["type"].isin([0, 1])]

    roads_sindex = roads_gdf.sindex if not roads_gdf.empty else None
    nodes_sindex = nodes_gdf.sindex if not nodes_gdf.empty else None

    gdf["adjacent_road_ids"] = ""
    gdf["adjacent_node_ids"] = ""

    for idx, poly in parcels_gdf.geometry.items():
        bnd = poly.boundary

        if roads_sindex is not None:
            r_pos = roads_sindex.query(bnd, predicate="intersects")
            r_ids = roads_gdf.iloc[r_pos].index.astype(str).tolist()
        else:
            r_ids = []

        if nodes_sindex is not None:
            n_pos = nodes_sindex.query(bnd, predicate="intersects")
            n_ids = nodes_gdf.iloc[n_pos].index.astype(str).tolist()
        else:
            n_ids = []

        gdf.at[idx, "adjacent_road_ids"] = ",".join(r_ids)
        gdf.at[idx, "adjacent_node_ids"] = ",".join(n_ids)

    print(gdf)
    return gdf


def summarize(parcels: List[Parcel], op_log: List[Dict], n: int, splits_target: int) -> dict:
    depths = [p.depth for p in parcels]
    areas = [float(p.geometry.area) for p in parcels]
    depth_hist: Dict[str, int] = {}
    for d in depths:
        key = str(d)
        depth_hist[key] = depth_hist.get(key, 0) + 1

    op_counts: Dict[str, int] = {}
    for x in op_log:
        name = x.get("operation", "unknown")
        op_counts[name] = op_counts.get(name, 0) + 1

    return {
        "n": n,
        "target_splits": splits_target,
        "actual_splits": len(op_log),
        "op_counts": op_counts,
        "corner_cuts": op_counts.get("corner_rect", 0),
        "diagonal_cuts": op_counts.get("diagonal", 0),
        "num_parcels": len(parcels),
        "depth_min": int(min(depths) if depths else 0),
        "depth_max": int(max(depths) if depths else 0),
        "depth_hist": depth_hist,
        "area_min": float(min(areas) if areas else 0.0),
        "area_mean": float(np.mean(areas) if areas else 0.0),
        "area_max": float(max(areas) if areas else 0.0),
    }


@dataclass(frozen=True)
class _SynthGenParams:
    """单任务参数（可 pickle，供多进程）。"""

    out_dir: str
    n: int
    map_size: float
    min_splits: int
    max_splits: int
    max_depth: int
    min_area_ratio: float
    cooldown_steps: int
    min_corner_cuts: int
    align_prob: float
    phase1_frac: float
    diagonal_weight: float
    base_seed: int


def _generate_one_sample(index: int, p: _SynthGenParams) -> dict:
    """
    生成并写入单个样本（模块级函数，供 ProcessPoolExecutor 调用）。
    返回 generation 元数据（与 index.json 条目一致）。
    """
    local_seed = p.base_seed + index * 1009
    rng = random.Random(local_seed)
    total_splits = rng.randint(p.min_splits, p.max_splits)

    parcels, op_log = generate_one(
        n=p.n,
        total_splits=total_splits,
        map_size=p.map_size,
        max_depth=p.max_depth,
        min_area_ratio=p.min_area_ratio,
        cooldown_steps=p.cooldown_steps,
        min_corner_cuts=p.min_corner_cuts,
        seed=local_seed,
        align_prob=p.align_prob,
        phase1_frac=p.phase1_frac,
        diagonal_weight=p.diagonal_weight,
    )

    gdf = build_topology_gdf(parcels)
    gen_meta = summarize(parcels, op_log, p.n, total_splits)
    gen_meta["sample_id"] = index

    out_dir = Path(p.out_dir)
    tile_dir = out_dir / f"{index:04d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = default_topology_gpkg(tile_dir)
    summary_path = default_summary_path(tile_dir)

    gdf.to_file(gpkg_path, driver="GPKG")

    topology_id = f"synth_{index:04d}"
    summary_doc = build_topology_summary(
        gdf,
        topology_id=topology_id,
        source="synthetic",
        extra={
            "sample_index": index,
            "generation": gen_meta,
            "operations": op_log,
        },
    )
    write_summary_json(summary_path, summary_doc)

    return gen_meta


def _add_args(parser):
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/topology2/synth"),
        help="合成拓扑根目录（其下为 0000/topology.gpkg 与 summary.json）",
    )
    parser.add_argument("--samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--n", type=int, default=5, help="初始网格大小 n（n×n）")
    parser.add_argument("--map-size", type=float, default=4000.0, help="地图边长（投影坐标单位）")
    parser.add_argument("--min-splits", type=int, default=60, help="每个样本最少切分次数")
    parser.add_argument("--max-splits", type=int, default=100, help="每个样本最多切分次数")
    parser.add_argument("--max-depth", type=int, default=5, help="最大地块切分深度")
    parser.add_argument("--min-area-ratio", type=float, default=0.08,
                        help="最小地块面积（初始格子面积的比例）")
    parser.add_argument("--cooldown-steps", type=int, default=2,
                        help="切分后子地块冷却步数")
    parser.add_argument("--min-corner-cuts", type=int, default=10,
                        help="每个样本最少角切次数")
    parser.add_argument("--align-prob", type=float, default=0.55,
                        help="新切分对齐已有道路线的概率（街道连贯性）")
    parser.add_argument("--phase1-frac", type=float, default=0.25,
                        help="Phase 1（主干道布局）占总切分步数的比例")
    parser.add_argument("--diagonal-weight", type=float, default=0.05,
                        help="斜切操作的相对权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="并发进程数；默认 min(CPU 核数, 样本数)；设为 1 则完全串行",
    )


def run(args) -> None:
    if args.min_splits > args.max_splits:
        raise ValueError("min-splits cannot be greater than max-splits")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_meta: List[dict] = []

    cpu = os.cpu_count()
    nw = min(cpu, max(1, args.workers))

    params = _SynthGenParams(
        out_dir=str(args.out_dir.resolve()),
        n=args.n,
        map_size=args.map_size,
        min_splits=args.min_splits,
        max_splits=args.max_splits,
        max_depth=args.max_depth,
        min_area_ratio=args.min_area_ratio,
        cooldown_steps=args.cooldown_steps,
        min_corner_cuts=args.min_corner_cuts,
        align_prob=args.align_prob,
        phase1_frac=args.phase1_frac,
        diagonal_weight=args.diagonal_weight,
        base_seed=args.seed,
    )

    def _log_done(gen_meta: dict) -> None:
        i = int(gen_meta["sample_id"])
        gpkg = args.out_dir / f"{i:04d}" / "topology.gpkg"
        print(
            f"[Sample {i:04d}] parcels={gen_meta['num_parcels']} "
            f"splits={gen_meta['actual_splits']}/{gen_meta['target_splits']} "
            f"corner={gen_meta['corner_cuts']} diag={gen_meta['diagonal_cuts']} "
            f"depth_max={gen_meta['depth_max']} -> {gpkg}"
        )

    if nw <= 1:
        for i in range(args.samples):
            gen_meta = _generate_one_sample(i, params)
            all_meta.append(gen_meta)
            _log_done(gen_meta)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = {
                ex.submit(_generate_one_sample, i, params): i for i in range(args.samples)
            }
            for fut in as_completed(futures):
                gen_meta = fut.result()
                all_meta.append(gen_meta)
                _log_done(gen_meta)

    all_meta.sort(key=lambda m: int(m["sample_id"]))

    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Done] Generated {args.samples} samples "
        f"(workers={nw}, cpu={cpu}), index: {index_path}"
    )


def add_subcommand(subparsers):
    parser = subparsers.add_parser(
        "gen-synth",
        help="程序化合成城市块区拓扑数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="程序化合成城市块区拓扑数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
