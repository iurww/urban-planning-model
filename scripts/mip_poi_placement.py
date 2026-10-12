#!/usr/bin/env python3
# coding: utf-8

"""
POI 选址 MIP（Gurobi）：在 K 个地块上设设施，最大化加权效用（面积与道路可达性）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

try:
    gp = importlib.import_module("gurobipy")
    GRB = gp.GRB
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("gurobipy is required.") from exc

from scripts.topology_io import load_topology, resolve_topology_path


def _ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty or gdf.crs is None:
        return gdf
    if getattr(gdf.crs, "is_geographic", False):
        return gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def poi_reward_scalar(raw_utility: float, k: int, weights: Dict[int, float]) -> float:
    """The fixed normalization used directly inside the Gurobi objective."""
    if k <= 0 or not weights:
        return 0.0
    w_max = max(float(v) for v in weights.values())
    denom = max(float(k) * w_max, 1e-12)
    return float(min(1.0, max(0.0, float(raw_utility) / denom)))


@dataclass
class PoiConfig:
    num_poi: int = 5
    max_runtime_sec: int = 120
    mip_gap: float = 0.02


def compute_poi_weights(
    parcels: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
) -> Dict[int, float]:
    road_union = unary_union(roads.geometry.tolist()) if not roads.empty else None
    int_union = unary_union(intersections.geometry.tolist()) if not intersections.empty else None
    max_area = max(float(parcels["area"].max()), 1e-9)
    w: Dict[int, float] = {}
    for _, row in parcels.iterrows():
        pid = int(row["parcel_id"])
        geom = row.geometry
        area_n = float(row["area"]) / max_area
        d_road = float(geom.distance(road_union)) if road_union is not None else 1000.0
        d_int = float(geom.distance(int_union)) if int_union is not None else 1000.0
        w[pid] = float(
            np.sqrt(float(row["area"]))
            * (0.5 * np.exp(-0.01 * d_road) + 0.5 * np.exp(-0.015 * d_int))
            * (0.6 + 0.4 * area_n)
        )
    return w


def solve_poi_mip(
    parcel_ids: List[int],
    weights: Dict[int, float],
    cfg: PoiConfig,
) -> tuple[Dict[int, float], float, float]:
    model = gp.Model("poi_placement_mip")
    model.Params.TimeLimit = cfg.max_runtime_sec
    model.Params.MIPGap = cfg.mip_gap

    K = min(cfg.num_poi, len(parcel_ids))
    z = model.addVars(parcel_ids, vtype=GRB.BINARY, name="z")
    model.addConstr(gp.quicksum(z[i] for i in parcel_ids) == K, name="cardinality")
    denominator = max(K * max((float(weights[i]) for i in parcel_ids), default=0.0), 1.0e-12)
    model.setObjective(
        gp.quicksum((weights[i] / denominator) * z[i] for i in parcel_ids),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed, status={model.Status}")

    z_val = {i: float(z[i].X) for i in parcel_ids}
    return z_val, float(model.ObjVal), float(model.MIPGap)


def build_poi_trajectory(
    decision_order: List[int],
    weights: Dict[int, float],
    z_val: Dict[int, float],
    final_reward: float,
) -> dict:
    k = max(1, sum(1 for i in decision_order if z_val.get(i, 0.0) > 0.5))
    steps: List[dict] = []
    placed = 0
    for t, pid in enumerate(decision_order, start=1):
        place_poi = bool(z_val.get(pid, 0.0) > 0.5)
        steps.append(
            {
                "t": t,
                "state": {
                    "placed_count": placed,
                    "remaining_slots": k - placed,
                    "remaining_parcels": len(decision_order) - t + 1,
                },
                "action": {"parcel_id": pid, "place_poi": place_poi},
                "reward": 0.0,
            }
        )
        placed += int(place_poi)

    if steps:
        steps[-1]["reward"] = float(final_reward)

    rtg = 0.0
    for step in reversed(steps):
        rtg += float(step["reward"])
        step["return_to_go"] = rtg

    return {
        "task_type": "poi_placement",
        "objective": "U_POI / (K * max(omega))",
        "reward_timing": "terminal_only",
        "num_poi": k,
        "metrics": {"r_POI": float(final_reward), "solver_objective": float(final_reward)},
        "num_steps": len(steps),
        "final_reward": float(final_reward),
        "steps": steps,
    }


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/topology/synth/0000/topology.gpkg"),
        help="拓扑 GeoPackage 或目录",
    )
    parser.add_argument("--solution-csv", type=Path, default=None, help="POI 选址 CSV")
    parser.add_argument("--trajectory-json", type=Path, default=None, help="轨迹 JSON")
    parser.add_argument("--num-poi", type=int, default=5, help="设施数量 K")
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.02)


def run(args: argparse.Namespace) -> None:
    topo_path = resolve_topology_path(args.input)
    parcels, roads, intersections = load_topology(topo_path)
    out_dir = topo_path.parent
    if args.solution_csv is None:
        args.solution_csv = out_dir / "poi_plan_solution.csv"
    if args.trajectory_json is None:
        args.trajectory_json = out_dir / "poi_plan_trajectory.json"

    weights = compute_poi_weights(parcels, roads, intersections)
    parcel_ids = parcels["parcel_id"].astype(int).tolist()

    eff_k = min(int(args.num_poi), len(parcel_ids))
    if eff_k < 1:
        raise ValueError("无可用地块用于 POI 选址。")
    cfg = PoiConfig(
        num_poi=eff_k,
        max_runtime_sec=int(args.time_limit),
        mip_gap=float(args.mip_gap),
    )

    z_val, obj, gap = solve_poi_mip(parcel_ids, weights, cfg)

    args.solution_csv.parent.mkdir(parents=True, exist_ok=True)
    source_by_pid = (
        dict(zip(parcels["parcel_id"].astype(int), parcels["source_id"].astype(str)))
        if "source_id" in parcels.columns
        else None
    )
    rows = []
    for i in parcel_ids:
        row = {"parcel_id": i, "place_poi": int(z_val.get(i, 0.0) > 0.5)}
        if source_by_pid is not None:
            row["source_id"] = source_by_pid[i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.solution_csv, index=False)

    raw_utility = sum(
        float(weights[i]) for i in parcel_ids if z_val.get(i, 0.0) > 0.5
    )
    r_poi = poi_reward_scalar(raw_utility, cfg.num_poi, weights)
    if not math.isclose(r_poi, obj, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise RuntimeError(
            "POI reward/model objective mismatch: "
            f"reward={r_poi:.12f}, objective={obj:.12f}"
        )
    decision_order = (
        parcels.sort_values("area", ascending=False)["parcel_id"].astype(int).tolist()
    )
    traj = build_poi_trajectory(decision_order, weights, z_val, r_poi)
    args.trajectory_json.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[Done] parcels={len(parcel_ids)}, K={cfg.num_poi}, "
        f"solver_objective={obj:.6f}, r_POI={r_poi:.6f}, mip_gap={gap:.6f}"
    )
    print(f"[Done] solution -> {args.solution_csv}")
    print(f"[Done] trajectory -> {args.trajectory_json}")


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "poi-plan",
        help="POI 选址 MIP：K 设施最大化加权效用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(p)
    p.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(description="POI 选址 MIP")
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
