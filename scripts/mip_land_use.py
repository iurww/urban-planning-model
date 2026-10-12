#!/usr/bin/env python3
# coding: utf-8

"""
Graph-based static land-use planning as a MILP/MIQP solved by Gurobi.

规划目标（与论文一致，土地奖励）::
    r_L = α · Service + Ecology

- Service（15 分钟城市，500m）: 对每个住宅区 RZ_i，计算到五类设施的最小欧氏距离；
  Service_i = (1/5) Σ_j 1[d(i,j) < 500m]；整体为面积加权平均（见实现）。
  五类设施在六类用地标签上的映射：
  - 教育 -> sch
  - 医疗 -> hos
  - 娱乐 -> grn
  - 工作 -> off
  - 购物 -> com

- Ecology: 公园与开敞空间周边 300m 缓冲区并集 ESR；Ecology 为住宅面积中被 ESR 覆盖的比例。
  实现中将「公园/开敞空间」近似为 **grn** 地块几何的 300m 缓冲，与住宅 **res** 相交面积加总，
  目标项为 Σ_{i,k} overlap_{ik} · x[i,res] · x[k,grn]（与分母一起见目标缩放说明）。

Input:
- Topology GeoPackage that contains at least:
  - type=1 polygons (parcels)
  - type=2 lines (roads, optional but recommended)
  - type=13 points (intersections, optional)

Output:
- Solution CSV: parcel -> land use label
- Trajectory JSON: sequence data for later training (DT-style)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

try:
    gp = importlib.import_module("gurobipy")
    GRB = gp.GRB
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "gurobipy is required. Install Gurobi and gurobipy first."
    ) from exc

from scripts.topology_io import load_topology, resolve_topology_path


LAND_USE_TYPES = ["grn", "com", "res", "sch", "hos", "off"]

LU_GREEN = "grn"
LU_COMMERCIAL = "com"
LU_RESIDENTIAL = "res"
LU_SCHOOL = "sch"
LU_HOSPITAL = "hos"
LU_OFFICE = "off"

# 五类服务：教育、医疗、娱乐、工作、购物
SERVICE_WEIGHTS = {
    LU_SCHOOL: 1.0 / 5.0,
    LU_HOSPITAL: 1.0 / 5.0,
    LU_GREEN: 1.0 / 5.0,
    LU_OFFICE: 1.0 / 5.0,
    LU_COMMERCIAL: 1.0 / 5.0,
}


@dataclass
class SolverConfig:
    quota_lower: Dict[str, float]
    quota_upper: Dict[str, float]
    alpha_service: float = 1.0
    service_radius_m: float = 500.0
    ecology_buffer_m: float = 300.0
    max_runtime_sec: int = 20
    mip_gap: float = 0.01


def build_default_config() -> SolverConfig:
    return SolverConfig(
        quota_lower={
            LU_GREEN: 0.08,
            LU_COMMERCIAL: 0.10,
            LU_RESIDENTIAL: 0.20,
            LU_SCHOOL: 0.08,
            LU_HOSPITAL: 0.08,
            LU_OFFICE: 0.08,
        },
        quota_upper={
            LU_GREEN: 0.30,
            LU_COMMERCIAL: 0.30,
            LU_RESIDENTIAL: 0.50,
            LU_SCHOOL: 0.20,
            LU_HOSPITAL: 0.20,
            LU_OFFICE: 0.30,
        },
        alpha_service=1.0,
        service_radius_m=500.0,
        ecology_buffer_m=300.0,
        max_runtime_sec=20,
        mip_gap=0.01,
    )


def _ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf
    if getattr(gdf.crs, "is_geographic", False):
        return gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def build_parcel_adjacency(parcels: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    if "adjacent_parcel_ids" in parcels.columns:
        edge_set: set[Tuple[int, int]] = set()
        valid_pids = {int(pid) for pid in parcels["parcel_id"].astype(int).tolist()}
        for _, row in parcels.iterrows():
            raw = row.get("adjacent_parcel_ids")
            if raw is None or str(raw).strip() == "":
                continue
            src = int(row["parcel_id"])
            for item in str(raw).split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    dst = int(float(item))
                except ValueError:
                    continue
                if dst == src or dst not in valid_pids:
                    continue
                edge_set.add(tuple(sorted((src, dst))))
        if edge_set:
            return sorted(edge_set)

    sindex = parcels.sindex
    edges: set[Tuple[int, int]] = set()
    eps = 1e-6

    def line_passes_through(line: LineString, geom) -> bool:
        if geom is None or geom.is_empty or not line.intersects(geom):
            return False
        if line.crosses(geom) or line.within(geom) or geom.contains(line):
            return True
        inter = line.intersection(geom)
        if inter.is_empty:
            return False
        if getattr(inter, "length", 0.0) > eps:
            return True
        if inter.geom_type == "GeometryCollection":
            return any(getattr(part, "length", 0.0) > eps for part in inter.geoms)
        return False

    for row_pos, (pid, geom) in enumerate(zip(parcels["parcel_id"], parcels.geometry)):
        if geom is None or geom.is_empty:
            continue
        center = geom.centroid
        for j in range(row_pos + 1, len(parcels)):
            other = parcels.geometry.iloc[j]
            if other is None or other.is_empty:
                continue
            other_center = other.centroid
            line = LineString([(center.x, center.y), (other_center.x, other_center.y)])
            if line.length <= 0:
                continue
            blocked = False
            for k in sindex.intersection(line.bounds):
                k = int(k)
                if k in {row_pos, j}:
                    continue
                if line_passes_through(line, parcels.geometry.iloc[k]):
                    blocked = True
                    break
            if not blocked:
                other_pid = int(parcels["parcel_id"].iloc[int(j)])
                edges.add(tuple(sorted((int(pid), other_pid))))

    return sorted(edges)


def precompute_service_neighbors(
    parcels: gpd.GeoDataFrame,
    radius_m: float,
) -> Dict[int, Set[int]]:
    """parcel_id -> 欧氏距离 < radius 的其他 parcel_id 集合（不含自身）。"""
    g = _ensure_metric_crs(parcels)
    geoms = g.geometry
    centers = np.array([[geom.centroid.x, geom.centroid.y] for geom in geoms])
    pids = g["parcel_id"].astype(int).tolist()
    r2 = float(radius_m) ** 2
    out: Dict[int, Set[int]] = {int(pid): set() for pid in pids}
    n = len(pids)
    for ia in range(n):
        dx = centers[:, 0] - centers[ia, 0]
        dy = centers[:, 1] - centers[ia, 1]
        d2 = dx * dx + dy * dy
        for ib in range(n):
            if ia == ib:
                continue
            if d2[ib] <= r2:
                out[int(pids[ia])].add(int(pids[ib]))
    return out


def precompute_ecology_overlap(
    parcels: gpd.GeoDataFrame,
    buffer_m: float,
) -> Dict[Tuple[int, int], float]:
    """
    overlap[i,k] = Area(RZ_i ∩ Buffer(PA_k, buffer_m))，其中 i 为住宅候选、k 为「开敞/公园」候选，
    实现中 k 仅与 grn 用地结合；预计算所有 (i,k) 供 x[i,res]*x[k,grn] 使用。
    """
    g = _ensure_metric_crs(parcels)
    pids = g["parcel_id"].astype(int).tolist()
    overlap: Dict[Tuple[int, int], float] = {}
    geoms = g.geometry.tolist()
    buf = float(buffer_m)
    for ik, geom_k in enumerate(geoms):
        try:
            bpoly = geom_k.buffer(buf)
        except Exception:
            continue
        k = int(pids[ik])
        for ii, geom_i in enumerate(geoms):
            i = int(pids[ii])
            try:
                inter = geom_i.intersection(bpoly)
                a = float(inter.area) if not inter.is_empty else 0.0
            except Exception:
                a = 0.0
            if a > 1e-12:
                overlap[(i, k)] = a
    return overlap


def solve_land_use_mip(
    parcels: gpd.GeoDataFrame,
    edges: List[Tuple[int, int]],
    service_neighbors: Dict[int, Set[int]],
    ecology_overlap: Dict[Tuple[int, int], float],
    cfg: SolverConfig,
) -> tuple[Dict[int, str], float, float]:
    """
    目标：max  α · Service_term + Ecology_term

    Service = (Σ_i x[i,res] · Σ_t w_t·s_t[i]) / Σ_i x[i,res]
      s_t[i]=1 当且仅当 ∃k∈N(i): x[k,t]=1 且 dist(i,k)<R

    Ecology = Σ_i covered_i / Σ_i area_i*x[i,res], where
      covered_i = min(area_i*x[i,res], Σ_k overlap[i,k]*x[k,grn])
    """
    model = gp.Model("land_use_paper_miqp")
    model.Params.TimeLimit = cfg.max_runtime_sec
    model.Params.MIPGap = cfg.mip_gap
    model.Params.NonConvex = 2
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.OptimalityTol = 1.0e-9
    model.Params.IntFeasTol = 1.0e-9
    model.Params.NumericFocus = 3

    parcel_ids = parcels["parcel_id"].astype(int).tolist()
    area_by_id = dict(zip(parcels["parcel_id"].astype(int), parcels["area"].astype(float)))
    total_area = float(parcels["area"].sum())
    area_fraction = {i: area_by_id[i] / total_area for i in parcel_ids}

    x = model.addVars(parcel_ids, LAND_USE_TYPES, vtype=GRB.BINARY, name="x")
    s_by_type = {
        lu: model.addVars(parcel_ids, vtype=GRB.BINARY, name=f"s_{lu}")
        for lu in SERVICE_WEIGHTS
    }
    service_active = model.addVars(
        parcel_ids,
        list(SERVICE_WEIGHTS),
        vtype=GRB.BINARY,
        name="service_active",
    )
    ecology_covered = model.addVars(
        parcel_ids,
        lb=0.0,
        ub=area_fraction,
        name="ecology_covered",
    )

    for i in parcel_ids:
        model.addConstr(gp.quicksum(x[i, k] for k in LAND_USE_TYPES) == 1, name=f"assign_{i}")

    for k in LAND_USE_TYPES:
        alloc_area = gp.quicksum(area_fraction[i] * x[i, k] for i in parcel_ids)
        model.addConstr(alloc_area >= cfg.quota_lower[k], name=f"quota_lb_{k}")
        model.addConstr(alloc_area <= cfg.quota_upper[k], name=f"quota_ub_{k}")

    # OR 线性化：s_t[i] = OR_{k∈N(i)} x[k,t]
    for i in parcel_ids:
        neigh = service_neighbors.get(i, set())
        if not neigh:
            for lu, s_var in s_by_type.items():
                model.addConstr(s_var[i] == 0, name=f"s_{lu}_empty_{i}")
            continue
        for lu, s_var in s_by_type.items():
            model.addConstr(
                s_var[i] <= gp.quicksum(x[k, lu] for k in neigh if k in area_by_id),
                name=f"s_{lu}_ub_{i}",
            )
            for k in neigh:
                if k in area_by_id:
                    model.addConstr(s_var[i] >= x[k, lu], name=f"s_{lu}_lb_{i}_{k}")

    # Exact AND linearization used by the actual scalar reward.
    for i in parcel_ids:
        for lu, s_var in s_by_type.items():
            active = service_active[i, lu]
            model.addConstr(active <= x[i, LU_RESIDENTIAL])
            model.addConstr(active <= s_var[i])
            model.addConstr(active >= x[i, LU_RESIDENTIAL] + s_var[i] - 1)

        green_overlap = gp.quicksum(
            (float(overlap) / total_area) * x[k, LU_GREEN]
            for (ii, k), overlap in ecology_overlap.items()
            if int(ii) == int(i) and k in area_by_id
        )
        model.addConstr(
            ecology_covered[i] <= area_fraction[i] * x[i, LU_RESIDENTIAL],
            name=f"ecology_residential_cap_{i}",
        )
        model.addConstr(
            ecology_covered[i] <= green_overlap,
            name=f"ecology_green_cap_{i}",
        )

    residential_count = gp.quicksum(x[i, LU_RESIDENTIAL] for i in parcel_ids)
    residential_area = gp.quicksum(
        area_fraction[i] * x[i, LU_RESIDENTIAL] for i in parcel_ids
    )
    service_sum = gp.quicksum(
        SERVICE_WEIGHTS[lu] * service_active[i, lu]
        for i in parcel_ids
        for lu in SERVICE_WEIGHTS
    )
    ecology_sum = gp.quicksum(ecology_covered[i] for i in parcel_ids)
    service_score = model.addVar(lb=0.0, ub=1.0, name="Service")
    ecology_score = model.addVar(lb=0.0, ub=1.0, name="Ecology")
    # These two equalities put the exact variable-denominator terminal reward
    # into the optimization model.  Gurobi solves the resulting non-convex
    # MIQCP directly; there is no constant-denominator proxy.
    model.addQConstr(
        service_score * residential_count == service_sum,
        name="service_ratio_exact",
    )
    model.addQConstr(
        ecology_score * residential_area == ecology_sum,
        name="ecology_ratio_exact",
    )

    model.setObjective(
        cfg.alpha_service * service_score + ecology_score,
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed, status={model.Status}")

    assignment: Dict[int, str] = {}
    for i in parcel_ids:
        best_k = max(LAND_USE_TYPES, key=lambda k: float(x[i, k].X))
        assignment[i] = best_k

    return assignment, float(model.ObjVal), float(model.MIPGap)


def evaluate_metrics(
    parcels: gpd.GeoDataFrame,
    assignment: Dict[int, str],
    service_neighbors: Dict[int, Set[int]],
    ecology_overlap: Dict[Tuple[int, int], float],
    cfg: SolverConfig,
) -> Dict[str, float]:
    """给定解，计算 Service、Ecology 与论文口径的标量（用于日志与轨迹）。"""
    area_by_id = dict(zip(parcels["parcel_id"].astype(int), parcels["area"].astype(float)))
    pids = parcels["parcel_id"].astype(int).tolist()

    res_area = 0.0
    res_count = 0
    service_num = 0.0
    eco_num = 0.0

    for i in pids:
        ai = area_by_id[int(i)]
        if assignment.get(int(i)) != LU_RESIDENTIAL:
            continue
        res_area += ai
        res_count += 1
        neigh = service_neighbors.get(int(i), set())
        si = sum(
            SERVICE_WEIGHTS[lu]
            * (1.0 if any(assignment.get(int(k)) == lu for k in neigh) else 0.0)
            for lu in SERVICE_WEIGHTS
        )
        service_num += si

    # 论文 Ecology：每块住宅被 ESR 覆盖面积 / 住宅总面积；多块 grn 重叠时按地块封顶
    for i in pids:
        if assignment.get(int(i)) != LU_RESIDENTIAL:
            continue
        ai = area_by_id[int(i)]
        covered = sum(
            float(ov)
            for (ii, k), ov in ecology_overlap.items()
            if int(ii) == int(i) and assignment.get(int(k)) == LU_GREEN
        )
        eco_num += min(covered, ai)

    service_val = service_num / float(res_count) if res_count > 0 else 0.0
    ecology_val = min(1.0, eco_num / res_area) if res_area > 1.0e-12 else 0.0
    r_l = cfg.alpha_service * service_val + ecology_val

    return {
        "Service": float(service_val),
        "Ecology": float(ecology_val),
        "alpha_Service": float(cfg.alpha_service * service_val),
        "r_L": float(r_l),
        "residential_area": float(res_area),
        "residential_count": float(res_count),
    }


def build_trajectory(
    parcels: gpd.GeoDataFrame,
    edges: List[Tuple[int, int]],
    assignment: Dict[int, str],
    final_reward: float,
    metrics: Dict[str, float],
    service_neighbors: Dict[int, Set[int]],
    ecology_overlap: Dict[Tuple[int, int], float],
    alpha_service: float,
) -> Dict:
    neighbors: Dict[int, List[int]] = {int(i): [] for i in parcels["parcel_id"]}
    for i, j in edges:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    order = (
        parcels.sort_values("area", ascending=False)["parcel_id"].astype(int).tolist()
    )

    steps: List[Dict] = []

    for t, pid in enumerate(order, start=1):
        lu = assignment[pid]
        steps.append(
            {
                "t": t,
                "state": {
                    "assigned_count": t,
                    "remaining_count": len(order) - t,
                    "parcel_id": pid,
                },
                "action": {
                    "parcel_id": pid,
                    "land_use": lu,
                },
                "reward": 0.0,
            }
        )

    if steps:
        steps[-1]["reward"] = float(final_reward)

    rtg = 0.0
    for step in reversed(steps):
        rtg += float(step["reward"])
        step["return_to_go"] = rtg

    return {
        "task_type": "land_use_planning",
        "objective": "alpha * Service + Ecology",
        "reward_timing": "terminal_only",
        "metrics": metrics,
        "num_steps": len(steps),
        "final_reward": float(final_reward),
        "steps": steps,
    }


def _add_args(parser):
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/topology/synth/0000/topology.gpkg"),
        help="拓扑文件（topology.json / topology.gpkg）或其所在目录",
    )
    parser.add_argument(
        "--solution-csv",
        type=Path,
        default=None,
        help="用地分配 CSV（默认：<input 父目录>/land_use_solution.csv）",
    )
    parser.add_argument(
        "--trajectory-json",
        type=Path,
        default=None,
        help="轨迹 JSON（默认：<input 父目录>/land_use_trajectory.json）",
    )
    parser.add_argument("--time-limit", type=int, default=20,
                        help="Gurobi 求解时间限制（秒）")
    parser.add_argument("--mip-gap", type=float, default=0.01,
                        help="Gurobi MIP gap")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="论文中 α：Service 项权重（Ecology 权重为 1）")
    parser.add_argument("--service-radius", type=float, default=500.0,
                        help="15 分钟服务半径（米，需投影坐标系）")
    parser.add_argument("--ecology-buffer", type=float, default=300.0,
                        help="生态服务范围 ESR 缓冲距离（米）")
    parser.add_argument("--max-parcels", type=int, default=None,
                        help="随机采样地块数上限（Gurobi 社区版约支持 N<=450）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")


def run(args) -> None:
    cfg = build_default_config()
    cfg.max_runtime_sec = int(getattr(args, "time_limit", cfg.max_runtime_sec))
    cfg.mip_gap = float(getattr(args, "mip_gap", cfg.mip_gap))
    cfg.alpha_service = float(getattr(args, "alpha", cfg.alpha_service))
    cfg.service_radius_m = float(getattr(args, "service_radius", cfg.service_radius_m))
    cfg.ecology_buffer_m = float(getattr(args, "ecology_buffer", cfg.ecology_buffer_m))

    topo_path = resolve_topology_path(args.input)
    parcels, roads, intersections = load_topology(topo_path)

    out_dir = topo_path.parent
    if args.solution_csv is None:
        args.solution_csv = out_dir / "land_use_solution.csv"
    if args.trajectory_json is None:
        args.trajectory_json = out_dir / "land_use_trajectory.json"

    if args.max_parcels is not None and len(parcels) > args.max_parcels:
        rng = np.random.default_rng(int(args.seed))
        keep = rng.choice(len(parcels), size=int(args.max_parcels), replace=False)
        parcels = parcels.iloc[sorted(keep)].reset_index(drop=True)
        parcels["parcel_id"] = parcels.index.astype(int)
        print(f"[Sampling] Kept {len(parcels)} parcels (from original).")

    edges = build_parcel_adjacency(parcels)
    service_neighbors = precompute_service_neighbors(parcels, cfg.service_radius_m)
    ecology_overlap = precompute_ecology_overlap(parcels, cfg.ecology_buffer_m)

    assignment, obj, gap = solve_land_use_mip(
        parcels, edges, service_neighbors, ecology_overlap, cfg
    )

    metrics = evaluate_metrics(parcels, assignment, service_neighbors, ecology_overlap, cfg)
    if not math.isclose(
        float(metrics["r_L"]), float(obj), rel_tol=1.0e-8, abs_tol=1.0e-8
    ):
        raise RuntimeError(
            "land reward/model objective mismatch: "
            f"reward={metrics['r_L']:.12f}, objective={obj:.12f}"
        )
    metrics["solver_objective"] = float(obj)

    keep_cols = ["parcel_id", "area", "geometry"]
    if "source_id" in parcels.columns:
        keep_cols.insert(1, "source_id")
    out_df = parcels[keep_cols].copy()
    out_df["land_use"] = out_df["parcel_id"].map(assignment)
    args.solution_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.drop(columns=["geometry"]).to_csv(args.solution_csv, index=False)

    traj = build_trajectory(
        parcels=parcels,
        edges=edges,
        assignment=assignment,
        final_reward=float(metrics["r_L"]),
        metrics=metrics,
        service_neighbors=service_neighbors,
        ecology_overlap=ecology_overlap,
        alpha_service=cfg.alpha_service,
    )
    args.trajectory_json.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_json.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] parcels={len(parcels)}, adjacency_edges={len(edges)}")
    print(f"[Done] solver_objective = r_L = {obj:.6f}, mip_gap={gap:.6f}")
    print(
        f"[Done] metrics: Service={metrics['Service']:.6f}, Ecology={metrics['Ecology']:.6f}, "
        f"r_L={metrics['r_L']:.6f}"
    )
    print(f"[Done] solution -> {args.solution_csv}")
    print(f"[Done] trajectory -> {args.trajectory_json}")


def add_subcommand(subparsers):
    parser = subparsers.add_parser(
        "land-use",
        help="用地规划 MILP/MIQP 求解（Gurobi），目标 α·Service+Ecology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用地规划 MILP/MIQP 求解（Gurobi）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
