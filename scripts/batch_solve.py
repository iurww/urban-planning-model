#!/usr/bin/env python3
# coding: utf-8

"""对单个拓扑目录依次运行用地、道路、POI 三类静态 MIP，并写出 CSV + 轨迹 JSON。"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import scripts.mip_land_use as mip_land_use
import scripts.mip_poi_placement as mip_poi_placement
import scripts.mip_road_planning as mip_road_planning
from scripts.topology_io import resolve_topology_path


def _parcel_count(tile_dir: Path) -> int | None:
    """优先从新格式 topology.json 数地块，其次读旧 summary.json。"""
    topo_json = tile_dir / "topology.json"
    if topo_json.is_file():
        try:
            doc = json.loads(topo_json.read_text(encoding="utf-8"))
            return len(doc.get("parcels", []))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    summary_path = tile_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
        c = doc.get("counts", {})
        return int(c.get("parcels_large", 0) or 0) + int(c.get("parcels_small", 0) or 0)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _is_solved(tile_dir: Path) -> bool:
    return (tile_dir / "land_use_solution.csv").is_file() and (
        tile_dir / "road_plan_solution.csv"
    ).is_file() and (tile_dir / "poi_plan_solution.csv").is_file()


STATIC_EXPERT_ARTIFACTS = (
    "land_use_solution.csv",
    "land_use_trajectory.json",
    "road_plan_solution.csv",
    "road_plan_trajectory.json",
    "poi_plan_solution.csv",
    "poi_plan_trajectory.json",
)


def _remove_existing_expert(tile_dir: Path) -> None:
    """Prevent a failed regeneration from leaving a mixed old/new expert set."""
    for name in STATIC_EXPERT_ARTIFACTS:
        (tile_dir / name).unlink(missing_ok=True)


def _run_solve_all(topo: Path, args: argparse.Namespace) -> None:
    if not topo.exists():
        raise FileNotFoundError(topo)

    ns = argparse.Namespace(
        input=topo,
        solution_csv=None,
        trajectory_json=None,
        time_limit=int(args.time_limit),
        mip_gap=float(args.mip_gap),
        alpha=float(args.alpha),
        service_radius=float(args.service_radius),
        ecology_buffer=float(args.ecology_buffer),
        max_parcels=args.max_parcels,
        seed=int(args.seed),
    )
    print("[solve-all] 1/3 用地规划 …")
    mip_land_use.run(ns)

    ns2 = argparse.Namespace(
        input=topo,
        solution_csv=None,
        trajectory_json=None,
        budget_frac=float(args.budget_frac),
        time_limit=int(args.time_limit),
        mip_gap=float(args.mip_gap),
    )
    print("[solve-all] 2/3 道路规划 …")
    mip_road_planning.run(ns2)

    ns3 = argparse.Namespace(
        input=topo,
        solution_csv=None,
        trajectory_json=None,
        num_poi=int(args.num_poi),
        time_limit=int(args.time_limit),
        mip_gap=float(args.mip_gap),
    )
    print("[solve-all] 3/3 POI 选址 …")
    mip_poi_placement.run(ns3)

    print("[solve-all] 全部完成。")


def _try_solve_one(tile_dir: Path, args: argparse.Namespace) -> tuple[bool, str | None]:
    """返回 (成功, 失败原因)。"""
    topo = resolve_topology_path(tile_dir)
    if not topo.is_file():
        return False, "缺少 topology.json / topology.gpkg"

    min_parcels = int(getattr(args, "min_parcels", 0))
    if min_parcels > 0:
        n = _parcel_count(tile_dir)
        if n is not None and n < min_parcels:
            return False, f"地块数 {n} < min_parcels={min_parcels}"

    if getattr(args, "skip_existing", False) and _is_solved(tile_dir):
        return False, "已求解（跳过）"

    try:
        _remove_existing_expert(tile_dir)
        _run_solve_all(tile_dir, args)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _iter_real_tile_dirs(base: Path) -> list[Path]:
    """新格式：所有含 topology.json 的目录；兼容旧 tile_*/topology.gpkg。"""
    # Prepared training directories are one-level directory symlinks.  pathlib
    # does not recurse through them, so include their canonical topology file
    # explicitly before scanning ordinary directories recursively.
    dirs = {
        child
        for child in base.iterdir()
        if child.is_dir() and (child / "topology.json").is_file()
    }
    dirs.update(p.parent for p in base.rglob("topology.json"))
    dirs.update(
        p.parent
        for p in base.rglob("topology.gpkg")
        if p.parent.name.startswith("tile_") and p.parent not in dirs
    )
    return sorted(dirs)


def run(args: argparse.Namespace) -> None:
    topo = Path(args.topology)
    tile_dir = topo.parent if topo.suffix.lower() in {".json", ".gpkg"} else topo
    ok, err = _try_solve_one(tile_dir, args)
    if not ok:
        raise SystemExit(f"solve-all 失败: {err}")


def run_batch(args: argparse.Namespace) -> None:
    base = Path(args.topology_base)
    for i in range(int(args.batch_from), int(args.batch_to) + 1):
        topo = base / f"{i:04d}"
        if not topo.is_dir():
            print(f"[solve-all-batch] 跳过（非目录或不存在）: {topo}")
            continue
        print(f"\n{'=' * 20} [{i:04d}] {'=' * 20}")
        ok, err = _try_solve_one(topo, args)
        if not ok:
            print(f"[FAIL] {topo}: {err}")


def run_batch_real(args: argparse.Namespace) -> None:
    """遍历 data/topology/real 下 tile_*/ 目录批量 solve-all，失败不中断。"""
    base = Path(args.topology_base)
    if not base.is_dir():
        raise FileNotFoundError(f"目录不存在: {base}")

    tiles = _iter_real_tile_dirs(base)
    if not tiles:
        print(f"[solve-all-real] 未找到 tile: {base}")
        return

    ok_count = 0
    skip_count = 0
    fail_count = 0
    failures: list[tuple[str, str]] = []

    for idx, tile_dir in enumerate(tiles, 1):
        rel = tile_dir.relative_to(base)
        print(f"\n[{idx}/{len(tiles)}] {rel}")
        ok, err = _try_solve_one(tile_dir, args)
        if ok:
            ok_count += 1
            print(f"[OK] {rel}")
        elif err and err.startswith("已求解"):
            skip_count += 1
            print(f"[SKIP] {rel}: {err}")
        else:
            fail_count += 1
            failures.append((str(rel), err or "unknown"))
            print(f"[FAIL] {rel}: {err}")
            if getattr(args, "verbose", False):
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(
        f"[solve-all-real] 完成: 成功 {ok_count}, 跳过 {skip_count}, "
        f"失败 {fail_count}, 共 {len(tiles)}"
    )
    if getattr(args, "fail_log", None):
        log_path = Path(args.fail_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{name}\t{reason}\n" for name, reason in failures]
        log_path.write_text("".join(lines), encoding="utf-8")
        if failures:
            print(f"[solve-all-real] 失败列表: {log_path}")
        else:
            print(f"[solve-all-real] 无失败，已清空: {log_path}")
    elif failures:
        print("[solve-all-real] 失败样例（前 10）:")
        for name, reason in failures[:10]:
            print(f"  {name}: {reason}")


def _add_solve_all_params(p: argparse.ArgumentParser) -> None:
    p.add_argument("--time-limit", type=int, default=20, help="各 MIP 时间上限（秒）")
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=1.0, help="用地规划中 Service 项权重 α")
    p.add_argument("--service-radius", type=float, default=500.0, help="用地规划服务半径（米）")
    p.add_argument("--ecology-buffer", type=float, default=300.0, help="用地规划生态缓冲距离（米）")
    p.add_argument("--max-parcels", type=int, default=None, help="用地 MIP 地块数上限（社区版等）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget-frac", type=float, default=0.85, help="低绕行道路网络的建设预算比例")
    p.add_argument("--num-poi", type=int, default=5, help="POI 设施数 K")
    p.add_argument(
        "--min-parcels",
        type=int,
        default=5,
        help="summary 中地块数低于此值的 tile 跳过（0 表示不限制）",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若三类 solution CSV 已存在则跳过",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="失败时打印完整 traceback")


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "topology",
        type=Path,
        help="拓扑目录（含 topology.gpkg）或 .gpkg 文件路径",
    )
    _add_solve_all_params(p)


def _add_numbered_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "topology_base",
        type=Path,
        help="拓扑父目录",
    )
    p.add_argument(
        "--from",
        dest="batch_from",
        type=int,
        default=0,
        metavar="N",
        help="起始编号（含），对应子目录名 {N:04d}（仅 solve-all-batch）",
    )
    p.add_argument(
        "--to",
        dest="batch_to",
        type=int,
        default=1000,
        metavar="N",
        help="结束编号（含）",
    )
    _add_solve_all_params(p)


def _add_real_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "topology_base",
        type=Path,
        nargs="?",
        default=Path("data/final"),
        help="拓扑根目录",
    )
    p.add_argument(
        "--fail-log",
        type=Path,
        default=Path("data/topology/real/solve_failures.tsv"),
        help="失败 tile 列表输出路径",
    )
    _add_solve_all_params(p)


def add_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "solve-all",
        help="对一拓扑运行用地 + 道路 + POI 三类 Gurobi 求解并生成轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(p)
    p.set_defaults(func=run)

    pb = subparsers.add_parser(
        "solve-all-batch",
        help="对父目录下 0000…N 各子目录依次执行 solve-all",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_numbered_batch_args(pb)
    pb.set_defaults(func=run_batch)

    pr = subparsers.add_parser(
        "solve-all-real",
        help="对根目录下全部拓扑（topology.json / tile_*）批量 solve-all，失败跳过并继续",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_real_batch_args(pr)
    pr.set_defaults(func=run_batch_real)

def main() -> None:
    parser = argparse.ArgumentParser(description="批量三类静态规划求解")
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
