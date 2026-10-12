#!/usr/bin/env python3
"""CLI and process-level orchestration for projected OSM topology blocks."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform

from scripts.osm import TileSpec, extract_osm_evidence, reconstruct_topology

@dataclass(frozen=True)
class MetricTile:
    column: int
    row: int
    min_x: float
    min_y: float
    tile_id: str


def build_metric_tile_spec(
    *,
    crs: str,
    min_x: float,
    min_y: float,
    size_m: float,
    tile_id: str,
    city: str,
    column: int,
    row: int,
) -> TileSpec:
    """Construct an exact square tile in a projected metric CRS."""
    metric_crs = CRS.from_user_input(crs)
    if not metric_crs.is_projected:
        raise ValueError(f"metric tile CRS must be projected, got {crs}")
    size = float(size_m)
    if size <= 0:
        raise ValueError(f"tile size must be positive, got {size_m}")
    bounds_m = (
        float(min_x),
        float(min_y),
        float(min_x) + size,
        float(min_y) + size,
    )
    inverse = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    polygon_wgs84 = transform(inverse.transform, box(*bounds_m))
    return TileSpec(
        tile_id=tile_id,
        city=city,
        crs=metric_crs.to_string(),
        size_m=size,
        column=int(column),
        row=int(row),
        bounds_m=bounds_m,
        polygon_wgs84=polygon_wgs84,
        stats={"tile_width_m": size, "tile_height_m": size},
    )


def generate_tiles(
    *,
    city_slug: str,
    origin_x: float,
    origin_y: float,
    size_m: float,
    col_range: tuple[int, int],
    row_range: tuple[int, int],
) -> list[MetricTile]:
    return [
        MetricTile(
            column=column,
            row=row,
            min_x=float(origin_x) + column * float(size_m),
            min_y=float(origin_y) + row * float(size_m),
            tile_id=f"{city_slug}_{int(size_m / 1000)}km_c{column}_r{row}",
        )
        for row in range(*row_range)
        for column in range(*col_range)
    ]


def _actual_dimensions(topology: dict[str, Any]) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for parcel in topology.get("parcels", []):
        for ring in parcel.get("geometry", []):
            for point in ring:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    if xs:
        return max(xs) - min(xs), max(ys) - min(ys)
    metadata = topology.get("metadata", {})
    width = metadata.get("tile_width_m")
    height = metadata.get("tile_height_m")
    if width is not None and height is not None:
        return float(width), float(height)
    return 0.0, 0.0


def validate_exact_square(topology_path: Path, expected_size_m: float) -> dict[str, int]:
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    width, height = _actual_dimensions(topology)
    tolerance = 0.01
    if abs(width - expected_size_m) > tolerance or abs(height - expected_size_m) > tolerance:
        raise ValueError(
            f"{topology_path}: expected {expected_size_m}m x {expected_size_m}m, "
            f"got {width:.3f}m x {height:.3f}m"
        )
    return {
        "parcels": len(topology.get("parcels", [])),
        "roads": len(topology.get("roads", [])),
        "intersections": len(topology.get("intersections", [])),
    }


def process_tile(
    tile: MetricTile,
    *,
    pbf: str,
    city: str,
    crs: str,
    size_m: float,
    cache_dir: str,
    force: bool,
    verbose: bool,
) -> tuple[str, str, dict[str, int], float]:
    started = time.monotonic()
    tile_dir = Path(cache_dir) / tile.tile_id
    topology_path = tile_dir / "abstract" / "topology.json"
    if topology_path.is_file() and not force:
        counts = validate_exact_square(topology_path, size_m)
        return tile.tile_id, "cached", counts, time.monotonic() - started
    if force and tile_dir.exists():
        shutil.rmtree(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    spec = build_metric_tile_spec(
        crs=crs,
        min_x=tile.min_x,
        min_y=tile.min_y,
        size_m=size_m,
        tile_id=tile.tile_id,
        city=city,
        column=tile.column,
        row=tile.row,
    )
    evidence_dir = tile_dir / "evidence"
    abstract_dir = tile_dir / "abstract"

    try:
        if verbose:
            report = extract_osm_evidence(Path(pbf), spec, evidence_dir)
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    report = extract_osm_evidence(Path(pbf), spec, evidence_dir)
        if int(report.get("counts", {}).get("logical_roads") or 0) == 0:
            return tile.tile_id, "no_roads", {}, time.monotonic() - started

        status = "generated"
        try:
            if verbose:
                reconstruct_topology(evidence_dir, abstract_dir)
            else:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        reconstruct_topology(evidence_dir, abstract_dir)
        except Exception:
            # v7 writes topology.json before building the optional advanced
            # traffic-signal contract.  The compact training signal contract
            # can be recovered from the retained v4 evidence later.
            if not topology_path.is_file():
                raise
            status = "generated_topology_only"
        counts = validate_exact_square(topology_path, size_m)
        return tile.tile_id, status, counts, time.monotonic() - started
    finally:
        gc.collect()


def _worker(tile: MetricTile, kwargs: dict[str, Any]) -> tuple[str, str, dict[str, int], float]:
    return process_tile(tile, **kwargs)


def run(args: argparse.Namespace) -> None:
    pbf = Path(args.pbf)
    if not pbf.is_file():
        raise FileNotFoundError(f"PBF not found: {pbf}")
    size_m = float(args.size_m)
    tiles = generate_tiles(
        city_slug=args.city_slug,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        size_m=size_m,
        col_range=tuple(args.col_range),
        row_range=tuple(args.row_range),
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[metric-grid] city={args.city} crs={args.crs} count={len(tiles)} "
        f"tile={size_m:.0f}m x {size_m:.0f}m cache={cache_dir}",
        flush=True,
    )
    kwargs = {
        "pbf": str(pbf),
        "city": args.city,
        "crs": args.crs,
        "size_m": size_m,
        "cache_dir": str(cache_dir),
        "force": bool(args.force),
        "verbose": bool(args.verbose),
    }
    workers = max(1, int(args.workers))
    statuses: dict[str, int] = {}
    failures: list[str] = []
    if workers == 1:
        for index, tile in enumerate(tiles, 1):
            try:
                name, status, counts, elapsed = process_tile(tile, **kwargs)
            except Exception as exc:
                failures.append(tile.tile_id)
                print(f"[{index}/{len(tiles)}] FAIL {tile.tile_id}: {exc}", flush=True)
                if args.verbose:
                    traceback.print_exc()
                continue
            statuses[status] = statuses.get(status, 0) + 1
            print(
                f"[{index}/{len(tiles)}] {status.upper()} {name} "
                f"P/R/I={counts.get('parcels', 0)}/{counts.get('roads', 0)}/"
                f"{counts.get('intersections', 0)} {elapsed:.1f}s",
                flush=True,
            )
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            future_to_tile = {
                executor.submit(_worker, tile, kwargs): tile for tile in tiles
            }
            pending = set(future_to_tile)
            completed = 0
            heartbeat = max(10.0, float(args.progress_interval))
            while pending:
                done, pending = wait(pending, timeout=heartbeat, return_when=FIRST_COMPLETED)
                if not done:
                    print(
                        f"[progress] completed={completed}/{len(tiles)} running={len(pending)}",
                        flush=True,
                    )
                    continue
                for future in done:
                    completed += 1
                    tile = future_to_tile[future]
                    try:
                        name, status, counts, elapsed = future.result()
                    except Exception as exc:
                        failures.append(tile.tile_id)
                        print(f"[{completed}/{len(tiles)}] FAIL {tile.tile_id}: {exc}", flush=True)
                        continue
                    statuses[status] = statuses.get(status, 0) + 1
                    print(
                        f"[{completed}/{len(tiles)}] {status.upper()} {name} "
                        f"P/R/I={counts.get('parcels', 0)}/{counts.get('roads', 0)}/"
                        f"{counts.get('intersections', 0)} {elapsed:.1f}s",
                        flush=True,
                    )
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    print(f"[done] statuses={statuses} failures={len(failures)}", flush=True)
    if failures:
        print("[failed] " + ", ".join(failures), flush=True)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--city", required=True, help="human-readable city name")
    parser.add_argument("--city-slug", required=True, help="lowercase ID prefix")
    parser.add_argument("--crs", required=True, help="projected metric CRS, e.g. EPSG:32650")
    parser.add_argument("--origin-x", type=float, required=True, help="grid column 0 easting")
    parser.add_argument("--origin-y", type=float, required=True, help="grid row 0 northing")
    parser.add_argument("--col-range", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    parser.add_argument("--row-range", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    parser.add_argument("--size-m", type=float, default=4000.0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", "-j", type=int, default=4)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "batch-osm",
        help="generate an exact projected-metric square topology grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact projected-metric square topology blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
