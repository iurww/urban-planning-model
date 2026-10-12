#!/usr/bin/env python3
"""Visualize every candidate topology and whichever expert artifacts exist."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle


LAND_COLORS = {
    "grn": "#66BB6A",
    "com": "#F4A460",
    "res": "#90CAF9",
    "sch": "#FFD54F",
    "hos": "#EF9A9A",
    "off": "#B39DDB",
}
LAND_LABELS = {
    "grn": "Green",
    "com": "Commercial",
    "res": "Residential",
    "sch": "School",
    "hos": "Hospital",
    "off": "Office",
}
TOPOLOGY_ROAD_COLOR = "#475569"
ROAD_LABEL_COLOR = "#0F766E"
JUNCTION_COLOR = "#1D4ED8"
SIGNAL_JUNCTION_COLOR = "#DC2626"
JUNCTION_LABEL_COLOR = "#0F172A"
BOUNDARY_PORTAL_COLOR = "#2563EB"
SELECTION_SPLITS = ("sft", "ppo", "test")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _csv_map(path: Path, key: str, value: str, cast=str) -> dict[int, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row[key]): cast(row[value])
            for row in csv.DictReader(handle)
        }


def _parcel_patches(topology: dict[str, Any]) -> tuple[list[Polygon], list[int]]:
    patches: list[Polygon] = []
    indices: list[int] = []
    for index, parcel in enumerate(topology["parcels"]):
        rings = parcel.get("geometry", [])
        if not rings or len(rings[0]) < 3:
            continue
        patches.append(Polygon(rings[0], closed=True))
        indices.append(index)
    return patches, indices


def _draw_parcels(
    ax,
    topology: dict[str, Any],
    colors: dict[int, str] | None = None,
    *,
    default: str = "#E8EEF3",
    alpha: float = 0.82,
) -> None:
    patches, indices = _parcel_patches(topology)
    collection = PatchCollection(
        patches,
        facecolor=[colors.get(index, default) if colors else default for index in indices],
        edgecolor="#FFFFFF",
        linewidth=0.55,
        alpha=alpha,
        zorder=1,
    )
    ax.add_collection(collection)


def _draw_roads(
    ax,
    topology: dict[str, Any],
    *,
    selected: set[int] | None = None,
) -> None:
    roads = topology["roads"]
    if selected is None:
        lines = [road["geometry"] for road in roads]
        ax.add_collection(
            LineCollection(
                lines,
                colors=TOPOLOGY_ROAD_COLOR,
                linewidths=1.0,
                alpha=0.9,
                zorder=3,
            )
        )
        return

    unselected_lines = [road["geometry"] for index, road in enumerate(roads) if index not in selected]
    selected_lines = [road["geometry"] for index, road in enumerate(roads) if index in selected]
    if unselected_lines:
        ax.add_collection(
            LineCollection(unselected_lines, colors="#94A3B8", linewidths=0.8, alpha=0.72, zorder=2)
        )
    if selected_lines:
        ax.add_collection(
            LineCollection(selected_lines, colors="#DC2626", linewidths=2.6, alpha=0.96, zorder=4)
        )


def _topology_bounds(topology: dict[str, Any]) -> tuple[float, float, float, float]:
    coordinates = [
        point
        for parcel in topology["parcels"]
        for ring in parcel.get("geometry", [])
        for point in ring
    ]
    if not coordinates:
        coordinates = [item["point"] for item in topology["intersections"]]
    if not coordinates:
        return 0.0, 0.0, 0.0, 0.0
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    return min(xs), min(ys), max(xs), max(ys)


def _finish_map(ax, topology: dict[str, Any], title: str) -> None:
    min_x, min_y, max_x, max_y = _topology_bounds(topology)
    coordinates = topology["parcels"] or topology["intersections"]
    if coordinates:
        width = max_x - min_x
        height = max_y - min_y
        pad = max(width, height, 1.0) * 0.025
        ax.set_xlim(min_x - pad, max_x + pad)
        ax.set_ylim(min_y - pad, max_y + pad)
        ax.add_patch(
            Rectangle(
                (min_x, min_y), width, height, fill=False,
                edgecolor="#0F172A", linewidth=0.9, linestyle="--", zorder=9,
            )
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_axis_off()


def _signal_ids(tile_dir: Path) -> set[str]:
    path = tile_dir / "traffic_signals.json"
    if not path.is_file():
        return set()
    payload = _json(path)
    result = {str(value) for value in payload.get("signalized_intersection_ids", [])}
    result.update(
        str(controller["intersection_id"])
        for controller in payload.get("controllers", [])
        if isinstance(controller, dict) and controller.get("intersection_id")
    )
    return result


def _short_entity_id(value: Any) -> str:
    """Keep labels compact while preserving the stable ID inside one topology."""
    text = str(value)
    return text.rsplit("/", 1)[-1]


def _compact_numeric_entity_id(value: Any) -> str:
    """Remove numeric zero-padding from a display label without changing its ID."""
    short_id = _short_entity_id(value)
    match = re.fullmatch(r"(.*?)(\d+)", short_id)
    if match is None:
        return short_id
    prefix, digits = match.groups()
    return f"{prefix}{int(digits)}"


def _polyline_midpoint(geometry: list[Any]) -> tuple[float, float] | None:
    """Return the length-weighted midpoint of a road polyline."""
    points = [
        (float(point[0]), float(point[1]))
        for point in geometry
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    segments = [
        (left, right, math.hypot(right[0] - left[0], right[1] - left[1]))
        for left, right in zip(points, points[1:])
    ]
    total_length = sum(length for _, _, length in segments)
    if total_length <= 0:
        return points[0]
    target = total_length / 2
    traversed = 0.0
    for left, right, length in segments:
        if traversed + length >= target:
            ratio = (target - traversed) / length
            return (
                left[0] + ratio * (right[0] - left[0]),
                left[1] + ratio * (right[1] - left[1]),
            )
        traversed += length
    return points[-1]


def _junctions(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only real road intersections, excluding portals and geometry bends."""
    return [
        item
        for item in topology.get("intersections", [])
        if item.get("kind") == "junction"
    ]


def _draw_topology_entity_labels(
    ax,
    topology: dict[str, Any],
    signal_ids: set[str],
) -> None:
    parcels = topology.get("parcels", [])
    parcel_font_size = 5.0 if len(parcels) <= 80 else 4.2 if len(parcels) <= 160 else 3.5
    for parcel in parcels:
        rings = parcel.get("geometry", [])
        if not rings or len(rings[0]) < 3:
            continue
        x, y = _polygon_centroid(rings[0])
        ax.text(
            x,
            y,
            _short_entity_id(parcel.get("id", "")),
            ha="center",
            va="center",
            fontsize=parcel_font_size,
            color="#334155",
            zorder=5,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "#FFFFFF",
                "edgecolor": "#64748B",
                "linewidth": 0.38,
                "alpha": 0.88,
            },
        )

    min_x, min_y, max_x, max_y = _topology_bounds(topology)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    roads = topology.get("roads", [])
    road_circle_radius = span * (0.0085 if len(roads) > 100 else 0.0095)
    road_font_size = 2.7 if len(roads) > 100 else 3.1
    for road in roads:
        midpoint = _polyline_midpoint(road.get("geometry", []))
        if midpoint is None:
            continue
        x, y = midpoint
        ax.add_patch(
            Circle(
                (x, y),
                radius=road_circle_radius,
                facecolor="#FFFFFF",
                edgecolor=TOPOLOGY_ROAD_COLOR,
                linewidth=0.72,
                alpha=0.94,
                zorder=5.4,
            )
        )
        ax.text(
            x,
            y,
            _compact_numeric_entity_id(road.get("id", "")),
            ha="center",
            va="center",
            fontsize=road_font_size,
            color=ROAD_LABEL_COLOR,
            fontweight="bold",
            zorder=5.5,
            clip_on=True,
        )

    junctions = _junctions(topology)
    circle_radius = span * (0.011 if len(junctions) > 80 else 0.0125)
    junction_font_size = 3.0 if len(junctions) > 80 else 3.4
    for junction in junctions:
        x, y = (float(value) for value in junction.get("point", [0.0, 0.0])[:2])
        signalized = str(junction.get("id")) in signal_ids
        color = SIGNAL_JUNCTION_COLOR if signalized else JUNCTION_COLOR
        ax.add_patch(
            Circle(
                (x, y),
                radius=circle_radius,
                fill=False,
                edgecolor=color,
                linewidth=0.9 if signalized else 0.72,
                zorder=6,
            )
        )
        ax.text(
            x,
            y,
            _compact_numeric_entity_id(junction.get("id", "")),
            ha="center",
            va="center",
            fontsize=junction_font_size,
            color=JUNCTION_LABEL_COLOR,
            fontweight="bold" if signalized else "normal",
            zorder=7,
            clip_on=True,
            path_effects=[
                path_effects.withStroke(linewidth=1.25, foreground="#FFFFFF")
            ],
        )


def _plot_topology(ax, tile_dir: Path, topology: dict[str, Any]) -> None:
    _draw_parcels(ax, topology, default="#EAF2F8")
    _draw_roads(ax, topology)
    signals = _signal_ids(tile_dir)
    _draw_topology_entity_labels(ax, topology, signals)
    portals = [
        item
        for item in topology.get("intersections", [])
        if item.get("kind") == "boundary_portal"
    ]
    if portals:
        ax.scatter(
            [item["point"][0] for item in portals],
            [item["point"][1] for item in portals],
            s=18,
            c=BOUNDARY_PORTAL_COLOR,
            marker="s",
            edgecolors="white",
            linewidths=0.35,
            zorder=8,
        )
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", color="none", markerfacecolor="none",
                markeredgecolor=JUNCTION_COLOR, label="Junction", markersize=6,
            ),
            Line2D(
                [], [], marker="o", color="none", markerfacecolor="none",
                markeredgecolor=SIGNAL_JUNCTION_COLOR,
                label="Signalized junction", markersize=6,
            ),
            Line2D(
                [], [], marker="s", color="none",
                markerfacecolor=BOUNDARY_PORTAL_COLOR,
                markeredgecolor="white", label="Boundary portal", markersize=6,
            ),
        ],
        loc="upper right",
        fontsize=7,
        frameon=True,
    )
    _finish_map(ax, topology, "Topology: parcels, roads and junctions")


def _plot_land(ax, tile_dir: Path, topology: dict[str, Any]) -> None:
    solution = _csv_map(tile_dir / "land_use_solution.csv", "parcel_id", "land_use")
    colors = {index: LAND_COLORS.get(value, "#CBD5E1") for index, value in solution.items()}
    _draw_parcels(ax, topology, colors)
    _draw_roads(ax, topology)
    ax.legend(
        handles=[Patch(facecolor=color, label=LAND_LABELS[key]) for key, color in LAND_COLORS.items()],
        ncol=2, loc="upper right", fontsize=6.5, frameon=True,
    )
    reward = float(_json(tile_dir / "land_use_trajectory.json").get("final_reward", 0.0))
    _finish_map(ax, topology, f"Land-use expert (reward={reward:.3f})")


def _plot_road(ax, tile_dir: Path, topology: dict[str, Any]) -> None:
    solution = _csv_map(tile_dir / "road_plan_solution.csv", "road_row_index", "upgraded", int)
    selected = {index for index, value in solution.items() if value == 1}
    _draw_parcels(ax, topology, default="#EDF2F7", alpha=0.62)
    _draw_roads(ax, topology, selected=selected)
    ax.legend(
        handles=[
            Line2D([], [], color="#DC2626", linewidth=2.6, label="Upgraded"),
            Line2D([], [], color="#94A3B8", linewidth=1.0, label="Unchanged"),
        ],
        loc="upper right", fontsize=7, frameon=True,
    )
    trajectory = _json(tile_dir / "road_plan_trajectory.json")
    reward = float(trajectory.get("final_reward", 0.0))
    metrics = trajectory.get("metrics", {})
    junction_coverage = metrics.get("junction_coverage")
    portal_coverage = metrics.get("boundary_portal_coverage")
    detour = metrics.get("observed_max_detour_factor")
    details = []
    if isinstance(junction_coverage, (int, float)):
        details.append(f"J={float(junction_coverage):.0%}")
    if isinstance(portal_coverage, (int, float)):
        details.append(f"B={float(portal_coverage):.0%}")
    if isinstance(detour, (int, float)):
        details.append(f"max detour={float(detour):.2f}×")
    suffix = " | " + ", ".join(details) if details else ""
    _finish_map(
        ax,
        topology,
        f"Road expert: {len(selected)}/{len(topology['roads'])} upgraded "
        f"(r={reward:.3f}){suffix}",
    )


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    if not ring:
        return 0.0, 0.0
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    if not points:
        return 0.0, 0.0
    if len(points) < 3:
        return (
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        )
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for left, right in zip(points, points[1:] + points[:1]):
        cross = float(left[0]) * float(right[1]) - float(right[0]) * float(left[1])
        twice_area += cross
        x_sum += (float(left[0]) + float(right[0])) * cross
        y_sum += (float(left[1]) + float(right[1])) * cross
    if abs(twice_area) < 1.0e-12:
        return (
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        )
    return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)


def _plot_poi(ax, tile_dir: Path, topology: dict[str, Any]) -> None:
    solution = _csv_map(tile_dir / "poi_plan_solution.csv", "parcel_id", "place_poi", int)
    selected = [index for index, value in solution.items() if value == 1]
    colors = {index: "#F0ABFC" for index in selected}
    _draw_parcels(ax, topology, colors, default="#DBEAFE")
    _draw_roads(ax, topology)
    centers = [
        _polygon_centroid(topology["parcels"][index].get("geometry", [[]])[0])
        for index in selected
    ]
    if centers:
        ax.scatter(
            [point[0] for point in centers], [point[1] for point in centers],
            s=90, c="#A21CAF", marker="*", edgecolors="white", linewidths=0.7, zorder=7,
        )
    reward = float(_json(tile_dir / "poi_plan_trajectory.json").get("final_reward", 0.0))
    _finish_map(ax, topology, f"POI expert: {len(selected)} facilities (reward={reward:.3f})")


def _signal_statistics(
    tile_dir: Path,
) -> tuple[dict[str, float], list[float], list[float], dict[str, Any], dict[str, Any]]:
    trajectory = _json(tile_dir / "signal_trajectory.json")
    queue_sum: dict[str, float] = {}
    queue_count: dict[str, int] = {}
    total_halt: list[float] = []
    running: list[float] = []
    observations = 0
    zero_pressure = 0
    queue_active = 0
    action_counts: dict[int, int] = {}
    last_action: dict[str, int] = {}
    switches = 0
    switch_opportunities = 0
    for step in trajectory.get("steps", []):
        global_state = step.get("global_state", {})
        total_halt.append(float(global_state.get("total_halt", 0.0)))
        running.append(float(global_state.get("running", 0.0)))
        for junction in step.get("junctions", []):
            source_id = str(junction.get("source_id", ""))
            state = junction.get("state", {})
            action = int(junction.get("action", {}).get("green_idx", -1))
            pressures = [float(value) for value in state.get("phase_pressure", [])]
            observations += 1
            zero_pressure += int(
                not pressures or all(abs(value) < 1.0e-9 for value in pressures)
            )
            queue_active += int(float(state.get("queue", 0.0)) > 0.0)
            if action >= 0:
                action_counts[action] = action_counts.get(action, 0) + 1
                if source_id in last_action:
                    switch_opportunities += 1
                    switches += int(last_action[source_id] != action)
                last_action[source_id] = action
            queue_sum[source_id] = queue_sum.get(source_id, 0.0) + float(
                state.get("queue", 0.0)
            )
            queue_count[source_id] = queue_count.get(source_id, 0) + 1
    mean_queue = {
        source_id: total / max(queue_count.get(source_id, 1), 1)
        for source_id, total in queue_sum.items()
    }
    action_total = max(sum(action_counts.values()), 1)
    complexity = {
        "zero_pressure_fraction": zero_pressure / max(observations, 1),
        "queue_active_fraction": queue_active / max(observations, 1),
        "dominant_action_share": max(action_counts.values(), default=0) / action_total,
        "switch_rate": switches / max(switch_opportunities, 1),
        "action_counts": action_counts,
        "demand_period": _json(tile_dir / "signal_meta.json")
        .get("demand", {})
        .get("period"),
    }
    return mean_queue, total_halt, running, trajectory, complexity


def _plot_signal_map(ax, tile_dir: Path, topology: dict[str, Any], mean_queue: dict[str, float]) -> None:
    _draw_parcels(ax, topology, default="#F8FAFC", alpha=0.48)
    _draw_roads(ax, topology)
    nodes = [item for item in topology["intersections"] if str(item["id"]) in mean_queue]
    values = [mean_queue[str(item["id"])] for item in nodes]
    if nodes:
        scatter = ax.scatter(
            [item["point"][0] for item in nodes],
            [item["point"][1] for item in nodes],
            s=[34 + 16 * math.sqrt(max(value, 0.0)) for value in values],
            c=values, cmap="magma", edgecolors="white", linewidths=0.55, zorder=7,
        )
        colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.035, pad=0.01)
        colorbar.set_label("Mean queue", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    _finish_map(ax, topology, "MaxPressure: mean queue by junction")


def _plot_signal_series(
    ax,
    total_halt: list[float],
    running: list[float],
    trajectory: dict[str, Any],
    complexity: dict[str, Any],
) -> None:
    x = list(range(len(total_halt)))
    ax.plot(x, total_halt, color="#DC2626", linewidth=1.8, label="Total halted")
    ax.fill_between(x, total_halt, color="#FCA5A5", alpha=0.25)
    ax.plot(x, running, color="#2563EB", linewidth=1.3, alpha=0.85, label="Running vehicles")
    ax.set_xlabel("Decision step", fontsize=8)
    ax.set_ylabel("Vehicles", fontsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=7)
    metrics = trajectory.get("metrics", {})
    reward = sum(float(step.get("reward", 0.0)) for step in trajectory.get("steps", []))
    reward += float(trajectory.get("final_reward", 0.0))
    period = complexity.get("demand_period")
    ax.set_title(
        "Signal trajectory\n"
        f"arrived={metrics.get('arrived', 0)}, mean_halt={metrics.get('mean_total_halt', 0)}, "
        f"return={reward:.1f}" + (f", period={period:g}s" if isinstance(period, (int, float)) else ""),
        fontsize=11, fontweight="bold",
    )
    action_counts = complexity.get("action_counts", {})
    action_text = ", ".join(
        f"a{action}={count}" for action, count in sorted(action_counts.items())
    )
    ax.text(
        0.99,
        0.98,
        f"zero pressure {complexity['zero_pressure_fraction']:.1%}\n"
        f"queue active {complexity['queue_active_fraction']:.1%}\n"
        f"dominant action {complexity['dominant_action_share']:.1%}\n"
        f"switch rate {complexity['switch_rate']:.1%}\n{action_text}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=6.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#CBD5E1"},
    )


def _tile_id(tile_dir: Path, topology: dict[str, Any]) -> str:
    metadata = topology.get("metadata", {})
    value = metadata.get("tile_id") if isinstance(metadata, dict) else None
    if value:
        return str(value)
    if tile_dir.name == "abstract":
        return tile_dir.parent.name
    return tile_dir.name


def _available_components(tile_dir: Path) -> list[str]:
    components = ["topology"]
    required = {
        "land": ("land_use_solution.csv", "land_use_trajectory.json"),
        "road": ("road_plan_solution.csv", "road_plan_trajectory.json"),
        "poi": ("poi_plan_solution.csv", "poi_plan_trajectory.json"),
        "signal": ("signal_trajectory.json", "signal_meta.json"),
    }
    for name, filenames in required.items():
        if all((tile_dir / filename).is_file() for filename in filenames):
            components.append(name)
    return components


def render_tile(tile_dir: Path, *, force: bool = False, dpi: int = 160) -> dict[str, Any]:
    output_dir = tile_dir / "visualization"
    output_path = output_dir / "overview.png"
    if output_path.is_file() and not force:
        topology = _json(tile_dir / "topology.json")
        return _tile_row(tile_dir, topology, output_path, skipped=True)

    topology = _json(tile_dir / "topology.json")
    components = _available_components(tile_dir)
    panel_count = 1 + sum(
        2 if component == "signal" else 1
        for component in components
        if component != "topology"
    )
    columns = min(3, panel_count)
    rows = math.ceil(panel_count / columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(6.0 * columns, 5.75 * rows),
    )
    flat_axes = list(axes.flat)
    panel_index = 0
    _plot_topology(flat_axes[panel_index], tile_dir, topology)
    panel_index += 1
    if "land" in components:
        _plot_land(flat_axes[panel_index], tile_dir, topology)
        panel_index += 1
    if "road" in components:
        _plot_road(flat_axes[panel_index], tile_dir, topology)
        panel_index += 1
    if "poi" in components:
        _plot_poi(flat_axes[panel_index], tile_dir, topology)
        panel_index += 1
    if "signal" in components:
        mean_queue, total_halt, running, trajectory, complexity = _signal_statistics(tile_dir)
        _plot_signal_map(flat_axes[panel_index], tile_dir, topology, mean_queue)
        panel_index += 1
        _plot_signal_series(
            flat_axes[panel_index], total_halt, running, trajectory, complexity
        )
        panel_index += 1
    for axis in flat_axes[panel_index:]:
        axis.set_axis_off()
    tile_id = _tile_id(tile_dir, topology)
    signal_count = len(_signal_ids(tile_dir))
    junction_count = len(_junctions(topology))
    figure.suptitle(
        f"{tile_id}  |  parcels={len(topology['parcels'])}, roads={len(topology['roads'])}, "
        f"junctions={junction_count}, signals={signal_count}  |  "
        f"size={_topology_bounds(topology)[2] - _topology_bounds(topology)[0]:.0f} × "
        f"{_topology_bounds(topology)[3] - _topology_bounds(topology)[1]:.0f} m",
        fontsize=15, fontweight="bold", y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return _tile_row(tile_dir, topology, output_path, skipped=False)


def _tile_row(
    tile_dir: Path,
    topology: dict[str, Any],
    output_path: Path,
    *,
    skipped: bool,
) -> dict[str, Any]:
    components = _available_components(tile_dir)
    metadata = topology.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    provenance = metadata.get("dataset_provenance", {})
    city = provenance.get("city") if isinstance(provenance, dict) else None
    if not city:
        city = metadata.get("city")
    tile_id = _tile_id(tile_dir, topology)
    if not city:
        city = tile_id.split("_", 1)[0].capitalize()
    min_x, min_y, max_x, max_y = _topology_bounds(topology)
    width = float(metadata.get("tile_width_m", max_x - min_x))
    height = float(metadata.get("tile_height_m", max_y - min_y))
    signals = len(_signal_ids(tile_dir))
    mean_total_halt = None
    signal_return = None
    demand_period = None
    zero_pressure_fraction = None
    queue_active_fraction = None
    dominant_action_share = None
    switch_rate = None
    if "signal" in components:
        trajectory = _json(tile_dir / "signal_trajectory.json")
        _, _, _, _, complexity = _signal_statistics(tile_dir)
        signals = int(trajectory.get("num_signalized", signals))
        mean_total_halt = float(
            trajectory.get("metrics", {}).get("mean_total_halt", 0.0)
        )
        signal_return = sum(
            float(step.get("reward", 0.0))
            for step in trajectory.get("steps", [])
        ) + float(trajectory.get("final_reward", 0.0))
        demand_period = complexity["demand_period"]
        zero_pressure_fraction = complexity["zero_pressure_fraction"]
        queue_active_fraction = complexity["queue_active_fraction"]
        dominant_action_share = complexity["dominant_action_share"]
        switch_rate = complexity["switch_rate"]
    return {
        "tile": tile_id,
        "city": str(city),
        "components": components,
        "parcels": len(topology["parcels"]),
        "roads": len(topology["roads"]),
        "intersections": len(_junctions(topology)),
        "topology_nodes": len(topology["intersections"]),
        "tile_width_m": width,
        "tile_height_m": height,
        "signals": signals,
        "mean_total_halt": mean_total_halt,
        "signal_return": signal_return,
        "demand_period": demand_period,
        "zero_pressure_fraction": zero_pressure_fraction,
        "queue_active_fraction": queue_active_fraction,
        "dominant_action_share": dominant_action_share,
        "switch_rate": switch_rate,
        "topology": str(tile_dir / "topology.json"),
        "image": str(output_path),
        "skipped": skipped,
    }


def _render_worker(payload: tuple[str, bool, int]) -> dict[str, Any]:
    path, force, dpi = payload
    return render_tile(Path(path), force=force, dpi=dpi)


def _write_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    result: list[Path] = []
    per_sheet = 20
    columns = 4
    for start in range(0, len(rows), per_sheet):
        chunk = rows[start:start + per_sheet]
        nrows = math.ceil(len(chunk) / columns)
        figure, axes = plt.subplots(nrows, columns, figsize=(20, 3.7 * nrows))
        flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, row in zip(flat_axes, chunk):
            ax.imshow(plt.imread(row["image"]))
            ax.set_title(
                f"{row['tile']}\nP/R/I/S={row['parcels']}/{row['roads']}/{row['intersections']}/{row['signals']}",
                fontsize=7,
            )
            ax.set_axis_off()
        for ax in flat_axes[len(chunk):]:
            ax.set_axis_off()
        figure.tight_layout()
        path = output_dir / f"contact_sheet_{start // per_sheet + 1:02d}.png"
        figure.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        result.append(path)
    return result


def _selection_for_html(
    path: Path,
    tile_ids: set[str],
) -> tuple[list[dict[str, str]], int]:
    """Read the CSV subset represented by the current visualization catalog."""
    if not path.is_file():
        return [], 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        if not ({"tile_id", "tile"} & fieldnames):
            raise ValueError(f"{path}: CSV must contain tile_id")
        selected: dict[str, dict[str, str]] = {}
        unknown: set[str] = set()
        for raw in reader:
            tile_id = str(raw.get("tile_id") or raw.get("tile") or "").strip()
            if not tile_id:
                continue
            split = str(raw.get("split") or "sft").strip().lower()
            if split not in SELECTION_SPLITS:
                raise ValueError(f"{path}: invalid split for {tile_id}: {split}")
            if tile_id in tile_ids:
                selected[tile_id] = {"tile_id": tile_id, "split": split}
            else:
                unknown.add(tile_id)
    return list(selected.values()), len(unknown)


def _write_html(
    rows: list[dict[str, Any]],
    output_dir: Path,
    sheets: list[Path],
    *,
    default_split: str = "sft",
    initial_selection: list[dict[str, str]] | None = None,
    selection_source: str | None = None,
    ignored_selection_count: int = 0,
) -> Path:
    component_labels = {
        "topology": "拓扑",
        "land": "用地专家解",
        "road": "道路专家解",
        "poi": "POI 专家解",
        "signal": "信号专家轨迹",
    }
    city_counts: dict[str, int] = {}
    for row in rows:
        city = str(row["city"])
        city_counts[city] = city_counts.get(city, 0) + 1
    cards = []
    for row in rows:
        image_rel = Path(os.path.relpath(Path(row["image"]), output_dir)).as_posix()
        components = [component_labels[name] for name in row["components"]]
        details = [
            f'P/R/I/S {row["parcels"]}/{row["roads"]}/{row["intersections"]}/{row["signals"]}',
            f'{row["tile_width_m"]:.0f} × {row["tile_height_m"]:.0f} m',
            "展示：" + "、".join(components),
        ]
        if row["mean_total_halt"] is not None:
            details.extend(
                [
                    f'mean halt {row["mean_total_halt"]:.2f}',
                    f'zero pressure {row["zero_pressure_fraction"]:.0%}',
                    f'queue active {row["queue_active_fraction"]:.0%}',
                    f'period {row["demand_period"]}s',
                ]
            )
        city = str(row["city"])
        tile = str(row["tile"])
        topology_path = str(row["topology"])
        cards.append(
            f'<article class="card" id="tile-{html.escape(tile)}" '
            f'data-tile="{html.escape(tile)}" data-city="{html.escape(city)}" '
            f'data-topology="{html.escape(topology_path)}">'
            f'<a href="{html.escape(image_rel)}">'
            f'<img loading="lazy" src="{html.escape(image_rel)}" alt="{html.escape(tile)}"></a>'
            f'<div class="card-title"><h3>{html.escape(tile)}</h3>'
            f'<span class="split-badge" data-split-label>未选择</span>'
            f'<button type="button" class="select-tile">加入选择</button></div>'
            f'<p>{html.escape(city)} · {html.escape(" · ".join(details))}</p></article>'
        )
    sheet_links = " ".join(
        f'<a href="{html.escape(path.name)}">{html.escape(path.stem)}</a>' for path in sheets
    ) or "未生成"
    city_options = "".join(
        f'<option value="{html.escape(city)}">{html.escape(city)}（{count}）</option>'
        for city, count in sorted(city_counts.items())
    )
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练数据可视化</title><style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;margin:0;background:#f5f7fa;color:#17202a}
button,input,select,textarea{font:inherit}.layout{display:grid;grid-template-columns:340px minmax(0,1fr);min-height:100vh}
.sidebar{position:sticky;top:0;height:100vh;overflow:auto;padding:18px;background:#0f172a;color:#e2e8f0}
.sidebar h2{font-size:18px;margin:0 0 8px}.sidebar p{color:#94a3b8}.sidebar button{cursor:pointer}
.sidebar-actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.sidebar-actions button,.add-row button,.batch-actions button{border:0;border-radius:7px;padding:8px;background:#2563eb;color:white}
.sidebar-actions .danger{background:#b91c1c}.add-row{display:flex;gap:6px;margin:10px 0}.add-row input{min-width:0;flex:1;padding:8px;border-radius:7px;border:1px solid #475569}
.selected-list{list-style:none;padding:0;margin:12px 0;display:grid;gap:7px}.selected-list li{background:#1e293b;border-radius:8px;padding:8px}
@keyframes selection-added{0%{opacity:0;transform:translateX(-16px) scale(.96);background:#2563eb;box-shadow:0 0 0 2px #60a5fa}65%{opacity:1;transform:translateX(3px) scale(1.01)}100%{transform:none;background:#1e293b;box-shadow:none}}
.selected-list li.selection-added{animation:selection-added .42s cubic-bezier(.22,1,.36,1)}
.selected-name{display:block;font-size:12px;word-break:break-all;margin-bottom:6px}.selected-buttons{display:flex;gap:6px}
.selected-buttons button{border:0;border-radius:5px;padding:4px 7px;font-size:11px}.jump{background:#dbeafe;color:#1d4ed8}.remove{background:#fee2e2;color:#b91c1c}
.batch-editor{margin-top:12px}.batch-editor summary{cursor:pointer;color:#bfdbfe}.batch-editor textarea{width:100%;height:180px;margin-top:8px;padding:8px;border-radius:7px;font:11px ui-monospace,monospace}
.batch-actions{display:flex;gap:7px;margin-top:7px}.status{min-height:34px;color:#93c5fd;font-size:11px;margin-top:8px}
.content{min-width:0;padding:24px}.summary{background:white;padding:16px 20px;border-radius:12px;margin-bottom:18px;box-shadow:0 2px 8px #0001}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0}select{min-width:220px;padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px;background:white}
.selection-filter{display:inline-flex;align-items:center;gap:6px;padding:6px 9px;border-radius:8px;background:#eff6ff;color:#1d4ed8;cursor:pointer;user-select:none}.selection-filter input{width:16px;height:16px;margin:0;accent-color:#2563eb}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}.card{background:white;border:2px solid transparent;border-radius:12px;padding:10px;box-shadow:0 2px 8px #0001}
.card.selected{border-color:#2563eb;background:#eff6ff}.card[hidden]{display:none}.card img{width:100%;border-radius:8px}.card-title{display:flex;align-items:center;gap:8px}.card-title h3{flex:1}
.split-badge{display:none;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:700;text-transform:uppercase;white-space:nowrap}.card.selected .split-badge{display:inline-block}.split-badge[data-split="sft"]{background:#dcfce7;color:#166534}.split-badge[data-split="ppo"]{background:#fef3c7;color:#92400e}.split-badge[data-split="test"]{background:#ede9fe;color:#6d28d9}
.select-tile{border:0;border-radius:7px;padding:6px 9px;background:#dbeafe;color:#1d4ed8;cursor:pointer;white-space:nowrap}.card.selected .select-tile{background:#2563eb;color:white}
h3{font-size:13px;margin:8px 2px 4px;word-break:break-all}p{font-size:12px;color:#52606d;margin:3px 2px}a{color:#2563eb;text-decoration:none}
@media(prefers-reduced-motion:reduce){.selected-list li.selection-added{animation:none}}
@media(max-width:900px){.layout{display:block}.sidebar{position:static;height:auto;max-height:65vh}.content{padding:14px}}
</style></head><body><div class="layout"><aside class="sidebar"><h2>已选择区域（<span id="selected-count">0</span>）</h2>
<p>生成页面时指定的选择 CSV 会直接载入；通过本地选择服务打开时还可重新加载和保存。</p>
<div class="sidebar-actions"><button type="button" id="load-server">从选择服务加载</button><button type="button" id="save-server">保存到选择服务</button>
<button type="button" id="import-csv">导入 CSV</button><button type="button" id="export-csv">导出 CSV</button>
<button type="button" id="clear-selection" class="danger">清空选择</button></div>
<input type="file" id="csv-file" accept=".csv,text/csv" hidden>
<div class="add-row"><input id="add-tile-id" placeholder="输入 tile_id"><button type="button" id="add-tile">增加</button></div>
<ul class="selected-list" id="selected-list"></ul>
<details class="batch-editor"><summary>批量编辑 CSV</summary><textarea id="csv-editor" spellcheck="false"></textarea>
<div class="batch-actions"><button type="button" id="apply-csv">应用批量编辑</button></div></details><div class="status" id="selection-status"></div>
</aside><section class="content"><section class="summary"><h1>训练数据可视化</h1>
<p>共 __TOTAL__ 个最终拓扑，来自 __CITY_TOTAL__ 个城市。每张图只展示当前已有的结果；未求解拓扑仅展示拓扑本身。</p>
<div class="controls"><label for="city-filter">选择城市</label>
<select id="city-filter"><option value="">全部城市（__TOTAL__）</option>__CITY_OPTIONS__</select>
<span>当前显示 <strong id="visible-count">__TOTAL__</strong> 个</span>
<label class="selection-filter"><input type="checkbox" id="selected-only">仅显示已选择</label>
<label class="selection-filter"><input type="checkbox" class="split-filter" value="sft" checked>显示 SFT</label>
<label class="selection-filter"><input type="checkbox" class="split-filter" value="ppo" checked>显示 PPO</label>
<label class="selection-filter"><input type="checkbox" class="split-filter" value="test" checked>显示 Test</label></div>
<p>联系表：__SHEET_LINKS__</p></section><main class="grid">__CARDS__</main></section></div>
<script>
const cityFilter = document.getElementById("city-filter");
const visibleCount = document.getElementById("visible-count");
const selectedOnlyFilter = document.getElementById("selected-only");
const splitFilters = Array.from(document.querySelectorAll(".split-filter"));
const validSplits = new Set(["sft","ppo","test"]);
const cards = Array.from(document.querySelectorAll(".card"));
const selected = new Map();
const initialSelection = __INITIAL_SELECTION__;
const initialSelectionSource = __INITIAL_SELECTION_SOURCE__;
const ignoredInitialSelectionCount = __IGNORED_SELECTION_COUNT__;
const catalog = new Map(cards.map(card => [card.dataset.tile, {
  tile_id: card.dataset.tile, city: card.dataset.city, topology_path: card.dataset.topology, split: "__DEFAULT_SPLIT__"
}]));
const selectionList = document.getElementById("selected-list");
const selectedCount = document.getElementById("selected-count");
const csvEditor = document.getElementById("csv-editor");
const statusBox = document.getElementById("selection-status");
const storageKey = "upmodel.topology-selection.v2";
function setStatus(message, error=false) { statusBox.textContent = message; statusBox.style.color = error ? "#fca5a5" : "#93c5fd"; }
function csvEscape(value) { const text=String(value ?? ""); return /[",\\r\\n]/.test(text) ? `"${text.replaceAll('"','""')}"` : text; }
function selectionCsv() {
  const lines = ["tile_id,city,topology_path,split"];
  for (const row of Array.from(selected.values()).sort((a,b) => a.tile_id.localeCompare(b.tile_id))) {
    lines.push([row.tile_id,row.city,row.topology_path,row.split || "sft"].map(csvEscape).join(","));
  }
  return lines.join("\\n") + "\\n";
}
function parseCsvLine(line) {
  const cells=[]; let cell=""; let quoted=false;
  for (let index=0; index<line.length; index+=1) {
    const char=line[index];
    if (quoted && char==='"' && line[index+1]==='"') { cell+='"'; index+=1; }
    else if (char==='"') quoted=!quoted;
    else if (char===',' && !quoted) { cells.push(cell); cell=""; }
    else cell+=char;
  }
  cells.push(cell); return cells;
}
function parseSelectionCsv(text) {
  const lines=text.replace(/^\\uFEFF/,"").split(/\\r?\\n/).filter(line => line.trim());
  if (!lines.length) return [];
  const first=parseCsvLine(lines[0]).map(value => value.trim());
  const hasHeader=first.includes("tile_id") || first.includes("tile");
  const header=hasHeader ? first : ["tile_id"];
  const tileIndex=Math.max(header.indexOf("tile_id"),header.indexOf("tile"));
  const splitIndex=header.indexOf("split");
  if (tileIndex < 0) throw new Error("CSV 缺少 tile_id 列");
  return lines.slice(hasHeader ? 1 : 0).map(line => {
    const cells=parseCsvLine(line); const split=(splitIndex >= 0 ? cells[splitIndex] : "sft").trim().toLowerCase() || "sft";
    if (!validSplits.has(split)) throw new Error(`非法 split：${split}`);
    return {tile_id:(cells[tileIndex] || "").trim(),split};
  }).filter(row => row.tile_id);
}
function persistLocal() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(
      Array.from(selected.values()).map(row => ({tile_id:row.tile_id,split:row.split}))
    ));
  } catch(error) {}
}
function renderSelection(animateTileId=null) {
  selectedCount.textContent=String(selected.size); selectionList.replaceChildren();
  for (const row of Array.from(selected.values()).sort((a,b) => a.tile_id.localeCompare(b.tile_id))) {
    const item=document.createElement("li"); const name=document.createElement("span"); name.className="selected-name"; name.textContent=`[${row.split}] ${row.city} · ${row.tile_id}`;
    if (row.tile_id === animateTileId) item.classList.add("selection-added");
    const buttons=document.createElement("div"); buttons.className="selected-buttons";
    const jump=document.createElement("button"); jump.className="jump"; jump.textContent="跳转"; jump.addEventListener("click",()=>jumpToTile(row.tile_id));
    const remove=document.createElement("button"); remove.className="remove"; remove.textContent="删除"; remove.addEventListener("click",()=>toggleTile(row.tile_id,false));
    const split=document.createElement("button"); split.className="jump"; split.textContent=row.split === "test" ? "改为训练" : "改为测试"; split.addEventListener("click",()=>{row.split=row.split === "test" ? "sft" : "test"; renderSelection();});
    buttons.append(jump,split,remove); item.append(name,buttons); selectionList.append(item);
  }
  for (const card of cards) {
    const row=selected.get(card.dataset.tile); const active=Boolean(row);
    card.classList.toggle("selected",active); card.querySelector(".select-tile").textContent=active ? "已选择" : "加入选择";
    const badge=card.querySelector("[data-split-label]"); const split=active && validSplits.has(row.split) ? row.split : "";
    badge.textContent=split || "未选择"; badge.dataset.split=split; card.dataset.split=split;
  }
  csvEditor.value=selectionCsv(); persistLocal(); applyFilters();
}
function replaceSelection(rows) {
  const unknown=[]; selected.clear();
  for (const input of rows) {
    const row=catalog.get(String(input.tile_id || "").trim());
    if (row) {
      const split=String(input.split || "__DEFAULT_SPLIT__").toLowerCase();
      selected.set(row.tile_id,{...row,split:validSplits.has(split) ? split : "__DEFAULT_SPLIT__"});
    } else if (input.tile_id) unknown.push(input.tile_id);
  }
  renderSelection(); setStatus(unknown.length ? `已忽略 ${unknown.length} 个未知 tile_id` : `已载入 ${selected.size} 个拓扑`,unknown.length>0);
}
function toggleTile(tileId,force) {
  const row=catalog.get(tileId); if (!row) return;
  const wasSelected=selected.has(tileId); const add=force ?? !wasSelected;
  if (add) selected.set(tileId,{...row,split:row.split || "__DEFAULT_SPLIT__"}); else selected.delete(tileId);
  renderSelection(add && !wasSelected ? tileId : null);
}
function applyFilters() {
  const city = cityFilter.value;
  const selectedOnly = selectedOnlyFilter.checked;
  const enabledSplits = new Set(splitFilters.filter(input => input.checked).map(input => input.value));
  const restrictSplits = enabledSplits.size < splitFilters.length;
  let visible = 0;
  for (const card of cards) {
    const selectedRow = selected.get(card.dataset.tile);
    const splitMatches = !restrictSplits || (selectedRow && enabledSplits.has(selectedRow.split));
    const show = (!city || card.dataset.city === city) && (!selectedOnly || selectedRow) && splitMatches;
    card.hidden = !show;
    if (show) visible += 1;
  }
  visibleCount.textContent = String(visible);
  const url = new URL(window.location.href);
  if (city) url.searchParams.set("city", city); else url.searchParams.delete("city");
  if (selectedOnly) url.searchParams.set("selected", "1"); else url.searchParams.delete("selected");
  if (restrictSplits) url.searchParams.set("splits",Array.from(enabledSplits).join(",")); else url.searchParams.delete("splits");
  window.history.replaceState(null, "", url);
}
function jumpToTile(tileId) { const card=document.getElementById(`tile-${tileId}`); if (!card) return; cityFilter.value=card.dataset.city; applyFilters(); card.scrollIntoView({behavior:"smooth",block:"center"}); }
async function loadServerSelection(silent=false) {
  try { const response=await fetch("/api/selection",{cache:"no-store"}); if (!response.ok) throw new Error(`HTTP ${response.status}`); const payload=await response.json(); replaceSelection(payload.tiles || []); setStatus(`已从 ${payload.path} 加载 ${selected.size} 个拓扑`); return true; }
  catch (error) { if (!silent) setStatus("本地选择服务不可用；仍可导入/导出 CSV",true); return false; }
}
async function saveServerSelection() {
  try { const response=await fetch("/api/selection",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({tiles:Array.from(selected.values()).map(row=>({tile_id:row.tile_id,split:row.split}))})}); const payload=await response.json(); if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`); replaceSelection(payload.tiles || []); setStatus(`已保存 ${selected.size} 个拓扑到 ${payload.path}`); }
  catch (error) { setStatus(`保存失败：${error.message}。请通过本地选择服务打开网页，或使用导出 CSV。`,true); }
}
const initialCity = new URLSearchParams(window.location.search).get("city");
if (initialCity && Array.from(cityFilter.options).some(option => option.value === initialCity)) {
  cityFilter.value = initialCity;
}
selectedOnlyFilter.checked = new URLSearchParams(window.location.search).get("selected") === "1";
const initialSplits = new URLSearchParams(window.location.search).get("splits");
if (initialSplits !== null) {
  const enabled = new Set(initialSplits.split(",").filter(value => validSplits.has(value)));
  for (const input of splitFilters) input.checked=enabled.has(input.value);
}
cityFilter.addEventListener("change", applyFilters);
selectedOnlyFilter.addEventListener("change", applyFilters);
for (const input of splitFilters) input.addEventListener("change",applyFilters);
for (const card of cards) card.querySelector(".select-tile").addEventListener("click",()=>toggleTile(card.dataset.tile));
document.getElementById("add-tile").addEventListener("click",()=>{ const input=document.getElementById("add-tile-id"); const id=input.value.trim(); if (!catalog.has(id)) setStatus(`未知 tile_id：${id}`,true); else { toggleTile(id,true); input.value=""; setStatus(`已增加 ${id}`); } });
document.getElementById("clear-selection").addEventListener("click",()=>{ if (!selected.size || confirm("清空全部已选择拓扑？")) replaceSelection([]); });
document.getElementById("apply-csv").addEventListener("click",()=>{ try { replaceSelection(parseSelectionCsv(csvEditor.value)); } catch(error) { setStatus(error.message,true); } });
document.getElementById("import-csv").addEventListener("click",()=>document.getElementById("csv-file").click());
document.getElementById("csv-file").addEventListener("change",async event=>{ const file=event.target.files[0]; if (!file) return; try { replaceSelection(parseSelectionCsv(await file.text())); } catch(error) { setStatus(error.message,true); } event.target.value=""; });
document.getElementById("export-csv").addEventListener("click",()=>{ const url=URL.createObjectURL(new Blob([selectionCsv()],{type:"text/csv;charset=utf-8"})); const link=document.createElement("a"); link.href=url; link.download="topology_selection.csv"; link.click(); URL.revokeObjectURL(url); setStatus(`已导出 ${selected.size} 个拓扑`); });
document.getElementById("load-server").addEventListener("click",()=>loadServerSelection(false));
document.getElementById("save-server").addEventListener("click",saveServerSelection);
if (initialSelectionSource !== null) {
  replaceSelection(initialSelection);
  const ignored = ignoredInitialSelectionCount ? `，忽略 ${ignoredInitialSelectionCount} 个不在当前候选根中的条目` : "";
  setStatus(`已从 ${initialSelectionSource} 载入 ${selected.size} 个拓扑${ignored}`);
} else {
  try {
    const localRows = JSON.parse(localStorage.getItem(storageKey) || "[]") || [];
    replaceSelection(localRows.map(value => typeof value === "string" ? {tile_id:value} : value));
  } catch(error) {
    replaceSelection([]);
  }
}
applyFilters();
loadServerSelection(true);
</script></body></html>"""
    selection_json = json.dumps(initial_selection or [], ensure_ascii=False).replace("</", "<\\/")
    source_json = json.dumps(selection_source, ensure_ascii=False).replace("</", "<\\/")
    document = (
        template.replace("__TOTAL__", str(len(rows)))
        .replace("__CITY_TOTAL__", str(len(city_counts)))
        .replace("__CITY_OPTIONS__", city_options)
        .replace("__SHEET_LINKS__", sheet_links)
        .replace("__CARDS__", "".join(cards))
        .replace("__DEFAULT_SPLIT__", default_split)
        .replace("__INITIAL_SELECTION__", selection_json)
        .replace("__INITIAL_SELECTION_SOURCE__", source_json)
        .replace("__IGNORED_SELECTION_COUNT__", str(ignored_selection_count))
    )
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


ROOT_INDEX_MARKER = "<!-- generated by visualize-training-data -->"


def _write_root_index(data_root: Path, visualization_index: Path) -> Path | None:
    """Write a safe server-root entry point when the visualization lives below it."""
    try:
        relative = visualization_index.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return None
    root_index = data_root / "index.html"
    if root_index.resolve() == visualization_index.resolve():
        return root_index
    if root_index.is_file() and ROOT_INDEX_MARKER not in root_index.read_text(
        encoding="utf-8", errors="replace"
    ):
        print(
            f"[visualize-training-data] keep existing non-generated root index: {root_index}",
            flush=True,
        )
        return None
    target = html.escape(relative, quote=True)
    root_index.write_text(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<title>训练数据可视化</title></head>
<body>{ROOT_INDEX_MARKER}<p>正在跳转到 <a href="{target}">{target}</a>…</p></body></html>
""",
        encoding="utf-8",
    )
    return root_index


def _discover_tile_dirs(data_root: Path) -> list[Path]:
    """Find all canonical candidates while excluding intermediate evidence."""
    result: list[Path] = []
    # ``Path.rglob`` does not descend into directory symlinks.  train/test are
    # intentionally direct symlink collections, so inspect their children
    # explicitly before handling nested real candidate directories.
    if data_root.is_dir():
        for child in data_root.iterdir():
            if child.name == "visualizations" or not child.is_dir():
                continue
            for relative in (Path("topology.json"), Path("abstract/topology.json")):
                topology_path = child / relative
                if topology_path.is_file():
                    result.append(topology_path.parent)
                    break
    for topology_path in data_root.rglob("topology.json"):
        tile_dir = topology_path.parent
        if tile_dir.name == "evidence":
            continue
        if tile_dir.name == "abstract" or tile_dir.parent == data_root or tile_dir == data_root:
            result.append(tile_dir)
    return sorted(set(result), key=lambda path: path.as_posix())


def _selection_csv_path(args: argparse.Namespace, candidate_root: Path) -> Path:
    raw_selection = getattr(args, "selection_csv", None)
    if raw_selection is not None:
        return Path(raw_selection)
    return candidate_root.parent / "topology_selection.csv"


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    tile_dirs = _discover_tile_dirs(data_root)
    if args.limit is not None:
        tile_dirs = tile_dirs[: max(int(args.limit), 0)]
    if not tile_dirs:
        raise SystemExit(f"no topology.json found below {data_root}")
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    workers = max(int(args.workers), 1)
    payloads = [(str(path), bool(args.force), int(args.dpi)) for path in tile_dirs]
    print(
        f"[visualize-training-data] discovered={len(payloads)} workers={workers} "
        f"force={bool(args.force)}",
        flush=True,
    )
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            try:
                row = _render_worker(payload)
                rows.append(row)
                print(f"[{index}/{len(payloads)}] OK {row['tile']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append((payload[0], f"{type(exc).__name__}: {exc}"))
                print(f"[{index}/{len(payloads)}] FAIL {payload[0]}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_render_worker, payload): payload[0] for payload in payloads}
            for index, future in enumerate(as_completed(future_map), 1):
                source = future_map[future]
                try:
                    row = future.result()
                    rows.append(row)
                    print(f"[{index}/{len(payloads)}] OK {row['tile']}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failures.append((source, f"{type(exc).__name__}: {exc}"))
                    print(f"[{index}/{len(payloads)}] FAIL {source}: {exc}", flush=True)

    rows.sort(key=lambda row: (row["city"], row["tile"]))
    sheets = [] if args.skip_contact_sheets else _write_contact_sheets(rows, output_dir)
    root_split = data_root.name.lower()
    default_split = root_split if root_split in SELECTION_SPLITS else "sft"
    selection_csv = _selection_csv_path(args, data_root)
    selection, ignored_selection_count = _selection_for_html(
        selection_csv, {str(row["tile"]) for row in rows}
    )
    selection_source = (
        os.path.relpath(selection_csv, Path.cwd()) if selection_csv.is_file() else None
    )
    if selection_source is None:
        print(
            f"[visualize-training-data] selection CSV not found; page starts without CSV selection: "
            f"{selection_csv}",
            flush=True,
        )
    else:
        print(
            f"[visualize-training-data] selection={len(selection)} "
            f"ignored={ignored_selection_count} csv={selection_source}",
            flush=True,
        )
    index_path = _write_html(
        rows,
        output_dir,
        sheets,
        default_split=default_split,
        initial_selection=selection,
        selection_source=selection_source,
        ignored_selection_count=ignored_selection_count,
    )
    root_index = _write_root_index(data_root, index_path)
    city_counts: dict[str, int] = {}
    component_counts: dict[str, int] = {}
    for row in rows:
        city = str(row["city"])
        city_counts[city] = city_counts.get(city, 0) + 1
        for component in row["components"]:
            component_counts[component] = component_counts.get(component, 0) + 1
    manifest = {
        "count": len(rows),
        "city_counts": city_counts,
        "component_counts": component_counts,
        "failures": [{"tile": tile, "reason": reason} for tile, reason in failures],
        "contact_sheets": [path.name for path in sheets],
        "selection_csv": selection_source,
        "selection_count": len(selection),
        "ignored_selection_count": ignored_selection_count,
        "root_index": str(root_index) if root_index is not None else None,
        "tiles": rows,
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[visualize-training-data] rendered={len(rows)} failures={len(failures)} "
        f"index={index_path}"
    )
    if failures:
        raise SystemExit(1)


SELECTION_CSV_FIELDS = ("tile_id", "city", "topology_path", "split")


def _selection_catalog(data_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = data_root / "visualizations" / "visualization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"visualization manifest not found: {manifest_path}; run visualize-training-data first"
        )
    manifest = _json(manifest_path)
    catalog: dict[str, dict[str, str]] = {}
    for raw in manifest.get("tiles", []):
        if not isinstance(raw, dict) or not raw.get("tile"):
            continue
        tile_id = str(raw["tile"])
        abstract = data_root / tile_id / "abstract" / "topology.json"
        root_topology = data_root / tile_id / "topology.json"
        topology_path = abstract if abstract.is_file() else root_topology
        if not topology_path.is_file():
            continue
        catalog[tile_id] = {
            "tile_id": tile_id,
            "city": str(raw.get("city") or tile_id.split("_", 1)[0].capitalize()),
            "topology_path": str(topology_path.resolve()),
        }
    return catalog


def _selection_rows(
    tile_ids: list[str],
    catalog: dict[str, dict[str, str]],
    *,
    relative_to: Path,
    split_by_id: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for raw_tile_id in tile_ids:
        tile_id = str(raw_tile_id).strip()
        if not tile_id or tile_id in seen:
            continue
        seen.add(tile_id)
        item = catalog.get(tile_id)
        if item is None:
            unknown.append(tile_id)
            continue
        topology_path = Path(item["topology_path"])
        rows.append(
            {
                "tile_id": tile_id,
                "city": item["city"],
                "topology_path": os.path.relpath(topology_path, relative_to.resolve()),
                "split": (split_by_id or {}).get(tile_id, "sft"),
            }
        )
    if unknown:
        examples = ", ".join(unknown[:5])
        raise ValueError(f"unknown tile_id values ({len(unknown)}): {examples}")
    return rows


def _read_selection_csv(
    path: Path,
    catalog: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not ({"tile_id", "tile"} & set(reader.fieldnames)):
            raise ValueError(f"{path}: CSV must contain tile_id")
        raw_rows = list(reader)
        tile_ids = [str(row.get("tile_id") or row.get("tile") or "") for row in raw_rows]
        split_by_id = {
            str(row.get("tile_id") or row.get("tile") or ""): str(row.get("split") or "sft").lower()
            for row in raw_rows
        }
    invalid = sorted(set(split_by_id.values()) - set(SELECTION_SPLITS))
    if invalid:
        raise ValueError(f"{path}: invalid split values: {invalid}")
    return _selection_rows(
        tile_ids, catalog, relative_to=path.parent, split_by_id=split_by_id
    )


def _write_selection_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def serve_selection(args: argparse.Namespace) -> None:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    from scripts.prepare_training_data import (
        _read_selection_history,
        _selection_history_path,
        _write_selection_history,
    )

    data_root = Path(args.data_root).resolve()
    selection_csv = _selection_csv_path(args, Path(args.data_root)).resolve()
    selection_history = _selection_history_path(selection_csv)
    index_path = data_root / "visualizations" / "index.html"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"visualization index not found: {index_path}; run visualize-training-data first"
        )
    catalog = _selection_catalog(data_root)

    class SelectionHandler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(data_root), **handler_kwargs)

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _selection_payload(self) -> dict[str, Any]:
            rows = _read_selection_csv(selection_csv, catalog)
            return {
                "tiles": rows,
                "count": len(rows),
                "path": os.path.relpath(selection_csv, Path.cwd()),
            }

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path == "/api/selection":
                try:
                    self._json_response(200, self._selection_payload())
                except Exception as exc:  # noqa: BLE001
                    self._json_response(400, {"error": str(exc)})
                return
            super().do_GET()

        def do_PUT(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/api/selection":
                self._json_response(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5_000_000:
                    raise ValueError("invalid request body size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                raw_tiles = payload.get("tiles") if isinstance(payload, dict) else None
                if raw_tiles is None and isinstance(payload, dict):
                    raw_tiles = [
                        {"tile_id": value, "split": "sft"}
                        for value in payload.get("tile_ids", [])
                    ]
                if not isinstance(raw_tiles, list):
                    raise ValueError("request must contain tiles list")
                tile_ids = [str(row.get("tile_id", "")) for row in raw_tiles if isinstance(row, dict)]
                split_by_id = {
                    str(row.get("tile_id", "")): str(row.get("split") or "sft").lower()
                    for row in raw_tiles
                    if isinstance(row, dict)
                }
                invalid = sorted(set(split_by_id.values()) - set(SELECTION_SPLITS))
                if invalid:
                    raise ValueError(f"invalid split values: {invalid}")
                previous_rows = _read_selection_csv(selection_csv, catalog)
                rows = _selection_rows(
                    [str(value) for value in tile_ids],
                    catalog,
                    relative_to=selection_csv.parent,
                    split_by_id=split_by_id,
                )
                history = _read_selection_history(selection_history)
                history.update(row["tile_id"] for row in previous_rows)
                history.update(row["tile_id"] for row in rows)
                _write_selection_history(
                    selection_history,
                    history,
                    data_root=data_root,
                )
                _write_selection_csv(selection_csv, rows)
                self._json_response(200, self._selection_payload())
            except Exception as exc:  # noqa: BLE001
                self._json_response(400, {"error": str(exc)})

        def log_message(self, message: str, *values: Any) -> None:
            print(f"[selection-server] {self.address_string()} {message % values}", flush=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), SelectionHandler)
    url = f"http://{args.host}:{args.port}/"
    print(
        f"[selection-server] url={url} catalog={len(catalog)} csv={selection_csv}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data/final"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "生成 HTML 时直接嵌入的 sft/ppo/test 选择表；默认 "
            "<data-root>/../topology_selection.csv，不存在时不预选"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
        help="只生成单拓扑图片、manifest 和 HTML，不额外拼接联系表",
    )


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize-training-data",
        help="绘制候选根中的全部 topology.json 与已有专家结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    parser.set_defaults(func=run)


def add_selection_server_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve-topology-selection",
        help="提供可直接读写选择 CSV 的本地拓扑可视化网页",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/final"))
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help="默认 <data-root>/../topology_selection.csv",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.set_defaults(func=serve_selection)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all candidate topologies and available expert artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
