"""Compile expert artifacts into fixed-what, learned-where static plans."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from upmodel.state import UrbanGraph
from upmodel.tasks import LAND_ACTIONS, LAND_TO_INDEX, Task


DEFAULT_LAND_USE_ORDER = ("off", "hos", "sch", "com", "res", "grn")


@dataclass(frozen=True)
class StaticExpertPlan:
    """A fixed what schedule and the expert entity selected at every step."""

    what: tuple[int, ...]
    where: tuple[int, ...]


def infer_land_use_order(graphs: Iterable[UrbanGraph]) -> tuple[str, ...]:
    """Order land-use blocks by aggregate training-set frequency."""
    counts = Counter({label: 0 for label in LAND_ACTIONS})
    for graph in graphs:
        for row in _csv_rows(graph.tile_dir / "land_use_solution.csv"):
            label = _required(row, "land_use", graph.tile_id)
            if label not in LAND_TO_INDEX:
                raise ValueError(f"{graph.tile_id}: unknown land use {label!r}")
            counts[label] += 1
    return tuple(
        sorted(LAND_ACTIONS, key=lambda label: (counts[label], LAND_TO_INDEX[label]))
    )


def load_land_area_targets(graph: UrbanGraph) -> tuple[float, ...]:
    """Return expert aggregate area shares as the external land demand."""
    rows = _csv_rows(graph.tile_dir / "land_use_solution.csv")
    parcel_count = len(graph.parcel_entities)
    if len(rows) != parcel_count:
        raise ValueError(
            f"{graph.tile_id}: land solution has {len(rows)} rows, "
            f"expected {parcel_count}"
        )
    assignment = {
        int(_required(row, "parcel_id", graph.tile_id)): LAND_TO_INDEX[
            _required(row, "land_use", graph.tile_id)
        ]
        for row in rows
    }
    if set(assignment) != set(range(parcel_count)):
        raise ValueError(
            f"{graph.tile_id}: land solution parcel ids must be 0.."
            f"{parcel_count - 1}"
        )
    areas = [
        float(parcel["area_m2"])
        for parcel in graph.raw_topology["parcels"]
    ]
    total_area = sum(areas)
    if total_area <= 0.0:
        raise ValueError(f"{graph.tile_id}: total parcel area must be positive")
    return tuple(
        sum(
            area
            for parcel_index, area in enumerate(areas)
            if assignment[parcel_index] == action
        )
        / total_area
        for action in range(len(LAND_ACTIONS))
    )


def load_static_expert_plan(
    graph: UrbanGraph,
    task: Task,
    *,
    land_use_order: Iterable[str] | None = None,
) -> StaticExpertPlan:
    if task == Task.LAND:
        return _land_plan(graph, tuple(land_use_order or DEFAULT_LAND_USE_ORDER))
    if task == Task.ROAD:
        return _road_plan(graph)
    raise ValueError(f"fixed-what static planning does not support {task.label}")


def _land_plan(graph: UrbanGraph, land_use_order: tuple[str, ...]) -> StaticExpertPlan:
    if set(land_use_order) != set(LAND_ACTIONS) or len(land_use_order) != len(
        LAND_ACTIONS
    ):
        raise ValueError("land_use_order must contain every land action exactly once")
    rows = _csv_rows(graph.tile_dir / "land_use_solution.csv")
    count = len(graph.parcel_entities)
    if len(rows) != count:
        raise ValueError(
            f"{graph.tile_id}: land solution has {len(rows)} rows, expected {count}"
        )
    assignment = {
        int(_required(row, "parcel_id", graph.tile_id)): LAND_TO_INDEX[
            _required(row, "land_use", graph.tile_id)
        ]
        for row in rows
    }
    if set(assignment) != set(range(count)):
        raise ValueError(f"{graph.tile_id}: land solution parcel ids must be 0..{count - 1}")

    trajectory = _json(graph.tile_dir / "land_use_trajectory.json")
    source_order = [
        int(step["action"]["parcel_id"]) for step in trajectory.get("steps", [])
    ]
    if sorted(source_order) != list(range(count)):
        raise ValueError(
            f"{graph.tile_id}: land trajectory does not cover every parcel exactly once"
        )
    where: list[int] = []
    what: list[int] = []
    for label in land_use_order:
        action = LAND_TO_INDEX[label]
        selected = [index for index in source_order if assignment[index] == action]
        where.extend(selected)
        what.extend([action] * len(selected))
    if len(where) != count:
        raise AssertionError("land plan did not schedule every parcel")
    return StaticExpertPlan(what=tuple(what), where=tuple(where))


def _road_plan(graph: UrbanGraph) -> StaticExpertPlan:
    rows = _csv_rows(graph.tile_dir / "road_plan_solution.csv")
    count = len(graph.road_entities)
    if len(rows) != count:
        raise ValueError(
            f"{graph.tile_id}: road solution has {len(rows)} rows, expected {count}"
        )
    row_indices = [
        int(_required(row, "road_row_index", graph.tile_id)) for row in rows
    ]
    if set(row_indices) != set(range(count)):
        raise ValueError(f"{graph.tile_id}: road solution indices are not contiguous")
    selected = {
        row_index
        for row_index, row in zip(row_indices, rows, strict=True)
        if int(_required(row, "upgraded", graph.tile_id)) == 1
    }
    if not selected or any(index < 0 or index >= count for index in selected):
        raise ValueError(f"{graph.tile_id}: road expert must select valid roads")
    deleted = tuple(sorted(set(range(count)) - selected))
    if not deleted:
        raise ValueError(f"{graph.tile_id}: road expert deletes no roads")
    # The learned action removes the small complement of a dense backbone.
    # This is the same final binary plan with a much shorter horizon.
    return StaticExpertPlan(what=(0,) * len(deleted), where=deleted)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError:
        raise FileNotFoundError(f"required expert file is missing: {path}") from None
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required expert file is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _required(row: dict[str, str], key: str, tile_id: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"{tile_id}: expert CSV field {key!r} is empty")
    return value
