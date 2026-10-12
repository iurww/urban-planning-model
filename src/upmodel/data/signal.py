"""Convert recorded or live signal observations to fixed-size model features."""

from __future__ import annotations

import math
from typing import Any

import torch

from upmodel.state import SIGNAL_DYNAMIC_DIM, UrbanGraph


def encode_signal_observation(
    graph: UrbanGraph,
    junction_records: list[dict[str, Any]],
    global_state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return entity history codes and dynamic features for one decision point."""
    if graph.signal_meta is None:
        raise ValueError(f"{graph.tile_id}: signal metadata is unavailable")
    by_source = {str(item.get("source_id")): item for item in junction_records}
    history = torch.zeros(graph.num_entities, dtype=torch.long)
    dynamic = torch.zeros((graph.num_entities, SIGNAL_DYNAMIC_DIM), dtype=torch.float32)
    global_state = global_state or {}
    running = math.log1p(max(float(global_state.get("running", 0.0)), 0.0)) / 8.0
    total_halt = math.log1p(max(float(global_state.get("total_halt", 0.0)), 0.0)) / 6.0
    arrived = math.log1p(max(float(global_state.get("arrived_cum", 0.0)), 0.0)) / 8.0

    for local_index, (meta, entity_tensor) in enumerate(
        zip(graph.signal_meta.get("junctions", []), graph.signal_entities)
    ):
        source_id = str(meta.get("source_id"))
        try:
            record = by_source[source_id]
        except KeyError as exc:
            raise ValueError(
                f"{graph.tile_id}: observation is missing signal junction {source_id!r}"
            ) from exc
        entity = int(entity_tensor.item())
        state = record.get("state", {})
        green_phases = [int(value) for value in meta.get("green_phases", [])]
        current_phase = int(state.get("cur_phase", 0))
        canonical = str(graph.signal_meta.get("phase_representation", "")).startswith(
            "canonical_action_id"
        )
        if canonical:
            current_green = current_phase if current_phase in green_phases else -1
        else:
            try:
                current_green = green_phases.index(current_phase)
            except ValueError:
                current_green = -1
        history[entity] = current_green + 1
        row = dynamic[entity]
        row[0] = max(current_green, 0) / (7.0 if canonical else max(len(green_phases) - 1, 1))
        delta_t = float(graph.signal_meta.get("control", {}).get("delta_t", 10.0))
        row[1] = min(max(float(state.get("phase_elapsed", 0.0)) / max(delta_t, 1.0), 0.0), 5.0)
        row[2] = math.log1p(max(float(state.get("queue", 0.0)), 0.0)) / 5.0
        row[3] = math.log1p(max(float(state.get("wait", 0.0)), 0.0)) / 10.0
        row[4] = math.log1p(max(float(state.get("veh", 0.0)), 0.0)) / 5.0
        for phase_index, pressure in enumerate(state.get("phase_pressure", [])[:8]):
            row[5 + phase_index] = math.tanh(float(pressure) / 10.0)
        row[13] = running
        row[14] = total_halt
        row[15] = arrived
        row[16] = local_index / max(len(graph.signal_entities) - 1, 1)
        row[17] = 1.0
    return history, dynamic
