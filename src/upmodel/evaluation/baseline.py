"""Expert reward baselines in exactly the same objective used by rollout environments."""

from __future__ import annotations

import json

from upmodel.state import UrbanGraph
from upmodel.tasks import Task


def expert_baseline(graph: UrbanGraph, task: Task) -> tuple[float, dict[str, float]]:
    filename = {
        Task.LAND: "land_use_trajectory.json",
        Task.ROAD: "road_plan_trajectory.json",
        Task.POI: "poi_plan_trajectory.json",
        Task.SIGNAL: "signal_trajectory.json",
    }[task]
    document = json.loads((graph.tile_dir / filename).read_text(encoding="utf-8"))
    if task == Task.SIGNAL:
        step_reward = sum(float(step.get("reward", 0.0)) for step in document["steps"])
        terminal_reward = float(document.get("final_reward", 0.0))
        metrics = {
            "step_reward": step_reward,
            "terminal_reward": terminal_reward,
            **{
                str(key): float(value)
                for key, value in document.get("metrics", {}).items()
                if isinstance(value, (int, float))
            },
        }
        return step_reward + terminal_reward, metrics
    metrics = {
        str(key): float(value)
        for key, value in document.get("metrics", {}).items()
        if isinstance(value, (int, float))
    }
    return float(document["final_reward"]), metrics
