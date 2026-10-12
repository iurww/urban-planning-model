"""Build fixed-what pointer and signal-action supervision from expert artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from tqdm.auto import tqdm

from upmodel.data.signal import encode_signal_observation
from upmodel.envs.static import StaticPlanningEnv
from upmodel.planning import load_static_expert_plan
from upmodel.state import PolicyState, UrbanGraph
from upmodel.tasks import TRAINING_TASKS, Task


@dataclass(frozen=True)
class ExpertStep:
    state: PolicyState
    action: int | torch.Tensor
    return_to_go: float
    return_scale: float
    target_mask: torch.Tensor | None = None


class ExpertDataset:
    """Eager, validated expert steps; topology tensors remain shared by reference."""

    def __init__(
        self,
        graphs: list[UrbanGraph],
        *,
        tasks: Iterable[Task] = TRAINING_TASKS,
        land_use_order: Iterable[str] | None = None,
        signal_gamma: float = 0.99,
        signal_history_steps: int = 4,
        road_permutations: int = 1,
        seed: int = 0,
        show_progress: bool = False,
    ) -> None:
        self.graphs = graphs
        self.tasks = tuple(dict.fromkeys(Task(task) for task in tasks))
        if not self.tasks:
            raise ValueError("expert dataset tasks cannot be empty")
        unsupported = set(self.tasks) - set(TRAINING_TASKS)
        if unsupported:
            raise ValueError(
                f"fixed-what training does not support tasks: "
                f"{sorted(task.label for task in unsupported)}"
            )
        if signal_history_steps < 0:
            raise ValueError("signal_history_steps cannot be negative")
        if road_permutations < 1:
            raise ValueError("road_permutations must be positive")
        generator = random.Random(seed)
        self.steps: list[ExpertStep] = []
        for graph in tqdm(
            graphs,
            desc="加载专家轨迹",
            unit="个",
            disable=not show_progress,
        ):
            if Task.LAND in self.tasks:
                self.steps.extend(
                    _load_static_steps(
                        graph,
                        Task.LAND,
                        land_use_order=land_use_order,
                    )
                )
            if Task.ROAD in self.tasks:
                expert_order = list(
                    load_static_expert_plan(graph, Task.ROAD).where
                )
                for permutation_index in range(road_permutations):
                    order = list(expert_order)
                    if permutation_index:
                        generator.shuffle(order)
                    self.steps.extend(
                        _load_static_steps(
                            graph,
                            Task.ROAD,
                            action_order=order,
                        )
                    )
            if Task.SIGNAL in self.tasks:
                self.steps.extend(
                    _load_signal_steps(
                        graph,
                        gamma=signal_gamma,
                        history_steps=signal_history_steps,
                    )
                )
        if not self.steps:
            raise ValueError("expert dataset is empty")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> ExpertStep:
        return self.steps[index]

    def by_task(self, task: Task) -> Iterator[ExpertStep]:
        return (step for step in self.steps if step.state.task == task)

    def summary(self) -> dict[str, int]:
        return {
            task.label: sum(step.state.task == task for step in self.steps)
            for task in self.tasks
        }

def _load_static_steps(
    graph: UrbanGraph,
    task: Task,
    *,
    land_use_order: Iterable[str] | None = None,
    action_order: Iterable[int] | None = None,
) -> list[ExpertStep]:
    plan = load_static_expert_plan(
        graph, task, land_use_order=land_use_order
    )
    trajectory_name = {
        Task.LAND: "land_use_trajectory.json",
        Task.ROAD: "road_plan_trajectory.json",
    }[task]
    trajectory = _json(graph.tile_dir / trajectory_name)
    final_reward = float(trajectory["final_reward"])
    ordered_actions = (
        tuple(int(action) for action in action_order)
        if action_order is not None
        else plan.where
    )
    if set(ordered_actions) != set(plan.where) or len(ordered_actions) != len(
        plan.where
    ):
        raise ValueError("static expert action order must be a permutation")
    env = StaticPlanningEnv(graph, task, what_schedule=plan.what)
    state = env.reset()
    result: list[ExpertStep] = []
    remaining_targets = set(ordered_actions)
    for step_index, action in enumerate(ordered_actions):
        action = int(action)
        if not bool(state.action_mask[action].item()):
            raise ValueError(
                f"{graph.tile_id}: expert {task.label} entity {action} is masked "
                f"at step {env.position}"
            )
        if state.query_action != plan.what[step_index]:
            raise AssertionError("static environment what schedule drifted")
        result.append(
            ExpertStep(
                state=state.snapshot(),
                action=action,
                # Static environments only emit a terminal reward. With
                # gamma=1, the true return-to-go is constant along the episode.
                return_to_go=float(final_reward),
                return_scale=max(abs(float(final_reward)), 1.0e-8),
                target_mask=(
                    torch.tensor(
                        [
                            candidate in remaining_targets
                            for candidate in range(len(state.action_mask))
                        ],
                        dtype=torch.bool,
                    )
                    if task == Task.ROAD
                    else None
                ),
            )
        )
        remaining_targets.remove(action)
        next_state, _, done, _ = env.step(action)
        if not done:
            assert next_state is not None
            state = next_state
    return result


def _load_signal_steps(
    graph: UrbanGraph, *, gamma: float, history_steps: int
) -> list[ExpertStep]:
    trajectory = _json(graph.tile_dir / "signal_trajectory.json")
    if graph.signal_meta is None or graph.signal_entities.numel() == 0:
        raise ValueError(f"{graph.tile_id}: signal trajectory exists but signal metadata is missing")
    raw_steps = trajectory.get("steps", [])
    rewards = [float(step.get("reward", 0.0)) for step in raw_steps]
    if rewards:
        rewards[-1] += float(trajectory.get("final_reward", 0.0))
    returns = [0.0] * len(rewards)
    return_scale = max(
        abs(sum(float(step.get("reward", 0.0)) for step in raw_steps)
            + float(trajectory.get("final_reward", 0.0))),
        1.0e-8,
    )
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        accumulator = rewards[index] + gamma * accumulator
        returns[index] = accumulator
    meta_sources = [str(item["source_id"]) for item in graph.signal_meta["junctions"]]
    result: list[ExpertStep] = []
    history_dynamic: list[torch.Tensor] = []
    history_actions: list[torch.Tensor] = []
    history_progress: list[float] = []
    for index, raw in enumerate(raw_steps):
        junctions = raw.get("junctions", [])
        by_source = {str(item.get("source_id")): item for item in junctions}
        if set(by_source) != set(meta_sources):
            raise ValueError(f"{graph.tile_id}: signal step {index + 1} junction set is inconsistent")
        history, dynamic = encode_signal_observation(
            graph, junctions, raw.get("global_state", {})
        )
        actions = torch.tensor(
            [
                int(by_source[source]["action"]["action_id"])
                if "action_id" in by_source[source]["action"]
                else int(by_source[source]["action"]["green_idx"])
                for source in meta_sources
            ],
            dtype=torch.long,
        )
        legal = graph.signal_action_mask.gather(1, actions[:, None]).all()
        if not bool(legal.item()):
            raise ValueError(f"{graph.tile_id}: signal step {index + 1} contains an illegal action")
        state = PolicyState(
            graph=graph,
            task=Task.SIGNAL,
            progress=index / max(len(raw_steps), 1),
            history=history,
            dynamic=dynamic,
            action_mask=graph.signal_action_mask,
            signal_entities=graph.signal_entities,
            signal_history_dynamic=(
                torch.stack(history_dynamic[-history_steps:])
                if history_steps and history_dynamic
                else torch.empty(
                    (0, len(graph.signal_entities), dynamic.shape[-1]),
                    dtype=torch.float32,
                )
            ),
            signal_history_actions=(
                torch.stack(history_actions[-history_steps:])
                if history_steps and history_actions
                else torch.empty(
                    (0, len(graph.signal_entities)), dtype=torch.long
                )
            ),
            signal_history_progress=torch.tensor(
                history_progress[-history_steps:] if history_steps else [],
                dtype=torch.float32,
            ),
        )
        result.append(
            ExpertStep(
                state=state,
                action=actions,
                return_to_go=returns[index],
                return_scale=return_scale,
            )
        )
        history_dynamic.append(dynamic[graph.signal_entities].clone())
        history_actions.append(actions.clone())
        history_progress.append(index / max(len(raw_steps), 1))
    return result


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required expert file is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value
