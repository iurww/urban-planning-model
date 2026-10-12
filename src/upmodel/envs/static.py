"""Fixed-what, learned-where environments for static planning tasks."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from upmodel.envs.rewards import LandReward, RewardResult, RoadReward
from upmodel.planning import load_static_expert_plan
from upmodel.state import PolicyState, SIGNAL_DYNAMIC_DIM, UrbanGraph
from upmodel.tasks import Task


class StaticPlanningEnv:
    """Select one unassigned entity for each externally supplied what token."""

    def __init__(
        self,
        graph: UrbanGraph,
        task: Task,
        *,
        what_schedule: Iterable[int] | None = None,
        land_use_order: Iterable[str] | None = None,
    ) -> None:
        if task not in {Task.LAND, Task.ROAD}:
            raise ValueError("StaticPlanningEnv supports only land and road")
        self.graph = graph
        self.task = task
        self.entities = (
            graph.parcel_entities if task == Task.LAND else graph.road_entities
        )
        if what_schedule is None:
            plan = load_static_expert_plan(
                graph, task, land_use_order=land_use_order
            )
            what_schedule = plan.what
        self.what_schedule = [int(value) for value in what_schedule]
        if not self.what_schedule:
            raise ValueError(f"{graph.tile_id}: {task.label} what schedule is empty")
        if task == Task.LAND:
            if len(self.what_schedule) != len(self.entities):
                raise ValueError("land what schedule must assign every parcel")
            if any(value < 0 or value >= 6 for value in self.what_schedule):
                raise ValueError("land what schedule contains an unknown land-use id")
            self.evaluator = LandReward(graph)
            self.default_action = -1
        else:
            if len(self.what_schedule) > len(self.entities):
                raise ValueError("road what schedule deletes more roads than exist")
            if any(value != 0 for value in self.what_schedule):
                raise ValueError("road what schedule can only contain the delete action")
            self.evaluator = RoadReward(graph)
            self.default_action = 1
        self.history = torch.zeros(graph.num_entities, dtype=torch.long)
        self.actions = [self.default_action] * len(self.entities)
        self.selected = torch.zeros(len(self.entities), dtype=torch.bool)
        self.decision_entities: list[int] = []
        self.decision_actions: list[int] = []
        self.position = 0
        self._done = False

    def reset(self) -> PolicyState:
        self.history.zero_()
        self.actions = [self.default_action] * len(self.entities)
        self.selected.zero_()
        self.decision_entities = []
        self.decision_actions = []
        self.position = 0
        self._done = False
        return self._state()

    def step(self, action: int) -> tuple[PolicyState | None, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called after episode termination")
        mask = self._action_mask()
        if action < 0 or action >= len(mask) or not bool(mask[action].item()):
            raise ValueError(
                f"illegal {self.task.label} entity {action} at step {self.position}"
            )
        entity = int(self.entities[action].item())
        what = self.what_schedule[self.position]
        self.actions[action] = what
        self.selected[action] = True
        self.history[entity] = what + 1
        self.decision_entities.append(entity)
        self.decision_actions.append(what)
        self.position += 1
        self._done = self.position == len(self.what_schedule)
        if self._done:
            result: RewardResult = self.evaluator.evaluate(self.actions)
            return None, result.reward, True, dict(result.metrics)
        if self.task == Task.ROAD and not bool(self._action_mask().any()):
            self._done = True
            result = self.evaluator.evaluate(self.actions)
            metrics = dict(result.metrics)
            metrics.update({"feasible": False, "stuck_before_budget": True})
            return None, 0.0, True, metrics
        return self._state(), 0.0, False, {}

    def _state(self) -> PolicyState:
        return PolicyState(
            graph=self.graph,
            task=self.task,
            progress=self.position / max(len(self.what_schedule), 1),
            history=self.history.clone(),
            dynamic=torch.zeros(
                (self.graph.num_entities, SIGNAL_DYNAMIC_DIM), dtype=torch.float32
            ),
            action_mask=self._action_mask(),
            candidate_entities=self.entities,
            query_action=self.what_schedule[self.position],
            decision_entities=torch.tensor(
                self.decision_entities, dtype=torch.long
            ),
            decision_actions=torch.tensor(
                self.decision_actions, dtype=torch.long
            ),
        )

    def _action_mask(self) -> torch.Tensor:
        if self.task == Task.ROAD:
            return self.evaluator.deletion_mask(
                self.actions,
                already_decided=self.selected,
            )
        return ~self.selected.clone()
