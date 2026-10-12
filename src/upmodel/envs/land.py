"""Area-constrained fixed-order inference for land-use planning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import gurobipy as gp
import torch

from upmodel.envs.rewards import (
    LAND_QUOTA_LOWER,
    LAND_QUOTA_UPPER,
    LandReward,
    RewardResult,
)
from upmodel.planning import DEFAULT_LAND_USE_ORDER, load_land_area_targets
from upmodel.state import PolicyState, SIGNAL_DYNAMIC_DIM, UrbanGraph
from upmodel.tasks import LAND_ACTIONS, LAND_TO_INDEX, Task


_GUROBI_ENV: gp.Env | None = None


class LandAreaPlanningEnv:
    """Select parcels by target area while preserving a legal completion."""

    def __init__(
        self,
        graph: UrbanGraph,
        *,
        land_use_order: Iterable[str] | None = None,
        target_ratios: Iterable[float] | None = None,
        feasibility_time_limit_s: float = 5.0,
    ) -> None:
        self.graph = graph
        self.task = Task.LAND
        self.entities = graph.parcel_entities
        self.order = tuple(land_use_order or DEFAULT_LAND_USE_ORDER)
        if len(self.order) != len(LAND_ACTIONS) or set(self.order) != set(
            LAND_ACTIONS
        ):
            raise ValueError(
                "land_use_order must contain every land action exactly once"
            )
        self.order_actions = tuple(LAND_TO_INDEX[label] for label in self.order)
        self.areas = tuple(
            float(parcel["area_m2"])
            for parcel in graph.raw_topology["parcels"]
        )
        if len(self.areas) != len(self.entities):
            raise ValueError("land parcel areas do not match parcel entities")
        self.total_area = sum(self.areas)
        if self.total_area <= 0.0:
            raise ValueError("total parcel area must be positive")
        self.area_ratios = tuple(area / self.total_area for area in self.areas)
        self.target_ratios = tuple(
            load_land_area_targets(graph)
            if target_ratios is None
            else (float(value) for value in target_ratios)
        )
        self._validate_targets()
        self.feasibility_time_limit_s = float(feasibility_time_limit_s)
        if self.feasibility_time_limit_s <= 0.0:
            raise ValueError("feasibility_time_limit_s must be positive")
        self.tolerance = 1.0e-9
        self.evaluator = LandReward(graph)
        self.history = torch.zeros(graph.num_entities, dtype=torch.long)
        self.actions = [-1] * len(self.entities)
        self.selected = torch.zeros(len(self.entities), dtype=torch.bool)
        self.decision_entities: list[int] = []
        self.decision_actions: list[int] = []
        self.position = 0
        self.order_position = 0
        self._done = False
        self._validated_action: int | None = None

    @property
    def current_action(self) -> int:
        return self.order_actions[self.order_position]

    def reset(self) -> PolicyState:
        self.history.zero_()
        self.actions = [-1] * len(self.entities)
        self.selected.zero_()
        self.decision_entities = []
        self.decision_actions = []
        self.position = 0
        self.order_position = 0
        self._done = False
        self._validated_action = None
        if not self._completion_exists(self.actions):
            raise ValueError(
                f"{self.graph.tile_id}: land quotas have no feasible assignment"
            )
        return self._state()

    def ranked_feasible_action(
        self, ranked_candidates: Sequence[int]
    ) -> int | None:
        """Return the highest-ranked action that keeps the remainder feasible.

        ``None`` means the current area block had to close below its soft
        target; callers must request a new model state for the next type.
        """
        if self._done:
            raise RuntimeError("cannot select an action after termination")
        mask = self._action_mask()
        for raw_action in ranked_candidates:
            action = int(raw_action)
            if action < 0 or action >= len(mask) or not bool(mask[action].item()):
                continue
            trial = list(self.actions)
            trial[action] = self.current_action
            if self._completion_exists(trial):
                self._validated_action = action
                return action
        if self._close_current_block(force=True):
            return None
        raise RuntimeError(
            f"{self.graph.tile_id}: no feasibility-preserving parcel for "
            f"{LAND_ACTIONS[self.current_action]} at decision {self.position}"
        )

    def current_state(self) -> PolicyState:
        if self._done:
            raise RuntimeError("terminal land environment has no current state")
        return self._state()

    def step(self, action: int) -> tuple[PolicyState | None, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called after episode termination")
        mask = self._action_mask()
        if action < 0 or action >= len(mask) or not bool(mask[action].item()):
            raise ValueError(
                f"illegal land entity {action} at decision {self.position}"
            )
        if self._validated_action != action:
            trial = list(self.actions)
            trial[action] = self.current_action
            if not self._completion_exists(trial):
                raise ValueError(
                    f"land entity {action} destroys quota feasibility at "
                    f"decision {self.position}"
                )
        self._validated_action = None
        assigned_action = self.current_action
        entity = int(self.entities[action].item())
        self.actions[action] = assigned_action
        self.selected[action] = True
        self.history[entity] = assigned_action + 1
        self.decision_entities.append(entity)
        self.decision_actions.append(assigned_action)
        self.position += 1
        self._done = self.position == len(self.entities)
        if self._done:
            result: RewardResult = self.evaluator.evaluate(self.actions)
            if not bool(result.metrics["feasible"]):
                raise AssertionError("area-constrained land rollout became infeasible")
            return None, result.reward, True, self._final_metrics(result)
        self._advance_ready_blocks()
        return self._state(), 0.0, False, {}

    def _state(self) -> PolicyState:
        return PolicyState(
            graph=self.graph,
            task=Task.LAND,
            progress=self.position / max(len(self.entities), 1),
            history=self.history.clone(),
            dynamic=torch.zeros(
                (self.graph.num_entities, SIGNAL_DYNAMIC_DIM), dtype=torch.float32
            ),
            action_mask=self._action_mask(),
            candidate_entities=self.entities,
            query_action=self.current_action,
            decision_entities=torch.tensor(
                self.decision_entities, dtype=torch.long
            ),
            decision_actions=torch.tensor(
                self.decision_actions, dtype=torch.long
            ),
        )

    def _action_mask(self) -> torch.Tensor:
        mask = ~self.selected.clone()
        current_area = self._assigned_ratio(self.current_action)
        upper = LAND_QUOTA_UPPER[self.current_action]
        for index in mask.nonzero(as_tuple=False).flatten().tolist():
            if current_area + self.area_ratios[index] > upper + self.tolerance:
                mask[index] = False
        return mask

    def _advance_ready_blocks(self) -> None:
        while self.order_position < len(self.order_actions) - 1:
            if not self._should_close_current_block():
                return
            if not self._close_current_block(force=False):
                return

    def _should_close_current_block(self) -> bool:
        action = self.current_action
        current = self._assigned_ratio(action)
        if current + self.tolerance < LAND_QUOTA_LOWER[action]:
            return False
        target = self.target_ratios[action]
        if current + self.tolerance >= target:
            return True
        candidates = [
            self.area_ratios[index]
            for index, selected in enumerate(self.selected.tolist())
            if not selected
            and current + self.area_ratios[index]
            <= LAND_QUOTA_UPPER[action] + self.tolerance
        ]
        if not candidates:
            return True
        current_error = abs(target - current)
        best_next_error = min(abs(target - current - area) for area in candidates)
        return best_next_error + self.tolerance >= current_error

    def _close_current_block(self, *, force: bool) -> bool:
        if self.order_position >= len(self.order_actions) - 1:
            return False
        action = self.current_action
        current = self._assigned_ratio(action)
        if current + self.tolerance < LAND_QUOTA_LOWER[action]:
            return False
        if not force and not self._should_close_current_block():
            return False
        if not self._completion_exists(self.actions, close_current=True):
            return False
        self.order_position += 1
        self._validated_action = None
        return True

    def _completion_exists(
        self,
        actions: Sequence[int],
        *,
        close_current: bool = False,
    ) -> bool:
        closed = set(self.order_actions[: self.order_position])
        if close_current:
            closed.add(self.current_action)
        fixed = [0.0] * len(LAND_ACTIONS)
        unassigned = []
        for index, action in enumerate(actions):
            if action < 0:
                unassigned.append(index)
            else:
                fixed[int(action)] += self.area_ratios[index]
        allowed = [action for action in range(len(LAND_ACTIONS)) if action not in closed]
        if unassigned and not allowed:
            return False
        remaining = sum(self.area_ratios[index] for index in unassigned)
        for action in range(len(LAND_ACTIONS)):
            if fixed[action] > LAND_QUOTA_UPPER[action] + self.tolerance:
                return False
            reachable = fixed[action] + (remaining if action in allowed else 0.0)
            if reachable + self.tolerance < LAND_QUOTA_LOWER[action]:
                return False
        required = sum(
            max(LAND_QUOTA_LOWER[action] - fixed[action], 0.0)
            for action in allowed
        )
        capacity = sum(
            max(LAND_QUOTA_UPPER[action] - fixed[action], 0.0)
            for action in allowed
        )
        if required > remaining + self.tolerance:
            return False
        if capacity + self.tolerance < remaining:
            return False
        if not unassigned:
            return True

        model = gp.Model(
            f"land-feasibility-{self.graph.tile_id}",
            env=_quiet_gurobi_env(),
        )
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.TimeLimit = self.feasibility_time_limit_s
        model.Params.FeasibilityTol = 1.0e-9
        model.Params.IntFeasTol = 1.0e-9
        model.Params.NumericFocus = 2
        variables = {
            (parcel, action): model.addVar(vtype=gp.GRB.BINARY)
            for parcel in unassigned
            for action in allowed
        }
        for parcel in unassigned:
            model.addConstr(
                gp.quicksum(variables[parcel, action] for action in allowed) == 1
            )
        for action in range(len(LAND_ACTIONS)):
            expression = fixed[action]
            if action in allowed:
                expression += gp.quicksum(
                    self.area_ratios[parcel] * variables[parcel, action]
                    for parcel in unassigned
                )
            model.addConstr(expression >= LAND_QUOTA_LOWER[action])
            model.addConstr(expression <= LAND_QUOTA_UPPER[action])
        model.setObjective(0.0)
        model.optimize()
        if model.SolCount > 0:
            return True
        if model.Status == gp.GRB.TIME_LIMIT:
            raise RuntimeError(
                f"{self.graph.tile_id}: land feasibility check timed out"
            )
        return False

    def _assigned_ratio(self, action: int) -> float:
        return sum(
            self.area_ratios[index]
            for index, assigned in enumerate(self.actions)
            if assigned == action
        )

    def _final_metrics(self, result: RewardResult) -> dict[str, float | bool]:
        metrics = dict(result.metrics)
        target_error = 0.0
        for action, label in enumerate(LAND_ACTIONS):
            actual = self._assigned_ratio(action)
            target = self.target_ratios[action]
            target_error += abs(actual - target)
            metrics[f"{label}_area_ratio"] = actual
            metrics[f"{label}_target_ratio"] = target
            metrics[f"{label}_parcel_count"] = float(
                sum(assigned == action for assigned in self.actions)
            )
        metrics["target_ratio_l1_error"] = target_error
        metrics["area_constrained"] = True
        return metrics

    def _validate_targets(self) -> None:
        if len(self.target_ratios) != len(LAND_ACTIONS):
            raise ValueError("land target_ratios must contain six values")
        if abs(sum(self.target_ratios) - 1.0) > 1.0e-6:
            raise ValueError("land target_ratios must sum to one")
        for action, target in enumerate(self.target_ratios):
            if not LAND_QUOTA_LOWER[action] - 1.0e-9 <= target <= LAND_QUOTA_UPPER[
                action
            ] + 1.0e-9:
                raise ValueError(
                    f"target ratio for {LAND_ACTIONS[action]}={target:.6f} "
                    "is outside the legal quota"
                )


def _quiet_gurobi_env() -> gp.Env:
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        _GUROBI_ENV = environment
    return _GUROBI_ENV
