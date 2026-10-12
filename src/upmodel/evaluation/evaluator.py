"""Run checkpoint policies and aggregate reward deltas against expert data."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical
from tqdm.auto import tqdm

from upmodel.data.split import topology_group_key
from upmodel.envs import LandAreaPlanningEnv, StaticPlanningEnv, SumoSignalEnv
from upmodel.evaluation.baseline import expert_baseline
from upmodel.model import MultiTaskActorCritic
from upmodel.model.policy import ranked_candidates
from upmodel.state import UrbanGraph
from upmodel.tasks import TRAINING_TASKS, Task


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationRow:
    tile_id: str
    task: str
    policy_reward: float
    expert_reward: float
    reward_delta: float
    relative_gap_pct: float
    episodes: int
    policy_metrics: dict[str, float]
    expert_metrics: dict[str, float]
    group_key: str = ""


@dataclass(frozen=True)
class RolloutTrace:
    task: str
    actions: list[int | list[int]]
    query_actions: list[int | None]
    step_rewards: list[float]
    step_metrics: list[dict[str, float]]
    final_assignment: list[int] | None


@dataclass(frozen=True)
class EvaluationReport:
    checkpoint: str
    deterministic: bool
    rows: list[EvaluationRow]
    summary: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "deterministic": self.deterministic,
            "rows": [asdict(row) for row in self.rows],
            "summary": self.summary,
        }

    def write_json(
        self,
        path: str | Path,
        *,
        fail_if_exists: bool = False,
    ) -> Path:
        return write_json_atomic(
            path,
            self.to_dict(),
            fail_if_exists=fail_if_exists,
        )


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    fail_if_exists: bool = False,
) -> Path:
    """Write complete JSON via a temporary file in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if fail_if_exists:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing JSON: {path}"
                ) from exc
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class PolicyEvaluator:
    def __init__(
        self,
        *,
        model: MultiTaskActorCritic,
        graphs: list[UrbanGraph],
        checkpoint: str | Path,
        seed: int = 0,
        land_use_order: list[str] | tuple[str, ...] = (),
        signal_history_steps: int = 4,
    ) -> None:
        self.model = model
        self.graphs = graphs
        self.checkpoint = str(checkpoint)
        self.seed = int(seed)
        self.land_use_order = tuple(land_use_order)
        self.signal_history_steps = int(signal_history_steps)

    def evaluate(
        self,
        *,
        tasks: list[Task],
        episodes: int = 1,
        deterministic: bool = True,
        show_progress: bool = True,
    ) -> EvaluationReport:
        if episodes < 1:
            raise ValueError("evaluation episodes must be positive")
        unsupported = set(tasks) - set(TRAINING_TASKS)
        if unsupported:
            raise ValueError(
                "evaluation supports only land, road, and signal"
            )
        self.model.eval()
        rows: list[EvaluationRow] = []
        supported = [
            (graph, task)
            for graph in self.graphs
            for task in tasks
            if task != Task.SIGNAL or graph.signal_entities.numel() > 0
        ]
        LOGGER.info(
            "开始 evaluate：graphs=%d task_graph_pairs=%d episodes=%d deterministic=%s",
            len(self.graphs),
            len(supported),
            episodes,
            deterministic,
        )
        with tqdm(
            total=len(supported) * episodes,
            desc="Evaluate",
            unit="episode",
            disable=not show_progress,
        ) as progress:
            for graph, task in supported:
                policy_rewards = []
                episode_metrics = []
                for episode in range(episodes):
                    progress.set_postfix(
                        task=task.label,
                        tile=graph.tile_id,
                        episode=f"{episode + 1}/{episodes}",
                        refresh=True,
                    )
                    reward, metrics = self._rollout(
                        graph,
                        task,
                        deterministic=deterministic,
                        seed=self.seed + episode,
                    )
                    policy_rewards.append(reward)
                    episode_metrics.append(metrics)
                    progress.update(1)
                policy_reward = sum(policy_rewards) / len(policy_rewards)
                expert_reward, expert_metrics = expert_baseline(graph, task)
                delta = policy_reward - expert_reward
                rows.append(
                    EvaluationRow(
                        tile_id=graph.tile_id,
                        task=task.label,
                        policy_reward=policy_reward,
                        expert_reward=expert_reward,
                        reward_delta=delta,
                        relative_gap_pct=_relative_gap_pct(
                            policy_reward, expert_reward
                        ),
                        episodes=episodes,
                        policy_metrics=_mean_metrics(episode_metrics),
                        expert_metrics=expert_metrics,
                        group_key=topology_group_key(graph),
                    )
                )
        report = EvaluationReport(
            checkpoint=self.checkpoint,
            deterministic=deterministic,
            rows=rows,
            summary=_summarize(rows),
        )
        LOGGER.info("evaluate 完成：rows=%d", len(rows))
        return report

    def _rollout(
        self,
        graph: UrbanGraph,
        task: Task,
        *,
        deterministic: bool,
        seed: int,
    ) -> tuple[float, dict[str, float]]:
        total_reward, metrics, _ = self.rollout(
            graph,
            task,
            deterministic=deterministic,
            seed=seed,
        )
        return total_reward, metrics

    def rollout(
        self,
        graph: UrbanGraph,
        task: Task,
        *,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> tuple[float, dict[str, float], RolloutTrace]:
        """Run one episode and retain the trace used by diagnostics."""
        self.model.eval()
        resolved_seed = self.seed if seed is None else int(seed)
        env = self._environment(graph, task, resolved_seed)
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        actions: list[int | list[int]] = []
        query_actions: list[int | None] = []
        step_rewards: list[float] = []
        step_metrics: list[dict[str, float]] = []
        try:
            state = env.reset()
            cached_entities = None
            if task != Task.SIGNAL:
                with torch.no_grad():
                    cached_entities = self.model.encode_entities(state)
            done = False
            while not done:
                if isinstance(env, LandAreaPlanningEnv):
                    with torch.no_grad():
                        logits = self.model.policy_logits_from_entities(
                            state, cached_entities
                        ).detach().cpu()
                    ranked = ranked_candidates(
                        logits,
                        state.action_mask,
                        deterministic=deterministic,
                    )
                    action = env.ranked_feasible_action(ranked)
                    if action is None:
                        state = env.current_state()
                        continue
                else:
                    with torch.no_grad():
                        if cached_entities is None:
                            action, _, _, _ = self.model.act(
                                state, deterministic=deterministic
                            )
                        else:
                            output = self.model.forward_from_entities(
                                state, cached_entities
                            )
                            logits = output.logits
                            action = int(
                                logits.argmax().item()
                                if deterministic
                                else Categorical(logits=logits).sample().item()
                            )
                query_actions.append(state.query_action)
                if isinstance(action, torch.Tensor):
                    actions.append([int(value) for value in action.tolist()])
                else:
                    actions.append(int(action))
                next_state, reward, done, final_info = env.step(action)
                total_reward += float(reward)
                step_rewards.append(float(reward))
                step_metrics.append(_numeric_metrics(final_info))
                if next_state is not None:
                    state = next_state
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        metrics = _numeric_metrics(final_info)
        raw_assignment = getattr(env, "actions", None)
        final_assignment = (
            [int(value) for value in raw_assignment]
            if isinstance(raw_assignment, list)
            else None
        )
        trace = RolloutTrace(
            task=task.label,
            actions=actions,
            query_actions=query_actions,
            step_rewards=step_rewards,
            step_metrics=step_metrics,
            final_assignment=final_assignment,
        )
        return total_reward, metrics, trace

    def _environment(self, graph: UrbanGraph, task: Task, seed: int):
        if task == Task.SIGNAL:
            return SumoSignalEnv(
                graph,
                seed=seed,
                history_steps=self.signal_history_steps,
            )
        if task == Task.LAND:
            return LandAreaPlanningEnv(
                graph,
                land_use_order=self.land_use_order,
            )
        return StaticPlanningEnv(
            graph,
            task,
            land_use_order=self.land_use_order,
        )


def _numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float, bool))
    }


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {
        key: sum(item[key] for item in items if key in item)
        / max(sum(key in item for item in items), 1)
        for key in keys
    }


def _summarize(rows: list[EvaluationRow]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for task in TRAINING_TASKS:
        selected = [row for row in rows if row.task == task.label]
        if not selected:
            continue
        groups: dict[str, list[EvaluationRow]] = {}
        for row in selected:
            groups.setdefault(row.group_key or row.tile_id, []).append(row)
        group_policy_rewards = [
            sum(row.policy_reward for row in group) / len(group)
            for group in groups.values()
        ]
        group_expert_rewards = [
            sum(row.expert_reward for row in group) / len(group)
            for group in groups.values()
        ]
        policy_reward = sum(group_policy_rewards) / len(group_policy_rewards)
        expert_reward = sum(group_expert_rewards) / len(group_expert_rewards)
        summary[task.label] = {
            "policy_reward": policy_reward,
            "expert_reward": expert_reward,
            "reward_delta": policy_reward - expert_reward,
            # Compute the percentage from the task-level means. Averaging
            # per-tile ratios would give small-expert tiles disproportionate
            # weight and would not match the two r values shown in W&B.
            "relative_gap_pct": _relative_gap_pct(policy_reward, expert_reward),
            "tiles": float(len(groups)),
            "members": float(len(selected)),
        }
    return summary


def _relative_gap_pct(policy_reward: float, expert_reward: float) -> float:
    """Absolute expert-relative reward difference in percentage points.

    A gap is a distance: 0% means the policy matches the expert and smaller
    values are better.  This remains meaningful for both positive rewards and
    negative traffic-control returns.
    """
    return float(
        100.0
        * abs(float(policy_reward) - float(expert_reward))
        / max(abs(float(expert_reward)), 1.0e-8)
    )


def expert_relative_score(policy_reward: float, expert_reward: float) -> float:
    """Directional expert-relative reward: expert=1 and larger is better."""
    return float(
        1.0
        + (float(policy_reward) - float(expert_reward))
        / max(abs(float(expert_reward)), 1.0e-8)
    )
