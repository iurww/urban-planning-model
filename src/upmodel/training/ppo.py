"""Multi-task PPO with task-local GAE normalization and a frozen KL reference."""

from __future__ import annotations

import logging
import math
import multiprocessing
import random
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical
from tqdm.auto import tqdm

from upmodel.config import PPOConfig
from upmodel.data.split import topology_group_key
from upmodel.envs import LandAreaPlanningEnv, StaticPlanningEnv, SumoSignalEnv
from upmodel.evaluation.baseline import expert_baseline
from upmodel.evaluation.evaluator import (
    _relative_gap_pct,
    expert_relative_score,
)
from upmodel.logging import RunLogger
from upmodel.model import MultiTaskActorCritic, PolicyOutput
from upmodel.model.policy import (
    categorical_kl,
    evaluate_action_components_from_output,
    ranked_action_log_prob,
    ranked_candidates,
)
from upmodel.state import PolicyState, UrbanGraph
from upmodel.tasks import Task
from upmodel.training.buffer import RolloutBuffer, Transition
from upmodel.training.checkpoint import save_checkpoint
from upmodel.training.road_baseline import ROAD_BASELINE_COMPONENTS


LOGGER = logging.getLogger(__name__)


# Preserve the legacy rollout seed mapping and domain-separate optimizer order.
_OPTIMIZATION_RNG_SEED_SALT = 0x9E37_79B9_7F4A_7C15


@dataclass
class _CollectedEpisode:
    episode_index: int
    graph: UrbanGraph
    task: Task
    expert_reward: float
    transitions: list[Transition]
    raw_total: float
    normalized_raw_total: float
    kl_total: float
    action_entropy_total: float
    final_info: dict[str, Any]


@dataclass
class _ActiveSignalEpisode:
    episode_index: int
    graph: UrbanGraph
    worker: "_SignalWorkerHandle"
    state: PolicyState
    action_generator: torch.Generator
    expert_reward: float
    reward_scale: float
    transitions: list[Transition] = field(default_factory=list)
    raw_total: float = 0.0
    normalized_raw_total: float = 0.0
    kl_total: float = 0.0
    action_entropy_total: float = 0.0
    final_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SignalStatePayload:
    progress: float
    history: torch.Tensor
    dynamic: torch.Tensor
    signal_history_dynamic: torch.Tensor | None
    signal_history_actions: torch.Tensor | None
    signal_history_progress: torch.Tensor | None


@dataclass
class _SignalWorkerHandle:
    process: Any
    connection: Any


def _make_signal_rollout_environment(
    graph: UrbanGraph,
    seed: int,
    history_steps: int,
) -> SumoSignalEnv:
    return SumoSignalEnv(
        graph,
        seed=seed,
        history_steps=history_steps,
        close_on_done=False,
    )


def _signal_state_payload(state: PolicyState) -> _SignalStatePayload:
    if state.task != Task.SIGNAL:
        raise ValueError("signal worker produced a non-signal policy state")
    return _SignalStatePayload(
        progress=float(state.progress),
        history=state.history,
        dynamic=state.dynamic,
        signal_history_dynamic=state.signal_history_dynamic,
        signal_history_actions=state.signal_history_actions,
        signal_history_progress=state.signal_history_progress,
    )


def _restore_signal_state(
    payload: _SignalStatePayload,
    graph: UrbanGraph,
) -> PolicyState:
    return PolicyState(
        graph=graph,
        task=Task.SIGNAL,
        progress=payload.progress,
        history=payload.history,
        dynamic=payload.dynamic,
        action_mask=graph.signal_action_mask,
        signal_entities=graph.signal_entities,
        signal_history_dynamic=payload.signal_history_dynamic,
        signal_history_actions=payload.signal_history_actions,
        signal_history_progress=payload.signal_history_progress,
    )


def _signal_rollout_worker(
    connection: Any,
    environment_factory: Callable[[UrbanGraph, int, int], Any],
) -> None:
    """Own one TraCI connection/SUMO process and expose a small pipe protocol."""
    env: Any | None = None
    episode_index = -1
    try:
        torch.set_num_threads(1)
        connection.send(("ready",))
        while True:
            command = connection.recv()
            operation = command[0]
            if operation == "shutdown":
                return
            if operation == "start":
                _, episode_index, graph, seed, history_steps = command
                if env is not None:
                    env.close()
                env = environment_factory(graph, seed, history_steps)
                state = env.reset()
                connection.send(
                    ("started", episode_index, _signal_state_payload(state))
                )
                continue
            if operation == "step":
                _, requested_episode, action = command
                if env is None or requested_episode != episode_index:
                    raise RuntimeError("signal worker received an out-of-order step")
                next_state, raw_reward, done, final_info = env.step(action)
                connection.send(
                    (
                        "stepped",
                        episode_index,
                        (
                            None
                            if next_state is None
                            else _signal_state_payload(next_state)
                        ),
                        float(raw_reward),
                        bool(done),
                        final_info,
                    )
                )
                if done:
                    env.close()
                    env = None
                continue
            raise RuntimeError(f"unknown signal worker command: {operation!r}")
    except BaseException:
        try:
            connection.send(
                ("error", episode_index, traceback.format_exc())
            )
        except BaseException:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        connection.close()


class PPOTrainer:
    def __init__(
        self,
        *,
        model: MultiTaskActorCritic,
        reference: MultiTaskActorCritic,
        graphs: list[UrbanGraph],
        test_graphs: list[UrbanGraph],
        config: PPOConfig,
        logger: RunLogger,
        output_dir: str | Path,
        full_config: dict,
        seed: int,
        land_use_order: list[str] | tuple[str, ...],
        signal_history_steps: int,
        log_step_offset: int = 0,
        road_baseline_rows: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.reference = reference
        self.graphs = graphs
        self.config = config
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.full_config = full_config
        # Rollout topology order must not depend on how many optimizer epochs
        # run before KL early stopping.
        self.rollout_random = random.Random(seed)
        self.optimization_random = random.Random(
            seed ^ _OPTIMIZATION_RNG_SEED_SALT
        )
        self.seed = seed
        self.land_use_order = tuple(land_use_order)
        self.signal_history_steps = int(signal_history_steps)
        self.log_step_offset = log_step_offset
        self.environment_step = 0
        self.optimizer_step = 0
        self._signal_environment_factory = _make_signal_rollout_environment
        # Spawn never inherits an initialized CUDA runtime into rollout workers.
        self._signal_rollout_start_method = "spawn"
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if not graphs or not test_graphs:
            raise ValueError("PPO requires nonempty training and test graphs")
        baseline_graphs = list(graphs) + list(test_graphs)
        enabled_tasks = tuple(Task.from_name(name) for name in config.enabled_tasks)
        missing_test_tasks = [
            task.label
            for task in enabled_tasks
            if not any(
                task != Task.SIGNAL or graph.signal_entities.numel() > 0
                for graph in test_graphs
            )
        ]
        if missing_test_tasks:
            raise ValueError(
                "test data has no evaluable topology for tasks: "
                + ", ".join(missing_test_tasks)
            )
        self.graph_expert_baselines = {
            (graph.tile_id, task): expert_baseline(graph, task)[0]
            for graph in baseline_graphs
            for task in enabled_tasks
            if task != Task.SIGNAL or graph.signal_entities.numel() > 0
        }
        self.test_graphs = list(test_graphs)
        self.training_graphs = list(graphs)
        self.road_baseline_rows = {
            str(group_key): dict(row)
            for group_key, row in (road_baseline_rows or {}).items()
        }
        if config.road_baseline_manifest and not self.road_baseline_rows:
            raise ValueError(
                "verified Road baseline rows are required for the configured manifest"
            )
        self.road_training_expert_scale: float | None = None
        if self.road_baseline_rows and Task.ROAD in enabled_tasks:
            expert_by_group = {
                topology_group_key(graph): self.graph_expert_baselines[
                    (graph.tile_id, Task.ROAD)
                ]
                for graph in self.training_graphs
            }
            self.road_training_expert_scale = max(
                sum(abs(value) for value in expert_by_group.values())
                / len(expert_by_group),
                1.0e-8,
            )
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        LOGGER.info(
            "PPO 训练器初始化完成：graphs=%d train_graphs=%d "
            "test_graphs=%d tasks=%s iterations=%d reward_normalization=%s "
            "road_center_rewards=%s road_expert_scale=%s",
            len(graphs),
            len(self.training_graphs),
            len(self.test_graphs),
            ",".join(config.enabled_tasks),
            config.iterations,
            config.normalize_rewards,
            config.road_center_rewards,
            self.road_training_expert_scale,
        )

    def train(self) -> Path:
        best_path = self.output_dir / "ppo_best.pt"
        LOGGER.info("开始二阶段 PPO 训练")
        # Test iteration 0 so PPO cannot silently replace its initialization
        # with a checkpoint that scores worse on the configured test tasks.
        initial_metrics, initial_evaluation = self._evaluate_test(0)
        best_score = self.logger.log_test_evaluation(
            initial_evaluation,
            step=self.log_step_offset,
            stage="ppo",
            stage_step=0,
        )
        if not math.isclose(
            best_score,
            initial_metrics["test/selection_score"],
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise AssertionError("PPO iteration-0 test logging drifted")
        save_checkpoint(
            best_path,
            model=self.model,
            optimizer=self.optimizer,
            step=0,
            config=self.full_config,
            metrics=initial_metrics,
        )
        LOGGER.info(
            "PPO iteration=0 test baseline=%.6f checkpoint=%s",
            best_score,
            best_path,
        )
        iterations = tqdm(
            range(1, self.config.iterations + 1),
            desc="Stage 2 / PPO",
            unit="iter",
        )
        for iteration in iterations:
            learning_rate = self._learning_rate(iteration)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            beta = self._kl_beta(iteration)
            buffer, rollout_metrics = self._collect(iteration, beta)
            update_metrics = self._update(buffer)
            update_metrics["ppo/learning_rate"] = learning_rate
            self.logger.log(
                {
                    "ppo/iteration": float(iteration),
                    **{
                        key: value
                        for key, value in rollout_metrics.items()
                        if key.startswith("rollout/")
                    },
                    **update_metrics,
                }
            )
            metrics = {
                **rollout_metrics,
                **update_metrics,
                "ppo/iteration": float(iteration),
                "ppo/kl_beta": beta,
            }
            test_score: float | None = None
            if (
                iteration == self.config.iterations
                or iteration % self.config.test_interval == 0
            ):
                test_metrics, test_evaluation = self._evaluate_test(iteration)
                metrics.update(test_metrics)
                test_score = self.logger.log_test_evaluation(
                    test_evaluation,
                    step=self.log_step_offset + iteration,
                    stage="ppo",
                    stage_step=iteration,
                )
                if not math.isclose(
                    test_score,
                    test_metrics["test/selection_score"],
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ):
                    raise AssertionError("PPO test logging drifted")
            task_rewards = " ".join(
                f"{Task.from_name(name).label}="
                f"{metrics[f'rollout/{_rollout_task_name(Task.from_name(name))}_reward_mean']:.4f}"
                for name in self.config.enabled_tasks
            )
            iterations.set_postfix(
                reward=f"{metrics['rollout/total_reward_mean']:.3f}"
            )
            LOGGER.info(
                "PPO iteration=%d/%d steps=%d normalized_reward=%.4f %s",
                iteration,
                self.config.iterations,
                int(metrics["_rollout_steps"]),
                metrics["rollout/total_reward_mean"],
                task_rewards,
            )
            if test_score is not None and test_score > best_score:
                best_score = test_score
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=iteration,
                    config=self.full_config,
                    metrics=metrics,
                )
                LOGGER.info("保存新的 PPO 最优 checkpoint：%s", best_path)
            save_checkpoint(
                self.output_dir / "ppo_last.pt",
                model=self.model,
                optimizer=self.optimizer,
                step=iteration,
                config=self.full_config,
                metrics=metrics,
            )
        self.logger.summary(
            {
                "ppo/best_test_score": best_score,
            }
        )
        LOGGER.info("二阶段 PPO 训练完成：best_test_score=%.6f", best_score)
        return best_path

    def _collect(
        self, iteration: int, beta: float
    ) -> tuple[RolloutBuffer, dict[str, float]]:
        self.model.eval()
        buffer = RolloutBuffer()
        totals: dict[str, list[float]] = defaultdict(list)
        collection_started = time.perf_counter()
        episode_total = sum(
            int(self.config.episodes_per_task[name])
            for name in self.config.enabled_tasks
        )
        LOGGER.info(
            "开始 PPO 采样：iteration=%d episodes=%d beta=%.6f",
            iteration,
            episode_total,
            beta,
        )
        with tqdm(
            total=episode_total,
            desc=f"PPO 采样 {iteration}",
            unit="episode",
            leave=False,
        ) as sampling_progress:
            for task_name in self.config.enabled_tasks:
                task = Task.from_name(task_name)
                candidates = [
                    graph
                    for graph in self.training_graphs
                    if task != Task.SIGNAL or graph.signal_entities.numel() > 0
                ]
                if not candidates:
                    raise ValueError(f"no graph supports PPO task {task.label}")
                episodes = int(self.config.episodes_per_task[task.label])
                LOGGER.info("开始采样任务：task=%s episodes=%d", task.label, episodes)
                rollout_graphs = self._rollout_topology_sequence(
                    candidates, episodes
                )
                if task == Task.SIGNAL:
                    effective_workers = min(
                        self.config.signal_rollout_workers,
                        len(rollout_graphs),
                    )

                    def signal_episode_complete(
                        result: _CollectedEpisode,
                    ) -> None:
                        sampling_progress.set_postfix(
                            task=task.label,
                            tile=result.graph.tile_id,
                            episode=f"{result.episode_index + 1}/{episodes}",
                            workers=effective_workers,
                            refresh=True,
                        )
                        self._record_collected_episode(buffer, totals, result)
                        sampling_progress.update(1)

                    signal_results, signal_elapsed = (
                        self._collect_signal_episodes(
                            rollout_graphs,
                            iteration=iteration,
                            beta=beta,
                            on_episode=signal_episode_complete,
                        )
                    )
                    signal_steps = sum(
                        len(result.transitions) for result in signal_results
                    )
                    LOGGER.info(
                        "Signal 并发采样完成：episodes=%d workers=%d "
                        "steps=%d elapsed=%.3fs throughput=%.3f steps/s",
                        len(signal_results),
                        effective_workers,
                        signal_steps,
                        signal_elapsed,
                        signal_steps / max(signal_elapsed, 1.0e-8),
                    )
                    continue
                for episode_index, graph in enumerate(rollout_graphs):
                    sampling_progress.set_postfix(
                        task=task.label,
                        tile=graph.tile_id,
                        episode=f"{episode_index + 1}/{episodes}",
                        refresh=True,
                    )
                    seed = self.seed + iteration * 10_000 + episode_index
                    result = self._collect_serial_episode(
                        episode_index=episode_index,
                        graph=graph,
                        task=task,
                        seed=seed,
                        beta=beta,
                    )
                    self._record_collected_episode(buffer, totals, result)
                    sampling_progress.update(1)
        normalized_task_stats = [
            _mean_std(
                totals[
                    f"{Task.from_name(name).label}_normalized_reward"
                ]
            )
            for name in self.config.enabled_tasks
        ]
        metrics = {
            "rollout/total_reward_mean": sum(
                mean for mean, _ in normalized_task_stats
            ),
            "rollout/total_reward_std": math.sqrt(
                sum(std * std for _, std in normalized_task_stats)
            ),
        }
        for task_name in self.config.enabled_tasks:
            task = Task.from_name(task_name)
            log_name = _rollout_task_name(task)
            reward_mean, reward_std = _mean_std(
                totals[f"{task.label}_raw_reward"]
            )
            metrics[f"rollout/{log_name}_reward_mean"] = reward_mean
            metrics[f"rollout/{log_name}_reward_std"] = reward_std
            metrics[f"rollout/{log_name}_action_entropy"] = sum(
                totals[f"{task.label}_action_entropy_total"]
            ) / max(sum(totals[f"{task.label}_length"]), 1.0)
            length_name = "signal" if task == Task.SIGNAL else task.label
            metrics[f"rollout/{length_name}_length_mean"] = sum(
                totals[f"{task.label}_length"]
            ) / max(len(totals[f"{task.label}_length"]), 1)
        rollout_steps = len(buffer.flattened(normalize=False))
        rollout_elapsed = time.perf_counter() - collection_started
        metrics["rollout/fps"] = rollout_steps / max(rollout_elapsed, 1.0e-8)
        metrics["_rollout_steps"] = float(rollout_steps)
        LOGGER.info(
            "PPO 采样完成：iteration=%d rollout_steps=%d elapsed=%.3fs "
            "throughput=%.3f steps/s",
            iteration,
            rollout_steps,
            rollout_elapsed,
            metrics["rollout/fps"],
        )
        return buffer, metrics

    def _collect_serial_episode(
        self,
        *,
        episode_index: int,
        graph: UrbanGraph,
        task: Task,
        seed: int,
        beta: float,
    ) -> _CollectedEpisode:
        if task == Task.SIGNAL:
            raise ValueError("signal episodes must use the signal rollout scheduler")
        env = self._environment(graph, task, seed=seed)
        expert_reward = self.graph_expert_baselines[(graph.tile_id, task)]
        road_random_reward: float | None = None
        if task == Task.ROAD and self.config.road_center_rewards:
            road_random_reward = self._road_random_reward(graph)
            if self.road_training_expert_scale is None:
                raise AssertionError("Road training expert scale is unavailable")
            reward_scale = self.road_training_expert_scale
        else:
            reward_scale = (
                max(abs(expert_reward), 1.0e-8)
                if self.config.normalize_rewards
                else 1.0
            )
        transitions: list[Transition] = []
        raw_total = 0.0
        normalized_raw_total = 0.0
        kl_total = 0.0
        action_entropy_total = 0.0
        final_info: dict[str, Any] = {}
        action_generator = torch.Generator(
            device=self.model.device
        ).manual_seed(seed)
        try:
            state = env.reset()
            with torch.no_grad():
                cached_entities = self.model.encode_entities(state)
                cached_reference_entities = (
                    self.reference.encode_entities(state) if beta else None
                )
            done = False
            while not done:
                with torch.no_grad():
                    current = self.model.forward_from_entities(
                        state, cached_entities
                    )
                    distribution = Categorical(logits=current.logits)
                    if beta:
                        reference = self.reference.forward_from_entities(
                            state, cached_reference_entities
                        )
                        kl = categorical_kl(
                            task, current.logits, reference.logits
                        )
                    else:
                        kl = current.logits.new_zeros(())
                stored_action: int | tuple[int, ...]
                if task == Task.LAND:
                    if not isinstance(env, LandAreaPlanningEnv):
                        raise AssertionError("land PPO environment drifted")
                    ranked = ranked_candidates(
                        current.logits,
                        state.action_mask,
                        deterministic=False,
                        generator=action_generator,
                    )
                    action = env.ranked_feasible_action(ranked)
                    if action is None:
                        state = env.current_state()
                        continue
                    prefix = tuple(ranked[: ranked.index(action) + 1])
                    log_prob = ranked_action_log_prob(
                        current.logits,
                        state.action_mask.to(self.model.device),
                        prefix,
                    )
                    stored_action = prefix
                else:
                    action_tensor = torch.multinomial(
                        distribution.probs,
                        num_samples=1,
                        replacement=True,
                        generator=action_generator,
                    ).squeeze(-1)
                    log_prob = distribution.log_prob(action_tensor)
                    action = int(action_tensor.item())
                    stored_action = action
                next_state, raw_reward, done, final_info = env.step(action)
                kl_value = float(kl.item())
                training_raw_reward = float(raw_reward)
                if road_random_reward is not None and done:
                    training_raw_reward -= road_random_reward
                normalized_raw_reward = training_raw_reward / reward_scale
                reward = normalized_raw_reward - beta * kl_value
                transitions.append(
                    Transition(
                        state=state.snapshot(),
                        action=stored_action,
                        old_log_prob=(
                            log_prob.detach().cpu().clone()
                            if log_prob.ndim
                            else float(log_prob.item())
                        ),
                        old_value=float(current.value.item()),
                        reward=reward,
                        raw_reward=float(raw_reward),
                        done=bool(done),
                        task=task,
                    )
                )
                raw_total += float(raw_reward)
                normalized_raw_total += normalized_raw_reward
                kl_total += kl_value
                action_entropy_total += float(
                    distribution.entropy().mean().item()
                )
                if next_state is not None:
                    state = next_state
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        return _CollectedEpisode(
            episode_index=episode_index,
            graph=graph,
            task=task,
            expert_reward=expert_reward,
            transitions=transitions,
            raw_total=raw_total,
            normalized_raw_total=normalized_raw_total,
            kl_total=kl_total,
            action_entropy_total=action_entropy_total,
            final_info=final_info,
        )

    def _collect_signal_episodes(
        self,
        rollout_graphs: list[UrbanGraph],
        *,
        iteration: int,
        beta: float,
        episode_seeds: list[int] | None = None,
        deterministic: bool = False,
        on_episode: Callable[[_CollectedEpisode], None] | None = None,
    ) -> tuple[list[_CollectedEpisode], float]:
        """Run process-owned SUMO episodes while inference stays on the trainer."""
        if not rollout_graphs:
            return [], 0.0
        seeds = (
            [
                self.seed + iteration * 10_000 + episode_index
                for episode_index in range(len(rollout_graphs))
            ]
            if episode_seeds is None
            else episode_seeds
        )
        worker_count = min(
            self.config.signal_rollout_workers,
            len(rollout_graphs),
        )
        started = time.perf_counter()
        collected: list[_CollectedEpisode] = []
        workers = self._start_signal_workers(worker_count)
        try:
            for wave_start in range(0, len(rollout_graphs), worker_count):
                wave_graphs = rollout_graphs[
                    wave_start : wave_start + worker_count
                ]
                wave_episodes: list[_ActiveSignalEpisode] = []
                wave_results: dict[int, _CollectedEpisode] = {}
                # Each worker owns its SUMO/TraCI connection. Only the short
                # process startup is serialized by the environment-level lock.
                for offset, graph in enumerate(wave_graphs):
                    episode_index = wave_start + offset
                    seed = seeds[episode_index]
                    workers[offset].connection.send(
                        (
                            "start",
                            episode_index,
                            graph,
                            seed,
                            self.signal_history_steps,
                        )
                    )
                for offset, graph in enumerate(wave_graphs):
                    worker = workers[offset]
                    episode_index = wave_start + offset
                    message = self._receive_signal_worker(worker, "started")
                    _, returned_episode, payload = message
                    if returned_episode != episode_index:
                        raise RuntimeError("signal worker started the wrong episode")
                    seed = seeds[episode_index]
                    expert_reward = self.graph_expert_baselines[
                        (graph.tile_id, Task.SIGNAL)
                    ]
                    reward_scale = (
                        max(abs(expert_reward), 1.0e-8)
                        if self.config.normalize_rewards
                        else 1.0
                    )
                    wave_episodes.append(
                        _ActiveSignalEpisode(
                            episode_index=episode_index,
                            graph=graph,
                            worker=worker,
                            state=_restore_signal_state(payload, graph),
                            action_generator=torch.Generator(
                                device=self.model.device
                            ).manual_seed(seed),
                            expert_reward=expert_reward,
                            reward_scale=reward_scale,
                        )
                    )

                active = list(wave_episodes)
                while active:
                    pending: list[
                        tuple[
                            _ActiveSignalEpisode,
                            torch.Tensor,
                            torch.Tensor,
                            float,
                            float,
                            float,
                            Any,
                        ]
                    ] = []
                    with torch.no_grad():
                        # Only the main process ever touches CUDA/model state.
                        current_outputs = self._forward_signal_rollout_states(
                            self.model, active
                        )
                        reference_outputs = (
                            self._forward_signal_rollout_states(
                                self.reference, active
                            )
                            if beta
                            else [None] * len(active)
                        )
                        for episode, current, reference in zip(
                            active,
                            current_outputs,
                            reference_outputs,
                            strict=True,
                        ):
                            distribution = Categorical(logits=current.logits)
                            action_tensor = (
                                distribution.probs.argmax(dim=-1)
                                if deterministic
                                else torch.multinomial(
                                    distribution.probs,
                                    num_samples=1,
                                    replacement=True,
                                    generator=episode.action_generator,
                                ).squeeze(-1)
                            )
                            log_prob = distribution.log_prob(action_tensor)
                            if beta:
                                if reference is None:
                                    raise AssertionError("reference batch is incomplete")
                                kl = categorical_kl(
                                    Task.SIGNAL,
                                    current.logits,
                                    reference.logits,
                                )
                            else:
                                kl = current.logits.new_zeros(())
                            action = action_tensor.detach().cpu()
                            pending.append(
                                (
                                    episode,
                                    action.clone(),
                                    log_prob.detach().cpu().clone(),
                                    float(current.value.item()),
                                    float(kl.item()),
                                    float(distribution.entropy().mean().item()),
                                    episode.state.snapshot(),
                                )
                            )
                            episode.worker.connection.send(
                                ("step", episode.episode_index, action)
                            )

                    next_active: list[_ActiveSignalEpisode] = []
                    for pending_step in pending:
                        (
                            episode,
                            stored_action,
                            old_log_prob,
                            old_value,
                            kl_value,
                            action_entropy,
                            state_snapshot,
                        ) = pending_step
                        message = self._receive_signal_worker(
                            episode.worker, "stepped"
                        )
                        (
                            _,
                            returned_episode,
                            next_payload,
                            raw_reward,
                            done,
                            final_info,
                        ) = message
                        if returned_episode != episode.episode_index:
                            raise RuntimeError("signal worker stepped the wrong episode")
                        normalized_raw_reward = (
                            float(raw_reward) / episode.reward_scale
                        )
                        episode.transitions.append(
                            Transition(
                                state=state_snapshot,
                                action=stored_action,
                                old_log_prob=old_log_prob,
                                old_value=old_value,
                                reward=normalized_raw_reward - beta * kl_value,
                                raw_reward=float(raw_reward),
                                done=bool(done),
                                task=Task.SIGNAL,
                            )
                        )
                        episode.raw_total += float(raw_reward)
                        episode.normalized_raw_total += normalized_raw_reward
                        episode.kl_total += kl_value
                        episode.action_entropy_total += action_entropy
                        episode.final_info = final_info
                        if done:
                            wave_results[episode.episode_index] = (
                                _CollectedEpisode(
                                    episode_index=episode.episode_index,
                                    graph=episode.graph,
                                    task=Task.SIGNAL,
                                    expert_reward=episode.expert_reward,
                                    transitions=episode.transitions,
                                    raw_total=episode.raw_total,
                                    normalized_raw_total=(
                                        episode.normalized_raw_total
                                    ),
                                    kl_total=episode.kl_total,
                                    action_entropy_total=(
                                        episode.action_entropy_total
                                    ),
                                    final_info=episode.final_info,
                                )
                            )
                        else:
                            if next_payload is None:
                                raise RuntimeError(
                                    "unfinished signal episode omitted its next state"
                                )
                            episode.state = _restore_signal_state(
                                next_payload, episode.graph
                            )
                            next_active.append(episode)
                    active = next_active

                expected = {
                    wave_start + offset for offset in range(len(wave_graphs))
                }
                if set(wave_results) != expected:
                    raise AssertionError("signal rollout wave lost an episode")
                for episode_index in sorted(wave_results):
                    result = wave_results[episode_index]
                    collected.append(result)
                    if on_episode is not None:
                        on_episode(result)
        finally:
            self._shutdown_signal_workers(workers)
        return collected, time.perf_counter() - started

    @staticmethod
    def _forward_signal_rollout_states(
        model: Any,
        episodes: list[_ActiveSignalEpisode],
    ) -> list[PolicyOutput]:
        states = [episode.state for episode in episodes]
        heterogeneous_forward = getattr(model, "forward_heterogeneous", None)
        if heterogeneous_forward is not None:
            return heterogeneous_forward(states)
        return [model(state) for state in states]

    def _start_signal_workers(
        self, worker_count: int
    ) -> list[_SignalWorkerHandle]:
        context = multiprocessing.get_context(self._signal_rollout_start_method)
        workers: list[_SignalWorkerHandle] = []
        try:
            for worker_index in range(worker_count):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_signal_rollout_worker,
                    args=(child_connection, self._signal_environment_factory),
                    name=f"upmodel-signal-{worker_index}",
                )
                try:
                    process.start()
                except BaseException:
                    child_connection.close()
                    parent_connection.close()
                    raise
                else:
                    child_connection.close()
                workers.append(
                    _SignalWorkerHandle(
                        process=process,
                        connection=parent_connection,
                    )
                )
            for worker in workers:
                self._receive_signal_worker(worker, "ready")
            return workers
        except BaseException:
            self._shutdown_signal_workers(workers)
            raise

    @staticmethod
    def _receive_signal_worker(
        worker: _SignalWorkerHandle,
        expected: str,
    ) -> tuple[Any, ...]:
        try:
            message = worker.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                "signal rollout worker exited before replying "
                f"(pid={worker.process.pid}, exitcode={worker.process.exitcode})"
            ) from exc
        if not isinstance(message, tuple) or not message:
            raise RuntimeError("signal rollout worker returned an invalid message")
        if message[0] == "error":
            _, episode_index, worker_traceback = message
            raise RuntimeError(
                "signal rollout worker failed "
                f"(pid={worker.process.pid}, episode={episode_index}):\n"
                f"{worker_traceback}"
            )
        if message[0] != expected:
            raise RuntimeError(
                f"signal rollout worker returned {message[0]!r}; "
                f"expected {expected!r}"
            )
        return message

    @staticmethod
    def _shutdown_signal_workers(workers: list[_SignalWorkerHandle]) -> None:
        for worker in workers:
            if worker.process.is_alive():
                try:
                    worker.connection.send(("shutdown",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        deadline = time.monotonic() + 10.0
        for worker in workers:
            worker.process.join(timeout=max(deadline - time.monotonic(), 0.0))
        for worker in workers:
            if worker.process.is_alive():
                worker.process.terminate()
        for worker in workers:
            # join() is required even when terminate() completed between the
            # liveness checks; otherwise the exited child can remain a zombie.
            worker.process.join(timeout=5.0)
            worker.connection.close()

    def _road_random_reward(self, graph: UrbanGraph) -> float:
        row = self.road_baseline_rows[topology_group_key(graph)]
        return float(row["uniform"]["reward"]["mean"])

    def _record_collected_episode(
        self,
        buffer: RolloutBuffer,
        totals: dict[str, list[float]],
        result: _CollectedEpisode,
    ) -> None:
        task = result.task
        transitions = result.transitions
        buffer.add_episode(
            transitions,
            gamma=float(self.config.gamma[task.label]),
            gae_lambda=float(self.config.gae_lambda[task.label]),
        )
        totals[f"{task.label}_raw_reward"].append(result.raw_total)
        totals[f"{task.label}_normalized_reward"].append(
            result.normalized_raw_total
        )
        totals[f"{task.label}_action_entropy_total"].append(
            result.action_entropy_total
        )
        totals[f"{task.label}_length"].append(float(len(transitions)))
        self.environment_step += len(transitions)
        if task == Task.SIGNAL and "final_wait_reward" in result.final_info:
            totals["ppo/signal_final_wait_reward"].append(
                float(result.final_info["final_wait_reward"])
            )

    def _evaluate_test(
        self, iteration: int
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Evaluate current weights on fixed test data for checkpointing."""
        self.model.eval()
        totals: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        LOGGER.info(
            "开始固定 test 推理：iteration=%d graphs=%d",
            iteration,
            len(self.test_graphs),
        )
        for task_name in self.config.enabled_tasks:
            task = Task.from_name(task_name)
            if task == Task.SIGNAL:
                signal_graphs = [
                    graph
                    for graph in self.test_graphs
                    if graph.signal_entities.numel() > 0
                    for _ in range(self.config.test_episodes)
                ]
                signal_seeds = [
                    episode
                    for graph in self.test_graphs
                    if graph.signal_entities.numel() > 0
                    for episode in range(self.config.test_episodes)
                ]
                results, elapsed = self._collect_signal_episodes(
                    signal_graphs,
                    iteration=iteration,
                    beta=0.0,
                    episode_seeds=signal_seeds,
                    deterministic=self.config.test_deterministic,
                )
                for result in results:
                    group_key = topology_group_key(result.graph)
                    totals["test_signal_raw_reward"][group_key].append(
                        result.raw_total
                    )
                    totals["test_signal_expert_reward"][group_key].append(
                        result.expert_reward
                    )
                LOGGER.info(
                    "固定 Signal test 推理完成：episodes=%d workers=%d elapsed=%.3fs",
                    len(results),
                    min(self.config.signal_rollout_workers, len(results)),
                    elapsed,
                )
                continue
            for graph in self.test_graphs:
                for episode in range(self.config.test_episodes):
                    # Episode 0 must match the provided signal expert's SUMO
                    # seed (0), so its relative score is an apples-to-apples
                    # comparison. Static tasks ignore this seed.
                    test_seed = episode
                    test_generator = torch.Generator(
                        device=self.model.device
                    ).manual_seed(test_seed)
                    env = self._environment(graph, task, seed=test_seed)
                    raw_total = 0.0
                    final_info: dict[str, Any] = {}
                    try:
                        state = env.reset()
                        cached_entities = None
                        if task != Task.SIGNAL:
                            with torch.no_grad():
                                cached_entities = self.model.encode_entities(state)
                        done = False
                        while not done:
                            with torch.no_grad():
                                if task == Task.LAND:
                                    if not isinstance(env, LandAreaPlanningEnv):
                                        raise AssertionError(
                                            "land test environment drifted"
                                        )
                                    output = self.model.forward_from_entities(
                                        state, cached_entities
                                    )
                                    ranked = ranked_candidates(
                                        output.logits,
                                        state.action_mask,
                                        deterministic=self.config.test_deterministic,
                                        generator=test_generator,
                                    )
                                    action = env.ranked_feasible_action(ranked)
                                    if action is None:
                                        state = env.current_state()
                                        continue
                                elif self.config.test_deterministic:
                                    if cached_entities is None:
                                        action, _, _, _ = self.model.act(
                                            state, deterministic=True
                                        )
                                    else:
                                        output = self.model.forward_from_entities(
                                            state, cached_entities
                                        )
                                        action = int(output.logits.argmax().item())
                                else:
                                    output = (
                                        self.model(state)
                                        if cached_entities is None
                                        else self.model.forward_from_entities(
                                            state, cached_entities
                                        )
                                    )
                                    action_tensor = torch.multinomial(
                                        output.logits.softmax(dim=-1),
                                        1,
                                        generator=test_generator,
                                    ).squeeze(-1)
                                    action = (
                                        action_tensor.detach().cpu()
                                        if task == Task.SIGNAL
                                        else int(action_tensor.item())
                                    )
                            next_state, reward, done, step_info = env.step(action)
                            raw_total += float(reward)
                            if done:
                                final_info = dict(step_info)
                            if next_state is not None:
                                state = next_state
                    finally:
                        close = getattr(env, "close", None)
                        if close is not None:
                            close()
                    expert_reward = self.graph_expert_baselines[(graph.tile_id, task)]
                    group_key = topology_group_key(graph)
                    totals[f"test_{task.label}_raw_reward"][
                        group_key
                    ].append(raw_total)
                    totals[f"test_{task.label}_expert_reward"][
                        group_key
                    ].append(expert_reward)
        metrics: dict[str, float] = {}
        evaluation_summary: dict[str, dict[str, float]] = {}
        scores = []
        for name in self.config.enabled_tasks:
            task = Task.from_name(name)
            raw_key = f"test_{task.label}_raw_reward"
            expert_key = f"test_{task.label}_expert_reward"
            task_summary, task_relative_score = _test_task_summary(
                totals[raw_key],
                totals[expert_key],
                test_episodes=self.config.test_episodes,
            )
            metrics[raw_key] = task_summary["policy_reward"]
            metrics[expert_key] = task_summary["expert_reward"]
            metrics[f"test/{task.label}_relative_score"] = (
                task_relative_score
            )
            evaluation_summary[task.label] = task_summary
            scores.append(task_relative_score)
        metrics["test/selection_score"] = float(
            sum(scores) / max(len(scores), 1)
        )
        LOGGER.info(
            "固定 test 推理完成：iteration=%d selection_score=%.6f",
            iteration,
            metrics["test/selection_score"],
        )
        return metrics, evaluation_summary

    def _update(self, buffer: RolloutBuffer) -> dict[str, float]:
        # Rollouts are collected with dropout disabled. Keeping eval mode here
        # is essential: enabling dropout changes log-probabilities before the
        # first optimizer step and creates a fake KL signal. Gradients remain
        # enabled in eval mode.
        self.model.eval()
        # cuDNN requires an RNN module to be in training mode for backward.
        # This one-layer GRU has no dropout, so its train/eval outputs are
        # identical while the surrounding Transformers remain deterministic.
        self.model.decision_transformer.signal_history_gru.train()
        transitions = buffer.flattened(normalize=True)
        evaluation_transitions = list(transitions)
        if self.config.task_balanced:
            transitions = self._task_balanced_transitions(transitions)
        totals: dict[str, float] = defaultdict(float)
        update_samples = 0
        batches_per_epoch = math.ceil(len(transitions) / self.config.minibatch_size)
        LOGGER.info(
            "开始 PPO 参数更新：transitions=%d update_epochs=%d task_balanced=%s",
            len(transitions),
            self.config.update_epochs,
            self.config.task_balanced,
        )
        with tqdm(
            total=self.config.update_epochs * batches_per_epoch,
            desc="PPO 更新",
            unit="batch",
            leave=False,
        ) as update_progress:
            for update_epoch in range(1, self.config.update_epochs + 1):
                epoch_kl = 0.0
                epoch_samples = 0
                self.optimization_random.shuffle(transitions)
                for start in range(0, len(transitions), self.config.minibatch_size):
                    minibatch = transitions[start : start + self.config.minibatch_size]
                    self.optimizer.zero_grad(set_to_none=True)
                    batch_loss = torch.zeros((), device=self.model.device)
                    batch_policy = torch.zeros((), device=self.model.device)
                    batch_value = torch.zeros((), device=self.model.device)
                    batch_entropy = torch.zeros((), device=self.model.device)
                    batch_kl = torch.zeros((), device=self.model.device)
                    batch_clip = torch.zeros((), device=self.model.device)
                    outputs = self._forward_minibatch(minibatch)
                    for transition, output in zip(minibatch, outputs, strict=True):
                        log_prob, value, entropy, _ = (
                            evaluate_action_components_from_output(
                                transition.state,
                                transition.action,
                                output,
                            )
                        )
                        old_log_prob = torch.as_tensor(
                            transition.old_log_prob,
                            dtype=torch.float32,
                            device=self.model.device,
                        )
                        advantage = torch.tensor(
                            transition.advantage,
                            dtype=torch.float32,
                            device=self.model.device,
                        )
                        policy_loss, approx_kl, clip_fraction = _ppo_policy_terms(
                            log_prob,
                            old_log_prob,
                            advantage,
                            clip_ratio=self.config.clip_ratio,
                        )
                        target = torch.tensor(
                            transition.return_target,
                            dtype=torch.float32,
                            device=self.model.device,
                        )
                        value_loss = (value - target).square()
                        loss = (
                            policy_loss
                            + self.config.value_coef * value_loss
                            - self.config.entropy_coef * entropy.mean()
                        )
                        weight = 1.0 / len(minibatch)
                        batch_loss = batch_loss + loss * weight
                        batch_policy = batch_policy + policy_loss.detach() * weight
                        batch_value = batch_value + value_loss.detach() * weight
                        batch_entropy = (
                            batch_entropy + entropy.detach().mean() * weight
                        )
                        # Non-negative Schulman KL estimator, less noisy than
                        # mean(old_log_prob - new_log_prob) on small batches.
                        batch_kl = batch_kl + approx_kl.detach() * weight
                        batch_clip = batch_clip + clip_fraction.detach() * weight
                    batch_loss.backward()
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    grad_norm_after = _gradient_norm(self.model.parameters())
                    self.optimizer.step()
                    batch_samples = len(minibatch)
                    totals["loss"] += (
                        float(batch_loss.detach().item()) * batch_samples
                    )
                    totals["policy"] += float(batch_policy.item()) * batch_samples
                    totals["value"] += float(batch_value.item()) * batch_samples
                    totals["entropy"] += float(batch_entropy.item()) * batch_samples
                    totals["approx_kl"] += float(batch_kl.item()) * batch_samples
                    totals["clip_fraction"] += (
                        float(batch_clip.item()) * batch_samples
                    )
                    totals["grad_norm_before"] += (
                        float(grad_norm_before.detach().item()) * batch_samples
                    )
                    totals["grad_norm_after"] += grad_norm_after * batch_samples
                    epoch_kl += float(batch_kl.item()) * batch_samples
                    epoch_samples += batch_samples
                    update_samples += batch_samples
                    self.optimizer_step += 1
                    update_progress.update(1)
                    update_progress.set_postfix(
                        loss=f"{float(batch_loss.detach().item()):.4f}",
                        kl=f"{float(batch_kl.item()):.4f}",
                    )
                mean_epoch_kl = epoch_kl / max(epoch_samples, 1)
                if mean_epoch_kl > 1.5 * self.config.target_kl:
                    LOGGER.warning(
                        "PPO 更新在完整 epoch 后因 KL 提前停止：epoch=%d "
                        "approx_kl=%.6f target_kl=%.6f",
                        update_epoch,
                        mean_epoch_kl,
                        self.config.target_kl,
                    )
                    break
        denominator = max(update_samples, 1)
        explained_variance = self._explained_variance(
            evaluation_transitions
        )
        entropy_mean = totals["entropy"] / denominator
        grad_norm_before = totals["grad_norm_before"] / denominator
        return {
            "ppo/policy_loss": totals["policy"] / denominator,
            "ppo/value_loss": totals["value"] / denominator,
            "ppo/entropy_loss": -self.config.entropy_coef * entropy_mean,
            "ppo/total_loss": totals["loss"] / denominator,
            "ppo/approx_kl": totals["approx_kl"] / denominator,
            "ppo/clip_fraction": totals["clip_fraction"] / denominator,
            "ppo/explained_variance": explained_variance,
            "ppo/grad_norm": grad_norm_before,
            "ppo/grad_norm_before_clip": grad_norm_before,
            "ppo/grad_norm_after_clip": (
                totals["grad_norm_after"] / denominator
            ),
            "ppo/learning_rate": float(
                self.optimizer.param_groups[0]["lr"]
            ),
        }

    def _explained_variance(
        self, transitions: list[Transition]
    ) -> float:
        if not transitions:
            return 0.0
        predictions: list[float] = []
        with torch.no_grad():
            for start in range(0, len(transitions), self.config.minibatch_size):
                minibatch = transitions[
                    start : start + self.config.minibatch_size
                ]
                predictions.extend(
                    float(output.value.item())
                    for output in self._forward_minibatch(minibatch)
                )
        returns = torch.tensor(
            [transition.return_target for transition in transitions],
            dtype=torch.float64,
        )
        values = torch.tensor(predictions, dtype=torch.float64)
        return _explained_variance(returns, values)

    def _environment(self, graph: UrbanGraph, task: Task, *, seed: int):
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

    def _task_balanced_transitions(
        self, transitions: list[Transition]
    ) -> list[Transition]:
        by_task = {
            task: [item for item in transitions if item.task == task]
            for task in (Task.from_name(name) for name in self.config.enabled_tasks)
        }
        active = {task: items for task, items in by_task.items() if items}
        target = max(len(items) for items in active.values())
        balanced: list[Transition] = []
        for items in active.values():
            sampled: list[Transition] = []
            while len(sampled) < target:
                shuffled = list(items)
                self.optimization_random.shuffle(shuffled)
                sampled.extend(shuffled)
            balanced.extend(sampled[:target])
        self.optimization_random.shuffle(balanced)
        return balanced

    def _rollout_topology_sequence(
        self, candidates: list[UrbanGraph], episodes: int
    ) -> list[UrbanGraph]:
        """Sample group-uniform cycles without consuming optimizer randomness."""
        grouped: dict[str, list[UrbanGraph]] = defaultdict(list)
        for graph in candidates:
            grouped[topology_group_key(graph)].append(graph)
        shuffled_groups = sorted(grouped.items())
        for _, members in shuffled_groups:
            members.sort(key=lambda graph: graph.tile_id)
            self.rollout_random.shuffle(members)
        self.rollout_random.shuffle(shuffled_groups)
        member_occurrences: dict[str, int] = defaultdict(int)
        selected: list[UrbanGraph] = []
        for episode_index in range(episodes):
            # Cover distinct physical/policy groups before sampling one twice.
            if episode_index and episode_index % len(shuffled_groups) == 0:
                self.rollout_random.shuffle(shuffled_groups)
            group_key, members = shuffled_groups[
                episode_index % len(shuffled_groups)
            ]
            occurrence = member_occurrences[group_key]
            selected.append(members[occurrence % len(members)])
            member_occurrences[group_key] += 1
        return selected

    def _forward_minibatch(
        self, minibatch: list[Transition]
    ) -> list[PolicyOutput]:
        """Batch model work by task and topology without reordering PPO samples."""
        grouped: dict[tuple[Task, int], list[tuple[int, Transition]]] = {}
        for position, transition in enumerate(minibatch):
            if transition.task != transition.state.task:
                raise ValueError("transition task does not match its policy state")
            key = (transition.task, id(transition.state.graph))
            grouped.setdefault(key, []).append((position, transition))

        outputs: list[PolicyOutput | None] = [None] * len(minibatch)
        static_entities: dict[int, torch.Tensor] = {}
        for (task, graph_id), indexed in grouped.items():
            states = [transition.state for _, transition in indexed]
            if task == Task.SIGNAL:
                entities = self.model.encode_entities_batch(states)
            else:
                entities = static_entities.get(graph_id)
                if entities is None:
                    entities = self.model.encode_entities(states[0])
                    static_entities[graph_id] = entities
            group_outputs = self.model.forward_batch_from_entities(states, entities)
            for (position, _), output in zip(indexed, group_outputs, strict=True):
                outputs[position] = output

        if any(output is None for output in outputs):
            raise AssertionError("grouped PPO forward omitted a transition")
        return [output for output in outputs if output is not None]

    def _kl_beta(self, iteration: int) -> float:
        if self.config.iterations <= 1:
            return float(self.config.kl_beta_end)
        fraction = (iteration - 1) / (self.config.iterations - 1)
        return float(
            self.config.kl_beta_start
            + fraction * (self.config.kl_beta_end - self.config.kl_beta_start)
        )

    def _learning_rate(self, iteration: int) -> float:
        if self.config.iterations <= 1:
            return float(self.config.learning_rate)
        fraction = (iteration - 1) / (self.config.iterations - 1)
        return float(
            self.config.learning_rate
            + fraction
            * (self.config.learning_rate_end - self.config.learning_rate)
        )

def _expert_relative_score(reward: float, expert_reward: float) -> float:
    """Map the expert return to 1 for both rewards and negative costs."""
    return expert_relative_score(reward, expert_reward)


def _test_task_summary(
    raw_by_group: dict[str, list[float]],
    expert_by_group: dict[str, list[float]],
    *,
    test_episodes: int,
) -> tuple[dict[str, float], float]:
    """Macro-average groups before deriving the checkpoint selection score."""
    if not raw_by_group or set(raw_by_group) != set(expert_by_group):
        raise ValueError("test rewards are missing or have mismatched groups")

    def group_macro_mean(values_by_group: dict[str, list[float]]) -> float:
        return sum(
            sum(values) / len(values) for values in values_by_group.values()
        ) / len(values_by_group)

    policy_reward = group_macro_mean(raw_by_group)
    expert_reward = group_macro_mean(expert_by_group)
    summary = {
        "policy_reward": policy_reward,
        "expert_reward": expert_reward,
        "reward_delta": policy_reward - expert_reward,
        "relative_gap_pct": _relative_gap_pct(policy_reward, expert_reward),
        "tiles": float(len(expert_by_group)),
        "members": float(
            sum(len(values) for values in expert_by_group.values())
            / test_episodes
        ),
    }
    return summary, _expert_relative_score(policy_reward, expert_reward)


def _validation_task_summary(
    raw_by_group: dict[str, list[float]],
    expert_by_group: dict[str, list[float]],
    *,
    validation_episodes: int,
) -> tuple[dict[str, float], float]:
    """Deprecated compatibility wrapper for the old helper name."""
    return _test_task_summary(
        raw_by_group,
        expert_by_group,
        test_episodes=validation_episodes,
    )


def _road_skill_summary(
    raw_by_group: dict[str, list[float]],
    expert_by_group: dict[str, list[float]],
    random_by_group: dict[str, float],
) -> dict[str, float]:
    """Close the macro random-to-expert Road gap without per-group reweighting."""
    policy_groups = {
        group: sum(values) / len(values) for group, values in raw_by_group.items()
    }
    expert_groups = {
        group: sum(expert_by_group[group]) / len(expert_by_group[group])
        for group in raw_by_group
    }
    policy_reward = sum(policy_groups.values()) / len(policy_groups)
    expert_reward = sum(expert_groups.values()) / len(expert_groups)
    random_reward = sum(random_by_group.values()) / len(random_by_group)
    expert_gap = expert_reward - random_reward
    if expert_gap <= 1.0e-8:
        raise ValueError("road expert baseline must exceed the random baseline")
    return {
        "random_reward": random_reward,
        "gap_closed": (policy_reward - random_reward) / expert_gap,
        "improvement_over_random": policy_reward - random_reward,
        "regret_to_expert": expert_reward - policy_reward,
        "win_rate_vs_random": sum(
            policy_groups[group] > random_by_group[group]
            for group in policy_groups
        )
        / len(policy_groups),
        "win_rate_vs_expert": sum(
            policy_groups[group] > expert_groups[group]
            for group in policy_groups
        )
        / len(policy_groups),
    }


def _road_component_summary(
    policy_by_component: Mapping[str, dict[str, list[float]]],
    random_by_component: Mapping[str, dict[str, float]],
    expert_by_component: Mapping[str, dict[str, float]],
) -> dict[str, float]:
    """Macro-average raw Road components for comparable test curves."""
    labels = {
        "Traffic": "traffic",
        "Tdensity": "density",
        "Tconnectivity": "connectivity",
        "Tspacing": "spacing",
    }
    summary: dict[str, float] = {}
    for component in ROAD_BASELINE_COMPONENTS:
        policy_groups = policy_by_component[component]
        random_groups = random_by_component[component]
        expert_groups = expert_by_component[component]
        policy_value = sum(
            sum(values) / len(values) for values in policy_groups.values()
        ) / len(policy_groups)
        random_value = sum(random_groups.values()) / len(random_groups)
        expert_value = sum(expert_groups.values()) / len(expert_groups)
        label = labels[component]
        summary[f"policy_{label}"] = policy_value
        summary[f"random_{label}"] = random_value
        summary[f"expert_{label}"] = expert_value
        summary[f"{label}_improvement_over_random"] = (
            policy_value - random_value
        )
        summary[f"{label}_regret_to_expert"] = expert_value - policy_value
    return summary


def _ppo_policy_terms(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute clipped terms per action factor before taking their mean."""
    log_ratio = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.minimum(
        ratio * advantage, clipped_ratio * advantage
    ).mean()
    approx_kl = (ratio - 1.0 - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return policy_loss, approx_kl, clip_fraction


def _rollout_task_name(task: Task) -> str:
    return "traffic" if task == Task.SIGNAL else task.label


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(variance, 0.0))


def _gradient_norm(parameters) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().norm(2).item())
        squared_norm += norm * norm
    return math.sqrt(squared_norm)


def _explained_variance(
    returns: torch.Tensor,
    values: torch.Tensor,
) -> float:
    if returns.numel() != values.numel() or returns.numel() == 0:
        raise ValueError("returns and values must be nonempty aligned vectors")
    return_variance = returns.var(unbiased=False)
    if float(return_variance.item()) <= 1.0e-12:
        return 0.0
    residual_variance = (returns - values).var(unbiased=False)
    return float((1.0 - residual_variance / return_variance).item())
