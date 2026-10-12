from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from upmodel.config import PPOConfig
from upmodel.model import PolicyOutput
from upmodel.state import PolicyState
from upmodel.tasks import Task
from upmodel.training.ppo import PPOTrainer


class _FakeSignalPolicy(torch.nn.Module):
    def __init__(self, action_bias: tuple[float, float]) -> None:
        super().__init__()
        self.action_bias = torch.nn.Parameter(torch.tensor(action_bias))

    @property
    def device(self) -> torch.device:
        return self.action_bias.device

    def forward(self, state: PolicyState) -> PolicyOutput:
        position = state.dynamic[:, 0]
        episode_marker = state.dynamic[:, 1] * 0.01
        logits = torch.stack(
            (
                0.15 * position + episode_marker,
                -0.1 * position - episode_marker,
            ),
            dim=-1,
        )
        logits = logits + self.action_bias
        value = self.action_bias.sum() * 0.0 + float(state.progress) * 0.25
        return PolicyOutput(logits=logits, value=value)


class _FakeSignalEnv:
    def __init__(
        self,
        graph: SimpleNamespace,
        *,
        seed: int,
        step_delay: float,
        episode_steps: int = 3,
    ) -> None:
        self.graph = graph
        self.seed = seed
        self.step_delay = step_delay
        self.episode_steps = episode_steps
        self.position = 0
        self.closed = False

    def reset(self) -> PolicyState:
        self.position = 0
        self.closed = False
        return self._state()

    def step(
        self, action: torch.Tensor
    ) -> tuple[PolicyState | None, float, bool, dict[str, float]]:
        time.sleep(self.step_delay)
        raw_reward = float(action.sum().item()) - float(self.seed % 7) * 0.1
        self.position += 1
        done = self.position >= self.episode_steps
        info = {
            "final_wait_reward": -float(self.seed % 11 + self.position),
        }
        return (None if done else self._state()), raw_reward, done, info

    def close(self) -> None:
        self.closed = True

    def _state(self) -> PolicyState:
        dynamic = torch.zeros((2, 18), dtype=torch.float32)
        dynamic[:, 0] = float(self.position)
        dynamic[:, 1] = float(self.seed % 13)
        dynamic[:, 2] = float(os.getpid())
        return PolicyState(
            graph=self.graph,
            task=Task.SIGNAL,
            progress=self.position / self.episode_steps,
            history=torch.empty(0),
            dynamic=dynamic,
            action_mask=torch.ones((2, 2), dtype=torch.bool),
            signal_entities=self.graph.signal_entities,
            signal_history_dynamic=torch.empty((0, 2, 18)),
            signal_history_actions=torch.empty((0, 2), dtype=torch.long),
            signal_history_progress=torch.empty(0),
        )


class ParallelSignalRolloutTest(unittest.TestCase):
    def test_parallel_rollout_matches_serial_order_rng_and_gae(self) -> None:
        serial_buffer, serial_metrics = _collect(workers=1, step_delay=0.02)
        serial_throughput = serial_metrics["rollout/fps"]
        self.assertEqual(
            set(serial_metrics),
            {
                "rollout/total_reward_mean",
                "rollout/total_reward_std",
                "rollout/traffic_reward_mean",
                "rollout/traffic_reward_std",
                "rollout/traffic_action_entropy",
                "rollout/signal_length_mean",
                "rollout/fps",
                "_rollout_steps",
            },
        )

        parallel_buffer, parallel_metrics = _collect(workers=4, step_delay=0.02)

        self.assertEqual(_buffer_signature(serial_buffer), _buffer_signature(parallel_buffer))
        stable_metric_keys = set(serial_metrics)
        stable_metric_keys.discard("rollout/fps")
        for key in stable_metric_keys:
            self.assertEqual(serial_metrics[key], parallel_metrics[key], key)

        self.assertEqual(len(_worker_pids(serial_buffer)), 1)
        self.assertEqual(len(_worker_pids(parallel_buffer)), 4)
        self.assertGreater(
            parallel_metrics["rollout/fps"],
            serial_throughput * 1.5,
        )

    def test_parallel_test_matches_serial_with_fixed_seeds(self) -> None:
        serial = _trainer(
            workers=1,
            step_delay=0.01,
            test_episodes=8,
        )
        parallel = _trainer(
            workers=4,
            step_delay=0.01,
            test_episodes=8,
        )

        serial_metrics, serial_summary = serial._evaluate_test(iteration=5)
        parallel_metrics, parallel_summary = parallel._evaluate_test(iteration=5)

        self.assertEqual(serial_metrics, parallel_metrics)
        self.assertEqual(serial_summary, parallel_summary)
        self.assertEqual(serial.environment_step, 0)
        self.assertEqual(parallel.environment_step, 0)


def _fake_signal_environment_factory(graph, seed: int, history_steps: int):
    del history_steps
    return _FakeSignalEnv(
        graph,
        seed=seed,
        step_delay=graph.step_delay,
        episode_steps=graph.episode_steps,
    )


def _collect(*, workers: int, step_delay: float):
    trainer = _trainer(workers=workers, step_delay=step_delay)
    return trainer._collect(iteration=2, beta=0.03)


def _trainer(
    *,
    workers: int,
    step_delay: float,
    test_episodes: int = 1,
) -> PPOTrainer:
    graph = SimpleNamespace(
        tile_id="signal-tile",
        raw_topology={
            "metadata": {
                "source_crs": "EPSG:32650",
                "local_origin_in_source_crs": [0.0, 0.0],
                "tile_width_m": 4000.0,
                "tile_height_m": 4000.0,
            }
        },
        signal_entities=torch.tensor([0, 1], dtype=torch.long),
        signal_action_mask=torch.ones((2, 2), dtype=torch.bool),
        step_delay=step_delay,
        episode_steps=12,
    )
    config = PPOConfig(
        enabled_tasks=["signal"],
        episodes_per_task={"signal": 8},
        signal_rollout_workers=workers,
        test_episodes=test_episodes,
    )
    with patch("upmodel.training.ppo.expert_baseline", return_value=(-10.0, {})):
        trainer = PPOTrainer(
            model=_FakeSignalPolicy((0.2, -0.1)),
            reference=_FakeSignalPolicy((-0.05, 0.15)),
            graphs=[graph],
            test_graphs=[graph],
            config=config,
            logger=Mock(),
            output_dir=Path("."),
            full_config={},
            seed=37,
            land_use_order=(),
            signal_history_steps=0,
        )
    trainer._signal_environment_factory = _fake_signal_environment_factory
    trainer._signal_rollout_start_method = "spawn"
    return trainer


def _buffer_signature(buffer) -> list[list[tuple]]:
    return [
        [
            (
                tuple(int(value) for value in transition.action.tolist()),
                tuple(float(value) for value in transition.old_log_prob.tolist()),
                transition.old_value,
                transition.reward,
                transition.raw_reward,
                transition.done,
                transition.advantage,
                transition.return_target,
                transition.state.progress,
                tuple(float(value) for value in transition.state.dynamic[:, 0]),
            )
            for transition in episode
        ]
        for episode in buffer.episodes
    ]


def _worker_pids(buffer) -> set[int]:
    return {
        int(episode[0].state.dynamic[0, 2].item())
        for episode in buffer.episodes
    }


if __name__ == "__main__":
    unittest.main()
