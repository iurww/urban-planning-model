from __future__ import annotations

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


class PPORandomStreamIsolationTest(unittest.TestCase):
    def test_optimizer_rng_consumption_does_not_change_rollout_topologies(self) -> None:
        control = _trainer(seed=41)
        optimizer_advanced = _trainer(seed=41)
        candidates = [
            SimpleNamespace(tile_id=f"tile-{index}") for index in range(7)
        ]

        with patch(
            "upmodel.training.ppo.topology_group_key",
            side_effect=lambda graph: graph.tile_id,
        ):
            self.assertEqual(
                _tile_ids(control._rollout_topology_sequence(candidates, 16)),
                _tile_ids(
                    optimizer_advanced._rollout_topology_sequence(candidates, 16)
                ),
            )

            for batch_size in (5, 11, 3, 19):
                minibatch_order = list(range(batch_size))
                optimizer_advanced.optimization_random.shuffle(minibatch_order)

            self.assertNotEqual(
                control.optimization_random.getstate(),
                optimizer_advanced.optimization_random.getstate(),
            )
            self.assertEqual(
                _tile_ids(control._rollout_topology_sequence(candidates, 23)),
                _tile_ids(
                    optimizer_advanced._rollout_topology_sequence(candidates, 23)
                ),
            )

    def test_road_episode_rng_is_independent_of_global_torch_state(self) -> None:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = _RoadPolicy()
        trainer.reference = _RoadPolicy()
        trainer.config = PPOConfig(
            enabled_tasks=["road"],
            episodes_per_task={"road": 1},
        )
        graph = SimpleNamespace(tile_id="road-rng")
        trainer.graph_expert_baselines = {(graph.tile_id, Task.ROAD): 10.0}
        trainer._environment = lambda graph, task, seed: _RoadEnv(graph)

        torch.manual_seed(1)
        first = trainer._collect_serial_episode(
            episode_index=0,
            graph=graph,
            task=Task.ROAD,
            seed=73,
            beta=0.0,
        )
        torch.manual_seed(999)
        torch.rand(1000)
        repeated = trainer._collect_serial_episode(
            episode_index=0,
            graph=graph,
            task=Task.ROAD,
            seed=73,
            beta=0.0,
        )
        different = trainer._collect_serial_episode(
            episode_index=0,
            graph=graph,
            task=Task.ROAD,
            seed=74,
            beta=0.0,
        )

        self.assertEqual(_actions(first), _actions(repeated))
        self.assertNotEqual(_actions(first), _actions(different))

    def test_road_centering_changes_only_the_terminal_training_reward(self) -> None:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = _RoadPolicy()
        trainer.reference = _RoadPolicy()
        trainer.config = PPOConfig(
            enabled_tasks=["road"],
            episodes_per_task={"road": 1},
            road_center_rewards=True,
        )
        graph = SimpleNamespace(tile_id="road-centered")
        trainer.graph_expert_baselines = {(graph.tile_id, Task.ROAD): 10.0}
        trainer.road_baseline_rows = {
            "road-group": {"uniform": {"reward": {"mean": 4.0}}}
        }
        trainer.road_training_expert_scale = 10.0
        trainer._environment = lambda graph, task, seed: _RoadEnv(graph)

        with patch(
            "upmodel.training.ppo.topology_group_key",
            return_value="road-group",
        ):
            result = trainer._collect_serial_episode(
                episode_index=0,
                graph=graph,
                task=Task.ROAD,
                seed=73,
                beta=0.0,
            )

        self.assertEqual([item.reward for item in result.transitions[:-1]], [0.0, 0.0])
        self.assertEqual(result.transitions[-1].raw_reward, result.raw_total)
        self.assertAlmostEqual(
            result.transitions[-1].reward,
            (result.raw_total - 4.0) / 10.0,
        )
        self.assertAlmostEqual(
            result.normalized_raw_total,
            (result.raw_total - 4.0) / 10.0,
        )

    def test_rollout_cycles_are_uniform_over_groups_not_duplicate_members(self) -> None:
        trainer = _trainer(seed=19)
        candidates = [
            SimpleNamespace(tile_id="a-1", group="a"),
            SimpleNamespace(tile_id="a-2", group="a"),
            SimpleNamespace(tile_id="b-1", group="b"),
            SimpleNamespace(tile_id="c-1", group="c"),
        ]

        with patch(
            "upmodel.training.ppo.topology_group_key",
            side_effect=lambda graph: graph.group,
        ):
            selected = trainer._rollout_topology_sequence(candidates, 6)

        group_counts = {
            group: sum(item.group == group for item in selected)
            for group in ("a", "b", "c")
        }
        self.assertEqual(group_counts, {"a": 2, "b": 2, "c": 2})
        self.assertEqual(
            {item.tile_id for item in selected if item.group == "a"},
            {"a-1", "a-2"},
        )


def _trainer(*, seed: int) -> PPOTrainer:
    graphs = [
        SimpleNamespace(
            tile_id="train",
            signal_entities=torch.empty(0, dtype=torch.long),
        )
    ]
    test_graphs = [
        SimpleNamespace(
            tile_id="test",
            signal_entities=torch.empty(0, dtype=torch.long),
        )
    ]
    model = torch.nn.Linear(1, 1)
    reference = torch.nn.Linear(1, 1)
    config = PPOConfig(
        enabled_tasks=["road"],
        episodes_per_task={"road": 1},
    )
    with patch("upmodel.training.ppo.expert_baseline", return_value=(1.0, {})):
        trainer = PPOTrainer(
            model=model,
            reference=reference,
            graphs=graphs,
            test_graphs=test_graphs,
            config=config,
            logger=Mock(),
            output_dir=Path("."),
            full_config={},
            seed=seed,
            land_use_order=(),
            signal_history_steps=0,
        )
    return trainer


def _tile_ids(graphs: list[SimpleNamespace]) -> list[str]:
    return [graph.tile_id for graph in graphs]


class _RoadPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def device(self) -> torch.device:
        return self.bias.device

    def encode_entities(self, state: PolicyState) -> torch.Tensor:
        return self.bias.new_zeros((5, 1))

    def forward_from_entities(
        self, state: PolicyState, entities: torch.Tensor
    ) -> PolicyOutput:
        del entities
        logits = self.bias + torch.linspace(0.0, 0.2, 5)
        logits = logits.masked_fill(~state.action_mask, -torch.inf)
        return PolicyOutput(logits=logits, value=self.bias * 0.0)


class _RoadEnv:
    def __init__(self, graph: SimpleNamespace) -> None:
        self.graph = graph
        self.selected = torch.zeros(5, dtype=torch.bool)
        self.position = 0

    def reset(self) -> PolicyState:
        self.selected.zero_()
        self.position = 0
        return self._state()

    def step(self, action: int):
        self.selected[action] = True
        self.position += 1
        done = self.position == 3
        reward = float(action) if done else 0.0
        return (None if done else self._state()), reward, done, {}

    def _state(self) -> PolicyState:
        return PolicyState(
            graph=self.graph,
            task=Task.ROAD,
            progress=self.position / 3.0,
            history=torch.zeros(5, dtype=torch.long),
            dynamic=torch.zeros((5, 18)),
            action_mask=~self.selected,
            candidate_entities=torch.arange(5),
            query_action=0,
        )


def _actions(result) -> list[int]:
    return [int(transition.action) for transition in result.transitions]


if __name__ == "__main__":
    unittest.main()
