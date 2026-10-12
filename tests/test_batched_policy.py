from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from upmodel.config import SFTConfig
from upmodel.model import MultiTaskActorCritic
from upmodel.state import (
    SIGNAL_DYNAMIC_DIM,
    STATIC_FEATURE_DIM,
    PolicyState,
    UrbanGraph,
)
from upmodel.tasks import MAX_ACTIONS, Task
from upmodel.training.buffer import Transition
from upmodel.training.ppo import PPOTrainer
from upmodel.training.sft import SFTTrainer


class BatchedPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.graph = _graph()
        self.model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()

    def test_land_and_road_batch_match_single_with_variable_histories(self) -> None:
        for task in (Task.LAND, Task.ROAD):
            with self.subTest(task=task.label):
                states = [
                    _static_state(self.graph, task, length)
                    for length in (0, 1, 2)
                ]
                with torch.no_grad():
                    single = [self.model(state) for state in states]
                    batched_entities = self.model.encode_entities_batch(states)
                    batched = self.model.forward_batch_from_entities(
                        states, batched_entities
                    )

                self.assertEqual(len(batched), len(states))
                for expected, actual in zip(single, batched, strict=True):
                    torch.testing.assert_close(
                        actual.logits, expected.logits, atol=2.0e-6, rtol=1.0e-5
                    )
                    torch.testing.assert_close(
                        actual.value, expected.value, atol=2.0e-6, rtol=1.0e-5
                    )

    def test_signal_entity_and_policy_batches_match_single_with_padding(self) -> None:
        states = [_signal_state(self.graph, length) for length in (0, 1, 3)]
        with torch.no_grad():
            single_entities = [self.model.encode_entities(state) for state in states]
            batched_entities = self.model.encode_entities_batch(states)
            single = [
                self.model.forward_from_entities(state, entities)
                for state, entities in zip(states, single_entities, strict=True)
            ]
            batched = self.model.forward_batch_from_entities(states, batched_entities)

        self.assertEqual(
            tuple(batched_entities.shape),
            (len(states), self.graph.num_entities, self.model.hidden_dim),
        )
        for expected_entities, actual_entities in zip(
            single_entities, batched_entities, strict=True
        ):
            torch.testing.assert_close(
                actual_entities, expected_entities, atol=2.0e-6, rtol=1.0e-5
            )
        for expected, actual in zip(single, batched, strict=True):
            torch.testing.assert_close(
                actual.logits, expected.logits, atol=2.0e-6, rtol=1.0e-5
            )
            torch.testing.assert_close(
                actual.value, expected.value, atol=2.0e-6, rtol=1.0e-5
            )

    def test_heterogeneous_signal_batch_matches_different_graph_sizes(self) -> None:
        short_graph = _short_graph(self.graph)
        states = [
            _signal_state(self.graph, 3),
            _signal_state(short_graph, 1),
        ]
        with torch.no_grad():
            expected = [self.model(state) for state in states]
            actual = self.model.forward_heterogeneous(states)

        for expected_output, actual_output in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                actual_output.logits,
                expected_output.logits,
                atol=2.0e-6,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                actual_output.value,
                expected_output.value,
                atol=2.0e-6,
                rtol=1.0e-5,
            )

    def test_detached_static_entity_cache_is_reusable(self) -> None:
        states = [_static_state(self.graph, Task.LAND, length) for length in (0, 2)]
        with torch.no_grad():
            cached_entities = self.model.encode_entities(states[0])
            expected = [self.model(state) for state in states]

        self.assertFalse(cached_entities.requires_grad)
        self.assertIsNone(cached_entities.grad_fn)
        cached_before = cached_entities.clone()

        actual = [
            self.model.forward_from_entities(state, cached_entities) for state in states
        ]
        torch.testing.assert_close(cached_entities, cached_before)
        for expected_output, actual_output in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                actual_output.logits,
                expected_output.logits,
                atol=2.0e-6,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                actual_output.value,
                expected_output.value,
                atol=2.0e-6,
                rtol=1.0e-5,
            )

        loss = sum(
            output.logits[state.action_mask].sum() + output.value
            for state, output in zip(states, actual, strict=True)
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.decision_transformer.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.model.entity_encoder.parameters()
            )
        )

    def test_ppo_grouped_forward_preserves_order_and_reuses_encoders(self) -> None:
        states = [
            _static_state(self.graph, Task.LAND, 0),
            _signal_state(self.graph, 0),
            _static_state(self.graph, Task.ROAD, 0),
            _static_state(self.graph, Task.LAND, 1),
            _signal_state(self.graph, 1),
            _static_state(self.graph, Task.ROAD, 1),
        ]
        transitions = [_transition(state) for state in states]
        with torch.no_grad():
            expected = [self.model(state) for state in states]

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = self.model
        with (
            patch.object(
                self.model,
                "encode_entities",
                wraps=self.model.encode_entities,
            ) as encode_static,
            patch.object(
                self.model,
                "encode_entities_batch",
                wraps=self.model.encode_entities_batch,
            ) as encode_signal,
        ):
            with torch.no_grad():
                actual = trainer._forward_minibatch(transitions)

        self.assertEqual(encode_static.call_count, 1)
        self.assertEqual(encode_signal.call_count, 1)
        for expected_output, actual_output in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                actual_output.logits,
                expected_output.logits,
                atol=2.0e-6,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                actual_output.value,
                expected_output.value,
                atol=2.0e-6,
                rtol=1.0e-5,
            )

    def test_sft_epoch_batches_are_full_and_topology_groupable(self) -> None:
        second_graph = replace(self.graph, tile_id="batch-unit-2")
        states = [
            *[_static_state(self.graph, Task.LAND, index % 3) for index in range(5)],
            *[
                _static_state(second_graph, Task.ROAD, index % 3)
                for index in range(4)
            ],
        ]

        class Dataset:
            def __init__(self) -> None:
                self.steps = [SimpleNamespace(state=state) for state in states]
                self.tasks = (Task.LAND, Task.ROAD)

            def __len__(self) -> int:
                return len(self.steps)

            def __getitem__(self, index: int):
                return self.steps[index]

        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.dataset = Dataset()
        trainer.config = SFTConfig(batch_size=4, task_balanced=False)
        trainer.generator = torch.Generator().manual_seed(3)

        batches = trainer._epoch_batches()

        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(9))
        )
        for batch in batches:
            for group in trainer._topology_groups(batch):
                first = trainer.dataset[group[0]].state
                self.assertTrue(
                    all(
                        trainer.dataset[index].state.graph is first.graph
                        and trainer.dataset[index].state.task == first.task
                        for index in group
                    )
                )

    def test_grouped_ppo_forward_matches_scalar_gradients(self) -> None:
        states = [
            _static_state(self.graph, Task.LAND, 0),
            _signal_state(self.graph, 0),
            _static_state(self.graph, Task.ROAD, 1),
            _static_state(self.graph, Task.LAND, 2),
            _signal_state(self.graph, 2),
        ]
        scalar_model = copy.deepcopy(self.model)
        grouped_model = copy.deepcopy(self.model)
        scalar_outputs = [scalar_model(state) for state in states]
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = grouped_model
        grouped_outputs = trainer._forward_minibatch(
            [_transition(state) for state in states]
        )

        _output_sum(states, scalar_outputs).backward()
        _output_sum(states, grouped_outputs).backward()

        for (scalar_name, scalar_parameter), (grouped_name, grouped_parameter) in zip(
            scalar_model.named_parameters(),
            grouped_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(scalar_name, grouped_name)
            if scalar_parameter.grad is None or grouped_parameter.grad is None:
                self.assertIsNone(scalar_parameter.grad)
                self.assertIsNone(grouped_parameter.grad)
                continue
            torch.testing.assert_close(
                grouped_parameter.grad,
                scalar_parameter.grad,
                atol=2.0e-6,
                rtol=1.0e-5,
            )


def _graph() -> UrbanGraph:
    generator = torch.Generator().manual_seed(11)
    signal_mask = torch.zeros((2, MAX_ACTIONS), dtype=torch.bool)
    signal_mask[0, :3] = True
    signal_mask[1, 1:5] = True
    return UrbanGraph(
        tile_id="batch-unit",
        tile_dir=Path("."),
        features=torch.randn((6, STATIC_FEATURE_DIM), generator=generator),
        entity_types=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
            ],
            dtype=torch.long,
        ),
        parcel_entities=torch.tensor([0, 1], dtype=torch.long),
        road_entities=torch.tensor([2, 3], dtype=torch.long),
        intersection_entities=torch.tensor([4, 5], dtype=torch.long),
        signal_entities=torch.tensor([4, 5], dtype=torch.long),
        signal_action_mask=signal_mask,
        raw_topology={},
        signal_meta=None,
    )


def _short_graph(graph: UrbanGraph) -> UrbanGraph:
    keep_edge = (graph.edge_index < 5).all(dim=0)
    return replace(
        graph,
        tile_id="batch-unit-short",
        features=graph.features[:5].clone(),
        entity_types=graph.entity_types[:5].clone(),
        edge_index=graph.edge_index[:, keep_edge].clone(),
        intersection_entities=torch.tensor([4], dtype=torch.long),
        signal_entities=torch.tensor([4], dtype=torch.long),
        signal_action_mask=graph.signal_action_mask[:1].clone(),
    )


def _static_state(graph: UrbanGraph, task: Task, history_length: int) -> PolicyState:
    candidates = graph.parcel_entities if task == Task.LAND else graph.road_entities
    entity_history = candidates[:history_length].clone()
    action_offset = 1 if task == Task.LAND else 0
    action_history = torch.arange(
        action_offset, action_offset + history_length, dtype=torch.long
    )
    action_mask = torch.tensor([history_length == 0, True], dtype=torch.bool)
    return PolicyState(
        graph=graph,
        task=task,
        progress=history_length / 3.0,
        history=torch.empty(0, dtype=torch.long),
        dynamic=torch.zeros((graph.num_entities, SIGNAL_DYNAMIC_DIM)),
        action_mask=action_mask,
        candidate_entities=candidates,
        query_action=(history_length + int(task)) % MAX_ACTIONS,
        decision_entities=entity_history,
        decision_actions=action_history,
    )


def _signal_state(graph: UrbanGraph, history_length: int) -> PolicyState:
    generator = torch.Generator().manual_seed(100 + history_length)
    dynamic = torch.randn(
        (graph.num_entities, SIGNAL_DYNAMIC_DIM), generator=generator
    )
    history_dynamic = torch.randn(
        (history_length, graph.signal_entities.numel(), SIGNAL_DYNAMIC_DIM),
        generator=generator,
    )
    history_actions = (
        torch.arange(history_length, dtype=torch.long).unsqueeze(1)
        + torch.arange(graph.signal_entities.numel(), dtype=torch.long)
    ) % 3
    return PolicyState(
        graph=graph,
        task=Task.SIGNAL,
        progress=history_length / 4.0,
        history=torch.empty(0, dtype=torch.long),
        dynamic=dynamic,
        action_mask=graph.signal_action_mask,
        signal_entities=graph.signal_entities,
        signal_history_dynamic=history_dynamic,
        signal_history_actions=history_actions,
        signal_history_progress=torch.linspace(0.0, 0.5, history_length),
    )


def _transition(state: PolicyState) -> Transition:
    action: int | torch.Tensor
    if state.task == Task.SIGNAL:
        if state.signal_entities is None:
            raise AssertionError("signal fixture is missing controlled entities")
        action = torch.zeros(state.signal_entities.numel(), dtype=torch.long)
    else:
        action = 0
    return Transition(
        state=state,
        action=action,
        old_log_prob=0.0,
        old_value=0.0,
        reward=0.0,
        raw_reward=0.0,
        done=True,
        task=state.task,
    )


def _output_sum(states, outputs) -> torch.Tensor:
    return sum(
        output.logits[state.action_mask].sum() + output.value
        for state, output in zip(states, outputs, strict=True)
    )


if __name__ == "__main__":
    unittest.main()
