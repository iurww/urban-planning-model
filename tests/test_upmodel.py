from __future__ import annotations

import csv
import fcntl
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
from torch.distributions import Categorical

from upmodel.config import PPOConfig, SFTConfig
from upmodel.data import ExpertDataset, load_graphs
from upmodel.data.expert import ExpertStep
from upmodel.envs import LandAreaPlanningEnv, StaticPlanningEnv
from upmodel.envs.signal import _start_traci
from upmodel.envs.rewards import (
    LAND_QUOTA_LOWER,
    LAND_QUOTA_UPPER,
    LandReward,
    PoiReward,
    RoadReward,
)
from upmodel.evaluation.baseline import expert_baseline
from upmodel.evaluation.evaluator import EvaluationRow, PolicyEvaluator, _summarize
from upmodel.model import (
    MultiTaskActorCritic,
    SASADecisionTransformer,
    UrbanEntityTransformer,
)
from upmodel.model.decision import NUM_TOKEN_TYPES
from upmodel.model.policy import (
    evaluate_action,
    ranked_action_log_prob,
    reduce_distribution_stats,
)
from upmodel.planning import (
    infer_land_use_order,
    load_land_area_targets,
    load_static_expert_plan,
)
from upmodel.state import PolicyState, UrbanGraph
from upmodel.tasks import LAND_TO_INDEX, TRAINING_TASKS, Task
from upmodel.training.ppo import (
    PPOTrainer,
    _expert_relative_score,
    _ppo_policy_terms,
    _road_component_summary,
    _road_skill_summary,
    _test_task_summary,
)
from upmodel.training.sft import (
    SFTTrainer,
    _masked_cross_entropy,
    _masked_set_cross_entropy,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "train"


class DataAndRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = next(
            graph for graph in load_graphs(DATA_ROOT)
            if graph.tile_id == "beijing_4km_c107_r1099"
        )

    def test_expert_dataset_covers_three_training_tasks(self) -> None:
        dataset = ExpertDataset([self.graph])
        signal_steps = int(
            json.loads(
                (self.graph.tile_dir / "signal_trajectory.json").read_text(
                    encoding="utf-8"
                )
            )["num_steps"]
        )
        road_steps = len(load_static_expert_plan(self.graph, Task.ROAD).where)
        self.assertEqual(
            dataset.summary(),
            {"land": 44, "road": road_steps, "signal": signal_steps},
        )

    def test_expert_rewards_match_trajectory(self) -> None:
        tile = self.graph.tile_dir
        cases = [
            (
                LandReward(self.graph),
                _actions(tile / "land_use_solution.csv", "parcel_id", "land_use", LAND_TO_INDEX),
                tile / "land_use_trajectory.json",
            ),
            (
                RoadReward(self.graph),
                _actions(tile / "road_plan_solution.csv", "road_row_index", "upgraded"),
                tile / "road_plan_trajectory.json",
            ),
            (
                PoiReward(self.graph),
                _actions(tile / "poi_plan_solution.csv", "parcel_id", "place_poi"),
                tile / "poi_plan_trajectory.json",
            ),
        ]
        for evaluator, actions, trajectory_path in cases:
            expected = json.loads(trajectory_path.read_text(encoding="utf-8"))["final_reward"]
            self.assertAlmostEqual(evaluator.evaluate(actions).reward, expected, places=10)

    def test_every_head_accepts_real_expert_state(self) -> None:
        dataset = ExpertDataset([self.graph])
        model = MultiTaskActorCritic(hidden_dim=32, num_layers=1, dropout=0.0)
        for task in TRAINING_TASKS:
            step = next(dataset.by_task(task))
            log_prob, value, entropy, _ = evaluate_action(model, step.state, step.action)
            self.assertTrue(torch.isfinite(log_prob))
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(entropy))

    def test_expert_critic_targets_are_dimensionless(self) -> None:
        dataset = ExpertDataset([self.graph])
        for task in (Task.LAND, Task.ROAD):
            for step in dataset.by_task(task):
                self.assertAlmostEqual(
                    step.return_to_go / step.return_scale, 1.0, places=7
                )
        for step in dataset.by_task(Task.SIGNAL):
            self.assertLessEqual(
                abs(step.return_to_go / step.return_scale), 1.0 + 1.0e-7
            )

    def test_model_is_hierarchical_entity_and_sasa_transformer(self) -> None:
        dataset = ExpertDataset([self.graph])
        model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=2,
            decision_layers=2,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        self.assertIsInstance(model.entity_encoder, UrbanEntityTransformer)
        self.assertIsInstance(model.decision_transformer, SASADecisionTransformer)

        land_steps = list(dataset.by_task(Task.LAND))
        _, sequence, _ = model.encode(land_steps[2].state)
        self.assertEqual(model.decision_transformer.token_type_embedding.num_embeddings, 5)
        self.assertEqual(NUM_TOKEN_TYPES, 5)
        self.assertEqual(sequence.token_kinds[0], "TASK")
        self.assertEqual(sequence.token_kinds[-1], "QUERY")
        self.assertEqual(
            sequence.token_kinds,
            ("TASK", "GLOBAL", "S", "A", "S", "A", "QUERY"),
        )
        self.assertEqual(sequence.decision_indices.numel(), 1)

        signal_step = next(dataset.by_task(Task.SIGNAL))
        _, signal_sequence, _ = model.encode(signal_step.state)
        self.assertEqual(signal_sequence.token_kinds[0:2], ("TASK", "GLOBAL"))
        self.assertEqual(signal_sequence.token_kinds[-1], "QUERY")
        self.assertEqual(
            signal_sequence.decision_indices.numel(),
            self.graph.signal_entities.numel(),
        )

    def test_softening_sft_actor_prior_preserves_argmax_and_increases_entropy(self) -> None:
        model = MultiTaskActorCritic(hidden_dim=32, num_layers=1, dropout=0.0)
        state = next(ExpertDataset([self.graph]).by_task(Task.ROAD)).state
        before = model(state).logits.detach()
        before_dist = Categorical(logits=before)

        model.soften_actor_prior_(0.7)
        after = model(state).logits.detach()
        after_dist = Categorical(logits=after)

        self.assertTrue(
            torch.allclose(
                after[state.action_mask], before[state.action_mask] * 0.7, atol=1.0e-6
            )
        )
        self.assertTrue(torch.equal(after.argmax(dim=-1), before.argmax(dim=-1)))
        self.assertGreaterEqual(
            float(after_dist.entropy().mean().detach()),
            float(before_dist.entropy().mean().detach()) - 1.0e-7,
        )

    def test_signal_baseline_combines_step_and_terminal_rewards(self) -> None:
        trajectory = json.loads(
            (self.graph.tile_dir / "signal_trajectory.json").read_text(
                encoding="utf-8"
            )
        )
        step_reward = sum(float(step["reward"]) for step in trajectory["steps"])
        terminal_reward = float(trajectory.get("final_reward", 0.0))
        reward, metrics = expert_baseline(self.graph, Task.SIGNAL)
        self.assertAlmostEqual(reward, step_reward + terminal_reward)
        self.assertAlmostEqual(metrics["step_reward"], step_reward)
        self.assertAlmostEqual(metrics["terminal_reward"], terminal_reward)

    def test_road_pointer_mask_selects_each_entity_at_most_once(self) -> None:
        env = StaticPlanningEnv(self.graph, Task.ROAD)
        plan = load_static_expert_plan(self.graph, Task.ROAD)
        state = env.reset()
        selected = []
        done = False
        info = {}
        while not done:
            action = int(state.action_mask.nonzero()[0].item())
            selected.append(action)
            next_state, reward, done, info = env.step(action)
            if next_state is not None:
                self.assertFalse(bool(next_state.action_mask[action]))
                state = next_state
        self.assertTrue(info["feasible"])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(
            info["selected_roads"],
            float(len(self.graph.road_entities) - len(plan.where)),
        )
        self.assertLessEqual(info["selected_roads"], info["budget"])
        self.assertGreater(reward, 0.0)

    def test_land_what_is_fixed_and_where_is_the_pointer_target(self) -> None:
        order = infer_land_use_order([self.graph])
        plan = load_static_expert_plan(
            self.graph, Task.LAND, land_use_order=order
        )
        env = StaticPlanningEnv(
            self.graph, Task.LAND, land_use_order=order
        )
        state = env.reset()
        self.assertEqual(state.query_action, plan.what[0])
        self.assertEqual(tuple(state.action_mask.shape), (len(plan.where),))
        next_state, _, done, _ = env.step(plan.where[0])
        self.assertFalse(done)
        assert next_state is not None
        self.assertEqual(next_state.query_action, plan.what[1])
        self.assertFalse(bool(next_state.action_mask[plan.where[0]]))

    def test_land_area_targets_are_legal_expert_demand(self) -> None:
        targets = load_land_area_targets(self.graph)
        self.assertEqual(len(targets), 6)
        self.assertAlmostEqual(sum(targets), 1.0)
        for index, target in enumerate(targets):
            self.assertGreaterEqual(target, LAND_QUOTA_LOWER[index] - 1.0e-9)
            self.assertLessEqual(target, LAND_QUOTA_UPPER[index] + 1.0e-9)

    def test_evaluator_land_rollout_is_area_constrained_and_feasible(self) -> None:
        order = infer_land_use_order([self.graph])
        model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        ).eval()
        evaluator = PolicyEvaluator(
            model=model,
            graphs=[self.graph],
            checkpoint="random",
            land_use_order=order,
        )
        reward, metrics, trace = evaluator.rollout(self.graph, Task.LAND)
        self.assertIsInstance(
            evaluator._environment(self.graph, Task.LAND, 0),
            LandAreaPlanningEnv,
        )
        self.assertTrue(metrics["feasible"])
        self.assertTrue(metrics["area_constrained"])
        self.assertGreater(reward, 0.0)
        self.assertEqual(len(trace.actions), len(self.graph.parcel_entities))
        self.assertEqual(len(trace.query_actions), len(trace.actions))
        self.assertIsNotNone(trace.final_assignment)

    def test_ppo_and_evaluator_use_the_same_land_environment(self) -> None:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.land_use_order = infer_land_use_order([self.graph])
        trainer.signal_history_steps = 4
        env = trainer._environment(self.graph, Task.LAND, seed=0)
        self.assertIsInstance(env, LandAreaPlanningEnv)

    def test_ppo_test_score_uses_macro_reward_ratio(self) -> None:
        summary, score = _test_task_summary(
            {"small": [0.5], "large": [90.0]},
            {"small": [1.0], "large": [100.0]},
            test_episodes=1,
        )

        self.assertAlmostEqual(summary["policy_reward"], 45.25)
        self.assertAlmostEqual(summary["expert_reward"], 50.5)
        self.assertAlmostEqual(score, 45.25 / 50.5)
        self.assertNotAlmostEqual(score, (0.5 + 0.9) / 2.0)

    def test_road_skill_uses_ratio_of_macro_means_without_clipping(self) -> None:
        summary = _road_skill_summary(
            {"small-gap": [1.2], "large-gap": [8.0]},
            {"small-gap": [1.1], "large-gap": [6.0]},
            {"small-gap": 1.0, "large-gap": 2.0},
        )

        expected = ((1.2 + 8.0) / 2.0 - 1.5) / (3.55 - 1.5)
        per_group_mean = ((1.2 - 1.0) / 0.1 + (8.0 - 2.0) / 4.0) / 2.0
        self.assertAlmostEqual(summary["gap_closed"], expected)
        self.assertNotAlmostEqual(summary["gap_closed"], per_group_mean)
        self.assertGreater(summary["gap_closed"], 1.0)

    def test_road_components_are_macro_averaged(self) -> None:
        components = ("Traffic", "Tdensity", "Tconnectivity", "Tspacing")
        policy = {
            component: {"duplicate": [1.0, 3.0], "single": [6.0]}
            for component in components
        }
        random = {
            component: {"duplicate": 1.0, "single": 3.0}
            for component in components
        }
        expert = {
            component: {"duplicate": 3.0, "single": 7.0}
            for component in components
        }

        summary = _road_component_summary(policy, random, expert)

        self.assertEqual(summary["policy_density"], 4.0)
        self.assertEqual(summary["random_density"], 2.0)
        self.assertEqual(summary["expert_density"], 5.0)

    def test_ppo_rejects_test_data_without_enabled_signal_task(self) -> None:
        no_signal = replace(
            self.graph,
            tile_id="no-signal-validation",
            signal_entities=torch.empty(0, dtype=torch.long),
            signal_action_mask=torch.empty((0, 8), dtype=torch.bool),
        )
        model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        reference = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        )

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "test data has no evaluable topology for tasks: signal",
            ):
                PPOTrainer(
                    model=model,
                    reference=reference,
                    graphs=[self.graph],
                    test_graphs=[no_signal],
                    config=PPOConfig(
                        enabled_tasks=["signal"],
                        episodes_per_task={"signal": 1},
                    ),
                    logger=Mock(),
                    output_dir=directory,
                    full_config={},
                    seed=0,
                    land_use_order=infer_land_use_order([self.graph]),
                    signal_history_steps=4,
                )

    def test_signal_history_is_limited_and_centralized(self) -> None:
        dataset = ExpertDataset([self.graph], signal_history_steps=4)
        steps = list(dataset.by_task(Task.SIGNAL))
        self.assertEqual(steps[0].state.signal_history_dynamic.shape[0], 0)
        self.assertEqual(steps[3].state.signal_history_dynamic.shape[0], 3)
        self.assertEqual(steps[-1].state.signal_history_dynamic.shape[0], 4)
        model = MultiTaskActorCritic(hidden_dim=32, num_layers=1, dropout=0.0)
        output = model(steps[-1].state)
        self.assertEqual(
            output.logits.shape,
            self.graph.signal_action_mask.shape,
        )

    def test_traci_startup_retry_does_not_write_to_progress_output(self) -> None:
        class FakeTraci:
            def __init__(self) -> None:
                self.command = None
                self.kwargs = None

            def start(self, command, **kwargs) -> None:
                self.command = command
                self.kwargs = kwargs
                print(" Retrying in 1 seconds")

        traci = FakeTraci()
        output = io.StringIO()
        with (
            patch("upmodel.envs.signal.fcntl.flock") as flock,
            redirect_stdout(output),
        ):
            _start_traci(traci, ["sumo", "-n", "net.xml"], "test-label")

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(flock.call_args.args[1], fcntl.LOCK_EX)
        self.assertEqual(traci.command, ["sumo", "-n", "net.xml"])
        self.assertEqual(traci.kwargs["label"], "test-label")
        self.assertIn("stdout", traci.kwargs)
        self.assertFalse(traci.kwargs["doSwitch"])

    def test_disconnected_road_plan_is_rejected_by_training_reward(self) -> None:
        evaluator = RoadReward(self.graph)
        actions = [0] * len(evaluator.endpoints)
        first = 0
        second = next(
            index
            for index, endpoints in enumerate(evaluator.endpoints)
            if not set(endpoints) & set(evaluator.endpoints[first])
        )
        actions[first] = 1
        actions[second] = 1
        result = evaluator.evaluate(actions)
        self.assertEqual(result.metrics["NCC"], 2.0)
        self.assertFalse(result.metrics["connected"])
        self.assertFalse(result.metrics["feasible"])
        self.assertFalse(result.metrics["structurally_feasible"])
        self.assertEqual(result.reward, 0.0)
        self.assertGreater(result.metrics["Traffic"], 0.0)
        self.assertAlmostEqual(
            result.metrics["Tconnectivity"],
            1.0 / result.metrics["NCC"] + 1.0 / (1.0 + result.metrics["dead_ends"]),
        )

    def test_signal_log_probability_is_mean_over_junctions(self) -> None:
        distribution = Categorical(
            logits=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        )
        actions = torch.tensor([0, 1, 0])
        log_prob, entropy = reduce_distribution_stats(
            Task.SIGNAL, distribution, actions
        )
        self.assertTrue(
            torch.allclose(log_prob, distribution.log_prob(actions).mean())
        )
        self.assertTrue(torch.allclose(entropy, distribution.entropy().mean()))

    def test_label_smoothing_never_assigns_mass_to_illegal_actions(self) -> None:
        logits = torch.tensor(
            [[1.0, torch.finfo(torch.float32).min, -0.5]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, False, True]])
        loss = _masked_cross_entropy(
            logits,
            torch.tensor([0]),
            mask,
            smoothing=0.05,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 10.0)

    def test_road_set_loss_uses_total_positive_probability(self) -> None:
        logits = torch.tensor([0.2, 1.1, -0.3, 0.7])
        mask = torch.tensor([True, True, False, True])
        targets = torch.tensor([True, False, False, True])
        loss = _masked_set_cross_entropy(logits, targets, mask, smoothing=0.0)
        log_prob = logits.masked_fill(~mask, -torch.inf).log_softmax(dim=-1)
        expected = -torch.logsumexp(log_prob[targets], dim=0)
        self.assertTrue(torch.allclose(loss, expected))

        road_steps = list(ExpertDataset([self.graph]).by_task(Task.ROAD))
        self.assertTrue(all(step.target_mask is not None for step in road_steps))
        self.assertEqual(int(road_steps[0].target_mask.sum()), len(road_steps))
        self.assertEqual(int(road_steps[-1].target_mask.sum()), 1)

    def test_expert_relative_score_handles_rewards_and_costs(self) -> None:
        self.assertAlmostEqual(_expert_relative_score(0.8, 0.8), 1.0)
        self.assertAlmostEqual(_expert_relative_score(-100.0, -100.0), 1.0)
        self.assertGreater(_expert_relative_score(1.2, 1.0), 1.0)
        self.assertGreater(_expert_relative_score(-80.0, -100.0), 1.0)
        self.assertLess(_expert_relative_score(-120.0, -100.0), 1.0)

    def test_land_proposal_prefix_uses_plackett_luce_log_probability(self) -> None:
        logits = torch.tensor([0.2, 9.0, 1.1, -0.4])
        mask = torch.tensor([True, False, True, True])
        actual = ranked_action_log_prob(logits, mask, (2, 0))
        first = logits.masked_fill(~mask, -torch.inf).log_softmax(dim=0)[2]
        remaining = mask.clone()
        remaining[2] = False
        second = logits.masked_fill(~remaining, -torch.inf).log_softmax(dim=0)[0]
        self.assertTrue(torch.allclose(actual, first + second))

    def test_signal_ppo_clips_each_controller_before_averaging(self) -> None:
        old_log_prob = torch.zeros(2)
        log_prob = torch.log(torch.tensor([2.0, 0.5]))
        loss, _, clip_fraction = _ppo_policy_terms(
            log_prob,
            old_log_prob,
            torch.tensor(1.0),
            clip_ratio=0.2,
        )
        self.assertAlmostEqual(float(loss), -0.85, places=6)
        self.assertAlmostEqual(float(clip_fraction), 1.0, places=6)

    def test_every_training_topology_is_exactly_4km_square(self) -> None:
        for topology_path in DATA_ROOT.glob("*/topology.json"):
            topology = json.loads(topology_path.read_text(encoding="utf-8"))
            points = [
                point
                for parcel in topology["parcels"]
                for ring in parcel["geometry"]
                for point in ring
            ]
            width = max(point[0] for point in points) - min(point[0] for point in points)
            height = max(point[1] for point in points) - min(point[1] for point in points)
            self.assertAlmostEqual(width, 4000.0, places=2, msg=str(topology_path))
            self.assertAlmostEqual(height, 4000.0, places=2, msg=str(topology_path))


class TrainingLoggingContractTest(unittest.TestCase):
    def test_sft_test_best_is_not_overwritten_between_evaluations(self) -> None:
        graph = UrbanGraph(
            tile_id="unit",
            tile_dir=Path("."),
            features=torch.zeros((1, 32)),
            entity_types=torch.zeros(1, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            parcel_entities=torch.tensor([0]),
            road_entities=torch.empty(0, dtype=torch.long),
            intersection_entities=torch.empty(0, dtype=torch.long),
            signal_entities=torch.empty(0, dtype=torch.long),
            signal_action_mask=torch.empty((0, 8), dtype=torch.bool),
            raw_topology={"parcels": [{"area_m2": 1.0}]},
            signal_meta=None,
        )
        step = ExpertStep(
            state=PolicyState(
                graph=graph,
                task=Task.LAND,
                progress=0.0,
                history=torch.tensor([0]),
                dynamic=torch.zeros((1, 18)),
                action_mask=torch.ones(1, dtype=torch.bool),
                candidate_entities=torch.tensor([0]),
                query_action=0,
                decision_entities=torch.empty(0, dtype=torch.long),
                decision_actions=torch.empty(0, dtype=torch.long),
            ),
            action=0,
            return_to_go=1.0,
            return_scale=1.0,
        )

        class OneStepDataset:
            steps = [step]
            tasks = (Task.LAND,)

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> ExpertStep:
                return self.steps[index]

        logged_rows: list[dict[str, float]] = []

        class FakeLogger:
            def log(self, values, *, step=None) -> None:
                logged_rows.append(dict(values))

            def summary(self, values) -> None:
                pass

        model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        with TemporaryDirectory() as directory:
            trainer = SFTTrainer(
                model=model,
                dataset=OneStepDataset(),  # type: ignore[arg-type]
                config=SFTConfig(
                    epochs=4,
                    batch_size=1,
                    test_interval=2,
                ),
                logger=FakeLogger(),  # type: ignore[arg-type]
                output_dir=directory,
                full_config={},
                seed=0,
                epoch_evaluator=lambda epoch, checkpoint: {
                    "selection_score": {2: 0.8, 4: 0.7}[epoch]
                },
            )
            losses = iter((4.0, 3.0, 2.0, 1.0))
            best_saves: list[int] = []

            def fake_save(path, *, model, optimizer, step, config, metrics):
                if Path(path).name == "sft_best.pt":
                    best_saves.append(step)
                return Path(path)

            with (
                patch.object(
                    trainer,
                    "_train_epoch",
                    side_effect=lambda epoch: {
                        "sft/loss": next(losses),
                        "sft/land_loss": float(epoch),
                    },
                ),
                patch("upmodel.training.sft.save_checkpoint", side_effect=fake_save),
            ):
                trainer.train()

        self.assertEqual(best_saves, [2])
        self.assertEqual(
            [set(row) for row in logged_rows],
            [
                {
                    "sft/epoch",
                    "sft/epoch/loss",
                    "sft/epoch/land_loss",
                }
            ]
            * 4,
        )
        self.assertTrue(
            all(
                not {
                    "sft/loss",
                    "sft/land_loss",
                    "sft/road_loss",
                    "sft/signal_loss",
                }
                & set(row)
                for row in logged_rows
            )
        )

    def test_summary_gap_percentage_uses_task_mean_rewards(self) -> None:
        rows = [
            EvaluationRow(
                tile_id="small",
                task="land",
                policy_reward=0.5,
                expert_reward=1.0,
                reward_delta=-0.5,
                relative_gap_pct=50.0,
                episodes=1,
                policy_metrics={},
                expert_metrics={},
            ),
            EvaluationRow(
                tile_id="large",
                task="land",
                policy_reward=90.0,
                expert_reward=100.0,
                reward_delta=-10.0,
                relative_gap_pct=10.0,
                episodes=1,
                policy_metrics={},
                expert_metrics={},
            ),
        ]
        summary = _summarize(rows)["land"]
        expected = 100.0 * abs(45.25 - 50.5) / 50.5
        self.assertAlmostEqual(summary["relative_gap_pct"], expected)
        self.assertNotAlmostEqual(summary["relative_gap_pct"], 30.0)

    def test_summary_macro_averages_duplicate_topology_groups(self) -> None:
        rows = [
            EvaluationRow(
                tile_id="variant-a",
                task="land",
                policy_reward=0.0,
                expert_reward=2.0,
                reward_delta=-2.0,
                relative_gap_pct=100.0,
                episodes=1,
                policy_metrics={},
                expert_metrics={},
                group_key="shared",
            ),
            EvaluationRow(
                tile_id="variant-b",
                task="land",
                policy_reward=2.0,
                expert_reward=2.0,
                reward_delta=0.0,
                relative_gap_pct=0.0,
                episodes=1,
                policy_metrics={},
                expert_metrics={},
                group_key="shared",
            ),
            EvaluationRow(
                tile_id="independent",
                task="land",
                policy_reward=5.0,
                expert_reward=4.0,
                reward_delta=1.0,
                relative_gap_pct=25.0,
                episodes=1,
                policy_metrics={},
                expert_metrics={},
                group_key="independent",
            ),
        ]

        summary = _summarize(rows)["land"]

        self.assertAlmostEqual(summary["policy_reward"], 3.0)
        self.assertAlmostEqual(summary["expert_reward"], 3.0)
        self.assertEqual(summary["tiles"], 2.0)
        self.assertEqual(summary["members"], 3.0)

    def test_sft_loss_does_not_train_value_head(self) -> None:
        graph = UrbanGraph(
            tile_id="unit",
            tile_dir=Path("."),
            features=torch.zeros((1, 32)),
            entity_types=torch.zeros(1, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            parcel_entities=torch.tensor([0]),
            road_entities=torch.empty(0, dtype=torch.long),
            intersection_entities=torch.empty(0, dtype=torch.long),
            signal_entities=torch.empty(0, dtype=torch.long),
            signal_action_mask=torch.empty((0, 8), dtype=torch.bool),
            raw_topology={"parcels": [{"area_m2": 1.0}]},
            signal_meta=None,
        )
        step = ExpertStep(
            state=PolicyState(
                graph=graph,
                task=Task.LAND,
                progress=0.0,
                history=torch.tensor([0]),
                dynamic=torch.zeros((1, 18)),
                action_mask=torch.ones(1, dtype=torch.bool),
                candidate_entities=torch.tensor([0]),
                query_action=0,
                decision_entities=torch.empty(0, dtype=torch.long),
                decision_actions=torch.empty(0, dtype=torch.long),
            ),
            action=0,
            return_to_go=1.0,
            return_scale=1.0,
        )

        class OneStepDataset:
            steps = [step]
            tasks = (Task.LAND,)

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> ExpertStep:
                return self.steps[index]

        model = MultiTaskActorCritic(
            hidden_dim=32,
            entity_layers=1,
            decision_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        trainer = SFTTrainer(
            model=model,
            dataset=OneStepDataset(),  # type: ignore[arg-type]
            config=SFTConfig(epochs=1, batch_size=1),
            logger=object(),  # type: ignore[arg-type]
            output_dir=".",
            full_config={},
            seed=0,
        )
        trainer._sample_loss(step).backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.value_head.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.static_pointer_query.parameters()
            )
        )


def _actions(
    path: Path,
    index_column: str,
    action_column: str,
    mapping: dict[str, int] | None = None,
) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actions = [0] * len(rows)
    for row in rows:
        raw = row[action_column]
        actions[int(row[index_column])] = mapping[raw] if mapping is not None else int(raw)
    return actions


if __name__ == "__main__":
    unittest.main()
