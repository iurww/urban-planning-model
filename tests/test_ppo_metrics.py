from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from upmodel.config import PPOConfig
from upmodel.model import PolicyOutput
from upmodel.state import PolicyState
from upmodel.tasks import Task
from upmodel.training.buffer import Transition
from upmodel.training.ppo import PPOTrainer, _explained_variance


class _UpdateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.decision_transformer = torch.nn.Module()
        self.decision_transformer.signal_history_gru = torch.nn.GRU(
            1, 1, batch_first=True
        )

    @property
    def device(self) -> torch.device:
        return self.anchor.device


class _Logger:
    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def log(self, values: dict[str, float], *, step=None) -> None:
        del step
        self.rows.append(dict(values))


class _Buffer:
    def __init__(self, transitions: list[Transition]) -> None:
        self.transitions = transitions

    def flattened(self, *, normalize: bool = True) -> list[Transition]:
        self.normalize = normalize
        return list(self.transitions)


class _NoShuffle:
    @staticmethod
    def shuffle(values) -> None:
        del values


class PPOOptimizerMetricsTest(unittest.TestCase):
    def test_train_logs_one_combined_event_per_iteration(self) -> None:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = Mock()
        trainer.optimizer = SimpleNamespace(
            param_groups=[{"lr": 0.0}]
        )
        trainer.config = PPOConfig(
            iterations=2,
            enabled_tasks=["land"],
            episodes_per_task={"land": 1},
            test_interval=2,
        )
        trainer.logger = Mock()
        trainer.logger.log_test_evaluation.side_effect = [0.0, 2.0]
        trainer.log_step_offset = 0
        trainer.full_config = {}
        trainer._learning_rate = lambda iteration: float(iteration)
        trainer._kl_beta = lambda iteration: 0.0
        trainer._collect = lambda iteration, beta: (
            object(),
            {
                "rollout/total_reward_mean": float(iteration),
                "rollout/land_reward_mean": float(iteration),
                "rollout/fps": 1.0,
                "_rollout_steps": 1.0,
            },
        )
        trainer._update = lambda buffer: {
            "ppo/policy_loss": 0.1,
            "ppo/value_loss": 0.2,
        }
        trainer._evaluate_test = lambda iteration: (
            {"test/selection_score": float(iteration)},
            {},
        )

        with (
            TemporaryDirectory() as directory,
            patch(
                "upmodel.training.ppo.save_checkpoint",
                side_effect=lambda path, **kwargs: Path(path),
            ),
        ):
            trainer.output_dir = Path(directory)
            trainer.train()

        rows = [call.args[0] for call in trainer.logger.log.call_args_list]
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["ppo/iteration"] for row in rows],
            [1.0, 2.0],
        )
        for row in rows:
            self.assertIn("rollout/total_reward_mean", row)
            self.assertIn("ppo/policy_loss", row)
            self.assertNotIn("train/policy_loss", row)

    def test_explained_variance_matches_definition(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0])
        predictions = torch.tensor([1.0, 1.0, 1.0])
        self.assertAlmostEqual(
            _explained_variance(returns, predictions),
            0.0,
        )
        self.assertAlmostEqual(
            _explained_variance(returns, returns),
            1.0,
        )

    def test_update_returns_one_summary_without_minibatch_logging(self) -> None:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = _UpdateModel()
        trainer.config = PPOConfig(
            enabled_tasks=["land", "road", "signal"],
            episodes_per_task={"land": 1, "road": 1, "signal": 1},
            update_epochs=1,
            minibatch_size=2,
            task_balanced=False,
            target_kl=10.0,
        )
        trainer.logger = _Logger()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.optimizer_step = 0
        trainer.optimization_random = _NoShuffle()
        transitions = [
            _transition(Task.LAND, progress=0.0, target=1.0, advantage=0.5),
            _transition(Task.ROAD, progress=1.0, target=1.0, advantage=-0.5),
            _transition(Task.SIGNAL, progress=2.0, target=0.0, advantage=1.0),
        ]
        buffer = _Buffer(transitions)

        def forward(items: list[Transition]) -> list[PolicyOutput]:
            outputs = []
            zero = trainer.model.anchor * 0.0
            for transition in items:
                logits = (
                    torch.stack((zero, zero)).unsqueeze(0)
                    if transition.task == Task.SIGNAL
                    else torch.stack((zero, zero))
                )
                outputs.append(
                    PolicyOutput(
                        logits=logits,
                        value=zero + float(transition.state.progress),
                    )
                )
            return outputs

        trainer._forward_minibatch = forward
        metrics = trainer._update(buffer)  # type: ignore[arg-type]

        self.assertTrue(buffer.normalize)
        self.assertAlmostEqual(metrics["ppo/value_loss"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["ppo/approx_kl"], 0.0)
        self.assertAlmostEqual(metrics["ppo/clip_fraction"], 0.0)
        self.assertAlmostEqual(metrics["ppo/explained_variance"], -6.0)
        self.assertAlmostEqual(
            metrics["ppo/entropy_loss"],
            -trainer.config.entropy_coef * math.log(2.0),
        )
        self.assertIn("ppo/grad_norm_before_clip", metrics)
        self.assertIn("ppo/grad_norm_after_clip", metrics)
        self.assertFalse(any(key.startswith("train/") for key in metrics))
        self.assertEqual(len(trainer.logger.rows), 0)


def _transition(
    task: Task,
    *,
    progress: float,
    target: float,
    advantage: float,
) -> Transition:
    signal = task == Task.SIGNAL
    state = PolicyState(
        graph=SimpleNamespace(),
        task=task,
        progress=progress,
        history=torch.empty(0, dtype=torch.long),
        dynamic=torch.zeros((1, 18)),
        action_mask=(
            torch.ones((1, 2), dtype=torch.bool)
            if signal
            else torch.ones(2, dtype=torch.bool)
        ),
        signal_entities=(torch.tensor([0]) if signal else None),
    )
    return Transition(
        state=state,
        action=torch.tensor([0]) if signal else 0,
        old_log_prob=(
            torch.tensor([-math.log(2.0)]) if signal else -math.log(2.0)
        ),
        old_value=progress,
        reward=0.0,
        raw_reward=0.0,
        done=True,
        task=task,
        advantage=advantage,
        return_target=target,
    )


if __name__ == "__main__":
    unittest.main()
