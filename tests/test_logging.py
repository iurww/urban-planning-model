from __future__ import annotations

import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from upmodel.config import TrainConfig
from upmodel.logging import RunLogger


class EvaluationLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = RunLogger.__new__(RunLogger)
        self.logger.log = Mock()

    def test_test_event_contains_only_rewards_score_and_explicit_axes(self) -> None:
        summary = {
            "land": {
                "policy_reward": 1.5,
                "expert_reward": 2.0,
            },
            "signal": {
                "policy_reward": -8.0,
                "expert_reward": -10.0,
            },
        }

        score = self.logger.log_test_evaluation(
            summary,
            step=12,
            stage="ppo",
            stage_step=3,
        )

        payload = self.logger.log.call_args.args[0]
        self.assertEqual(
            payload,
            {
                "test/training_step": 12.0,
                "test/stage_step": 3.0,
                "test/land_reward": 1.5,
                "test/traffic_reward": -8.0,
                "test/selection_score": 0.975,
                "test/stage": "ppo",
            },
        )
        self.assertAlmostEqual(score, 0.975)

    def test_validation_logging_is_rejected(self) -> None:
        report = SimpleNamespace(summary={})
        with self.assertRaisesRegex(ValueError, "no longer has a validation"):
            self.logger.log_evaluation(report, split="validation")

    def test_metric_families_use_one_iteration_axis(self) -> None:
        wandb = Mock()
        run = Mock()
        run.url = None
        wandb.init.return_value = run
        with TemporaryDirectory() as directory:
            config = TrainConfig(output_dir=directory)
            with patch.dict(sys.modules, {"wandb": wandb}):
                logger = RunLogger(config)

        wandb.define_metric.assert_any_call("ppo/iteration")
        wandb.define_metric.assert_any_call(
            "rollout/*", step_metric="ppo/iteration"
        )
        wandb.define_metric.assert_any_call(
            "ppo/policy_loss", step_metric="ppo/iteration"
        )
        self.assertFalse(
            any(
                call.args and call.args[0] == "train/*"
                for call in wandb.define_metric.call_args_list
            )
        )
        wandb.define_metric.assert_any_call("test/training_step")
        wandb.define_metric.assert_any_call(
            "test/land_reward", step_metric="test/training_step"
        )
        wandb.define_metric.assert_any_call(
            "test/road_reward", step_metric="test/training_step"
        )
        wandb.define_metric.assert_any_call(
            "test/traffic_reward", step_metric="test/training_step"
        )
        logger.finish()


if __name__ == "__main__":
    unittest.main()
