from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from upmodel.config import TrainConfig


ROOT = Path(__file__).resolve().parents[1]


class TrainConfigCompatibilityTest(unittest.TestCase):
    def test_removed_all_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "stage must be one of: sft, ppo"
        ):
            self._load({"stage": "all"})

    def test_old_validation_names_migrate_to_test_config(self) -> None:
        config = self._load(
            {
                "sft": {"validation_interval": 3},
                "ppo": {
                    "validation_interval": 7,
                    "validation_episodes": 2,
                    "validation_deterministic": False,
                    "validation_topologies": 9,
                },
                "validation_topologies": 5,
                "split_seed": 123,
            }
        )

        self.assertEqual(config.sft.test_interval, 3)
        self.assertEqual(config.ppo.test_interval, 7)
        self.assertEqual(config.ppo.test_episodes, 2)
        self.assertFalse(config.ppo.test_deterministic)
        self.assertNotIn("validation", str(config.to_dict()))

    def test_conflicting_old_and_new_test_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts with"):
            self._load(
                {
                    "ppo": {
                        "validation_interval": 5,
                        "test_interval": 7,
                    },
                }
            )

    def test_signal_rollout_workers_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "ppo.signal_rollout_workers must be positive"
        ):
            self._load({"ppo": {"signal_rollout_workers": 0}})

    def test_road_centering_requires_a_frozen_baseline(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "frozen.*road_baseline_manifest.*required"
        ):
            self._load({"ppo": {"road_center_rewards": True}})

    def test_road_baseline_path_and_sha_are_atomic(self) -> None:
        for values in (
            {"road_baseline_manifest": "baseline.json"},
            {"road_baseline_manifest_sha256": "a" * 64},
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "configured together"):
                    self._load({"ppo": values})

    def test_road_centering_requires_normalized_rewards(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalize_rewards must be true"):
            self._load(
                {
                    "ppo": {
                        "road_baseline_manifest": "baseline.json",
                        "road_baseline_manifest_sha256": "a" * 64,
                        "road_center_rewards": True,
                        "normalize_rewards": False,
                    }
                }
            )

    def test_road_baseline_sha_must_be_lowercase_hex(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "road_baseline_manifest_sha256"
        ):
            self._load(
                {
                    "ppo": {
                        "road_baseline_manifest": "baseline.json",
                        "road_baseline_manifest_sha256": "INVALID",
                    }
                }
            )

    def test_paired_ppo_pilot_contract_is_frozen(self) -> None:
        config = TrainConfig.from_json(
            ROOT / "experiments/configs/paired-ppo-pilot-v1.json"
        )

        self.assertEqual(config.stage, "ppo")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.ppo.iterations, 10)
        self.assertEqual(
            config.ppo.episodes_per_task,
            {"land": 80, "road": 256, "signal": 48},
        )
        self.assertEqual(config.ppo.test_interval, 1)
        self.assertEqual(config.ppo.initial_policy_logit_scale, 1.0)
        self.assertEqual(config.ppo.signal_rollout_workers, 4)
        self.assertTrue(config.ppo.road_center_rewards)
        self.assertEqual(
            config.ppo.road_baseline_manifest_sha256,
            "57d7b4732f6bf7f580b21220e18bed27301ae529340cbc246987b9fdb99df021",
        )
        self.assertFalse(hasattr(config.evaluation, "enabled"))
        self.assertEqual(config.wandb.mode, "offline")

    def test_clean_sft_config_bytes_remain_frozen(self) -> None:
        path = ROOT / "experiments/configs/init-ablation-v1.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "e74bd45bd69db0cb47068fdbefcfd245e685a77d07b5671d52d602731c5cb14b",
        )

    def _load(self, values: dict) -> TrainConfig:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            return TrainConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
