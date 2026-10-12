from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from upmodel.config import TrainConfig
from upmodel.tasks import Task
from upmodel.train import (
    _checkpoint_initialization_record,
    _infer_checkpoint_initialization_type,
    _ppo_initialization_summary,
    _resolve_init_checkpoint,
    _write_data_snapshot,
    _write_initialization_manifest,
    build_parser,
    main,
    run,
)


class TrainInitializationTest(unittest.TestCase):
    def test_parser_rejects_removed_all_stage(self) -> None:
        with redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--stage", "all"])

        self.assertIn("invalid choice", stderr.getvalue())

    def test_parser_accepts_generic_init_checkpoint(self) -> None:
        args = build_parser().parse_args(
            ["--stage", "ppo", "--init-checkpoint", "model.pt"]
        )

        self.assertEqual(args.init_checkpoint, Path("model.pt"))
        self.assertIsNone(args.sft_checkpoint)

    def test_deprecated_split_seed_is_accepted_but_not_stored(self) -> None:
        config = TrainConfig(seed=1)
        with (
            patch.object(
                sys,
                "argv",
                [
                    "up-model-train",
                    "--seed",
                    "73",
                    "--split-seed",
                    "20260718",
                ],
            ),
            patch.object(TrainConfig, "from_json", return_value=config),
            patch("upmodel.train.configure_logging"),
            patch("upmodel.train.run", return_value={}) as run_mock,
        ):
            main()

        self.assertEqual(config.seed, 73)
        self.assertFalse(hasattr(config, "split_seed"))
        run_mock.assert_called_once()

    def test_output_guard_refuses_an_existing_directory(self) -> None:
        with TemporaryDirectory() as directory:
            config = TrainConfig(output_dir=directory)
            with (
                patch.object(
                    sys,
                    "argv",
                    ["up-model-train", "--fail-if-output-exists"],
                ),
                patch.object(TrainConfig, "from_json", return_value=config),
                patch("upmodel.train.configure_logging"),
                patch("upmodel.train.run") as run_mock,
            ):
                with self.assertRaisesRegex(
                    FileExistsError, "refusing to overwrite"
                ):
                    main()

            run_mock.assert_not_called()

    def test_parser_rejects_both_checkpoint_options(self) -> None:
        with redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "--init-checkpoint",
                        "generic.pt",
                        "--sft-checkpoint",
                        "legacy.pt",
                    ]
                )

        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_programmatic_checkpoint_options_conflict(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _resolve_init_checkpoint(
                init_checkpoint="generic.pt",
                sft_checkpoint="legacy.pt",
            )

    def test_external_checkpoint_is_only_allowed_for_ppo_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid with --stage ppo"):
            run(TrainConfig(stage="sft"), init_checkpoint="model.pt")

    def test_payload_inference_distinguishes_checkpoint_stages(self) -> None:
        self.assertEqual(
            _infer_checkpoint_initialization_type(
                {
                    "metrics": {"sft/loss": 0.25},
                    "config": {"stage": "sft"},
                }
            ),
            "sft",
        )
        self.assertEqual(
            _infer_checkpoint_initialization_type(
                {"metrics": {"ppo/iteration": 3.0}}
            ),
            "ppo",
        )
        self.assertEqual(
            _infer_checkpoint_initialization_type(
                {
                    "metrics": {"stage": "random_init"},
                    "config": {"stage": "sft"},
                }
            ),
            "random_init",
        )
        self.assertEqual(
            _infer_checkpoint_initialization_type(
                {"metrics": {}, "config": {"stage": "sft"}}
            ),
            "unknown_checkpoint",
        )

    def test_initialization_record_has_absolute_path_and_sha256(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sft_best.pt"
            payload = {
                "model": {"weight": torch.tensor([1.0, 2.0])},
                "metrics": {"sft/loss": 0.25},
                "config": {"stage": "sft"},
            }
            torch.save(payload, checkpoint)

            record = _checkpoint_initialization_record(
                checkpoint,
                payload,
                cli_option="--init-checkpoint",
            )
            manifest = _write_initialization_manifest(directory, record)
            saved = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(saved["type"], "sft")
            self.assertEqual(saved["checkpoint_path"], str(checkpoint.resolve()))
            self.assertTrue(Path(saved["checkpoint_path"]).is_absolute())
            self.assertEqual(
                saved["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(saved["cli_option"], "--init-checkpoint")

            summary = _ppo_initialization_summary(
                record,
                initial_policy_logit_scale=0.7,
            )
            self.assertEqual(summary["ppo/initialization"], "sft")
            self.assertEqual(
                summary["ppo/init_checkpoint_path"],
                str(checkpoint.resolve()),
            )
            self.assertEqual(
                summary["ppo/init_checkpoint_sha256"],
                saved["checkpoint_sha256"],
            )
            self.assertEqual(
                summary["ppo/initial_policy_logit_scale"], 0.7
            )

    def test_data_snapshot_hashes_consumed_artifact_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tile_dir = root / "tile"
            first_output = root / "first"
            second_output = root / "second"
            tile_dir.mkdir()
            first_output.mkdir()
            second_output.mkdir()
            for relative_path, content in (
                ("topology.json", "{}"),
                ("land_use_solution.csv", "parcel_id,land_use\n0,res\n"),
                ("land_use_trajectory.json", '{"final_reward": 1.0}'),
            ):
                (tile_dir / relative_path).write_text(content, encoding="utf-8")
            graph = SimpleNamespace(
                tile_id="tile-a",
                tile_dir=tile_dir,
                signal_entities=torch.empty(0, dtype=torch.long),
                signal_meta=None,
            )

            first_path, first = _write_data_snapshot(
                first_output,
                {"training": [graph]},
                tasks=(Task.LAND,),
            )
            (tile_dir / "land_use_trajectory.json").write_text(
                '{"final_reward": 2.0}', encoding="utf-8"
            )
            _, second = _write_data_snapshot(
                second_output,
                {"training": [graph]},
                tasks=(Task.LAND,),
            )

            saved = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["aggregate_sha256"], first["aggregate_sha256"])
            self.assertEqual(saved["algorithm"], "sha256")
            self.assertEqual(
                {record["relative_path"] for record in saved["files"]},
                {
                    "topology.json",
                    "land_use_solution.csv",
                    "land_use_trajectory.json",
                },
            )
            self.assertNotEqual(
                first["aggregate_sha256"], second["aggregate_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
