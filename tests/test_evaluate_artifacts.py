from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from upmodel.evaluate import build_parser, evaluate_checkpoint, ledger_path_for
from upmodel.evaluation import EvaluationReport
from upmodel.tasks import Task


class EvaluationArtifactTest(unittest.TestCase):
    def test_report_json_refuses_to_replace_existing_file(self) -> None:
        report = EvaluationReport(
            checkpoint="checkpoint.pt",
            deterministic=True,
            rows=[],
            summary={},
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            output.write_text("original", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                report.write_json(output, fail_if_exists=True)

            self.assertEqual(output.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(output.parent.glob(".evaluation.json.*.tmp")), [])

    def test_report_json_replaces_existing_file_by_default(self) -> None:
        report = EvaluationReport(
            checkpoint="new.pt",
            deterministic=True,
            rows=[],
            summary={},
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            output.write_text("old", encoding="utf-8")

            report.write_json(output)

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                report.to_dict(),
            )

    def test_atomic_write_preserves_old_report_if_replace_fails(self) -> None:
        report = EvaluationReport(
            checkpoint="new.pt",
            deterministic=True,
            rows=[],
            summary={},
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            output.write_text("old", encoding="utf-8")

            with patch(
                "upmodel.evaluation.evaluator.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    report.write_json(output)

            self.assertEqual(output.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(output.parent.glob(".evaluation.json.*.tmp")), [])

    def test_evaluate_writes_provenance_ledger_next_to_report(self) -> None:
        report = EvaluationReport(
            checkpoint="checkpoint.pt",
            deterministic=True,
            rows=[],
            summary={},
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint bytes")
            data_root = root / "final-test"
            data_root.mkdir()
            output = root / "formal.json"
            evaluator = Mock()
            evaluator.evaluate.return_value = report

            with (
                patch(
                    "upmodel.evaluate.torch.load",
                    return_value={"model": {}},
                ),
                patch("upmodel.evaluate.MultiTaskActorCritic", return_value=Mock()),
                patch(
                    "upmodel.evaluate.load_graphs",
                    return_value=[SimpleNamespace(tile_id="tile-a")],
                ),
                patch("upmodel.evaluate.PolicyEvaluator", return_value=evaluator),
                patch("upmodel.evaluate._git_commit", return_value="a" * 40),
            ):
                evaluate_checkpoint(
                    checkpoint=checkpoint,
                    data_root=data_root,
                    tasks=[Task.LAND, Task.SIGNAL],
                    output=output,
                    device="cpu",
                    episodes=3,
                    deterministic=False,
                    seed=17,
                    tile_ids=["tile-a"],
                    command="up-model-eval --formal-example",
                )

            ledger = json.loads(
                ledger_path_for(output).read_text(encoding="utf-8")
            )
            self.assertEqual(
                ledger["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint bytes").hexdigest(),
            )
            self.assertEqual(ledger["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(ledger["data_root"], str(data_root.resolve()))
            self.assertEqual(ledger["git_commit"], "a" * 40)
            self.assertEqual(ledger["tasks"], ["land", "signal"])
            self.assertEqual(ledger["episodes"], 3)
            self.assertEqual(ledger["seed"], 17)
            self.assertFalse(ledger["deterministic"])
            self.assertEqual(ledger["tile_ids"], ["tile-a"])
            self.assertEqual(ledger["requested_tile_ids"], ["tile-a"])
            self.assertEqual(ledger["command"], "up-model-eval --formal-example")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report.to_dict())

    def test_fail_if_output_exists_checks_report_and_ledger_before_loading(self) -> None:
        for existing_artifact in ("report", "ledger"):
            with self.subTest(existing_artifact=existing_artifact):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    checkpoint = root / "checkpoint.pt"
                    checkpoint.write_bytes(b"checkpoint")
                    output = root / "evaluation.json"
                    existing = (
                        output
                        if existing_artifact == "report"
                        else ledger_path_for(output)
                    )
                    existing.write_text("existing", encoding="utf-8")

                    with patch("upmodel.evaluate.torch.load") as load:
                        with self.assertRaisesRegex(
                            FileExistsError, "evaluation artifact"
                        ):
                            evaluate_checkpoint(
                                checkpoint=checkpoint,
                                data_root=root,
                                tasks=[Task.LAND],
                                output=output,
                                fail_if_output_exists=True,
                            )

                    load.assert_not_called()
                    self.assertEqual(existing.read_text(encoding="utf-8"), "existing")

    def test_cli_flag_is_opt_in(self) -> None:
        default_args = build_parser().parse_args(["--checkpoint", "model.pt"])
        guarded_args = build_parser().parse_args(
            ["--checkpoint", "model.pt", "--fail-if-output-exists"]
        )

        self.assertFalse(default_args.fail_if_output_exists)
        self.assertTrue(guarded_args.fail_if_output_exists)


if __name__ == "__main__":
    unittest.main()
