"""CLI for comparing a trained policy checkpoint with expert trajectories."""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import torch

from upmodel.data import load_graphs
from upmodel.evaluation import PolicyEvaluator
from upmodel.evaluation.evaluator import write_json_atomic
from upmodel.evaluation.reporting import log_report_to_wandb, print_report
from upmodel.logging import configure_logging
from upmodel.model import MultiTaskActorCritic
from upmodel.tasks import Task


LOGGER = logging.getLogger(__name__)


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    data_root: str | Path,
    tasks: list[Task],
    output: str | Path,
    device: str = "auto",
    episodes: int = 1,
    deterministic: bool = True,
    seed: int = 0,
    tile_ids: list[str] | None = None,
    fail_if_output_exists: bool = False,
    command: str | None = None,
):
    checkpoint = Path(checkpoint)
    data_root = Path(data_root)
    output = Path(output)
    ledger_output = ledger_path_for(output)
    if fail_if_output_exists:
        _require_new_outputs(output, ledger_output)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    git_commit = _git_commit()
    resolved_device = _device(device)
    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=False,
    )
    saved_config = payload.get("config", {})
    model_config = saved_config.get("model", {})
    kwargs = {
        "hidden_dim": int(model_config.get("hidden_dim", 128)),
        "entity_layers": int(model_config.get("entity_layers", 2)),
        "decision_layers": int(model_config.get("decision_layers", 2)),
        "attention_heads": int(model_config.get("attention_heads", 4)),
        "dropout": float(model_config.get("dropout", 0.1)),
    }
    if "num_layers" in model_config and "entity_layers" not in model_config:
        kwargs["num_layers"] = int(model_config["num_layers"])
    model = MultiTaskActorCritic(**kwargs).to(resolved_device)
    LOGGER.info("加载 evaluate checkpoint：%s", checkpoint)
    model.load_state_dict(payload["model"])
    LOGGER.info("开始加载 evaluate 数据：%s", data_root)
    graphs = load_graphs(data_root, show_progress=True)
    if tile_ids:
        requested = set(tile_ids)
        graphs = [graph for graph in graphs if graph.tile_id in requested]
        missing = requested - {graph.tile_id for graph in graphs}
        if missing:
            raise ValueError(f"requested tile_ids not found: {sorted(missing)}")
    LOGGER.info("evaluate 数据加载完成：graphs=%d", len(graphs))
    evaluator = PolicyEvaluator(
        model=model,
        graphs=graphs,
        checkpoint=checkpoint,
        seed=seed,
        land_use_order=saved_config.get("land_use_order", ()),
        signal_history_steps=int(model_config.get("signal_history_steps", 4)),
    )
    report = evaluator.evaluate(
        tasks=tasks,
        episodes=episodes,
        deterministic=deterministic,
    )
    report.write_json(output, fail_if_exists=fail_if_output_exists)
    write_json_atomic(
        ledger_output,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_path": str(output.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "data_root": str(data_root.resolve()),
            "git_commit": git_commit,
            "tasks": [task.label for task in tasks],
            "episodes": int(episodes),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "tile_ids": [graph.tile_id for graph in graphs],
            "requested_tile_ids": None if tile_ids is None else list(tile_ids),
            "command": command if command is not None else shlex.join(sys.argv),
        },
        fail_if_exists=fail_if_output_exists,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a policy checkpoint with expert trajectory baselines"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/final"))
    parser.add_argument("--tile-ids", nargs="+", default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("land", "road", "signal"),
        default=["land", "road", "signal"],
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="SUMO seed; default matches the provided expert trajectory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/upmodel/evaluation.json"))
    parser.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        help="fail before evaluation if the report or its ledger already exists",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="up-model-ppo")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    report = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        tasks=[Task.from_name(name) for name in args.tasks],
        output=args.output,
        device=args.device,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        tile_ids=args.tile_ids,
        fail_if_output_exists=args.fail_if_output_exists,
        command=shlex.join(sys.argv),
    )
    print_report(report)
    LOGGER.info("evaluate JSON 已保存：%s", args.output)
    LOGGER.info("evaluate ledger 已保存：%s", ledger_path_for(args.output))
    log_report_to_wandb(
        report,
        mode=args.wandb_mode,
        project=args.wandb_project,
        name=args.wandb_name,
    )


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def ledger_path_for(output: str | Path) -> Path:
    return Path(output).with_suffix(".ledger.json")


def _require_new_outputs(*paths: Path) -> None:
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite evaluation artifact: {path}")


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) not in (40, 64):
        return None
    try:
        int(commit, 16)
    except ValueError:
        return None
    return commit


if __name__ == "__main__":
    main()
