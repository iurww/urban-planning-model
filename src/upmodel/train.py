"""Command-line orchestration for SFT pretraining and multi-task PPO."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from upmodel.config import TrainConfig
from upmodel.data import (
    canonical_road_fingerprint,
    ExpertDataset,
    GraphSplits,
    group_aware_splits,
    infer_land_use_order,
    load_graphs,
    topology_group_key,
)
from upmodel.evaluation import PolicyEvaluator
from upmodel.logging import RunLogger, add_file_logging, configure_logging
from upmodel.model import MultiTaskActorCritic
from upmodel.tasks import TRAINING_TASKS, Task
from upmodel.training import PPOTrainer, SFTTrainer
from upmodel.training.checkpoint import load_checkpoint, save_checkpoint
from upmodel.training.road_baseline import (
    index_road_baseline_manifest,
    load_road_baseline_manifest,
)


LOGGER = logging.getLogger(__name__)


def run(
    config: TrainConfig,
    *,
    init_checkpoint: str | Path | None = None,
    sft_checkpoint: str | Path | None = None,
) -> dict[str, Path]:
    checkpoint_path, checkpoint_option = _resolve_init_checkpoint(
        init_checkpoint=init_checkpoint,
        sft_checkpoint=sft_checkpoint,
    )
    if checkpoint_path is not None and config.stage != "ppo":
        raise ValueError(
            f"{checkpoint_option} is only valid with --stage ppo; "
            f"stage={config.stage!r} manages its own initialization"
        )
    config.validate()
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = add_file_logging(output_dir / "train.log")
    LOGGER.info(
        "训练流程初始化：stage=%s device=%s data_root=%s output_dir=%s",
        config.stage,
        device,
        config.data_root,
        output_dir,
    )
    LOGGER.info("训练日志写入：%s", log_path)
    LOGGER.info("开始加载训练拓扑池")
    training_pool = load_graphs(config.data_root, show_progress=True)
    if config.tile_ids:
        requested = set(config.tile_ids)
        training_pool = [
            graph for graph in training_pool if graph.tile_id in requested
        ]
        missing = requested - {graph.tile_id for graph in training_pool}
        if missing:
            raise ValueError(f"requested tile_ids not found: {sorted(missing)}")
        if not training_pool:
            raise ValueError("tile_ids filter selected no topologies")
    LOGGER.info("训练拓扑池加载完毕：graphs=%d", len(training_pool))
    LOGGER.info("开始加载测试拓扑数据：test_data_root=%s", config.test_data_root)
    test_graphs = load_graphs(config.test_data_root, show_progress=True)
    if config.tile_ids:
        # Explicit single-topology diagnostics intentionally reuse the same
        # topology; normal runs always use the disjoint data/test split.
        splits = GraphSplits(
            training=training_pool,
            test=training_pool,
            excluded_test_duplicates=[],
        )
    else:
        splits = group_aware_splits(training_pool, test_graphs)
    graphs = splits.training
    test_graphs = splits.test
    split_manifest = splits.manifest()
    split_manifest["diagnostic_reuse"] = bool(config.tile_ids)
    split_manifest["data_root"] = str(Path(config.data_root).resolve())
    split_manifest["test_data_root"] = str(
        Path(config.test_data_root).resolve()
    )
    data_split_path = output_dir / "data_split.json"
    data_split_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshot_tasks = (
        TRAINING_TASKS
        if config.stage == "sft"
        else tuple(Task.from_name(name) for name in config.ppo.enabled_tasks)
    )
    data_snapshot_path, data_snapshot = _write_data_snapshot(
        output_dir,
        {"training": graphs, "test": test_graphs},
        tasks=snapshot_tasks,
    )
    road_baseline_path, road_baseline_rows = _load_road_baseline_contract(
        config,
        graphs,
    )
    road_baseline_snapshot_path: Path | None = None
    if road_baseline_path is not None:
        road_baseline_snapshot_path = output_dir / "road_baseline_manifest.json"
        shutil.copyfile(road_baseline_path, road_baseline_snapshot_path)
    LOGGER.info(
        "数据归属确定：train=%d test=%d excluded_test_duplicates=%d "
        "（未划分 validation）manifest=%s",
        len(graphs),
        len(test_graphs),
        len(splits.excluded_test_duplicates),
        data_split_path,
    )
    if splits.excluded_test_duplicates:
        LOGGER.warning(
            "训练池中的以下样本与 test 表示同一地理范围、等价路网或完全相同的"
            "模型输入，因此从训练中排除以避免测试泄漏：%s",
            ",".join(
                graph.tile_id for graph in splits.excluded_test_duplicates
            ),
        )
    if not config.land_use_order:
        config.land_use_order = list(infer_land_use_order(graphs))
    config.validate()
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("土地 fixed-what 类型顺序：%s", ",".join(config.land_use_order))
    dataset: ExpertDataset | None = None
    if config.stage == "sft":
        signal_gamma = float(config.ppo.gamma.get("signal", 0.99))
        LOGGER.info("开始加载专家轨迹")
        dataset = ExpertDataset(
            graphs,
            tasks=TRAINING_TASKS,
            land_use_order=config.land_use_order,
            signal_gamma=signal_gamma,
            signal_history_steps=config.model.signal_history_steps,
            road_permutations=config.sft.road_permutations,
            seed=config.seed,
            show_progress=True,
        )
        LOGGER.info(
            "专家轨迹加载完毕：steps=%d summary=%s",
            len(dataset),
            dataset.summary(),
        )
    else:
        LOGGER.info(
            "纯 PPO 阶段跳过 SFT 专家 step 展开；按拓扑读取专家回报用于归一化"
        )
    initialization_description = (
        "checkpoint (type inferred from payload)"
        if config.stage == "ppo" and checkpoint_path is not None
        else "random"
        if config.stage == "ppo"
        else "random SFT"
    )
    model = MultiTaskActorCritic(
        hidden_dim=config.model.hidden_dim,
        entity_layers=config.model.entity_layers,
        decision_layers=config.model.decision_layers,
        attention_heads=config.model.attention_heads,
        dropout=config.model.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "两级 Transformer 初始化完成：entity_layers=%d decision_layers=%d "
        "attention_heads=%d hidden_dim=%d parameters=%d initialization=%s",
        config.model.entity_layers,
        config.model.decision_layers,
        config.model.attention_heads,
        config.model.hidden_dim,
        parameter_count,
        initialization_description,
    )
    logger = RunLogger(config)
    LOGGER.info("W&B run 已初始化：mode=%s url=%s", config.wandb.mode, logger.run_url)
    artifacts: dict[str, Path] = {
        "data_split": data_split_path,
        "data_snapshot": data_snapshot_path,
    }
    if road_baseline_snapshot_path is not None:
        artifacts["road_baseline_manifest"] = road_baseline_snapshot_path
    initialization_record: dict[str, str | None] = {
        "type": "not_applicable",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "cli_option": None,
    }
    soften_actor_prior = False
    initialization_summary = _ppo_initialization_summary(
        initialization_record,
        initial_policy_logit_scale=1.0,
    )
    ppo: PPOTrainer | None = None
    try:
        if config.stage == "sft":
            assert dataset is not None
            artifacts["random_init"] = save_checkpoint(
                output_dir / "random_init.pt",
                model=model,
                optimizer=None,
                step=0,
                config=config.to_dict(),
                metrics={"stage": "random_init"},
            )

            def evaluate_sft_epoch(epoch: int, checkpoint: Path) -> dict[str, float]:
                result_path, evaluation_metrics = _evaluate_stage(
                    model=model,
                    graphs=test_graphs,
                    checkpoint=checkpoint,
                    stage=f"sft_epoch_{epoch:03d}",
                    config=config,
                    logger=logger,
                    output_dir=output_dir,
                    step=epoch,
                    stage_name="sft",
                    stage_step=epoch,
                )
                artifacts["sft_test_evaluation"] = result_path
                return evaluation_metrics

            sft = SFTTrainer(
                model=model,
                dataset=dataset,
                config=config.sft,
                logger=logger,
                output_dir=output_dir,
                full_config=config.to_dict(),
                seed=config.seed,
                epoch_evaluator=evaluate_sft_epoch,
            )
            artifacts["sft"] = sft.train()
        elif config.stage == "ppo":
            if checkpoint_path is not None:
                resolved_checkpoint = checkpoint_path.expanduser().resolve()
                artifacts["init_checkpoint"] = resolved_checkpoint
                checkpoint_payload = load_checkpoint(
                    resolved_checkpoint, model=model, map_location=device
                )
                if checkpoint_option == "--sft-checkpoint":
                    # Preserve the old programmatic return key while keeping
                    # the generic checkpoint artifact accurately named.
                    artifacts["sft"] = resolved_checkpoint

        if config.stage == "ppo":
            if checkpoint_path is not None:
                initialization_record = _checkpoint_initialization_record(
                    artifacts["init_checkpoint"],
                    checkpoint_payload,
                    cli_option=checkpoint_option,
                )
            else:
                initialization_record = {
                    "type": "random",
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "cli_option": None,
                }
            initialization_path = _write_initialization_manifest(
                output_dir, initialization_record
            )
            artifacts["initialization"] = initialization_path
            LOGGER.info(
                "PPO 初始化已记录：type=%s checkpoint=%s sha256=%s manifest=%s",
                initialization_record["type"],
                initialization_record["checkpoint_path"],
                initialization_record["checkpoint_sha256"],
                initialization_path,
            )
            soften_actor_prior = initialization_record["type"] == "sft"
            initialization_summary = _ppo_initialization_summary(
                initialization_record,
                initial_policy_logit_scale=(
                    config.ppo.initial_policy_logit_scale
                    if soften_actor_prior
                    else 1.0
                ),
            )
            # Record provenance before the potentially long PPO stage, so an
            # interrupted run still identifies the exact starting weights.
            logger.summary(initialization_summary)
            if soften_actor_prior:
                model.soften_actor_prior_(config.ppo.initial_policy_logit_scale)
                LOGGER.info(
                    "检测到 SFT 初始化，PPO 起始策略先验软化："
                    "logit_scale=%.3f，greedy 动作不变，仅提高采样探索熵",
                    config.ppo.initial_policy_logit_scale,
                )
            else:
                LOGGER.info(
                    "PPO 初始化类型为 %s，不应用 SFT actor 先验软化",
                    initialization_record["type"],
                )
            reference = copy.deepcopy(model).to(device)
            ppo = PPOTrainer(
                model=model,
                reference=reference,
                graphs=graphs,
                test_graphs=test_graphs,
                config=config.ppo,
                logger=logger,
                output_dir=output_dir,
                full_config=config.to_dict(),
                seed=config.seed,
                land_use_order=config.land_use_order,
                signal_history_steps=config.model.signal_history_steps,
                log_step_offset=0,
                road_baseline_rows=road_baseline_rows,
            )
            artifacts["ppo"] = ppo.train()
            load_checkpoint(artifacts["ppo"], model=model, map_location=device)
        summary = {
            "data/graphs": len(graphs),
            "data/test_graphs": len(test_graphs),
            "data/snapshot_sha256": data_snapshot["aggregate_sha256"],
            "data/excluded_test_duplicates": len(
                splits.excluded_test_duplicates
            ),
            "model/parameters": parameter_count,
            **initialization_summary,
        }
        if dataset is not None:
            summary["data/expert_steps"] = len(dataset)
            summary.update(
                {
                    f"data/{key}_steps": value
                    for key, value in dataset.summary().items()
                }
            )
        if road_baseline_path is not None:
            summary["ppo/road_baseline_manifest_sha256"] = (
                config.ppo.road_baseline_manifest_sha256
            )
        if ppo is not None and ppo.road_training_expert_scale is not None:
            summary["ppo/road_training_expert_scale"] = (
                ppo.road_training_expert_scale
            )
        logger.summary(summary)
        return artifacts
    finally:
        LOGGER.info("训练流程结束，关闭 W&B run")
        logger.finish()


def _evaluate_stage(
    *,
    model: MultiTaskActorCritic,
    graphs,
    checkpoint: Path,
    stage: str,
    config: TrainConfig,
    logger: RunLogger,
    output_dir: Path,
    step: int,
    stage_name: str,
    stage_step: int,
) -> tuple[Path, dict[str, float]]:
    tasks = [Task.from_name(name) for name in config.evaluation.tasks]
    result_path = output_dir / f"evaluation_test_{stage}.json"
    LOGGER.info(
        "开始 %s checkpoint test 推理：checkpoint=%s tasks=%s",
        stage,
        checkpoint,
        ",".join(task.label for task in tasks),
    )
    evaluator = PolicyEvaluator(
        model=model,
        graphs=graphs,
        checkpoint=checkpoint,
        seed=config.evaluation.seed,
        land_use_order=config.land_use_order,
        signal_history_steps=config.model.signal_history_steps,
    )
    report = evaluator.evaluate(
        tasks=tasks,
        episodes=config.evaluation.episodes,
        deterministic=config.evaluation.deterministic,
        show_progress=True,
    )
    report.write_json(result_path)
    evaluation_metrics = logger.log_evaluation(
        report,
        step=step,
        split="test",
        stage=stage_name,
        stage_step=stage_step,
    )
    LOGGER.info(
        "%s evaluate 完成并写入 W&B：result=%s summary=%s",
        stage,
        result_path,
        report.summary,
    )
    return result_path, evaluation_metrics


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_data_snapshot(
    output_dir: str | Path,
    split_graphs: dict[str, list],
    *,
    tasks: tuple[Task, ...],
) -> tuple[Path, dict[str, Any]]:
    """Hash every training/test artifact consumed by this run."""
    records: list[dict[str, str]] = []
    for split, graphs in sorted(split_graphs.items()):
        for graph in sorted(graphs, key=lambda item: item.tile_id):
            for relative_path in _training_artifact_paths(graph, tasks):
                path = graph.tile_dir / relative_path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{graph.tile_id}: required training artifact is missing: {path}"
                    )
                records.append(
                    {
                        "split": split,
                        "tile_id": graph.tile_id,
                        "relative_path": relative_path.as_posix(),
                        "sha256": _file_sha256(path),
                    }
                )
    records.sort(
        key=lambda item: (
            item["split"],
            item["tile_id"],
            item["relative_path"],
        )
    )
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(record["split"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["tile_id"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["relative_path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "sha256",
        "aggregate_encoding": (
            "sorted split + NUL + tile_id + NUL + relative_path + NUL + "
            "file_sha256 + LF"
        ),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }
    path = Path(output_dir) / "data_snapshot.json"
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path, snapshot


def _load_road_baseline_contract(
    config: TrainConfig,
    training_graphs: list,
) -> tuple[Path | None, dict[str, dict[str, Any]] | None]:
    """Load a frozen Road baseline and prove complete training coverage."""
    configured_path = config.ppo.road_baseline_manifest
    if configured_path is None:
        return None, None
    path = Path(configured_path).expanduser().resolve()
    manifest = load_road_baseline_manifest(path)
    recorded_sha = str(manifest["aggregate_sha256"])
    expected_sha = config.ppo.road_baseline_manifest_sha256
    if recorded_sha != expected_sha:
        raise ValueError(
            "Road baseline manifest SHA-256 mismatch: "
            f"expected={expected_sha} recorded={recorded_sha} path={path}"
        )

    rows = index_road_baseline_manifest(manifest)
    expected_groups = {
        topology_group_key(graph) for graph in training_graphs
    }
    for graph in training_graphs:
        group_key = topology_group_key(graph)
        row = rows.get(group_key)
        if row is None or row["road_fingerprint"] != canonical_road_fingerprint(
            graph
        ):
            raise ValueError(
                f"Road baseline does not match topology group {group_key!r}"
            )
    # Version-1 manifests stored the now-merged training data under separate
    # "training" and "validation" ownership labels. Their union is the exact
    # training set after validation removal.
    recorded_groups = set().union(
        *(
            set(manifest["split_ownership"].get(split, []))
            for split in ("training", "validation")
        )
    )
    if recorded_groups != expected_groups:
        raise ValueError("Road baseline does not exactly cover training groups")

    LOGGER.info(
        "Road 随机基线契约验证完成：groups=%d samples_per_group=%d "
        "sha256=%s path=%s",
        len(rows),
        manifest["sampling"]["samples_per_group"],
        recorded_sha,
        path,
    )
    return path, rows


def _training_artifact_paths(graph, tasks: tuple[Task, ...]) -> list[Path]:
    paths = {Path("topology.json")}
    task_set = set(tasks)
    if Task.LAND in task_set:
        paths.update(
            {Path("land_use_solution.csv"), Path("land_use_trajectory.json")}
        )
    if Task.ROAD in task_set:
        paths.update(
            {Path("road_plan_solution.csv"), Path("road_plan_trajectory.json")}
        )
    signal_meta_path = graph.tile_dir / "signal_meta.json"
    if signal_meta_path.is_file():
        paths.add(Path("signal_meta.json"))
    if Task.SIGNAL in task_set and graph.signal_entities.numel() > 0:
        if graph.signal_meta is None:
            raise ValueError(f"{graph.tile_id}: signal task is missing signal metadata")
        paths.add(Path("signal_trajectory.json"))
        for field, fallback in (
            ("net_file", "sumo/net.xml"),
            ("route_file", "sumo/routes.xml"),
        ):
            relative_path = Path(str(graph.signal_meta.get(field, fallback)))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"{graph.tile_id}: signal {field} must stay inside its tile"
                )
            paths.add(relative_path)
    return sorted(paths, key=lambda path: path.as_posix())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_init_checkpoint(
    *,
    init_checkpoint: str | Path | None,
    sft_checkpoint: str | Path | None,
) -> tuple[Path | None, str | None]:
    if init_checkpoint is not None and sft_checkpoint is not None:
        raise ValueError(
            "--init-checkpoint conflicts with deprecated --sft-checkpoint; "
            "provide only one"
        )
    if init_checkpoint is not None:
        return Path(init_checkpoint), "--init-checkpoint"
    if sft_checkpoint is not None:
        LOGGER.warning(
            "--sft-checkpoint is deprecated; use --init-checkpoint instead"
        )
        return Path(sft_checkpoint), "--sft-checkpoint"
    return None, None


def _infer_checkpoint_initialization_type(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return "unknown_checkpoint"
    stage = metrics.get("stage")
    if stage == "random_init":
        return "random_init"
    if isinstance(stage, str):
        normalized_stage = stage.lower()
        if normalized_stage == "sft" or normalized_stage.startswith("sft_"):
            return "sft"
        if normalized_stage == "ppo" or normalized_stage.startswith("ppo_"):
            return "ppo"
    metric_names = tuple(key for key in metrics if isinstance(key, str))
    if any(name.startswith("sft/") for name in metric_names):
        return "sft"
    if any(name.startswith("ppo/") for name in metric_names):
        return "ppo"
    return "unknown_checkpoint"


def _checkpoint_initialization_record(
    path: str | Path,
    payload: dict[str, Any],
    *,
    cli_option: str | None,
) -> dict[str, str | None]:
    resolved_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "type": _infer_checkpoint_initialization_type(payload),
        "checkpoint_path": str(resolved_path),
        "checkpoint_sha256": digest.hexdigest(),
        "cli_option": cli_option,
    }


def _write_initialization_manifest(
    output_dir: str | Path,
    record: dict[str, str | None],
) -> Path:
    path = Path(output_dir) / "initialization.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _ppo_initialization_summary(
    record: dict[str, str | None],
    *,
    initial_policy_logit_scale: float,
) -> dict[str, str | float | None]:
    return {
        "ppo/initialization": record["type"],
        "ppo/init_checkpoint_path": record["checkpoint_path"],
        "ppo/init_checkpoint_sha256": record["checkpoint_sha256"],
        "ppo/initial_policy_logit_scale": initial_policy_logit_scale,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the shared urban graph policy with expert SFT and PPO"
    )
    parser.add_argument("--config", type=Path, default=Path("src/configs/train.json"))
    parser.add_argument("--stage", choices=("sft", "ppo"), default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--test-data-root", type=Path, default=None)
    parser.add_argument(
        "--tile-ids",
        nargs="+",
        default=None,
        help="train/evaluate only these topology ids (useful for convergence diagnostics)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=None, help="training random seed")
    parser.add_argument(
        "--split-seed", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--validation-topologies",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, or ERROR")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help=(
            "deprecated post-stage switch; periodic test inference used for "
            "checkpoint selection always runs"
        ),
    )
    parser.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        help="refuse to start when the resolved output directory already exists",
    )
    initialization_group = parser.add_mutually_exclusive_group()
    initialization_group.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="initialize a standalone PPO run from this checkpoint",
    )
    initialization_group.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="deprecated alias for --init-checkpoint",
    )
    parser.add_argument("--sft-epochs", type=int, default=None)
    parser.add_argument("--sft-test-interval", type=int, default=None)
    parser.add_argument("--ppo-iterations", type=int, default=None)
    parser.add_argument(
        "--ppo-tasks",
        nargs="+",
        choices=("land", "road", "signal"),
        default=None,
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="override rollout episodes for every enabled PPO task",
    )
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--ppo-learning-rate-end", type=float, default=None)
    parser.add_argument("--ppo-update-epochs", type=int, default=None)
    parser.add_argument("--ppo-minibatch-size", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument(
        "--ppo-initial-logit-scale",
        type=float,
        default=None,
        help="soften SFT actor logits before PPO; in (0, 1], greedy actions unchanged",
    )
    parser.add_argument("--kl-beta-start", type=float, default=None)
    parser.add_argument("--kl-beta-end", type=float, default=None)
    parser.add_argument(
        "--ppo-test-interval",
        "--ppo-validation-interval",
        dest="ppo_test_interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ppo-validation-topologies",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ppo-test-episodes",
        "--ppo-validation-episodes",
        dest="ppo_test_episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--signal-rollout-workers",
        type=int,
        default=None,
        help="overlap this many independent SUMO signal episodes",
    )
    parser.add_argument(
        "--ppo-stochastic-test",
        "--ppo-stochastic-validation",
        dest="ppo_stochastic_test",
        action="store_true",
        help="checkpoint on fixed-seed sampled test return instead of greedy return",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    config = TrainConfig.from_json(args.config)
    if args.stage is not None:
        config.stage = args.stage
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    if args.test_data_root is not None:
        config.test_data_root = str(args.test_data_root)
    if args.tile_ids is not None:
        config.tile_ids = args.tile_ids
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.wandb_mode is not None:
        config.wandb.mode = args.wandb_mode
    if args.wandb_name is not None:
        config.wandb.name = args.wandb_name
    if args.sft_epochs is not None:
        config.sft.epochs = args.sft_epochs
    if args.sft_test_interval is not None:
        config.sft.test_interval = args.sft_test_interval
    if args.ppo_iterations is not None:
        config.ppo.iterations = args.ppo_iterations
    if args.ppo_tasks is not None:
        config.ppo.enabled_tasks = args.ppo_tasks
    if args.episodes_per_task is not None:
        for task in config.ppo.enabled_tasks:
            config.ppo.episodes_per_task[task] = args.episodes_per_task
    if args.ppo_learning_rate is not None:
        config.ppo.learning_rate = args.ppo_learning_rate
    if args.ppo_learning_rate_end is not None:
        config.ppo.learning_rate_end = args.ppo_learning_rate_end
    if args.ppo_update_epochs is not None:
        config.ppo.update_epochs = args.ppo_update_epochs
    if args.ppo_minibatch_size is not None:
        config.ppo.minibatch_size = args.ppo_minibatch_size
    if args.entropy_coef is not None:
        config.ppo.entropy_coef = args.entropy_coef
    if args.ppo_initial_logit_scale is not None:
        config.ppo.initial_policy_logit_scale = args.ppo_initial_logit_scale
    if args.kl_beta_start is not None:
        config.ppo.kl_beta_start = args.kl_beta_start
    if args.kl_beta_end is not None:
        config.ppo.kl_beta_end = args.kl_beta_end
    if args.ppo_test_interval is not None:
        config.ppo.test_interval = args.ppo_test_interval
    if args.ppo_test_episodes is not None:
        config.ppo.test_episodes = args.ppo_test_episodes
    if args.signal_rollout_workers is not None:
        config.ppo.signal_rollout_workers = args.signal_rollout_workers
    if args.ppo_stochastic_test:
        config.ppo.test_deterministic = False
    if args.fail_if_output_exists and Path(config.output_dir).exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {config.output_dir}"
        )
    artifacts = run(
        config,
        init_checkpoint=args.init_checkpoint,
        sft_checkpoint=args.sft_checkpoint,
    )
    for name, path in artifacts.items():
        LOGGER.info("训练产物 %s：%s", name, path)


if __name__ == "__main__":
    main()
