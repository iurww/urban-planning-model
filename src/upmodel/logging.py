"""Standard process logging plus a small W&B lifecycle adapter."""

from __future__ import annotations

import json
import logging as std_logging
from pathlib import Path
from typing import Any

from upmodel.config import TrainConfig
from upmodel.evaluation.evaluator import expert_relative_score
from upmodel.tasks import TRAINING_TASKS

def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(std_logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"invalid logging level: {level!r}")
    std_logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def add_file_logging(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = std_logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        std_logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    std_logging.getLogger().addHandler(handler)
    return path


class RunLogger:
    def __init__(self, config: TrainConfig) -> None:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError("wandb is required for training logs") from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            mode=config.wandb.mode,
            tags=config.wandb.tags,
            config=config.to_dict(),
            job_type="training",
        )
        wandb.define_metric("sft/optimizer_step")
        wandb.define_metric(
            "sft/train/*", step_metric="sft/optimizer_step"
        )
        wandb.define_metric("sft/epoch")
        wandb.define_metric("sft/epoch/*", step_metric="sft/epoch")
        wandb.define_metric("ppo/iteration")
        wandb.define_metric("rollout/*", step_metric="ppo/iteration")
        for metric in (
            "ppo/policy_loss",
            "ppo/value_loss",
            "ppo/entropy_loss",
            "ppo/total_loss",
            "ppo/approx_kl",
            "ppo/clip_fraction",
            "ppo/explained_variance",
            "ppo/grad_norm",
            "ppo/grad_norm_before_clip",
            "ppo/grad_norm_after_clip",
            "ppo/learning_rate",
        ):
            wandb.define_metric(metric, step_metric="ppo/iteration")
        wandb.define_metric("test/training_step")
        for metric in (
            "test/land_reward",
            "test/road_reward",
            "test/traffic_reward",
            "test/selection_score",
        ):
            wandb.define_metric(metric, step_metric="test/training_step")
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._local_history_path = output_dir / "wandb_history.jsonl"
        self._local_summary_path = output_dir / "wandb_summary.json"
        self._local_history_path.write_text("", encoding="utf-8")
        self._local_summary: dict[str, Any] = {}
        self._event_step = 0

    @property
    def run_url(self) -> str | None:
        return self._run.url

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        payload = dict(values)
        # W&B's global step is only the event sequence. Explicit stage axes are
        # included in each payload and bound through define_metric above.
        self._run.log(payload)
        local_row = {
            key: value
            for key, value in payload.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
        local_row["_step"] = self._event_step
        if step is not None:
            local_row["_trainer_step"] = step
        if local_row:
            with self._local_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(local_row, ensure_ascii=False) + "\n")
        self._event_step += 1

    def summary(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self._run.summary[key] = value
        self._local_summary.update(values)
        self._local_summary_path.write_text(
            json.dumps(self._local_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_test_evaluation(
        self,
        summary: dict[str, dict[str, float]],
        *,
        step: int,
        stage: str,
        stage_step: int,
    ) -> float:
        """Log one compact test event and return its checkpoint score."""
        scalars: dict[str, Any] = {
            "test/training_step": float(step),
            "test/stage_step": float(stage_step),
        }
        relative_scores: list[float] = []
        for task in TRAINING_TASKS:
            metrics = summary.get(task.label)
            if metrics is None:
                continue
            policy_reward = float(metrics["policy_reward"])
            expert_reward = float(metrics["expert_reward"])
            relative_score = expert_relative_score(policy_reward, expert_reward)
            log_name = "traffic" if task.label == "signal" else task.label
            scalars[f"test/{log_name}_reward"] = policy_reward
            relative_scores.append(relative_score)
        score = float(sum(relative_scores) / max(len(relative_scores), 1))
        scalars["test/selection_score"] = score
        scalars["test/stage"] = stage
        self.log(scalars)
        return score

    def log_evaluation(
        self,
        report: Any,
        *,
        split: str = "test",
        step: int | None = None,
        stage: str = "evaluate",
        stage_step: int | None = None,
    ) -> dict[str, float]:
        if split != "test":
            raise ValueError("training no longer has a validation split")
        selection_score = self.log_test_evaluation(
            report.summary,
            step=int(step or 0),
            stage=stage,
            stage_step=int(stage_step if stage_step is not None else step or 0),
        )
        return {"selection_score": selection_score}

    def finish(self) -> None:
        self._run.finish()
