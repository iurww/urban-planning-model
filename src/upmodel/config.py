"""Typed configuration with JSON loading and conservative defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    entity_layers: int = 2
    decision_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    signal_history_steps: int = 4


@dataclass
class SFTConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0
    task_balanced: bool = True
    test_interval: int = 5
    road_permutations: int = 8
    log_interval_steps: int = 1


@dataclass
class PPOConfig:
    iterations: int = 300
    episodes_per_task: dict[str, int] = field(
        default_factory=lambda: {"land": 80, "road": 256, "signal": 48}
    )
    enabled_tasks: list[str] = field(
        default_factory=lambda: ["land", "road", "signal"]
    )
    learning_rate: float = 3.0e-5
    learning_rate_end: float = 3.0e-6
    weight_decay: float = 1.0e-4
    update_epochs: int = 2
    minibatch_size: int = 128
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # A positive factor below one is equivalent to a one-time temperature
    # increase for an SFT-initialized actor.  It keeps every greedy action but
    # prevents sharp heuristic logits from starving PPO exploration.
    initial_policy_logit_scale: float = 0.7
    max_grad_norm: float = 1.0
    gamma: dict[str, float] = field(
        default_factory=lambda: {"land": 1.0, "road": 1.0, "signal": 0.99}
    )
    gae_lambda: dict[str, float] = field(
        default_factory=lambda: {"land": 0.95, "road": 0.95, "signal": 0.95}
    )
    # SFT is an initialization, not a permanent policy anchor.  Diagnostics
    # therefore default to no reference-policy reward penalty; experiments may
    # opt into a small decaying value explicitly.
    kl_beta_start: float = 0.0
    kl_beta_end: float = 0.0
    target_kl: float = 0.05
    task_balanced: bool = True
    normalize_rewards: bool = True
    test_interval: int = 5
    test_episodes: int = 1
    test_deterministic: bool = True
    # SUMO runs as one independent process/TraCI connection per episode. Model
    # inference remains on the training process and environment steps overlap.
    signal_rollout_workers: int = 4
    road_baseline_manifest: str | None = None
    road_baseline_manifest_sha256: str | None = None
    road_center_rewards: bool = False


@dataclass
class WandbConfig:
    project: str = "up-model-ppo"
    entity: str | None = None
    name: str | None = None
    mode: str = "offline"
    tags: list[str] = field(default_factory=lambda: ["ppo", "multi-task"])


@dataclass
class EvaluationConfig:
    tasks: list[str] = field(
        default_factory=lambda: ["land", "road", "signal"]
    )
    episodes: int = 1
    deterministic: bool = True
    seed: int = 0


@dataclass
class TrainConfig:
    data_root: str = "data/sft"
    test_data_root: str = "data/test"
    tile_ids: list[str] = field(default_factory=list)
    output_dir: str = "outputs/upmodel"
    seed: int = 42
    device: str = "auto"
    stage: str = "sft"
    num_workers: int = 0
    land_use_order: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_json(cls, path: str | Path | None) -> "TrainConfig":
        config = cls()
        if path is None:
            return config
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("training config must be a JSON object")
        raw = dict(raw)
        _migrate_validation_config(raw)
        _update_dataclass(config, raw, prefix="config")
        config.validate()
        return config

    def validate(self) -> None:
        if self.stage not in {"sft", "ppo"}:
            raise ValueError("stage must be one of: sft, ppo")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if not self.ppo.enabled_tasks:
            raise ValueError("ppo.enabled_tasks cannot be empty")
        training_tasks = {"land", "road", "signal"}
        unknown = set(self.ppo.enabled_tasks) - training_tasks
        if unknown:
            raise ValueError(f"unknown PPO tasks: {sorted(unknown)}")
        for name in self.ppo.enabled_tasks:
            if self.ppo.episodes_per_task.get(name, 0) < 1:
                raise ValueError(f"episodes_per_task[{name!r}] must be positive")
        if self.ppo.test_interval < 1:
            raise ValueError("ppo.test_interval must be positive")
        if self.sft.test_interval < 1:
            raise ValueError("sft.test_interval must be positive")
        if self.sft.road_permutations < 1:
            raise ValueError("sft.road_permutations must be positive")
        if self.sft.log_interval_steps < 1:
            raise ValueError("sft.log_interval_steps must be positive")
        if self.ppo.test_episodes < 1:
            raise ValueError("ppo.test_episodes must be positive")
        if self.ppo.signal_rollout_workers < 1:
            raise ValueError("ppo.signal_rollout_workers must be positive")
        road_manifest = self.ppo.road_baseline_manifest
        road_manifest_sha = self.ppo.road_baseline_manifest_sha256
        if road_manifest is not None and not road_manifest.strip():
            raise ValueError("ppo.road_baseline_manifest cannot be empty")
        if (road_manifest is None) != (road_manifest_sha is None):
            raise ValueError(
                "ppo.road_baseline_manifest and "
                "ppo.road_baseline_manifest_sha256 must be configured together"
            )
        if self.ppo.road_center_rewards and road_manifest is None:
            raise ValueError(
                "a frozen ppo.road_baseline_manifest and SHA-256 are required "
                "when road_center_rewards is true"
            )
        if self.ppo.road_center_rewards and not self.ppo.normalize_rewards:
            raise ValueError(
                "ppo.normalize_rewards must be true when road_center_rewards is true"
            )
        if road_manifest is not None and "road" not in self.ppo.enabled_tasks:
            raise ValueError(
                "ppo.road_baseline_manifest requires the Road task to be enabled"
            )
        if road_manifest_sha is not None and (
            len(road_manifest_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in road_manifest_sha
            )
        ):
            raise ValueError(
                "ppo.road_baseline_manifest_sha256 must be 64 lowercase hex digits"
            )
        if self.ppo.learning_rate <= 0.0 or self.ppo.learning_rate_end <= 0.0:
            raise ValueError("PPO learning rates must be positive")
        if not 0.0 < self.ppo.initial_policy_logit_scale <= 1.0:
            raise ValueError("ppo.initial_policy_logit_scale must be in (0, 1]")
        if self.model.hidden_dim % self.model.attention_heads != 0:
            raise ValueError("model.hidden_dim must be divisible by model.attention_heads")
        if self.model.entity_layers < 1 or self.model.decision_layers < 1:
            raise ValueError("both Transformer layer counts must be positive")
        if self.model.signal_history_steps < 0:
            raise ValueError("model.signal_history_steps cannot be negative")
        if self.land_use_order and (
            len(self.land_use_order) != 6
            or set(self.land_use_order) != {"grn", "com", "res", "sch", "hos", "off"}
        ):
            raise ValueError(
                "land_use_order must contain all six land-use labels exactly once"
            )
        if self.evaluation.episodes < 1:
            raise ValueError("evaluation.episodes must be positive")
        if any(not str(tile_id).strip() for tile_id in self.tile_ids):
            raise ValueError("tile_ids cannot contain empty values")
        if not str(self.test_data_root).strip():
            raise ValueError("test_data_root cannot be empty")
        evaluation_unknown = set(self.evaluation.tasks) - training_tasks
        if evaluation_unknown:
            raise ValueError(f"unknown evaluation tasks: {sorted(evaluation_unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _migrate_validation_config(raw: dict[str, Any]) -> None:
    """Accept old configuration names while resolving to test-only fields."""
    raw.pop("split_seed", None)
    raw.pop("validation_topologies", None)
    evaluation_values = raw.get("evaluation")
    if isinstance(evaluation_values, dict):
        evaluation_values = dict(evaluation_values)
        for name in ("enabled", "after_sft", "after_ppo"):
            evaluation_values.pop(name, None)
        raw["evaluation"] = evaluation_values
    for section_name, aliases in (
        ("sft", {"validation_interval": "test_interval"}),
        (
            "ppo",
            {
                "validation_interval": "test_interval",
                "validation_episodes": "test_episodes",
                "validation_deterministic": "test_deterministic",
            },
        ),
    ):
        values = raw.get(section_name)
        if not isinstance(values, dict):
            continue
        values = dict(values)
        values.pop("validation_topologies", None)
        if section_name == "ppo":
            values.pop("log_interval_steps", None)
        for old_name, new_name in aliases.items():
            if old_name not in values:
                continue
            if (
                new_name in values
                and values[new_name] != values[old_name]
            ):
                raise ValueError(
                    f"config.{section_name}.{old_name} conflicts with "
                    f"config.{section_name}.{new_name}"
                )
            values.setdefault(new_name, values.pop(old_name))
        raw[section_name] = values


def _update_dataclass(target: T, values: dict[str, Any], *, prefix: str) -> T:
    known = {item.name: item for item in fields(target)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"{prefix}: unknown keys: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}.{key} must be an object")
            _update_dataclass(current, value, prefix=f"{prefix}.{key}")
        else:
            setattr(target, key, value)
    return target
