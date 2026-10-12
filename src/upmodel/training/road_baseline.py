"""Reproducible uniform-legal Road baselines for reward calibration.

The baseline deliberately supplies its own deletion schedule.  It never reads
expert actions; expert artifacts are used only for the terminal reference
reward stored next to the sampled baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from upmodel.data.split import (
    GROUP_KEY_VERSION,
    ROAD_FINGERPRINT_VERSION,
    canonical_road_fingerprint,
    topology_fingerprint,
    topology_group_key,
)
from upmodel.envs.static import StaticPlanningEnv
from upmodel.evaluation.baseline import expert_baseline
from upmodel.road_metrics import DEFAULT_ROAD_BUDGET_FRAC, road_budget
from upmodel.state import UrbanGraph
from upmodel.tasks import Task


ROAD_BASELINE_SCHEMA_VERSION = 1
ROAD_BASELINE_VERSION = "road-random-feasible-v1"
ROAD_BASELINE_SEED_SCHEME = "sha256-version-group-sample-uint64be-v1"
ROAD_BASELINE_SEED_DERIVATION = (
    "uint64_be(sha256(version + NUL + group_key + NUL + "
    "decimal_sample_index)[0:8])"
)
ROAD_BASELINE_AGGREGATE_ENCODING = (
    "canonical-json-without-aggregate_sha256-v1"
)
ROAD_BASELINE_COMPONENTS = (
    "Traffic",
    "Tdensity",
    "Tconnectivity",
    "Tspacing",
)
ROAD_BASELINE_SPLITS = ("training", "validation")
DEFAULT_ROAD_BASELINE_SAMPLES = 256
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RoadBaselineSample:
    """One terminal sample, retaining both environment reward and raw metrics."""

    reward: float
    traffic: float
    density: float
    connectivity: float
    spacing: float
    stuck: bool
    steps: int
    expected_steps: int

    @property
    def components(self) -> dict[str, float]:
        return {
            "Traffic": self.traffic,
            "Tdensity": self.density,
            "Tconnectivity": self.connectivity,
            "Tspacing": self.spacing,
        }


EnvironmentFactory = Callable[..., Any]
RolloutFunction = Callable[..., RoadBaselineSample]
ExpertFunction = Callable[[UrbanGraph, Task], tuple[float, dict[str, float]]]


def road_baseline_seed(group_key: str, sample_index: int) -> int:
    """Derive the frozen episode seed from version, group, and sample index."""
    if not isinstance(group_key, str) or not group_key:
        raise ValueError("road baseline group key must be a nonempty string")
    if (
        not isinstance(sample_index, int)
        or isinstance(sample_index, bool)
        or sample_index < 0
    ):
        raise ValueError("road baseline sample index must be a nonnegative integer")
    encoded = b"\0".join(
        (
            ROAD_BASELINE_VERSION.encode("ascii"),
            group_key.encode("utf-8"),
            str(int(sample_index)).encode("ascii"),
        )
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def road_deletion_schedule(
    graph: UrbanGraph,
) -> list[int]:
    """Return the explicit fixed-what schedule for the Road baseline."""
    road_count = int(graph.road_entities.numel())
    deletion_count = road_count - road_budget(
        road_count, DEFAULT_ROAD_BUDGET_FRAC
    )
    if deletion_count < 1:
        raise ValueError(
            f"{graph.tile_id}: uniform Road baseline needs at least one deletion; "
            f"roads={road_count} budget_fraction={DEFAULT_ROAD_BUDGET_FRAC:g}"
        )
    return [0] * deletion_count


def rollout_uniform_legal_road(
    graph: UrbanGraph,
    *,
    seed: int,
    env_factory: EnvironmentFactory = StaticPlanningEnv,
    environment: Any | None = None,
) -> RoadBaselineSample:
    """Uniformly select among legal deletions until the fixed budget is met."""
    schedule = road_deletion_schedule(graph)
    env = environment
    if env is None:
        env = env_factory(
            graph,
            Task.ROAD,
            what_schedule=schedule,
        )
    return _rollout_uniform_legal_environment(
        env,
        seed=int(seed),
        expected_steps=len(schedule),
    )


def estimate_uniform_legal_road(
    graph: UrbanGraph,
    *,
    group_key: str | None = None,
    samples: int = DEFAULT_ROAD_BASELINE_SAMPLES,
    env_factory: EnvironmentFactory = StaticPlanningEnv,
    rollout_fn: RolloutFunction | None = None,
) -> dict[str, Any]:
    """Estimate raw Road reward/component moments for one canonical graph."""
    if samples < 1:
        raise ValueError("road baseline samples must be positive")
    resolved_group_key = group_key or topology_group_key(graph)
    environment = None
    if rollout_fn is None:
        schedule = road_deletion_schedule(graph)
        environment = env_factory(
            graph,
            Task.ROAD,
            what_schedule=schedule,
        )

    results: list[RoadBaselineSample] = []
    for sample_index in range(samples):
        seed = road_baseline_seed(resolved_group_key, sample_index)
        if rollout_fn is None:
            result = rollout_uniform_legal_road(
                graph,
                seed=seed,
                env_factory=env_factory,
                environment=environment,
            )
        else:
            result = rollout_fn(graph, seed=seed)
        if not isinstance(result, RoadBaselineSample):
            raise TypeError("road rollout function must return RoadBaselineSample")
        results.append(result)

    return {
        "samples": samples,
        "reward": _sample_statistics([result.reward for result in results]),
        "components": {
            component: _sample_statistics(
                [result.components[component] for result in results]
            )
            for component in ROAD_BASELINE_COMPONENTS
        },
        "stuck_count": sum(result.stuck for result in results),
        "stuck_rate": float(
            sum(result.stuck for result in results) / samples
        ),
    }


def build_road_baseline_manifest(
    split_graphs: Mapping[str, Sequence[UrbanGraph]],
    *,
    samples_per_group: int = DEFAULT_ROAD_BASELINE_SAMPLES,
    split_seed: int | None = None,
    validation_size: int | None = None,
    data_root: str | Path | None = None,
    final_test_root: str | Path | None = None,
    env_factory: EnvironmentFactory = StaticPlanningEnv,
    rollout_fn: RolloutFunction | None = None,
    expert_fn: ExpertFunction = expert_baseline,
) -> dict[str, Any]:
    """Build one baseline row per topology group in train and validation."""
    supplied_splits = set(split_graphs)
    required_splits = set(ROAD_BASELINE_SPLITS)
    if supplied_splits != required_splits:
        raise ValueError(
            "road baseline requires exactly training and validation splits; "
            f"received={sorted(supplied_splits)}"
        )
    if samples_per_group < 1:
        raise ValueError("road baseline samples_per_group must be positive")
    grouped: dict[str, list[UrbanGraph]] = defaultdict(list)
    ownership: dict[str, str] = {}
    member_counts: dict[str, int] = {}
    for split in ROAD_BASELINE_SPLITS:
        graphs = list(split_graphs[split])
        if not graphs:
            raise ValueError(f"road baseline {split} split must be nonempty")
        member_counts[split] = len(graphs)
        for graph in graphs:
            group_key = topology_group_key(graph)
            prior_owner = ownership.setdefault(group_key, split)
            if prior_owner != split:
                raise ValueError(
                    f"road baseline group {group_key!r} belongs to both "
                    f"{prior_owner} and {split}"
                )
            grouped[group_key].append(graph)

    rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda key: (ownership[key], key)):
        members = grouped[group_key]
        road_fingerprints = {
            canonical_road_fingerprint(graph) for graph in members
        }
        if None in road_fingerprints:
            raise ValueError(
                f"road baseline group {group_key!r} has no canonical Road fingerprint"
            )
        if len(road_fingerprints) != 1:
            raise ValueError(
                f"road baseline group {group_key!r} contains different Road networks"
            )
        road_fingerprint = next(iter(road_fingerprints))
        assert road_fingerprint is not None
        canonical = min(
            members,
            key=lambda graph: (topology_fingerprint(graph), graph.tile_id),
        )
        uniform = estimate_uniform_legal_road(
            canonical,
            group_key=group_key,
            samples=samples_per_group,
            env_factory=env_factory,
            rollout_fn=rollout_fn,
        )
        expert_reward, expert_metrics = expert_fn(canonical, Task.ROAD)
        expert_reward = _finite_float(expert_reward, label="expert reward")
        expert_components = {
            component: _finite_float(
                expert_metrics.get(
                    component,
                    expert_reward if component == "Traffic" else 0.0,
                ),
                label=f"expert {component}",
            )
            for component in ROAD_BASELINE_COMPONENTS
        }
        uniform_mean = float(uniform["reward"]["mean"])
        gap = expert_reward - uniform_mean
        rows.append(
            {
                "group_key": group_key,
                "road_fingerprint": road_fingerprint,
                "split": ownership[group_key],
                "canonical_tile_id": canonical.tile_id,
                "member_tile_ids": sorted(graph.tile_id for graph in members),
                "member_count": len(members),
                "policy_fingerprints": sorted(
                    {topology_fingerprint(graph) for graph in members}
                ),
                "uniform": uniform,
                "expert": {
                    "reward": expert_reward,
                    "components": expert_components,
                },
                "gap": {
                    "expert_minus_uniform_mean": float(gap),
                    "expert_relative_pct": float(
                        100.0 * gap / max(abs(expert_reward), 1.0e-12)
                    ),
                },
            }
        )

    split_ownership = {
        split: sorted(
            row["group_key"] for row in rows if row["split"] == split
        )
        for split in ROAD_BASELINE_SPLITS
    }
    manifest: dict[str, Any] = {
        "schema_version": ROAD_BASELINE_SCHEMA_VERSION,
        "baseline_version": ROAD_BASELINE_VERSION,
        "group_key_version": GROUP_KEY_VERSION,
        "road_fingerprint_version": ROAD_FINGERPRINT_VERSION,
        "seed_scheme": ROAD_BASELINE_SEED_SCHEME,
        "sampling": {
            "samples_per_group": int(samples_per_group),
            "budget_fraction": DEFAULT_ROAD_BUDGET_FRAC,
            "selection": "uniform-over-current-legal-deletions",
            "seed_derivation": ROAD_BASELINE_SEED_DERIVATION,
            "std_ddof": 1,
            "stuck_policy": "reward-zero-no-retry",
        },
        "split": {
            "seed": None if split_seed is None else int(split_seed),
            "requested_validation_topologies": (
                None if validation_size is None else int(validation_size)
            ),
            "data_root": _resolved_path(data_root),
            "final_test_root": _resolved_path(final_test_root),
            "final_test_role": "split-reproduction-only-not-sampled",
            "member_counts": member_counts,
            "group_counts": {
                split: len(split_ownership[split])
                for split in ROAD_BASELINE_SPLITS
            },
        },
        "sampled_splits": list(ROAD_BASELINE_SPLITS),
        "excluded_splits": ["test", "quarantined_training"],
        "split_ownership": split_ownership,
        "groups": rows,
        "aggregate_algorithm": "sha256",
        "aggregate_encoding": ROAD_BASELINE_AGGREGATE_ENCODING,
    }
    manifest["aggregate_sha256"] = road_baseline_manifest_sha256(manifest)
    verify_road_baseline_manifest(manifest)
    return manifest


def road_baseline_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash all semantic manifest content except the digest field itself."""
    payload = dict(manifest)
    payload.pop("aggregate_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def verify_road_baseline_manifest(manifest: Mapping[str, Any]) -> None:
    """Check integrity, version compatibility, and fields used by training."""
    if not isinstance(manifest, Mapping):
        raise ValueError("road baseline manifest must be a JSON object")
    aggregate = manifest.get("aggregate_sha256")
    if not isinstance(aggregate, str) or not _SHA256_RE.fullmatch(aggregate):
        raise ValueError("road baseline aggregate_sha256 is invalid")
    expected = road_baseline_manifest_sha256(manifest)
    if aggregate != expected:
        raise ValueError(
            "road baseline aggregate SHA-256 mismatch: "
            f"expected={expected} recorded={aggregate}"
        )
    expected_metadata = {
        "schema_version": ROAD_BASELINE_SCHEMA_VERSION,
        "baseline_version": ROAD_BASELINE_VERSION,
        "group_key_version": GROUP_KEY_VERSION,
        "road_fingerprint_version": ROAD_FINGERPRINT_VERSION,
        "seed_scheme": ROAD_BASELINE_SEED_SCHEME,
        "aggregate_algorithm": "sha256",
        "aggregate_encoding": ROAD_BASELINE_AGGREGATE_ENCODING,
    }
    for key, value in expected_metadata.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"road baseline {key} is incompatible: "
                f"expected={value!r} actual={manifest.get(key)!r}"
            )
    sampling = manifest.get("sampling")
    samples = (
        sampling.get("samples_per_group")
        if isinstance(sampling, Mapping)
        else None
    )
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("road baseline sampling contract is incompatible")
    rows = manifest.get("groups")
    if not isinstance(rows, list) or not rows:
        raise ValueError("road baseline groups must be a nonempty list")

    observed_ownership = {split: [] for split in ROAD_BASELINE_SPLITS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("road baseline group row must be an object")
        group_key = row.get("group_key")
        split = row.get("split")
        if not isinstance(group_key, str) or split not in ROAD_BASELINE_SPLITS:
            raise ValueError(
                "road baseline group identity or split is invalid"
            )
        observed_ownership[split].append(group_key)
        uniform = row.get("uniform")
        expert = row.get("expert")
        if not isinstance(uniform, Mapping) or not isinstance(expert, Mapping):
            raise ValueError(f"road baseline values are missing for {group_key!r}")
        _finite_float(uniform["reward"]["mean"], label="uniform reward")
        _finite_float(expert["reward"], label="expert reward")
        for component in ROAD_BASELINE_COMPONENTS:
            _finite_float(
                uniform["components"][component]["mean"],
                label=f"uniform {component}",
            )
            _finite_float(
                expert["components"][component],
                label=f"expert {component}",
            )

    if len({key for keys in observed_ownership.values() for key in keys}) != len(
        rows
    ):
        raise ValueError("road baseline group keys must be unique")
    recorded_ownership = manifest.get("split_ownership")
    expected_ownership = {
        split: sorted(observed_ownership[split]) for split in ROAD_BASELINE_SPLITS
    }
    if recorded_ownership != expected_ownership:
        raise ValueError("road baseline split ownership is inconsistent")


def write_road_baseline_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    fail_if_exists: bool = False,
) -> Path:
    """Verify and atomically write a Road baseline manifest."""
    verify_road_baseline_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if fail_if_exists:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite road baseline manifest: {path}"
                ) from exc
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_road_baseline_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest only after checking its complete semantic digest."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"road baseline manifest does not exist: {path}"
        ) from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: road baseline manifest must be a JSON object")
    verify_road_baseline_manifest(payload)
    return payload


def index_road_baseline_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index a manifest already validated by the loader or builder."""
    return {
        str(row["group_key"]): dict(row)
        for row in manifest["groups"]
    }


def _rollout_uniform_legal_environment(
    env: Any,
    *,
    seed: int,
    expected_steps: int,
) -> RoadBaselineSample:
    generator = random.Random(seed)
    state = env.reset()
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    steps = 0
    done = False
    while not done:
        mask = state.action_mask
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
            raise ValueError("Road baseline environment action mask must be 1-D")
        legal = torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten().tolist()
        if not legal:
            return _sample_from_terminal(
                reward=0.0,
                metrics=_partial_road_metrics(env),
                stuck=True,
                steps=steps,
                expected_steps=expected_steps,
            )
        action = int(legal[generator.randrange(len(legal))])
        next_state, reward, done, info = env.step(action)
        total_reward += float(reward)
        final_info = dict(info)
        steps += 1
        if next_state is not None:
            state = next_state
    stuck = bool(final_info.get("stuck_before_budget", False)) or (
        steps != expected_steps
    )
    return _sample_from_terminal(
        reward=total_reward,
        metrics=final_info,
        stuck=stuck,
        steps=steps,
        expected_steps=expected_steps,
    )


def _partial_road_metrics(env: Any) -> dict[str, Any]:
    evaluator = getattr(env, "evaluator", None)
    actions = getattr(env, "actions", None)
    if evaluator is None or not isinstance(actions, list):
        return {}
    result = evaluator.evaluate(actions)
    return dict(result.metrics)


def _sample_from_terminal(
    *,
    reward: float,
    metrics: Mapping[str, Any],
    stuck: bool,
    steps: int,
    expected_steps: int,
) -> RoadBaselineSample:
    values = {
        component: _finite_float(
            metrics.get(component, 0.0), label=f"Road baseline {component}"
        )
        for component in ROAD_BASELINE_COMPONENTS
    }
    return RoadBaselineSample(
        reward=_finite_float(reward, label="Road baseline reward"),
        traffic=values["Traffic"],
        density=values["Tdensity"],
        connectivity=values["Tconnectivity"],
        spacing=values["Tspacing"],
        stuck=bool(stuck),
        steps=int(steps),
        expected_steps=int(expected_steps),
    )


def _sample_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Road baseline statistics require at least one sample")
    finite = [_finite_float(value, label="Road baseline sample") for value in values]
    standard_deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return {
        "mean": float(statistics.fmean(finite)),
        "std": float(standard_deviation),
        "standard_error": float(standard_deviation / math.sqrt(len(finite))),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _finite_float(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"road baseline {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"road baseline {label} must be finite")
    return result


def _resolved_path(value: str | Path | None) -> str | None:
    return None if value is None else str(Path(value).expanduser().resolve())


__all__ = [
    "DEFAULT_ROAD_BASELINE_SAMPLES",
    "ROAD_BASELINE_AGGREGATE_ENCODING",
    "ROAD_BASELINE_COMPONENTS",
    "ROAD_BASELINE_SCHEMA_VERSION",
    "ROAD_BASELINE_SEED_DERIVATION",
    "ROAD_BASELINE_SEED_SCHEME",
    "ROAD_BASELINE_SPLITS",
    "ROAD_BASELINE_VERSION",
    "RoadBaselineSample",
    "build_road_baseline_manifest",
    "estimate_uniform_legal_road",
    "index_road_baseline_manifest",
    "load_road_baseline_manifest",
    "road_baseline_manifest_sha256",
    "road_baseline_seed",
    "road_deletion_schedule",
    "rollout_uniform_legal_road",
    "verify_road_baseline_manifest",
    "write_road_baseline_manifest",
]
