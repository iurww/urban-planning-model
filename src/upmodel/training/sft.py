"""Behavior cloning for the shared encoder, decision model, and actor heads."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm

from upmodel.config import SFTConfig
from upmodel.data.expert import ExpertDataset, ExpertStep
from upmodel.logging import RunLogger
from upmodel.model import MultiTaskActorCritic
from upmodel.training.checkpoint import save_checkpoint
from upmodel.tasks import Task


LOGGER = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(
        self,
        *,
        model: MultiTaskActorCritic,
        dataset: ExpertDataset,
        config: SFTConfig,
        logger: RunLogger,
        output_dir: str | Path,
        full_config: dict,
        seed: int,
        epoch_evaluator: Callable[[int, Path], dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.full_config = full_config
        # The value head is deliberately excluded from SFT. PPO initializes
        # and trains its critic from rollout returns instead of fitting expert
        # return-to-go targets before exploration starts.
        policy_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("value_head.")
        ]
        self.optimizer = torch.optim.AdamW(
            policy_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch_evaluator = epoch_evaluator
        self.optimizer_step = 0
        LOGGER.info(
            "SFT 训练器初始化完成：expert_steps=%d batch_size=%d epochs=%d "
            "task_balanced=%s",
            len(dataset),
            config.batch_size,
            config.epochs,
            config.task_balanced,
        )

    def train(self) -> Path:
        best_loss = math.inf
        best_test_score = -math.inf
        best_path = self.output_dir / "sft_best.pt"
        last_path = self.output_dir / "sft_last.pt"
        LOGGER.info("开始一阶段 SFT 训练")
        epochs = tqdm(
            range(1, self.config.epochs + 1),
            desc="Stage 1 / SFT",
            unit="epoch",
        )
        for epoch in epochs:
            metrics = self._train_epoch(epoch)
            metrics["sft/epoch"] = epoch
            # Keep the SFT dashboard to a single optimization curve. Reward
            # quality is reported by the test-set evaluator below.
            # Keep the global objective for a compact overview, but expose the
            # per-task means as well.  With task-balanced sampling they make it
            # possible to see a stalled task even when the aggregate decreases.
            epoch_log = {
                "sft/epoch": epoch,
                "sft/epoch/loss": metrics["sft/loss"],
            }
            for task in self.dataset.tasks:
                epoch_log[f"sft/epoch/{task.label}_loss"] = metrics[
                    f"sft/{task.label}_loss"
                ]
            self.logger.log(epoch_log, step=epoch)
            save_checkpoint(
                last_path,
                model=self.model,
                optimizer=self.optimizer,
                step=epoch,
                config=self.full_config,
                metrics=metrics,
            )
            test_score: float | None = None
            should_evaluate = self.epoch_evaluator is not None and (
                epoch % self.config.test_interval == 0
                or epoch == self.config.epochs
            )
            if should_evaluate:
                LOGGER.info("开始 SFT epoch=%d 测试集 evaluate", epoch)
                test_metrics = self.epoch_evaluator(epoch, last_path)
                metrics.update(test_metrics)
                test_score = test_metrics["selection_score"]
                LOGGER.info(
                    "SFT epoch=%d 测试集 evaluate 完成：r=%.6f",
                    epoch,
                    test_score,
                )
            epochs.set_postfix(
                loss=f"{metrics['sft/loss']:.4f}",
            )
            LOGGER.info(
                "SFT epoch=%d/%d loss=%.6f",
                epoch,
                self.config.epochs,
                metrics["sft/loss"],
            )
            if not math.isfinite(metrics["sft/loss"]):
                raise FloatingPointError(f"non-finite SFT loss at epoch {epoch}: {metrics}")
            # With periodic test inference configured, only its selection score
            # may replace the best checkpoint. Training loss is never mixed
            # into that decision on the intervening epochs.
            improved = (
                test_score is not None
                and test_score > best_test_score
                if self.epoch_evaluator is not None
                else metrics["sft/loss"] < best_loss
            )
            best_loss = min(best_loss, metrics["sft/loss"])
            if test_score is not None:
                best_test_score = max(best_test_score, test_score)
            if improved:
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=epoch,
                    config=self.full_config,
                    metrics=metrics,
                )
                LOGGER.info("保存新的 SFT 最优 checkpoint：%s", best_path)
        summary = {"sft/best_loss": best_loss}
        if self.epoch_evaluator is not None:
            summary["sft/best_test_score"] = best_test_score
        self.logger.summary(summary)
        LOGGER.info(
            "一阶段 SFT 训练完成：best_loss=%.6f best_test_score=%s",
            best_loss,
            f"{best_test_score:.6f}"
            if self.epoch_evaluator is not None
            else "n/a",
        )
        return best_path

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        epoch_batches = self._epoch_batches()
        totals = {
            "loss": 0.0,
            "samples": 0.0,
        }
        task_totals = {
            task.label: {"loss": 0.0, "samples": 0.0}
            for task in self.dataset.tasks
        }
        batches = tqdm(
            epoch_batches,
            total=len(epoch_batches),
            desc=f"SFT epoch {epoch}",
            unit="batch",
            leave=False,
        )
        for indices in batches:
            batch_started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=self.model.device)
            batch_task_loss: dict[str, float] = {}
            batch_task_samples: dict[str, int] = {}
            for group_indices in self._topology_groups(indices):
                steps = [self.dataset[index] for index in group_indices]
                losses = self._batch_losses(steps)
                batch_loss = batch_loss + torch.stack(losses).sum() / len(indices)
                for step, loss in zip(steps, losses, strict=True):
                    loss_value = float(loss.detach().item())
                    totals["loss"] += loss_value
                    totals["samples"] += 1.0
                    task = step.state.task.label
                    task_totals[task]["loss"] += loss_value
                    task_totals[task]["samples"] += 1.0
                    batch_task_loss[task] = (
                        batch_task_loss.get(task, 0.0) + loss_value
                    )
                    batch_task_samples[task] = batch_task_samples.get(task, 0) + 1
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer_step += 1
            if self.optimizer_step % self.config.log_interval_steps == 0:
                elapsed = max(time.perf_counter() - batch_started, 1.0e-9)
                step_metrics = {
                    "sft/optimizer_step": self.optimizer_step,
                    "sft/train/loss": float(batch_loss.detach().item()),
                    "sft/train/grad_norm": float(grad_norm.detach().item()),
                    "sft/train/learning_rate": float(
                        self.optimizer.param_groups[0]["lr"]
                    ),
                    "sft/train/samples_per_second": len(indices) / elapsed,
                }
                for task, task_loss in batch_task_loss.items():
                    step_metrics[f"sft/train/{task}_loss"] = (
                        task_loss / batch_task_samples[task]
                    )
                self.logger.log(step_metrics)
            batches.set_postfix(loss=f"{float(batch_loss.detach().item()):.4f}")
        sample_count = max(totals["samples"], 1.0)
        metrics = {"sft/loss": totals["loss"] / sample_count}
        for task in self.dataset.tasks:
            values = task_totals[task.label]
            metrics[f"sft/{task.label}_loss"] = values["loss"] / max(
                values["samples"], 1.0
            )
        return metrics

    def _epoch_batches(self) -> list[list[int]]:
        if not self.config.task_balanced:
            sampled = torch.randperm(
                len(self.dataset), generator=self.generator
            ).tolist()
        else:
            by_task = {
                task: [
                    index
                    for index, step in enumerate(self.dataset.steps)
                    if step.state.task == task
                ]
                for task in self.dataset.tasks
            }
            target = max(len(indices) for indices in by_task.values())
            sampled = []
            for indices in by_task.values():
                task_sampled: list[int] = []
                while len(task_sampled) < target:
                    permutation = torch.randperm(
                        len(indices), generator=self.generator
                    ).tolist()
                    task_sampled.extend(indices[index] for index in permutation)
                sampled.extend(task_sampled[:target])

        by_topology: dict[tuple[Task, int], list[int]] = {}
        for index in sampled:
            state = self.dataset[index].state
            by_topology.setdefault((state.task, id(state.graph)), []).append(index)
        chunks: list[list[int]] = []
        for group_indices in by_topology.values():
            permutation = torch.randperm(
                len(group_indices), generator=self.generator
            ).tolist()
            shuffled = [group_indices[index] for index in permutation]
            chunks.extend(
                shuffled[start : start + self.config.batch_size]
                for start in range(0, len(shuffled), self.config.batch_size)
            )
        chunk_order = torch.randperm(
            len(chunks), generator=self.generator
        ).tolist()

        batches: list[list[int]] = []
        current: list[int] = []
        for chunk_index in chunk_order:
            chunk = chunks[chunk_index]
            offset = 0
            while offset < len(chunk):
                room = self.config.batch_size - len(current)
                take = min(room, len(chunk) - offset)
                current.extend(chunk[offset : offset + take])
                offset += take
                if len(current) == self.config.batch_size:
                    batches.append(current)
                    current = []
        if current:
            batches.append(current)
        return batches

    def _topology_groups(self, indices: list[int]) -> list[list[int]]:
        grouped: dict[tuple[Task, int], list[int]] = {}
        for index in indices:
            state = self.dataset[index].state
            grouped.setdefault((state.task, id(state.graph)), []).append(index)
        return list(grouped.values())

    def _batch_losses(self, steps: list[ExpertStep]) -> list[torch.Tensor]:
        if not steps:
            raise ValueError("SFT batch cannot be empty")
        graph = steps[0].state.graph
        task = steps[0].state.task
        if any(step.state.graph is not graph or step.state.task != task for step in steps):
            raise ValueError("SFT optimized batches must share one graph and task")
        states = [step.state for step in steps]
        entity_batch = self.model.encode_entities_batch(states)
        logits = self.model.policy_logits_batch_from_entities(states, entity_batch)
        return [
            self._actor_loss(step, step_logits)
            for step, step_logits in zip(steps, logits, strict=True)
        ]

    def _sample_loss(self, step: ExpertStep) -> torch.Tensor:
        return self._actor_loss(step, self.model(step.state).logits)

    def _actor_loss(
        self, step: ExpertStep, logits: torch.Tensor
    ) -> torch.Tensor:
        target = torch.as_tensor(step.action, dtype=torch.long, device=self.model.device)
        if step.target_mask is not None:
            actor_loss = _masked_set_cross_entropy(
                logits,
                step.target_mask.to(self.model.device),
                step.state.action_mask.to(self.model.device),
                smoothing=self.config.label_smoothing,
            )
        elif step.state.task == Task.SIGNAL:
            actor_loss = _masked_cross_entropy(
                logits,
                target,
                step.state.action_mask.to(self.model.device),
                smoothing=self.config.label_smoothing,
            )
        else:
            actor_loss = _masked_cross_entropy(
                logits.unsqueeze(0),
                target.reshape(1),
                step.state.action_mask.to(self.model.device).unsqueeze(0),
                smoothing=self.config.label_smoothing,
            )
        return actor_loss


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    smoothing: float,
) -> torch.Tensor:
    """Label smoothing over legal actions only; invalid logits never enter the loss."""
    log_prob = logits.masked_fill(~mask.bool(), -torch.inf).log_softmax(dim=-1)
    negative_log_likelihood = -log_prob.gather(1, target[:, None]).squeeze(1)
    legal_log_prob = log_prob.masked_fill(~mask, 0.0)
    smooth_loss = -legal_log_prob.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    return ((1.0 - smoothing) * negative_log_likelihood + smoothing * smooth_loss).mean()


def _masked_set_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    smoothing: float,
) -> torch.Tensor:
    """Supervise an unordered set by its total legal probability mass."""
    if logits.ndim != 1 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("set targets, mask, and logits must be aligned vectors")
    positives = targets.bool() & mask.bool()
    if not bool(positives.any()):
        raise ValueError("set supervision requires at least one legal target")
    log_prob = logits.masked_fill(~mask.bool(), -torch.inf).log_softmax(dim=-1)
    negative_log_likelihood = -torch.logsumexp(log_prob[positives], dim=0)
    smooth_loss = -log_prob.masked_fill(~mask.bool(), 0.0).sum() / mask.sum().clamp_min(1)
    return (1.0 - smoothing) * negative_log_likelihood + smoothing * smooth_loss
