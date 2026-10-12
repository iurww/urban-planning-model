"""In-memory graph and policy-state containers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from upmodel.tasks import Task


STATIC_FEATURE_DIM = 32
SIGNAL_DYNAMIC_DIM = 18


@dataclass(frozen=True)
class UrbanGraph:
    tile_id: str
    tile_dir: Path
    features: torch.Tensor
    entity_types: torch.Tensor
    edge_index: torch.Tensor
    parcel_entities: torch.Tensor
    road_entities: torch.Tensor
    intersection_entities: torch.Tensor
    signal_entities: torch.Tensor
    signal_action_mask: torch.Tensor
    raw_topology: dict[str, Any]
    signal_meta: dict[str, Any] | None

    @property
    def num_entities(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class PolicyState:
    graph: UrbanGraph
    task: Task
    progress: float
    history: torch.Tensor
    dynamic: torch.Tensor
    action_mask: torch.Tensor
    candidate_entities: torch.Tensor | None = None
    query_action: int | None = None
    decision_entities: torch.Tensor | None = None
    decision_actions: torch.Tensor | None = None
    signal_entities: torch.Tensor | None = None
    signal_history_dynamic: torch.Tensor | None = None
    signal_history_actions: torch.Tensor | None = None
    signal_history_progress: torch.Tensor | None = None

    def snapshot(self) -> "PolicyState":
        return replace(
            self,
            history=self.history.detach().cpu().clone(),
            dynamic=self.dynamic.detach().cpu().clone(),
            action_mask=self.action_mask.detach().cpu().clone(),
            candidate_entities=(
                None
                if self.candidate_entities is None
                else self.candidate_entities.detach().cpu().clone()
            ),
            decision_entities=(
                None
                if self.decision_entities is None
                else self.decision_entities.detach().cpu().clone()
            ),
            decision_actions=(
                None
                if self.decision_actions is None
                else self.decision_actions.detach().cpu().clone()
            ),
            signal_entities=(
                None
                if self.signal_entities is None
                else self.signal_entities.detach().cpu().clone()
            ),
            signal_history_dynamic=(
                None
                if self.signal_history_dynamic is None
                else self.signal_history_dynamic.detach().cpu().clone()
            ),
            signal_history_actions=(
                None
                if self.signal_history_actions is None
                else self.signal_history_actions.detach().cpu().clone()
            ),
            signal_history_progress=(
                None
                if self.signal_history_progress is None
                else self.signal_history_progress.detach().cpu().clone()
            ),
        )
