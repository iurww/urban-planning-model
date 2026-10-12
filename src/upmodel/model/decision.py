"""Task-conditioned decision sequences for fixed-what and fixed-where tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from upmodel.state import PolicyState, SIGNAL_DYNAMIC_DIM
from upmodel.tasks import MAX_ACTIONS, NUM_TASKS, Task


TASK_TOKEN = 0
GLOBAL_TOKEN = 1
STATE_TOKEN = 2
ACTION_TOKEN = 3
QUERY_TOKEN = 4
NUM_TOKEN_TYPES = 5


@dataclass(frozen=True)
class DecisionSequence:
    hidden: torch.Tensor
    decision_indices: torch.Tensor
    query_index: torch.Tensor
    token_kinds: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedDecision:
    tokens: torch.Tensor
    decision_indices: tuple[int, ...]
    query_index: int
    token_kinds: tuple[str, ...]


class SASADecisionTransformer(nn.Module):
    """Encode ``TASK,GLOBAL,(S,A)*,QUERY`` with exactly five token roles."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_embedding = nn.Embedding(NUM_TASKS, hidden_dim)
        self.action_embedding = nn.Embedding(MAX_ACTIONS, hidden_dim)
        self.query_embedding = nn.Embedding(NUM_TASKS, hidden_dim)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, hidden_dim)
        self.progress_projection = nn.Linear(3, hidden_dim, bias=False)
        self.signal_history_gru = nn.GRU(
            SIGNAL_DYNAMIC_DIM + hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        self.history_global_projection = nn.Linear(
            SIGNAL_DYNAMIC_DIM + hidden_dim, hidden_dim
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        entity_hidden: torch.Tensor,
        state: PolicyState,
        progress_features: torch.Tensor,
    ) -> DecisionSequence:
        return self.forward_batch(
            entity_hidden.unsqueeze(0),
            [state],
            progress_features.unsqueeze(0),
        )[0]

    def forward_batch(
        self,
        entity_hidden: torch.Tensor,
        states: list[PolicyState],
        progress_features: torch.Tensor,
    ) -> list[DecisionSequence]:
        """Causally encode padded variable-length decision sequences."""
        if entity_hidden.ndim != 3 or progress_features.shape != (len(states), 3):
            raise ValueError("decision batch tensors have invalid shapes")
        if entity_hidden.shape[0] != len(states) or not states:
            raise ValueError("decision batch size must match states")
        return self.forward_heterogeneous(
            list(entity_hidden), states, progress_features
        )

    def forward_heterogeneous(
        self,
        entity_hidden: list[torch.Tensor],
        states: list[PolicyState],
        progress_features: torch.Tensor,
    ) -> list[DecisionSequence]:
        """Batch decision sequences whose underlying graphs have different sizes."""
        if not states or len(entity_hidden) != len(states):
            raise ValueError("decision batch size must match states")
        if progress_features.shape != (len(states), 3):
            raise ValueError("decision progress batch has an invalid shape")
        if any(item.ndim != 2 for item in entity_hidden):
            raise ValueError("decision entity inputs must be matrices")
        prepared = [
            self._prepare(entity, state, progress)
            for entity, state, progress in zip(
                entity_hidden, states, progress_features, strict=True
            )
        ]
        lengths = torch.tensor(
            [item.tokens.shape[0] for item in prepared],
            dtype=torch.long,
            device=entity_hidden[0].device,
        )
        padded = nn.utils.rnn.pad_sequence(
            [item.tokens for item in prepared], batch_first=True
        )
        positions = torch.arange(padded.shape[1], device=padded.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(
                (padded.shape[1], padded.shape[1]),
                dtype=torch.bool,
                device=padded.device,
            ),
            diagonal=1,
        )
        contextual = self.transformer(
            padded,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return [
            DecisionSequence(
                hidden=contextual[index, : item.tokens.shape[0]],
                decision_indices=torch.tensor(
                    item.decision_indices,
                    dtype=torch.long,
                    device=padded.device,
                ),
                query_index=torch.tensor(
                    item.query_index, dtype=torch.long, device=padded.device
                ),
                token_kinds=item.token_kinds,
            )
            for index, item in enumerate(prepared)
        ]

    def _prepare(
        self,
        entity_hidden: torch.Tensor,
        state: PolicyState,
        progress_features: torch.Tensor,
    ) -> _PreparedDecision:
        device = entity_hidden.device
        task_id = torch.tensor(int(state.task), dtype=torch.long, device=device)
        token_type = self.token_type_embedding.weight
        tokens = [
            self.task_embedding(task_id)
            + token_type[TASK_TOKEN]
            + self.progress_projection(progress_features)
        ]
        kinds = ["TASK"]
        tokens.append(
            entity_hidden.mean(dim=0)
            + token_type[GLOBAL_TOKEN]
            + self.progress_projection(progress_features)
        )
        kinds.append("GLOBAL")

        if state.task == Task.SIGNAL:
            decision_indices = self._append_signal_tokens(
                tokens, kinds, entity_hidden, state
            )
            query_index = len(tokens)
            tokens.append(
                self.query_embedding(task_id) + token_type[QUERY_TOKEN]
            )
            kinds.append("QUERY")
        else:
            self._append_static_history(tokens, kinds, entity_hidden, state)
            query_index = len(tokens)
            if state.query_action is None:
                raise ValueError(f"{state.task.label} state is missing query_action")
            query_action = int(state.query_action)
            if query_action < 0 or query_action >= MAX_ACTIONS:
                raise ValueError(f"query action {query_action} is out of range")
            tokens.append(
                self.query_embedding(task_id)
                + self.action_embedding.weight[query_action]
                + token_type[QUERY_TOKEN]
            )
            kinds.append("QUERY")
            decision_indices = [query_index]

        sequence = torch.stack(tokens, dim=0)
        sequence = sequence + _sinusoidal_positions(
            sequence.shape[0], self.hidden_dim, device=device, dtype=sequence.dtype
        )
        return _PreparedDecision(
            tokens=sequence,
            decision_indices=tuple(decision_indices),
            query_index=query_index,
            token_kinds=tuple(kinds),
        )

    def _append_static_history(
        self,
        tokens: list[torch.Tensor],
        kinds: list[str],
        entity_hidden: torch.Tensor,
        state: PolicyState,
    ) -> None:
        if state.task not in {Task.LAND, Task.ROAD}:
            raise ValueError(f"unsupported training task: {state.task.label}")
        if state.decision_entities is None or state.decision_actions is None:
            raise ValueError(f"{state.task.label} state is missing decision history")
        entities = state.decision_entities.to(entity_hidden.device)
        actions = state.decision_actions.to(entity_hidden.device)
        if entities.ndim != 1 or actions.ndim != 1 or entities.shape != actions.shape:
            raise ValueError("static decision entity/action history must be aligned vectors")
        if actions.numel() and bool(
            ((actions < 0) | (actions >= MAX_ACTIONS)).any().item()
        ):
            raise ValueError("static decision history contains an invalid action")
        token_type = self.token_type_embedding.weight
        for entity, action in zip(entities, actions, strict=True):
            tokens.append(entity_hidden[entity] + token_type[STATE_TOKEN])
            kinds.append("S")
            tokens.append(self.action_embedding(action) + token_type[ACTION_TOKEN])
            kinds.append("A")

    def _append_signal_tokens(
        self,
        tokens: list[torch.Tensor],
        kinds: list[str],
        entity_hidden: torch.Tensor,
        state: PolicyState,
    ) -> list[int]:
        if state.signal_entities is None or state.signal_entities.numel() == 0:
            raise ValueError("signal state is missing controlled signal entities")
        entities = state.signal_entities.to(entity_hidden.device)
        junction_history, historical_globals = self._signal_history(state, entities)
        token_type = self.token_type_embedding.weight
        for historical_global in historical_globals:
            tokens.append(historical_global + token_type[GLOBAL_TOKEN])
            kinds.append("GLOBAL")
        decisions: list[int] = []
        for local_index, entity in enumerate(entities):
            decisions.append(len(tokens))
            tokens.append(
                entity_hidden[entity]
                + junction_history[local_index]
                + token_type[STATE_TOKEN]
            )
            kinds.append("S")
        return decisions

    def _signal_history(
        self, state: PolicyState, signal_entities: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        device = signal_entities.device
        junction_count = signal_entities.numel()
        zeros = self.token_type_embedding.weight.new_zeros(
            (junction_count, self.hidden_dim)
        )
        if state.signal_history_dynamic is None or state.signal_history_actions is None:
            return zeros, []
        dynamic = state.signal_history_dynamic.to(device)
        actions = state.signal_history_actions.to(device)
        if dynamic.ndim != 3 or dynamic.shape[1:] != (
            junction_count,
            SIGNAL_DYNAMIC_DIM,
        ):
            raise ValueError("signal history dynamic tensor has an invalid shape")
        if actions.shape != dynamic.shape[:2]:
            raise ValueError("signal history actions do not align with dynamic history")
        if dynamic.shape[0] == 0:
            return zeros, []
        if bool(((actions < 0) | (actions >= MAX_ACTIONS)).any().item()):
            raise ValueError("signal history contains an invalid action")

        action_hidden = self.action_embedding(actions)
        history_input = torch.cat((dynamic, action_hidden), dim=-1)
        # The GRU is shared across junctions; junctions form the batch axis.
        output, _ = self.signal_history_gru(history_input.transpose(0, 1))
        junction_history = output[:, -1]

        progress = state.signal_history_progress
        historical_globals: list[torch.Tensor] = []
        for index in range(history_input.shape[0]):
            pooled = history_input[index].mean(dim=0)
            value = self.history_global_projection(pooled)
            if progress is not None and progress.numel():
                scalar = float(progress[index].item())
                features = torch.tensor(
                    [scalar, math.sin(math.pi * scalar), math.cos(math.pi * scalar)],
                    dtype=value.dtype,
                    device=device,
                )
                value = value + self.progress_projection(features)
            historical_globals.append(value)
        return junction_history, historical_globals


def _sinusoidal_positions(
    length: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    even_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
    frequencies = torch.exp(even_indices * (-math.log(10_000.0) / hidden_dim))
    encoding = torch.zeros((length, hidden_dim), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    odd_count = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_count])
    return encoding.to(dtype=dtype)
