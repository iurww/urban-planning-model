"""Hierarchical fixed-what/fixed-where multi-task actor-critic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from upmodel.model.decision import DecisionSequence, SASADecisionTransformer
from upmodel.model.encoder import UrbanEntityTransformer
from upmodel.state import PolicyState
from upmodel.tasks import MAX_ACTIONS, Task


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


class MultiTaskActorCritic(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        entity_layers: int = 2,
        decision_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers is not None:
            entity_layers = num_layers
            decision_layers = num_layers
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.entity_encoder = UrbanEntityTransformer(
            hidden_dim, entity_layers, attention_heads, dropout
        )
        self.decision_transformer = SASADecisionTransformer(
            hidden_dim, decision_layers, attention_heads, dropout
        )
        self.static_pointer_query = _projection(hidden_dim)
        self.static_pointer_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.signal_actor = _head(hidden_dim * 2, hidden_dim, MAX_ACTIONS)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def encoder(self) -> UrbanEntityTransformer:
        """Compatibility alias for the lower-level entity encoder."""
        return self.entity_encoder

    @torch.no_grad()
    def soften_actor_prior_(self, scale: float) -> None:
        """Scale SFT logits while preserving every unmasked greedy action."""
        if not 0.0 < scale <= 1.0:
            raise ValueError("actor prior logit scale must be in (0, 1]")
        _scale_last_linear(self.static_pointer_query, scale)
        _scale_last_linear(self.signal_actor, scale)

    def encode(
        self, state: PolicyState
    ) -> tuple[torch.Tensor, DecisionSequence, torch.Tensor]:
        """Return entity features and the upper causal decision sequence."""
        entities = self.encode_entities(state)
        sequence, progress = self._decision_context(state, entities)
        return entities, sequence, progress

    def encode_entities(self, state: PolicyState) -> torch.Tensor:
        """Encode one graph state, allowing static rollouts to cache the result."""
        device = self.device
        graph = state.graph
        return self.entity_encoder(
            features=graph.features.to(device),
            entity_types=graph.entity_types.to(device),
            edge_index=graph.edge_index.to(device),
            dynamic=state.dynamic.to(device),
        )

    def encode_entities_batch(
        self, states: Sequence[PolicyState]
    ) -> torch.Tensor:
        """Encode dynamic states sharing one topology as [B,N,H]."""
        if not states:
            raise ValueError("entity batch cannot be empty")
        graph = states[0].graph
        if any(state.graph is not graph for state in states):
            raise ValueError("entity batch states must share one UrbanGraph")
        device = self.device
        dynamic = torch.stack([state.dynamic for state in states]).to(device)
        return self.entity_encoder.forward_batch(
            features=graph.features.to(device),
            entity_types=graph.entity_types.to(device),
            edge_index=graph.edge_index.to(device),
            dynamic=dynamic,
        )

    def _decision_context(
        self, state: PolicyState, entities: torch.Tensor
    ) -> tuple[DecisionSequence, torch.Tensor]:
        progress = self._progress_features(state, device=entities.device)
        sequence = self.decision_transformer(entities, state, progress)
        return sequence, progress

    def _progress_features(
        self, state: PolicyState, *, device: torch.device
    ) -> torch.Tensor:
        return torch.tensor(
            [
                float(state.progress),
                math.sin(math.pi * float(state.progress)),
                math.cos(math.pi * float(state.progress)),
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, state: PolicyState) -> PolicyOutput:
        return self.forward_from_entities(state, self.encode_entities(state))

    def forward_heterogeneous(
        self, states: Sequence[PolicyState]
    ) -> list[PolicyOutput]:
        """Evaluate differently sized graphs in one padded inference batch."""
        if not states:
            raise ValueError("heterogeneous policy batch cannot be empty")
        device = self.device
        entities = self.entity_encoder.forward_heterogeneous(
            features=[state.graph.features.to(device) for state in states],
            entity_types=[state.graph.entity_types.to(device) for state in states],
            edge_indices=[state.graph.edge_index.to(device) for state in states],
            dynamics=[state.dynamic.to(device) for state in states],
        )
        progress = torch.stack(
            [self._progress_features(state, device=device) for state in states]
        )
        sequences = self.decision_transformer.forward_heterogeneous(
            entities, list(states), progress
        )
        return [
            self._output_from_context(state, entity, sequence, item_progress)
            for state, entity, sequence, item_progress in zip(
                states, entities, sequences, progress, strict=True
            )
        ]

    def policy_logits_from_entities(
        self, state: PolicyState, entities: torch.Tensor
    ) -> torch.Tensor:
        """Compute actor logits without building the critic graph."""
        sequence, _ = self._decision_context(state, entities)
        return self._actor_logits(state, entities, sequence)

    def policy_logits_batch_from_entities(
        self,
        states: Sequence[PolicyState],
        entities: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Batch the upper Transformer while retaining variable actor shapes."""
        entity_batch, sequences, _ = self._batch_decision_context(states, entities)
        return [
            self._actor_logits(state, entity, sequence)
            for state, entity, sequence in zip(
                states, entity_batch, sequences, strict=True
            )
        ]

    def forward_batch_from_entities(
        self,
        states: Sequence[PolicyState],
        entities: torch.Tensor,
    ) -> list[PolicyOutput]:
        """Evaluate actor and critic for a topology-local training batch."""
        entity_batch, sequences, progress = self._batch_decision_context(
            states, entities
        )
        return [
            self._output_from_context(state, entity, sequence, item_progress)
            for state, entity, sequence, item_progress in zip(
                states, entity_batch, sequences, progress, strict=True
            )
        ]

    def _batch_decision_context(
        self,
        states: Sequence[PolicyState],
        entities: torch.Tensor,
    ) -> tuple[torch.Tensor, list[DecisionSequence], torch.Tensor]:
        if not states:
            raise ValueError("policy batch cannot be empty")
        graph = states[0].graph
        if any(state.graph is not graph for state in states):
            raise ValueError("policy batch states must share one UrbanGraph")
        if entities.ndim not in {2, 3}:
            raise ValueError("policy entity batch has an invalid shape")
        if entities.shape[-2] != graph.num_entities:
            raise ValueError("policy entity batch does not match the graph")
        if entities.ndim == 2:
            entity_batch = entities.unsqueeze(0).expand(len(states), -1, -1)
        elif entities.ndim == 3 and entities.shape[0] == len(states):
            entity_batch = entities
        else:
            raise ValueError("policy entity batch has an invalid shape")
        progress = torch.stack(
            [
                self._progress_features(state, device=entity_batch.device)
                for state in states
            ]
        )
        sequences = self.decision_transformer.forward_batch(
            entity_batch,
            list(states),
            progress,
        )
        return entity_batch, sequences, progress

    def forward_from_entities(
        self, state: PolicyState, entities: torch.Tensor
    ) -> PolicyOutput:
        sequence, progress = self._decision_context(state, entities)
        return self._output_from_context(state, entities, sequence, progress)

    def _output_from_context(
        self,
        state: PolicyState,
        entities: torch.Tensor,
        sequence: DecisionSequence,
        progress: torch.Tensor,
    ) -> PolicyOutput:
        device = entities.device
        graph = state.graph
        logits = self._actor_logits(state, entities, sequence)

        pooled = []
        for entity_indices in (
            graph.parcel_entities,
            graph.road_entities,
            graph.intersection_entities,
        ):
            if entity_indices.numel() == 0:
                pooled.append(torch.zeros(entities.shape[-1], device=device))
            else:
                pooled.append(entities[entity_indices.to(device)].mean(dim=0))
        query = sequence.hidden[sequence.query_index]
        critic_input = torch.cat(
            (*pooled, sequence.hidden[0], query, progress), dim=-1
        )
        value = self.value_head(critic_input).squeeze(-1)
        return PolicyOutput(logits=logits, value=value)

    def _actor_logits(
        self,
        state: PolicyState,
        entities: torch.Tensor,
        sequence: DecisionSequence,
    ) -> torch.Tensor:
        device = entities.device
        query = sequence.hidden[sequence.query_index]
        if state.task in {Task.LAND, Task.ROAD}:
            if state.candidate_entities is None:
                raise ValueError(f"{state.task.label} state is missing pointer candidates")
            candidates = state.candidate_entities.to(device)
            if candidates.ndim != 1 or candidates.numel() == 0:
                raise ValueError("static pointer candidates must be a nonempty vector")
            query_hidden = self.static_pointer_query(query)
            keys = self.static_pointer_key(entities[candidates])
            logits = torch.matmul(keys, query_hidden) / math.sqrt(self.hidden_dim)
        elif state.task == Task.SIGNAL:
            junctions = sequence.hidden[sequence.decision_indices]
            centralized_query = query.expand(junctions.shape[0], -1)
            logits = self.signal_actor(
                torch.cat((junctions, centralized_query), dim=-1)
            )
        else:
            raise ValueError(f"unsupported training task: {state.task.label}")

        mask = state.action_mask
        if mask.shape != logits.shape:
            raise ValueError(
                f"action mask shape {tuple(mask.shape)} does not match logits "
                f"{tuple(logits.shape)}"
            )
        if not bool(mask.any(dim=-1).all().item()):
            raise ValueError("each action distribution must contain at least one legal action")
        mask = mask.to(device)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        return logits

    def distribution(self, state: PolicyState) -> tuple[Categorical, torch.Tensor]:
        output = self(state)
        return Categorical(logits=output.logits), output.value

    def act(
        self, state: PolicyState, *, deterministic: bool = False
    ) -> tuple[int | torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(state)
        action_tensor = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        log_prob, entropy = reduce_distribution_stats(
            state.task, distribution, action_tensor
        )
        if state.task == Task.SIGNAL:
            action: int | torch.Tensor = action_tensor.detach().cpu()
        else:
            action = int(action_tensor.item())
        return action, log_prob, value, entropy


def evaluate_action(
    model: MultiTaskActorCritic,
    state: PolicyState,
    action: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, value = model.distribution(state)
    action_tensor = torch.as_tensor(action, dtype=torch.long, device=model.device)
    log_prob, entropy = reduce_distribution_stats(state.task, distribution, action_tensor)
    return log_prob, value, entropy, distribution.logits


def evaluate_action_components(
    model: MultiTaskActorCritic,
    state: PolicyState,
    action: int | tuple[int, ...] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unclumped action factors for a mathematically valid PPO ratio."""
    return evaluate_action_components_from_output(state, action, model(state))


def evaluate_action_components_from_output(
    state: PolicyState,
    action: int | tuple[int, ...] | torch.Tensor,
    output: PolicyOutput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate an action from a precomputed, potentially batched output."""
    distribution = Categorical(logits=output.logits)
    if state.task == Task.LAND and isinstance(action, tuple):
        log_prob = ranked_action_log_prob(
            distribution.logits,
            state.action_mask.to(output.logits.device),
            action,
        )
    else:
        action_tensor = torch.as_tensor(
            action, dtype=torch.long, device=output.logits.device
        )
        log_prob = distribution.log_prob(action_tensor)
    return log_prob, output.value, distribution.entropy(), distribution.logits


def ranked_action_log_prob(
    logits: torch.Tensor,
    mask: torch.Tensor,
    prefix: tuple[int, ...],
) -> torch.Tensor:
    """Log probability of a without-replacement Plackett-Luce prefix."""
    if logits.ndim != 1 or mask.shape != logits.shape:
        raise ValueError("ranked action logits and mask must be aligned vectors")
    if not prefix:
        raise ValueError("ranked action prefix cannot be empty")
    available = mask.bool().clone()
    total = logits.new_zeros(())
    for raw_action in prefix:
        action = int(raw_action)
        if action < 0 or action >= logits.numel() or not bool(available[action]):
            raise ValueError("ranked action prefix contains an illegal or repeated action")
        step_logits = logits.masked_fill(~available, -torch.inf)
        total = total + step_logits.log_softmax(dim=0)[action]
        available[action] = False
    return total


def ranked_candidates(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Rank legal proposals under a Plackett-Luce policy."""
    if logits.ndim != 1 or mask.shape != logits.shape:
        raise ValueError("candidate logits and mask must be aligned vectors")
    indices = mask.to(logits.device).bool().nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return []
    legal_logits = logits[indices]
    if deterministic:
        order = torch.argsort(legal_logits, descending=True)
    else:
        order = torch.multinomial(
            legal_logits.softmax(dim=0),
            num_samples=len(indices),
            replacement=False,
            generator=generator,
        )
    return [int(indices[index].item()) for index in order]


def reduce_distribution_stats(
    task: Task, distribution: Categorical, action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    per_action_log_prob = distribution.log_prob(action)
    if task == Task.SIGNAL:
        # A centralized signal transition has a variable number of factorized
        # categorical outputs. The mean keeps the PPO scale topology-invariant.
        return per_action_log_prob.mean(), distribution.entropy().mean()
    return per_action_log_prob, distribution.entropy()


def categorical_kl(
    task: Task,
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    current_log_prob = current_logits.log_softmax(dim=-1)
    reference_log_prob = reference_logits.log_softmax(dim=-1)
    current_prob = current_log_prob.exp()
    per_distribution = (
        current_prob * (current_log_prob - reference_log_prob)
    ).sum(dim=-1)
    return per_distribution.mean() if task == Task.SIGNAL else per_distribution


def _projection(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _scale_last_linear(module: nn.Sequential, scale: float) -> None:
    output_layer = module[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("actor decoder must end in nn.Linear")
    output_layer.weight.mul_(scale)
    if output_layer.bias is not None:
        output_layer.bias.mul_(scale)
