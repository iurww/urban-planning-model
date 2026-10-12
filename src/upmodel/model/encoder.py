"""Lower-level Transformer that encodes urban entities and graph structure."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from upmodel.state import SIGNAL_DYNAMIC_DIM, STATIC_FEATURE_DIM
from upmodel.tasks import NUM_ENTITY_TYPES


class UrbanEntityTransformer(nn.Module):
    """Encode entities without leaking task, action-history, or decision information."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.static_projection = nn.Sequential(
            nn.Linear(STATIC_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dynamic_projection = nn.Linear(SIGNAL_DYNAMIC_DIM, hidden_dim, bias=False)
        self.entity_embedding = nn.Embedding(NUM_ENTITY_TYPES, hidden_dim)
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
        *,
        features: torch.Tensor,
        entity_types: torch.Tensor,
        edge_index: torch.Tensor,
        dynamic: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_batch(
            features=features,
            entity_types=entity_types,
            edge_index=edge_index,
            dynamic=dynamic.unsqueeze(0),
        ).squeeze(0)

    def forward_batch(
        self,
        *,
        features: torch.Tensor,
        entity_types: torch.Tensor,
        edge_index: torch.Tensor,
        dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """Encode several dynamic states of one topology in a real GPU batch."""
        if features.ndim != 2 or dynamic.ndim != 3:
            raise ValueError("batched entity features must have shapes [N,F] and [B,N,D]")
        if (
            features.shape[0] != entity_types.shape[0]
            or features.shape[0] != dynamic.shape[1]
        ):
            raise ValueError("entity feature tensors must contain the same number of entities")
        hidden = (
            self.static_projection(features).unsqueeze(0)
            + self.dynamic_projection(dynamic)
            + self.entity_embedding(entity_types).unsqueeze(0)
        )
        attention_mask = _graph_attention_mask(
            hidden.shape[1], edge_index, device=hidden.device
        )
        return self.transformer(hidden, mask=attention_mask)

    def forward_heterogeneous(
        self,
        *,
        features: Sequence[torch.Tensor],
        entity_types: Sequence[torch.Tensor],
        edge_indices: Sequence[torch.Tensor],
        dynamics: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Encode differently sized topologies in one padded Transformer batch."""
        batch_size = len(features)
        if not batch_size or not (
            len(entity_types)
            == len(edge_indices)
            == len(dynamics)
            == batch_size
        ):
            raise ValueError("heterogeneous entity inputs must be nonempty and aligned")
        lengths = torch.tensor(
            [item.shape[0] for item in features],
            dtype=torch.long,
            device=features[0].device,
        )
        for static, types, dynamic in zip(
            features, entity_types, dynamics, strict=True
        ):
            if static.ndim != 2 or dynamic.ndim != 2:
                raise ValueError("heterogeneous entity features must be matrices")
            if static.shape[0] != types.shape[0] or static.shape[0] != dynamic.shape[0]:
                raise ValueError("heterogeneous graph entity counts must align")

        padded_features = nn.utils.rnn.pad_sequence(
            list(features), batch_first=True
        )
        padded_dynamic = nn.utils.rnn.pad_sequence(
            list(dynamics), batch_first=True
        )
        padded_types = nn.utils.rnn.pad_sequence(
            list(entity_types), batch_first=True
        )
        hidden = (
            self.static_projection(padded_features)
            + self.dynamic_projection(padded_dynamic)
            + self.entity_embedding(padded_types)
        )
        max_entities = hidden.shape[1]
        positions = torch.arange(max_entities, device=hidden.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        blocked = torch.zeros(
            (batch_size, max_entities, max_entities),
            dtype=torch.bool,
            device=hidden.device,
        )
        for index, (length, edge_index) in enumerate(
            zip(lengths.tolist(), edge_indices, strict=True)
        ):
            blocked[index, :length, :length] = _graph_attention_mask(
                length, edge_index, device=hidden.device
            )
            blocked[index, :length, length:] = True
        attention_mask = blocked.repeat_interleave(self.num_heads, dim=0)
        encoded = self.transformer(
            hidden,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        return [
            encoded[index, :length]
            for index, length in enumerate(lengths.tolist())
        ]


def _graph_attention_mask(
    num_entities: int,
    edge_index: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Allow each entity to attend to itself and its direct graph neighbours."""
    blocked = torch.ones(
        (num_entities, num_entities), dtype=torch.bool, device=device
    )
    blocked.fill_diagonal_(False)
    if edge_index.numel() > 0:
        source, destination = edge_index
        blocked[destination, source] = False
    return blocked


# Import compatibility for code that used the old encoder class name.
UrbanGraphEncoder = UrbanEntityTransformer
