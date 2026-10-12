"""Atomic checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "step": int(step),
            "config": config,
            "metrics": metrics or {},
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
