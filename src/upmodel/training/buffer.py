"""Trajectory storage with per-episode GAE and per-task normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from upmodel.state import PolicyState
from upmodel.tasks import TRAINING_TASKS, Task


@dataclass
class Transition:
    state: PolicyState
    action: int | tuple[int, ...] | torch.Tensor
    old_log_prob: float | torch.Tensor
    old_value: float
    reward: float
    raw_reward: float
    done: bool
    task: Task
    advantage: float = 0.0
    return_target: float = 0.0


class RolloutBuffer:
    def __init__(self) -> None:
        self.episodes: list[list[Transition]] = []

    def add_episode(
        self,
        transitions: list[Transition],
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if not transitions or not transitions[-1].done:
            raise ValueError("PPO buffer accepts complete episodes only")
        advantage = 0.0
        next_value = 0.0
        for transition in reversed(transitions):
            non_terminal = 0.0 if transition.done else 1.0
            delta = (
                transition.reward
                + gamma * next_value * non_terminal
                - transition.old_value
            )
            advantage = delta + gamma * gae_lambda * non_terminal * advantage
            transition.advantage = float(advantage)
            transition.return_target = float(advantage + transition.old_value)
            next_value = transition.old_value
        self.episodes.append(transitions)

    def flattened(self, *, normalize: bool = True) -> list[Transition]:
        transitions = [item for episode in self.episodes for item in episode]
        if normalize:
            for task in TRAINING_TASKS:
                task_items = [item for item in transitions if item.task == task]
                if not task_items:
                    continue
                values = torch.tensor([item.advantage for item in task_items])
                mean = float(values.mean().item())
                std = float(values.std(unbiased=False).item())
                for item in task_items:
                    item.advantage = (item.advantage - mean) / max(std, 1.0e-8)
        return transitions
