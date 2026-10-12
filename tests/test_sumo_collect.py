from __future__ import annotations

import copy

import pytest

from scripts.sumo_collect import (
    DECISION_INTERVAL_S,
    HORIZON_S,
    NUM_DECISIONS,
    _validate_fixed_horizon_episode,
)


def _complete_steps() -> list[dict[str, float | int]]:
    return [
        {
            "t": index + 1,
            "sim_time": index * DECISION_INTERVAL_S,
        }
        for index in range(NUM_DECISIONS)
    ]


def test_fixed_horizon_accepts_unfinished_and_pending_vehicles() -> None:
    _validate_fixed_horizon_episode(
        _complete_steps(),
        final_sim_time_s=HORIZON_S,
        expected_vehicles=2000,
        departed=1992,
        arrived=1149,
    )


def test_fixed_horizon_still_rejects_incomplete_or_inconsistent_trajectory() -> None:
    with pytest.raises(RuntimeError, match="decisions=79"):
        _validate_fixed_horizon_episode(
            _complete_steps()[:-1],
            final_sim_time_s=HORIZON_S,
            expected_vehicles=2000,
            departed=1992,
            arrived=1149,
        )

    wrong_time = copy.deepcopy(_complete_steps())
    wrong_time[20]["sim_time"] = -1.0
    with pytest.raises(RuntimeError, match=r"step\[20\]\.sim_time"):
        _validate_fixed_horizon_episode(
            wrong_time,
            final_sim_time_s=HORIZON_S,
            expected_vehicles=2000,
            departed=1992,
            arrived=1149,
        )

    with pytest.raises(RuntimeError, match="arrived=1993"):
        _validate_fixed_horizon_episode(
            _complete_steps(),
            final_sim_time_s=HORIZON_S,
            expected_vehicles=2000,
            departed=1992,
            arrived=1993,
        )
