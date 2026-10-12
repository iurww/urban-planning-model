from types import SimpleNamespace

import torch

from upmodel.envs.signal import SumoSignalEnv


class _FakeTrafficLight:
    def __init__(self, owner: "_FakeConnection") -> None:
        self.owner = owner
        self.states: list[tuple[float, str, str]] = []

    def setRedYellowGreenState(self, tls_id: str, state: str) -> None:
        self.states.append((self.owner.time, tls_id, state))


class _FakeSimulation:
    @staticmethod
    def getArrivedNumber() -> int:
        return 0

    @staticmethod
    def getDepartedNumber() -> int:
        return 0


class _FakeLane:
    @staticmethod
    def getLastStepHaltingNumber(_lane_id: str) -> int:
        return 0

    @staticmethod
    def getWaitingTime(_lane_id: str) -> float:
        return 0.0


class _FakeConnection:
    def __init__(self) -> None:
        self.time = 0.0
        self.trafficlight = _FakeTrafficLight(self)
        self.simulation = _FakeSimulation()
        self.lane = _FakeLane()

    def simulationStep(self) -> None:
        self.time += 1.0


def _environment() -> tuple[SumoSignalEnv, _FakeConnection]:
    connection = _FakeConnection()
    environment = SumoSignalEnv.__new__(SumoSignalEnv)
    environment.connection = connection
    environment._closed = False
    environment.graph = SimpleNamespace(
        signal_entities=torch.tensor([0]),
        signal_action_mask=torch.tensor([[True, True]]),
    )
    environment.meta = {
        "junctions": [
            {
                "tls_id": "tls0",
                "phase_states": ["G0", "G1"],
                "yellow_states": ["Y0", "Y1"],
                "incoming_lanes": [],
            }
        ]
    }
    environment.history_steps = 0
    environment._last_dynamic = torch.zeros((1, 18))
    environment._last_progress = 0.0
    environment.current_action = {"tls0": 0}
    environment.phase_elapsed = {"tls0": 5.0}
    environment.num_substeps = 3
    environment.yellow_s = 1.0
    environment.step_length = 1.0
    environment.position = 0
    environment.num_steps = 1
    environment.arrived = 0
    environment.departed = 0
    environment.total_wait_accumulated = 0.0
    environment.reward_type = "negative_mean_queue"
    environment.close_on_done = False
    return environment, connection


def test_signal_env_holds_same_action_without_yellow() -> None:
    environment, connection = _environment()

    environment.step([0])

    assert connection.time == 3.0
    assert connection.trafficlight.states == []
    assert environment.current_action == {"tls0": 0}
    assert environment.phase_elapsed == {"tls0": 8.0}


def test_signal_env_uses_yellow_only_when_action_changes() -> None:
    environment, connection = _environment()

    environment.step([1])

    assert connection.time == 3.0
    assert connection.trafficlight.states == [
        (0.0, "tls0", "Y0"),
        (1.0, "tls0", "G1"),
    ]
    assert environment.current_action == {"tls0": 1}
    assert environment.phase_elapsed == {"tls0": 2.0}
