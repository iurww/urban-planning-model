"""Online SUMO environment for centralized multi-intersection signal PPO."""

from __future__ import annotations

import fcntl
import io
import logging
import os
import subprocess
import uuid
from collections import deque
from contextlib import redirect_stdout
from typing import Any

import torch

from upmodel.data.signal import encode_signal_observation
from upmodel.state import PolicyState, UrbanGraph
from upmodel.sumo_tls import validate_traci_controlled_links
from upmodel.tasks import Task

LOGGER = logging.getLogger(__name__)
_TRACI_START_LOCK = f"/tmp/upmodel-traci-{os.getuid()}.lock"


class SumoSignalEnv:
    def __init__(
        self,
        graph: UrbanGraph,
        *,
        seed: int | None = None,
        history_steps: int = 4,
        close_on_done: bool = True,
    ) -> None:
        if graph.signal_meta is None or graph.signal_entities.numel() == 0:
            raise ValueError(f"{graph.tile_id}: signal metadata is unavailable")
        if history_steps < 0:
            raise ValueError("signal history_steps cannot be negative")
        self.graph = graph
        self.meta = graph.signal_meta
        self.history_steps = int(history_steps)
        self.close_on_done = bool(close_on_done)
        self.seed = int(self.meta.get("demand", {}).get("seed", 0) if seed is None else seed)
        control = self.meta.get("control", {})
        self.delta_t = float(control.get("delta_t", 10.0))
        self.step_length = float(control.get("step_length", 1.0))
        self.horizon = float(control.get("horizon", 300.0))
        self.warmup = float(control.get("warmup", 0.0))
        self.num_steps = max(int(round(self.horizon / self.delta_t)), 1)
        self.num_substeps = max(int(round(self.delta_t / self.step_length)), 1)
        self.yellow_s = float(control.get("yellow_s", 3.0))
        self.green_s = float(control.get("green_s", self.delta_t - self.yellow_s))
        self.time_to_teleport_s = int(control.get("time_to_teleport_s", 600))
        self.num_warmup_steps = max(int(round(self.warmup / self.step_length)), 0)
        self.reward_type = str(self.meta.get("reward", {}).get("type", "legacy_wait"))
        self.connection: Any | None = None
        self.links: dict[str, Any] = {}
        self.incoming_lanes: set[str] = set()
        self.phase_elapsed: dict[str, float] = {}
        self.current_action: dict[str, int] = {}
        self.position = 0
        self.arrived = 0
        self.departed = 0
        self.total_wait_accumulated = 0.0
        self.history_dynamic: deque[torch.Tensor] = deque(maxlen=self.history_steps)
        self.history_actions: deque[torch.Tensor] = deque(maxlen=self.history_steps)
        self.history_progress: deque[float] = deque(maxlen=self.history_steps)
        self._last_dynamic: torch.Tensor | None = None
        self._last_progress = 0.0
        self._closed = True

    def reset(self) -> PolicyState:
        self.close()
        try:
            import sumolib
            import traci
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SUMO PPO requires the eclipse-sumo package (sumolib and traci)"
            ) from exc
        net_path = self.graph.tile_dir / str(self.meta.get("net_file", "sumo/net.xml"))
        route_path = self.graph.tile_dir / str(self.meta.get("route_file", "sumo/routes.xml"))
        if not net_path.is_file() or not route_path.is_file():
            raise FileNotFoundError(
                f"{self.graph.tile_id}: missing SUMO net/route file: {net_path}, {route_path}"
            )
        binary = sumolib.checkBinary("sumo")
        label = f"upmodel-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        _start_traci(
            traci,
            [
                binary,
                "-n",
                str(net_path),
                "-r",
                str(route_path),
                "--step-length",
                str(self.step_length),
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
                "--time-to-teleport",
                str(self.time_to_teleport_s),
                "--seed",
                str(self.seed),
                "--duration-log.disable",
                "true",
            ],
            label,
        )
        self.connection = traci.getConnection(label)
        self._closed = False
        available = set(self.connection.trafficlight.getIDList())
        self.links = {}
        self.incoming_lanes = set()
        self.phase_elapsed = {}
        self.current_action = {}
        for junction in self.meta["junctions"]:
            tls_id = str(junction["tls_id"])
            if tls_id not in available:
                self.close()
                raise ValueError(f"{self.graph.tile_id}: SUMO is missing traffic light {tls_id}")
            self.links[tls_id] = list(junction.get("controlled_links", []))
            try:
                validate_traci_controlled_links(
                    tls_id,
                    self.links[tls_id],
                    self.connection.trafficlight.getControlledLinks(tls_id),
                )
            except ValueError:
                self.close()
                raise
            self.incoming_lanes.update(str(lane) for lane in junction.get("incoming_lanes", []))
            self.phase_elapsed[tls_id] = 0.0
            initial = int(junction.get("initial_action_id", junction["green_phases"][0]))
            self.current_action[tls_id] = initial
            self.connection.trafficlight.setRedYellowGreenState(
                tls_id, str(junction["phase_states"][initial])
            )
        self.position = 0
        self.arrived = 0
        self.departed = 0
        self.total_wait_accumulated = 0.0
        self.history_dynamic.clear()
        self.history_actions.clear()
        self.history_progress.clear()
        self._last_dynamic = None
        self._last_progress = 0.0
        for _ in range(self.num_warmup_steps):
            self.connection.simulationStep()
            self.arrived += int(self.connection.simulation.getArrivedNumber())
            self.departed += int(self.connection.simulation.getDepartedNumber())
        return self._state()

    def step(
        self, action: torch.Tensor | list[int]
    ) -> tuple[PolicyState | None, float, bool, dict[str, float]]:
        if self.connection is None or self._closed:
            raise RuntimeError("SUMO environment must be reset before step")
        actions = torch.as_tensor(action, dtype=torch.long).flatten()
        if actions.shape != self.graph.signal_entities.shape:
            raise ValueError(
                f"signal action shape {tuple(actions.shape)} does not match "
                f"{tuple(self.graph.signal_entities.shape)}"
            )
        legal = self.graph.signal_action_mask.gather(1, actions[:, None]).all()
        if not bool(legal.item()):
            raise ValueError("signal action contains an invalid green-phase index")
        if self._last_dynamic is None:
            raise RuntimeError("signal state was not observed before step")
        if self.history_steps:
            self.history_dynamic.append(
                self._last_dynamic[self.graph.signal_entities].clone()
            )
            self.history_actions.append(actions.clone())
            self.history_progress.append(self._last_progress)

        selected_by_tls = {
            str(junction["tls_id"]): int(actions[local_index].item())
            for local_index, junction in enumerate(self.meta["junctions"])
        }
        changed_tls = {
            tls_id
            for tls_id, selected in selected_by_tls.items()
            if selected != self.current_action[tls_id]
        }
        for junction in self.meta["junctions"]:
            tls_id = str(junction["tls_id"])
            if tls_id not in changed_tls:
                continue
            previous = self.current_action[tls_id]
            yellow_states = junction.get("yellow_states")
            if yellow_states is None:
                yellow_states = [
                    "".join("y" if char in "Gg" else "r" for char in state)
                    for state in junction["phase_states"]
                ]
            self.connection.trafficlight.setRedYellowGreenState(
                tls_id, str(yellow_states[previous])
            )

        for substep in range(self.num_substeps):
            if substep == int(round(self.yellow_s / self.step_length)):
                for junction in self.meta["junctions"]:
                    tls_id = str(junction["tls_id"])
                    if tls_id not in changed_tls:
                        continue
                    selected = selected_by_tls[tls_id]
                    self.phase_elapsed[tls_id] = 0.0
                    self.current_action[tls_id] = selected
                    self.connection.trafficlight.setRedYellowGreenState(
                        tls_id, str(junction["phase_states"][selected])
                    )
            self.connection.simulationStep()
            self.arrived += int(self.connection.simulation.getArrivedNumber())
            self.departed += int(self.connection.simulation.getDepartedNumber())
            for tls_id in self.phase_elapsed:
                if (
                    tls_id not in changed_tls
                    or substep >= int(round(self.yellow_s / self.step_length))
                ):
                    self.phase_elapsed[tls_id] += self.step_length

        halt_after = 0
        wait_after = 0.0
        for junction in self.meta["junctions"]:
            for lane in junction.get("incoming_lanes", []):
                try:
                    halt_after += int(self.connection.lane.getLastStepHaltingNumber(lane))
                    wait_after += float(self.connection.lane.getWaitingTime(lane))
                except Exception as exc:
                    if not _is_traci_exception(exc):
                        raise
        self.total_wait_accumulated += wait_after
        self.position += 1
        done = self.position >= self.num_steps
        if self.reward_type == "negative_mean_queue":
            terminal_reward = 0.0
            reward = -float(halt_after) / max(len(self.meta["junctions"]), 1)
        else:
            terminal_reward = -float(self.total_wait_accumulated) if done else 0.0
            reward = -float(halt_after) + terminal_reward
        info = {
            "total_halt": float(halt_after),
            "total_wait": float(wait_after),
            "arrived": float(self.arrived),
            "departed": float(self.departed),
            "total_wait_accumulated": float(self.total_wait_accumulated),
            "terminal_reward": terminal_reward,
        }
        if self.reward_type != "negative_mean_queue":
            info["final_wait_reward"] = -float(self.total_wait_accumulated)
        if done:
            if self.close_on_done:
                self.close()
            return None, reward, True, info
        return self._state(), reward, False, info

    def close(self) -> None:
        if self.connection is not None and not self._closed:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = None
        self._closed = True

    def _state(self) -> PolicyState:
        junctions, global_state = self._observation()
        history, dynamic = encode_signal_observation(self.graph, junctions, global_state)
        progress = self.position / max(self.num_steps, 1)
        self._last_dynamic = dynamic
        self._last_progress = progress
        signal_count = len(self.graph.signal_entities)
        return PolicyState(
            graph=self.graph,
            task=Task.SIGNAL,
            progress=progress,
            history=history,
            dynamic=dynamic,
            action_mask=self.graph.signal_action_mask,
            signal_entities=self.graph.signal_entities,
            signal_history_dynamic=(
                torch.stack(tuple(self.history_dynamic))
                if self.history_dynamic
                else torch.empty((0, signal_count, dynamic.shape[-1]))
            ),
            signal_history_actions=(
                torch.stack(tuple(self.history_actions))
                if self.history_actions
                else torch.empty((0, signal_count), dtype=torch.long)
            ),
            signal_history_progress=torch.tensor(
                tuple(self.history_progress), dtype=torch.float32
            ),
        )

    def _observation(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        assert self.connection is not None
        halt: dict[str, int] = {}
        wait: dict[str, float] = {}
        vehicles: dict[str, int] = {}
        for lane in self.incoming_lanes:
            try:
                halt[lane] = int(self.connection.lane.getLastStepHaltingNumber(lane))
                wait[lane] = float(self.connection.lane.getWaitingTime(lane))
                vehicles[lane] = int(self.connection.lane.getLastStepVehicleNumber(lane))
            except Exception as exc:
                if not _is_traci_exception(exc):
                    raise
                halt[lane] = 0
                wait[lane] = 0.0
                vehicles[lane] = 0

        records = []
        for junction in self.meta["junctions"]:
            tls_id = str(junction["tls_id"])
            controlled_links = self.links[tls_id]
            pressures = []
            states = list(junction.get("phase_states", []))
            canonical = str(self.meta.get("phase_representation", "")).startswith(
                "canonical_action_id"
            )
            phase_ids = range(8) if canonical else junction.get("green_phases", [])
            action_mask = [bool(value) for value in junction.get("available_phase_mask", [])]
            for phase in phase_ids:
                if canonical and action_mask and not action_mask[int(phase)]:
                    pressures.append(0)
                    continue
                state = states[int(phase)]
                pressure = 0
                for link_index, character in enumerate(state):
                    if character not in "Gg" or link_index >= len(controlled_links):
                        continue
                    connection = controlled_links[link_index]
                    incoming = str(connection["from_lane_id"])
                    outgoing = str(connection["to_lane_id"])
                    pressure += halt.get(incoming, 0)
                    if outgoing:
                        try:
                            pressure -= int(
                                self.connection.lane.getLastStepHaltingNumber(outgoing)
                            )
                        except Exception as exc:
                            if not _is_traci_exception(exc):
                                raise
                pressures.append(pressure)
            incoming_lanes = [str(value) for value in junction.get("incoming_lanes", [])]
            records.append(
                {
                    "junction_id": junction.get("junction_id"),
                    "source_id": junction.get("source_id"),
                    "state": {
                        "cur_phase": int(self.current_action[tls_id]),
                        "phase_elapsed": float(self.phase_elapsed[tls_id]),
                        "phase_pressure": pressures,
                        "queue": sum(halt.get(lane, 0) for lane in incoming_lanes),
                        "wait": sum(wait.get(lane, 0.0) for lane in incoming_lanes),
                        "veh": sum(vehicles.get(lane, 0) for lane in incoming_lanes),
                    },
                }
            )
        global_state = {
            "running": float(self.connection.vehicle.getIDCount()),
            "arrived_cum": float(self.arrived),
            "departed_cum": float(self.departed),
            "total_halt": float(sum(halt.values())),
        }
        return records, global_state

    def __enter__(self) -> "SumoSignalEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _is_traci_exception(exc: Exception) -> bool:
    return exc.__class__.__module__.startswith("traci")


def _start_traci(traci: Any, command: list[str], label: str) -> None:
    """Serialize SUMO startup and keep retry output away from progress bars."""
    startup_output = io.StringIO()
    try:
        with open(_TRACI_START_LOCK, "w", encoding="ascii") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with redirect_stdout(startup_output):
                traci.start(
                    command,
                    label=label,
                    stdout=subprocess.DEVNULL,
                    # Never install an episode connection as TraCI's process-global
                    # default. Parallel rollout only accesses named connections.
                    doSwitch=False,
                )
    except Exception:
        captured = startup_output.getvalue().strip()
        if captured:
            LOGGER.error("TraCI 启动失败前的输出：%s", captured)
        raise
    captured = startup_output.getvalue().strip()
    if captured:
        LOGGER.debug("TraCI 启动重试后连接成功：%s", captured)
