#!/usr/bin/env python3
# coding: utf-8

"""Replay a saved signal-control expert trajectory in SUMO-GUI."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upmodel.sumo_tls import validate_traci_controlled_links


@dataclass(frozen=True)
class ReplayPlan:
    tile_dir: Path
    net_path: Path
    route_path: Path
    junctions: tuple[dict[str, Any], ...]
    selected_actions: tuple[dict[str, int], ...]
    step_length_s: float
    decision_steps: int
    yellow_steps: int
    time_to_teleport_s: float
    seed: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"缺少回放文件: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: JSON 无效: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: 顶层必须是 JSON 对象")
    return value


def _resolve_tile_dir(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"tile 目录不存在: {path}")
    return path


def _whole_steps(duration_s: float, step_length_s: float, label: str) -> int:
    if duration_s < 0.0 or step_length_s <= 0.0:
        raise ValueError(
            f"{label}={duration_s}、step_length={step_length_s} 必须为有效正数"
        )
    ratio = duration_s / step_length_s
    result = int(round(ratio))
    if not math.isclose(ratio, result, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{label}={duration_s} 不是 step_length={step_length_s} 的整数倍")
    return result


def load_replay_plan(value: str | Path) -> ReplayPlan:
    """Load and fully validate replay inputs before SUMO-GUI is launched."""
    tile_dir = _resolve_tile_dir(value)
    meta = _read_json(tile_dir / "signal_meta.json")
    trajectory = _read_json(tile_dir / "signal_trajectory.json")

    net_path = tile_dir / str(meta.get("net_file", "sumo/net.xml"))
    route_path = tile_dir / str(meta.get("route_file", "sumo/routes.xml"))
    for path in (net_path, route_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少回放文件: {path}")

    control = meta.get("control")
    if not isinstance(control, dict):
        raise ValueError("signal_meta.json 缺少 control")
    step_length_s = float(control["step_length"])
    delta_t_s = float(control["delta_t"])
    yellow_s = float(control["yellow_s"])
    green_s = float(control["green_s"])
    decision_steps = _whole_steps(delta_t_s, step_length_s, "delta_t")
    yellow_steps = _whole_steps(yellow_s, step_length_s, "yellow_s")
    if yellow_steps >= decision_steps:
        raise ValueError("yellow_s 必须小于 delta_t")
    if not math.isclose(
        yellow_s + green_s, delta_t_s, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("yellow_s + green_s 必须等于 delta_t")

    raw_junctions = meta.get("junctions")
    if not isinstance(raw_junctions, list) or not raw_junctions:
        raise ValueError("signal_meta.json 缺少 junctions")
    junctions = tuple(raw_junctions)
    junction_ids = [str(row["junction_id"]) for row in junctions]
    if len(set(junction_ids)) != len(junction_ids):
        raise ValueError("signal_meta.json 包含重复 junction_id")

    current_actions: dict[str, int] = {}
    for row in junctions:
        junction_id = str(row["junction_id"])
        initial_action = int(row["initial_action_id"])
        phase_states = row.get("phase_states")
        yellow_states = row.get("yellow_states")
        mask = row.get("available_phase_mask")
        if not isinstance(phase_states, list) or not isinstance(yellow_states, list):
            raise ValueError(f"{junction_id}: 缺少 phase_states/yellow_states")
        if not isinstance(mask, list) or len(mask) != len(phase_states):
            raise ValueError(f"{junction_id}: available_phase_mask 长度不一致")
        if not 0 <= initial_action < len(phase_states) or not bool(mask[initial_action]):
            raise ValueError(f"{junction_id}: initial_action_id 非法")
        current_actions[junction_id] = initial_action

    raw_steps = trajectory.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("signal_trajectory.json 缺少 steps")
    expected_num_steps = int(control["num_steps"])
    if len(raw_steps) != expected_num_steps:
        raise ValueError(
            f"专家轨迹步数={len(raw_steps)}，control.num_steps={expected_num_steps}"
        )

    selected_actions: list[dict[str, int]] = []
    expected_junction_ids = set(junction_ids)
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise ValueError(f"轨迹第 {index + 1} 步不是 JSON 对象")
        if int(step.get("t", -1)) != index + 1:
            raise ValueError(f"轨迹第 {index + 1} 步的 t 不连续")
        expected_time = index * delta_t_s
        actual_time = float(step.get("sim_time", float("nan")))
        if not math.isclose(
            actual_time, expected_time, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"轨迹第 {index + 1} 步 sim_time={actual_time}，预期 {expected_time}"
            )

        records = step.get("junctions")
        if not isinstance(records, list):
            raise ValueError(f"轨迹第 {index + 1} 步缺少 junctions")
        by_junction: dict[str, dict[str, Any]] = {}
        for record in records:
            junction_id = str(record["junction_id"])
            if junction_id in by_junction:
                raise ValueError(f"轨迹第 {index + 1} 步重复路口 {junction_id}")
            by_junction[junction_id] = record
        if set(by_junction) != expected_junction_ids:
            missing = sorted(expected_junction_ids - set(by_junction))
            extra = sorted(set(by_junction) - expected_junction_ids)
            raise ValueError(
                f"轨迹第 {index + 1} 步路口集合不一致: missing={missing}, extra={extra}"
            )

        selections: dict[str, int] = {}
        for row in junctions:
            junction_id = str(row["junction_id"])
            record = by_junction[junction_id]
            recorded_current = int(record["state"]["cur_phase"])
            if recorded_current != current_actions[junction_id]:
                raise ValueError(
                    f"轨迹第 {index + 1} 步 {junction_id} cur_phase="
                    f"{recorded_current}，预期 {current_actions[junction_id]}"
                )
            selected = int(record["action"]["action_id"])
            mask = row["available_phase_mask"]
            if not 0 <= selected < len(mask) or not bool(mask[selected]):
                raise ValueError(
                    f"轨迹第 {index + 1} 步 {junction_id} action_id={selected} 非法"
                )
            selections[junction_id] = selected
        selected_actions.append(selections)
        current_actions.update(selections)

    demand = meta.get("demand", {})
    seed = int(demand.get("seed", 0)) if isinstance(demand, dict) else 0
    return ReplayPlan(
        tile_dir=tile_dir,
        net_path=net_path,
        route_path=route_path,
        junctions=junctions,
        selected_actions=tuple(selected_actions),
        step_length_s=step_length_s,
        decision_steps=decision_steps,
        yellow_steps=yellow_steps,
        time_to_teleport_s=float(control["time_to_teleport_s"]),
        seed=seed,
    )


def replay(
    plan: ReplayPlan,
    *,
    delay_ms: float,
    traci_module: Any | None = None,
    sumolib_module: Any | None = None,
    sumo_binary: str = "sumo-gui",
) -> None:
    """Drive SUMO-GUI with the recorded expert actions."""
    if delay_ms < 0.0:
        raise ValueError("--delay-ms 不能为负数")
    if traci_module is None or sumolib_module is None:
        import sumo
        import sumolib
        import traci

        if getattr(sumo, "SUMO_HOME", None):
            os.environ["SUMO_HOME"] = str(sumo.SUMO_HOME)
        elif "SUMO_HOME" not in os.environ:
            raise RuntimeError("找不到 SUMO_HOME")
        traci_module = traci
        sumolib_module = sumolib

    if sumo_binary not in {"sumo", "sumo-gui"}:
        raise ValueError(f"不支持的 SUMO binary: {sumo_binary}")
    binary = sumolib_module.checkBinary(sumo_binary)
    command = [
        binary,
        "-n",
        str(plan.net_path),
        "-r",
        str(plan.route_path),
        "--step-length",
        str(plan.step_length_s),
        "--time-to-teleport",
        str(plan.time_to_teleport_s),
        "--seed",
        str(plan.seed),
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
        "--duration-log.disable",
        "true",
    ]
    if sumo_binary == "sumo-gui":
        command.extend(["--delay", str(float(delay_ms))])

    started = False
    try:
        traci_module.start(command)
        started = True
        expected_tls = {str(row["tls_id"]) for row in plan.junctions}
        actual_tls = set(traci_module.trafficlight.getIDList())
        if actual_tls != expected_tls:
            raise ValueError(
                "SUMO TLS 与专家轨迹不一致: "
                f"missing={sorted(expected_tls - actual_tls)}, "
                f"extra={sorted(actual_tls - expected_tls)}"
            )
        for row in plan.junctions:
            tls_id = str(row["tls_id"])
            validate_traci_controlled_links(
                tls_id,
                row["controlled_links"],
                traci_module.trafficlight.getControlledLinks(tls_id),
            )

        current_actions = {
            str(row["junction_id"]): int(row["initial_action_id"])
            for row in plan.junctions
        }
        for row in plan.junctions:
            junction_id = str(row["junction_id"])
            traci_module.trafficlight.setRedYellowGreenState(
                str(row["tls_id"]),
                str(row["phase_states"][current_actions[junction_id]]),
            )

        for decision, selections in enumerate(plan.selected_actions):
            expected_time = decision * plan.decision_steps * plan.step_length_s
            actual_time = float(traci_module.simulation.getTime())
            if not math.isclose(
                actual_time, expected_time, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise RuntimeError(
                    f"SUMO 时间={actual_time}，专家第 {decision + 1} 步预期 "
                    f"{expected_time}"
                )

            changed_junctions = {
                junction_id
                for junction_id, selected in selections.items()
                if selected != current_actions[junction_id]
            }
            for row in plan.junctions:
                junction_id = str(row["junction_id"])
                if junction_id not in changed_junctions:
                    continue
                traci_module.trafficlight.setRedYellowGreenState(
                    str(row["tls_id"]),
                    str(row["yellow_states"][current_actions[junction_id]]),
                )
            for substep in range(plan.decision_steps):
                if substep == plan.yellow_steps:
                    for row in plan.junctions:
                        junction_id = str(row["junction_id"])
                        if junction_id not in changed_junctions:
                            continue
                        selected = selections[junction_id]
                        traci_module.trafficlight.setRedYellowGreenState(
                            str(row["tls_id"]),
                            str(row["phase_states"][selected]),
                        )
                        current_actions[junction_id] = selected
                traci_module.simulationStep()
    finally:
        if started:
            try:
                traci_module.close()
            except Exception:  # noqa: BLE001
                pass


def run(args: argparse.Namespace) -> None:
    plan = load_replay_plan(args.input)
    print(
        f"[sumo-replay] tile={plan.tile_dir}，专家步数={len(plan.selected_actions)}，"
        f"仿真时长={len(plan.selected_actions) * plan.decision_steps * plan.step_length_s:g}s"
    )
    replay(plan, delay_ms=float(args.delay_ms))
    print("[sumo-replay] 播放完成")


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sumo-replay",
        help="用 SUMO-GUI 只读回放已有 MaxPressure 专家轨迹",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="tile 目录或其中任一专家文件")
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=50.0,
        help="每个 1 秒仿真步的 GUI 播放延迟（毫秒）；越大播放越慢",
    )
    parser.set_defaults(func=run)


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO-GUI 专家轨迹回放")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subcommand(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
