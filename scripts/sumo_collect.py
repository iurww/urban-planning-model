#!/usr/bin/env python3
# coding: utf-8

"""Collect canonical eight-action MaxPressure expert trajectories in SUMO."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.sumo_demand import (
    DEFAULT_DEPART_END_S,
    DEFAULT_VEHICLE_COUNT,
    build_sumo_net,
    generate_demand,
)
from upmodel.sumo_tls import (
    TlsLinkOrder,
    adapt_controller_to_compiled,
    compiled_tls_link_orders,
    validate_traci_controlled_links,
)

SCHEMA_VERSION = 6
DECISION_INTERVAL_S = 15.0
YELLOW_S = 3.0
GREEN_S = 12.0
STEP_LENGTH_S = 1.0
HORIZON_S = 1200.0
NUM_DECISIONS = 80
TIME_TO_TELEPORT_S = 600


def _sumo_home() -> Path:
    import sumo  # noqa: WPS433

    if getattr(sumo, "SUMO_HOME", None):
        return Path(sumo.SUMO_HOME)
    if "SUMO_HOME" in os.environ:
        return Path(os.environ["SUMO_HOME"])
    raise RuntimeError("找不到 SUMO_HOME。")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"缺少文件: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: 顶层必须是 JSON 对象")
    return value


def _controller_meta(
    controller: dict[str, Any], order: TlsLinkOrder
) -> dict[str, Any]:
    controller = adapt_controller_to_compiled(controller, order)
    program = controller["program"]
    actions = sorted(program["actions"], key=lambda row: int(row["action_id"]))
    if [int(row["action_id"]) for row in actions] != list(range(8)):
        raise ValueError(f"{controller['id']}: canonical action ids 必须严格为 0..7")
    mask = [bool(value) for value in controller["available_phase_mask"]]
    if len(mask) != 8 or not any(mask):
        raise ValueError(f"{controller['id']}: 非法 available_phase_mask={mask}")
    for index, action in enumerate(actions):
        if bool(action["available"]) != mask[index]:
            raise ValueError(f"{controller['id']}: action/mask 在 {index} 不一致")
        if mask[index] and (
            not action.get("green_state") or not action.get("yellow_state")
        ):
            raise ValueError(f"{controller['id']}: 合法 action {index} 缺少灯态")

    incoming_lanes: list[str] = []
    for slot in controller.get("approach_slots", {}).values():
        incoming_lanes.extend(str(value) for value in slot.get("incoming_lane_ids", []))
    incoming_lanes = list(dict.fromkeys(incoming_lanes))
    states = [
        str(row.get("green_state") or program["all_red_state"]) for row in actions
    ]
    yellow_states = [
        str(row.get("yellow_state") or program["all_red_state"]) for row in actions
    ]
    return {
        "junction_id": str(controller["intersection_id"]),
        "source_id": str(controller["intersection_id"]),
        "controller_id": str(controller["id"]),
        "tls_id": str(controller["sumo_tl_id"]),
        "available_phase_mask": mask,
        # Canonical ids, not compact local indices or raw SUMO phases.
        "green_phases": [index for index, available in enumerate(mask) if available],
        "phase_states": states,
        "yellow_states": yellow_states,
        "initial_action_id": int(program["initial_action_id"]),
        "all_red_state": str(program["all_red_state"]),
        "incoming_lanes": incoming_lanes,
        "observation_lane_ids": [
            str(value) for value in controller.get("observation_lane_ids", [])
        ],
        "controlled_links": list(controller["controlled_links"]),
        "actions": actions,
    }


def _build_signal_meta(
    tile_dir: Path, contract: dict[str, Any], demand: dict[str, Any]
) -> dict[str, Any]:
    orders = compiled_tls_link_orders(contract, tile_dir / "sumo" / "net.xml")
    junctions = [
        _controller_meta(row, orders[str(row["sumo_tl_id"])])
        for row in contract["controllers"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_type": "signal_control",
        "topology_ref": "topology.json",
        "signals_ref": "traffic_signals.json",
        "net_file": "sumo/net.xml",
        "route_file": "sumo/routes.xml",
        "phase_representation": "canonical_action_id_0_to_7",
        "transition": (
            "changed_action:3_seconds_current_yellow_then_12_seconds_target_green;"
            "unchanged_action:hold_current_green_for_15_seconds"
        ),
        "control": {
            "delta_t": DECISION_INTERVAL_S,
            "yellow_s": YELLOW_S,
            "green_s": GREEN_S,
            "step_length": STEP_LENGTH_S,
            "horizon": HORIZON_S,
            "warmup": 0.0,
            "num_steps": NUM_DECISIONS,
            "time_to_teleport_s": TIME_TO_TELEPORT_S,
        },
        "termination": {
            "type": "fixed_horizon",
            "require_all_departed": False,
            "require_all_arrived": False,
            "require_actual_road_coverage": False,
        },
        "demand": demand,
        "reward": {
            "type": "negative_mean_queue",
            "normalization": "number_of_signalized_junctions",
            "terminal_wait_weight": 0.0,
        },
        "junctions": junctions,
    }


def _lane_observation(traci: Any, lanes: set[str]) -> tuple[dict[str, int], dict[str, float], dict[str, int]]:
    halt: dict[str, int] = {}
    wait: dict[str, float] = {}
    vehicles: dict[str, int] = {}
    for lane in lanes:
        try:
            halt[lane] = int(traci.lane.getLastStepHaltingNumber(lane))
            wait[lane] = float(traci.lane.getWaitingTime(lane))
            vehicles[lane] = int(traci.lane.getLastStepVehicleNumber(lane))
        except traci.TraCIException:
            halt[lane] = 0
            wait[lane] = 0.0
            vehicles[lane] = 0
    return halt, wait, vehicles


def _action_pressure(
    junction: dict[str, Any], action_id: int, halt: dict[str, int]
) -> int:
    state = str(junction["phase_states"][action_id])
    links = junction["controlled_links"]
    pressure = 0
    for link_index, character in enumerate(state):
        if character not in "Gg" or link_index >= len(links):
            continue
        link = links[link_index]
        pressure += halt.get(str(link["from_lane_id"]), 0)
        pressure -= halt.get(str(link["to_lane_id"]), 0)
    return int(pressure)


def _validate_fixed_horizon_episode(
    steps: list[dict[str, Any]],
    *,
    final_sim_time_s: float,
    expected_vehicles: int,
    departed: int,
    arrived: int,
) -> None:
    """Validate trajectory integrity without requiring demand completion."""
    errors: list[str] = []
    if len(steps) != NUM_DECISIONS:
        errors.append(f"decisions={len(steps)} != {NUM_DECISIONS}")
    if not math.isclose(
        final_sim_time_s,
        HORIZON_S,
        rel_tol=0.0,
        abs_tol=max(STEP_LENGTH_S * 1.0e-6, 1.0e-9),
    ):
        errors.append(f"final_sim_time={final_sim_time_s} != {HORIZON_S}")
    for index, step in enumerate(steps):
        expected_t = index + 1
        expected_sim_time = index * DECISION_INTERVAL_S
        if int(step.get("t", -1)) != expected_t:
            errors.append(f"step[{index}].t={step.get('t')} != {expected_t}")
            break
        if not math.isclose(
            float(step.get("sim_time", -1.0)),
            expected_sim_time,
            rel_tol=0.0,
            abs_tol=max(STEP_LENGTH_S * 1.0e-6, 1.0e-9),
        ):
            errors.append(
                f"step[{index}].sim_time={step.get('sim_time')} "
                f"!= {expected_sim_time}"
            )
            break
    if expected_vehicles < 0:
        errors.append(f"expected_vehicles={expected_vehicles} < 0")
    if departed < 0 or departed > expected_vehicles:
        errors.append(
            f"departed={departed} outside 0..expected={expected_vehicles}"
        )
    if arrived < 0 or arrived > departed:
        errors.append(f"arrived={arrived} outside 0..departed={departed}")
    if errors:
        raise RuntimeError(
            "SUMO fixed-horizon episode contract failed: " + "; ".join(errors)
        )


def _collect(tile_dir: Path, *, seed: int, gui: bool = False) -> tuple[dict, dict]:
    import sumolib  # noqa: WPS433
    import traci  # noqa: WPS433

    contract = _read_json(tile_dir / "traffic_signals.json")
    demand = _read_json(tile_dir / "sumo" / "demand_manifest.json")
    meta = _build_signal_meta(tile_dir, contract, demand)
    net_path = tile_dir / "sumo" / "net.xml"
    route_path = tile_dir / "sumo" / "routes.xml"

    home = _sumo_home()
    os.environ["SUMO_HOME"] = str(home)
    binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    traci.start(
        [
            binary,
            "-n",
            str(net_path),
            "-r",
            str(route_path),
            "--step-length",
            str(STEP_LENGTH_S),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            str(TIME_TO_TELEPORT_S),
            "--seed",
            str(int(seed)),
            "--duration-log.disable",
            "true",
        ]
    )

    junctions = meta["junctions"]
    expected_tls = {str(row["tls_id"]) for row in junctions}
    actual_tls = set(traci.trafficlight.getIDList())
    if actual_tls != expected_tls:
        traci.close()
        raise ValueError(
            f"SUMO TLS 与清洗契约不一致: missing={sorted(expected_tls-actual_tls)[:5]}, "
            f"extra={sorted(actual_tls-expected_tls)[:5]}"
        )
    for row in junctions:
        controlled = traci.trafficlight.getControlledLinks(str(row["tls_id"]))
        try:
            validate_traci_controlled_links(
                str(row["tls_id"]), row["controlled_links"], controlled
            )
        except ValueError:
            traci.close()
            raise

    observed_lanes: set[str] = set()
    for row in junctions:
        observed_lanes.update(str(value) for value in row["incoming_lanes"])
        observed_lanes.update(
            str(link["to_lane_id"]) for link in row["controlled_links"]
        )
    edge_to_road = {
        str(row["id"]): str(row["topology_road_id"])
        for row in contract["sumo_network"]["directed_edges"]
    }
    required_roads = set(edge_to_road.values())
    visited_edges: set[str] = set()
    departed_route_edges: set[str] = set()

    current_action = {
        str(row["tls_id"]): int(row["initial_action_id"]) for row in junctions
    }
    green_elapsed = {str(row["tls_id"]): 0.0 for row in junctions}
    for row in junctions:
        action = current_action[str(row["tls_id"])]
        traci.trafficlight.setRedYellowGreenState(
            str(row["tls_id"]), str(row["phase_states"][action])
        )

    arrived = 0
    departed = 0
    total_wait_accumulated = 0.0
    teleport_started = 0
    teleport_ended = 0
    steps: list[dict[str, Any]] = []
    remaining_by_edge: dict[str, int] = {}
    final_sim_time_s = 0.0
    running_at_horizon = 0
    pending_at_horizon = 0
    min_expected_at_horizon = 0

    try:
        for decision in range(NUM_DECISIONS):
            sim_time = float(traci.simulation.getTime())
            halt, wait, vehicles = _lane_observation(traci, observed_lanes)
            records: list[dict[str, Any]] = []
            selected_actions: dict[str, int] = {}
            for row in junctions:
                tls_id = str(row["tls_id"])
                mask = [bool(value) for value in row["available_phase_mask"]]
                pressures = [
                    _action_pressure(row, action_id, halt) if mask[action_id] else 0
                    for action_id in range(8)
                ]
                previous = current_action[tls_id]
                legal_actions = [index for index, legal in enumerate(mask) if legal]
                selected = max(
                    legal_actions,
                    key=lambda action_id: (
                        pressures[action_id],
                        action_id == previous,
                        -action_id,
                    ),
                )
                selected_actions[tls_id] = selected
                incoming = [str(value) for value in row["incoming_lanes"]]
                records.append(
                    {
                        "junction_id": row["junction_id"],
                        "source_id": row["source_id"],
                        "state": {
                            "cur_phase": int(previous),
                            "phase_elapsed": float(green_elapsed[tls_id]),
                            "phase_pressure": pressures,
                            "queue": int(sum(halt.get(lane, 0) for lane in incoming)),
                            "wait": round(sum(wait.get(lane, 0.0) for lane in incoming), 3),
                            "veh": int(sum(vehicles.get(lane, 0) for lane in incoming)),
                        },
                        "action": {
                            "action_id": int(selected),
                            "green_idx": int(selected),
                        },
                    }
                )

            changed_tls = {
                tls_id
                for tls_id, selected in selected_actions.items()
                if selected != current_action[tls_id]
            }
            # Only a real action change needs a yellow transition.  Keeping the
            # same action preserves its complete green/red state for all 15 s.
            for row in junctions:
                tls_id = str(row["tls_id"])
                if tls_id not in changed_tls:
                    continue
                previous = current_action[tls_id]
                traci.trafficlight.setRedYellowGreenState(
                    tls_id, str(row["yellow_states"][previous])
                )
            for substep in range(int(DECISION_INTERVAL_S)):
                if substep == int(YELLOW_S):
                    for row in junctions:
                        tls_id = str(row["tls_id"])
                        if tls_id not in changed_tls:
                            continue
                        selected = selected_actions[tls_id]
                        green_elapsed[tls_id] = 0.0
                        current_action[tls_id] = selected
                        traci.trafficlight.setRedYellowGreenState(
                            tls_id, str(row["phase_states"][selected])
                        )
                traci.simulationStep()
                arrived += int(traci.simulation.getArrivedNumber())
                departed += int(traci.simulation.getDepartedNumber())
                teleport_started += int(traci.simulation.getStartingTeleportNumber())
                teleport_ended += int(traci.simulation.getEndingTeleportNumber())
                for vehicle_id in traci.simulation.getDepartedIDList():
                    try:
                        departed_route_edges.update(traci.vehicle.getRoute(vehicle_id))
                    except traci.TraCIException:
                        pass
                for vehicle_id in traci.vehicle.getIDList():
                    try:
                        edge_id = str(traci.vehicle.getRoadID(vehicle_id))
                    except traci.TraCIException:
                        continue
                    if edge_id in edge_to_road:
                        visited_edges.add(edge_id)
                for tls_id in green_elapsed:
                    if tls_id not in changed_tls or substep >= int(YELLOW_S):
                        green_elapsed[tls_id] += STEP_LENGTH_S

            halt_after, wait_after, _ = _lane_observation(traci, observed_lanes)
            total_halt = int(sum(halt_after.values()))
            total_wait = float(sum(wait_after.values()))
            total_wait_accumulated += total_wait
            reward = -float(total_halt) / max(len(junctions), 1)
            steps.append(
                {
                    "t": decision + 1,
                    "sim_time": sim_time,
                    "global_state": {
                        "running": int(traci.vehicle.getIDCount()),
                        "arrived_cum": int(arrived),
                        "departed_cum": int(departed),
                        "total_halt": total_halt,
                    },
                    "junctions": records,
                    "reward": reward,
                }
            )
        for vehicle_id in traci.vehicle.getIDList():
            edge_id = str(traci.vehicle.getRoadID(vehicle_id))
            remaining_by_edge[edge_id] = remaining_by_edge.get(edge_id, 0) + 1
        final_sim_time_s = float(traci.simulation.getTime())
        running_at_horizon = int(traci.vehicle.getIDCount())
        pending_at_horizon = len(traci.simulation.getPendingVehicles())
        min_expected_at_horizon = int(traci.simulation.getMinExpectedNumber())
    finally:
        traci.close()

    route_roads = {edge_to_road[edge] for edge in departed_route_edges if edge in edge_to_road}
    visited_roads = {edge_to_road[edge] for edge in visited_edges if edge in edge_to_road}
    expected_vehicles = int(demand["vehicle_count"])
    _validate_fixed_horizon_episode(
        steps,
        final_sim_time_s=final_sim_time_s,
        expected_vehicles=expected_vehicles,
        departed=departed,
        arrived=arrived,
    )

    trajectory = {
        "task_type": "signal_control",
        "topology_ref": "topology.json",
        "meta_ref": "signal_meta.json",
        "expert": "max_pressure_canonical_8_action",
        "objective": "MaxPressure over legal canonical actions",
        "metrics": {
            "arrived": int(arrived),
            "departed": int(departed),
            "expected_vehicles": expected_vehicles,
            "unfinished_vehicles": max(expected_vehicles - arrived, 0),
            "pending_departure_at_horizon": int(pending_at_horizon),
            "running_at_horizon": int(running_at_horizon),
            "sumo_min_expected_at_horizon": int(min_expected_at_horizon),
            "final_sim_time_s": float(final_sim_time_s),
            "mean_total_halt": round(
                sum(step["global_state"]["total_halt"] for step in steps)
                / max(len(steps), 1),
                3,
            ),
            "total_wait_accumulated": round(total_wait_accumulated, 3),
            "teleport_started": int(teleport_started),
            "teleport_ended": int(teleport_ended),
            "required_topology_roads": len(required_roads),
            "route_covered_topology_roads": len(route_roads),
            "actually_visited_topology_roads": len(visited_roads),
            "road_coverage": len(visited_roads) / max(len(required_roads), 1),
            "remaining_by_edge": dict(sorted(remaining_by_edge.items())),
        },
        "num_signalized": len(junctions),
        "num_steps": len(steps),
        "final_reward": 0.0,
        "steps": steps,
    }
    return meta, trajectory


def _remove_existing_signal_expert(tile_dir: Path) -> None:
    for name in ("signal_meta.json", "signal_trajectory.json"):
        (tile_dir / name).unlink(missing_ok=True)


def _try_one(tile_dir: Path, args: argparse.Namespace) -> tuple[bool, str | None]:
    if not (tile_dir / "topology.json").is_file():
        return False, "缺少 topology.json"
    trajectory_path = tile_dir / "signal_trajectory.json"
    if getattr(args, "skip_existing", False) and trajectory_path.is_file():
        return False, "已存在（跳过）"
    try:
        _remove_existing_signal_expert(tile_dir)
        net_path = build_sumo_net(tile_dir, force=bool(args.force))
        generate_demand(
            tile_dir,
            net_path,
            seed=int(args.seed),
            vehicle_count=int(args.vehicle_count),
            depart_end=float(args.depart_end),
            force=True,
        )
        meta, trajectory = _collect(
            tile_dir, seed=int(args.seed), gui=bool(getattr(args, "gui", False))
        )
        (tile_dir / "signal_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        trajectory_path.write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> None:
    tile_dir = Path(args.input)
    if tile_dir.suffix.lower() in {".json", ".gpkg"}:
        tile_dir = tile_dir.parent
    ok, error = _try_one(tile_dir, args)
    if not ok:
        raise SystemExit(f"[sumo-collect] 失败: {error}")
    print(f"[sumo-collect] OK: {tile_dir}")


def _worker(payload: tuple[str, dict[str, Any]]) -> tuple[str, bool, str | None]:
    path, values = payload
    args = argparse.Namespace(**values)
    good, error = _try_one(Path(path), args)
    return path, good, error


def run_batch(args: argparse.Namespace) -> None:
    base = Path(args.topology_base)
    tiles = sorted(
        child for child in base.iterdir()
        if child.is_dir() and (child / "topology.json").is_file()
    )
    if args.limit:
        tiles = tiles[: int(args.limit)]
    values = vars(args).copy()
    values.pop("func", None)
    payloads = [(str(tile), values) for tile in tiles]
    workers = max(1, int(args.workers))
    successes = skips = failures = 0
    failure_rows: list[tuple[str, str]] = []

    def consume(path: str, good: bool, error: str | None) -> None:
        nonlocal successes, skips, failures
        relative = str(Path(path).relative_to(base))
        if good:
            successes += 1
            print(f"[{successes + skips + failures}/{len(tiles)}] OK {relative}")
        elif error and error.endswith("（跳过）"):
            skips += 1
            print(f"[{successes + skips + failures}/{len(tiles)}] SKIP {relative}")
        else:
            failures += 1
            failure_rows.append((relative, error or "unknown"))
            print(f"[{successes + skips + failures}/{len(tiles)}] FAIL {relative}: {error}")

    if workers == 1:
        for payload in payloads:
            consume(*_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                consume(*future.result())
    print(
        f"[sumo-collect-batch] 成功 {successes}, 跳过 {skips}, "
        f"失败 {failures}, 共 {len(tiles)}"
    )
    if failure_rows:
        for name, reason in failure_rows[:20]:
            print(f"  {name}: {reason}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=DEFAULT_VEHICLE_COUNT,
        help="OD 中的精确车辆数",
    )
    parser.add_argument(
        "--depart-end",
        type=float,
        default=DEFAULT_DEPART_END_S,
        help="最后一辆车之前的发车时间窗（秒）",
    )
    parser.add_argument("--force", action="store_true", help="重建路网并覆盖旧专家数据")
    parser.add_argument("--gui", action="store_true", help="使用 sumo-gui")


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sumo-collect", help="生成固定 80 步的 MaxPressure 专家轨迹"
    )
    parser.add_argument("input", help="tile 目录或其 topology.json 路径")
    _add_common(parser)
    parser.set_defaults(func=run)

    batch = subparsers.add_parser(
        "sumo-collect-batch", help="批量生成固定契约的 MaxPressure 专家轨迹"
    )
    batch.add_argument("--topology-base", default="data/train")
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--skip-existing", action="store_true")
    batch.add_argument("--verbose", action="store_true")
    _add_common(batch)
    batch.set_defaults(func=run_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO MaxPressure 轨迹采集")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_subcommand(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
