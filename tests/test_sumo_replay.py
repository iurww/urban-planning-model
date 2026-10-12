from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sumo_replay import add_subcommand, load_replay_plan, replay


class _FakeTrafficLight:
    def __init__(self, owner: "_FakeTraci") -> None:
        self.owner = owner
        self.states: list[tuple[float, str, str]] = []

    def getIDList(self) -> list[str]:
        return ["tls0"]

    def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str]]]:
        assert tls_id == "tls0"
        return [
            [("in_0", "out_0", ":via_0")],
            [("in_1", "out_1", ":via_1")],
        ]

    def setRedYellowGreenState(self, tls_id: str, state: str) -> None:
        self.states.append((self.owner.time, tls_id, state))


class _FakeSimulation:
    def __init__(self, owner: "_FakeTraci") -> None:
        self.owner = owner

    def getTime(self) -> float:
        return self.owner.time


class _FakeTraci:
    def __init__(self) -> None:
        self.time = 0.0
        self.command: list[str] | None = None
        self.closed = False
        self.trafficlight = _FakeTrafficLight(self)
        self.simulation = _FakeSimulation(self)

    def start(self, command: list[str]) -> None:
        self.command = command

    def simulationStep(self) -> None:
        self.time += 1.0

    def close(self) -> None:
        self.closed = True


class _FakeSumolib:
    @staticmethod
    def checkBinary(name: str) -> str:
        return f"/fake/{name}"


class SumoReplayTest(unittest.TestCase):
    def _write_tile(self, root: Path) -> Path:
        tile = root / "tile"
        sumo_dir = tile / "sumo"
        sumo_dir.mkdir(parents=True)
        (sumo_dir / "net.xml").write_text("<net/>", encoding="utf-8")
        (sumo_dir / "routes.xml").write_text("<routes/>", encoding="utf-8")
        meta = {
            "net_file": "sumo/net.xml",
            "route_file": "sumo/routes.xml",
            "control": {
                "delta_t": 3.0,
                "yellow_s": 1.0,
                "green_s": 2.0,
                "step_length": 1.0,
                "num_steps": 2,
                "time_to_teleport_s": 600.0,
            },
            "demand": {"seed": 7},
            "junctions": [
                {
                    "junction_id": "j0",
                    "tls_id": "tls0",
                    "available_phase_mask": [True, True],
                    "phase_states": ["G0", "G1"],
                    "yellow_states": ["Y0", "Y1"],
                    "initial_action_id": 0,
                    "controlled_links": [
                        {
                            "link_index": 0,
                            "from_lane_id": "in_0",
                            "to_lane_id": "out_0",
                        },
                        {
                            "link_index": 1,
                            "from_lane_id": "in_1",
                            "to_lane_id": "out_1",
                        },
                    ],
                }
            ],
        }
        trajectory = {
            "steps": [
                {
                    "t": 1,
                    "sim_time": 0.0,
                    "junctions": [
                        {
                            "junction_id": "j0",
                            "state": {"cur_phase": 0},
                            "action": {"action_id": 0},
                        }
                    ],
                },
                {
                    "t": 2,
                    "sim_time": 3.0,
                    "junctions": [
                        {
                            "junction_id": "j0",
                            "state": {"cur_phase": 0},
                            "action": {"action_id": 1},
                        }
                    ],
                },
            ]
        }
        (tile / "signal_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        (tile / "signal_trajectory.json").write_text(
            json.dumps(trajectory), encoding="utf-8"
        )
        return tile

    def test_cli_accepts_tile_and_delay(self) -> None:
        parser = argparse.ArgumentParser()
        add_subcommand(parser.add_subparsers(dest="command", required=True))

        args = parser.parse_args(["sumo-replay", "datadoc/train/tile", "--delay-ms", "75"])

        self.assertEqual(args.input, Path("datadoc/train/tile"))
        self.assertEqual(args.delay_ms, 75.0)

    def test_replay_holds_same_action_and_only_yellows_on_change(self) -> None:
        with TemporaryDirectory() as directory:
            tile = self._write_tile(Path(directory))
            plan = load_replay_plan(tile)
            fake_traci = _FakeTraci()

            replay(
                plan,
                delay_ms=25.0,
                traci_module=fake_traci,
                sumolib_module=_FakeSumolib(),
            )

            self.assertEqual(fake_traci.time, 6.0)
            self.assertTrue(fake_traci.closed)
            self.assertIn("--delay", fake_traci.command or [])
            teleport_index = (fake_traci.command or []).index("--time-to-teleport")
            self.assertEqual((fake_traci.command or [])[teleport_index + 1], "600.0")
            self.assertEqual(
                fake_traci.trafficlight.states,
                [
                    (0.0, "tls0", "G0"),
                    (3.0, "tls0", "Y0"),
                    (4.0, "tls0", "G1"),
                ],
            )

    def test_plan_rejects_broken_phase_continuity(self) -> None:
        with TemporaryDirectory() as directory:
            tile = self._write_tile(Path(directory))
            path = tile / "signal_trajectory.json"
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            trajectory["steps"][1]["junctions"][0]["state"]["cur_phase"] = 1
            path.write_text(json.dumps(trajectory), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cur_phase"):
                load_replay_plan(tile)


if __name__ == "__main__":
    unittest.main()
