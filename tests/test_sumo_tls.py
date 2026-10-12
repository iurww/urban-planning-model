import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.sumo_collect import _build_signal_meta
from scripts.sumo_demand import _plain_xml
from upmodel.sumo_tls import (
    compiled_tls_link_orders,
    validate_traci_controlled_links,
)

TLS_ID = "tls_intersection_i0010"
MOVEMENTS = ("N_S", "N_E", "E_N", "E_S", "S_E", "S_N")
CONNECTIONS = (
    ("north_in", 1, "south_out", 1),
    ("north_in", 2, "east_out", 2),
    ("east_in", 0, "north_out", 0),
    ("east_in", 2, "south_out", 2),
    ("south_in", 0, "east_out", 0),
    ("south_in", 1, "north_out", 1),
)
# The observed i0010 class of netconvert reorder:
# contract [N_S,N_E,E_N,E_S,S_E,S_N] -> actual [E_N,E_S,S_E,S_N,N_S,N_E].
CONTRACT_TO_ACTUAL = (4, 5, 0, 1, 2, 3)


def _contract() -> dict:
    links = []
    network_connections = []
    for index, (movement, connection) in enumerate(
        zip(MOVEMENTS, CONNECTIONS, strict=True)
    ):
        from_edge, from_lane, to_edge, to_lane = connection
        link = {
            "link_index": index,
            "connection_id": f"connection_{movement}",
            "from_edge_id": from_edge,
            "to_edge_id": to_edge,
            "from_lane_id": f"{from_edge}_{from_lane}",
            "to_lane_id": f"{to_edge}_{to_lane}",
            "from_lane_index": from_lane,
            "to_lane_index": to_lane,
            "movement_key": movement,
        }
        links.append(link)
        network_connections.append(
            {
                **link,
                "sumo_tl_id": TLS_ID,
            }
        )
    actions = []
    for action_id in range(8):
        available = action_id in {1, 2}
        actions.append(
            {
                "action_id": action_id,
                "available": available,
                "green_state": "Ggrrrg" if available else None,
                "yellow_state": "yyrrrr" if available else None,
                "protected_link_indices": [0, 1] if available else [],
                "permissive_right_link_indices": [4, 5] if available else [],
            }
        )
    controller = {
        "id": "signal_controller/i0010",
        "intersection_id": "intersection_i0010",
        "sumo_tl_id": TLS_ID,
        "available_phase_mask": [action_id in {1, 2} for action_id in range(8)],
        "approach_slots": {"N": {"incoming_lane_ids": ["north_in_1", "north_in_2"]}},
        "observation_lane_ids": ["north_in_1", "north_in_2"],
        "controlled_links": links,
        "program": {
            "actions": actions,
            "initial_action_id": 1,
            "all_red_state": "rrrrrr",
            "bootstrap_tl_logic": {
                "type": "static",
                "program_id": "rl_bootstrap",
                "offset_s": 0,
                "phases": [
                    {
                        "duration_s": 1000,
                        "state": "Ggrrrg",
                        "name": "unsafe_contract_order",
                    }
                ],
            },
        },
    }
    return {
        "controllers": [controller],
        "sumo_network": {
            "nodes": [],
            "directed_edges": [],
            "connections": network_connections,
        },
    }


def _write_net(
    path: Path,
    contract: dict,
    *,
    actual_indices: tuple[int, ...] = CONTRACT_TO_ACTUAL,
) -> None:
    root = ET.Element("net")
    controller = contract["controllers"][0]
    for link, actual_index in zip(
        controller["controlled_links"], actual_indices, strict=True
    ):
        ET.SubElement(
            root,
            "connection",
            {
                "from": link["from_edge_id"],
                "to": link["to_edge_id"],
                "fromLane": str(link["from_lane_index"]),
                "toLane": str(link["to_lane_index"]),
                "tl": TLS_ID,
                "linkIndex": str(actual_index),
            },
        )
    logic = ET.SubElement(root, "tlLogic", {"id": TLS_ID})
    ET.SubElement(logic, "phase", {"duration": "1000", "state": "rrrrrr"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_i0010_non_identity_order_preserves_each_physical_movement(
    tmp_path: Path,
) -> None:
    contract = _contract()
    sumo_dir = tmp_path / "sumo"
    sumo_dir.mkdir()
    _write_net(sumo_dir / "net.xml", contract)

    meta = _build_signal_meta(tmp_path, contract, {"seed": 7})
    junction = meta["junctions"][0]
    assert meta["schema_version"] == 6
    assert meta["control"]["time_to_teleport_s"] == 600
    assert "unchanged_action:hold_current_green_for_15_seconds" in meta["transition"]
    assert meta["termination"] == {
        "type": "fixed_horizon",
        "require_all_departed": False,
        "require_all_arrived": False,
        "require_actual_road_coverage": False,
    }
    assert [link["movement_key"] for link in junction["controlled_links"]] == [
        "E_N",
        "E_S",
        "S_E",
        "S_N",
        "N_S",
        "N_E",
    ]
    assert [link["link_index"] for link in junction["controlled_links"]] == list(
        range(6)
    )

    original_action = contract["controllers"][0]["program"]["actions"][1]
    adapted_action = junction["actions"][1]
    assert junction["phase_states"][1] == "rrrgGg"
    assert junction["yellow_states"][1] == "rrrryy"
    assert adapted_action["protected_link_indices"] == [4, 5]
    assert adapted_action["permissive_right_link_indices"] == [2, 3]
    for contract_index, actual_index in enumerate(CONTRACT_TO_ACTUAL):
        assert (
            junction["controlled_links"][actual_index]["movement_key"]
            == MOVEMENTS[contract_index]
        )
        assert (
            junction["phase_states"][1][actual_index]
            == original_action["green_state"][contract_index]
        )
        assert (
            junction["yellow_states"][1][actual_index]
            == original_action["yellow_state"][contract_index]
        )


def test_plain_tls_bootstrap_is_order_independent_and_covers_every_link(
    tmp_path: Path,
) -> None:
    contract = _contract()
    paths = _plain_xml(contract, tmp_path, speed=None, lanes=None)
    phases = ET.parse(paths[3]).getroot().findall("tlLogic/phase")
    assert phases[0].attrib["state"] == "rrrrrr"
    assert phases[0].attrib["name"] == "bootstrap_all_red"
    assert [phase.attrib["state"] for phase in phases[1:]] == [
        state
        for index in range(6)
        for state in (
            "r" * index + "G" + "r" * (5 - index),
            "r" * index + "y" + "r" * (5 - index),
        )
    ]


@pytest.mark.parametrize(
    ("mutate_contract", "actual_indices", "match"),
    [
        (
            lambda contract: contract["controllers"][0]["controlled_links"][1].update(
                {
                    key: value
                    for key, value in contract["controllers"][0]["controlled_links"][
                        0
                    ].items()
                    if key != "link_index"
                }
            ),
            CONTRACT_TO_ACTUAL,
            "重复 controlled-link",
        ),
        (lambda contract: None, (4, 5, 0, 1, 2, 2), "唯一且连续"),
        (lambda contract: None, (4, 5, 0, 1, 2, 6), "唯一且连续"),
    ],
)
def test_compiled_tls_order_rejects_duplicate_or_non_continuous_links(
    tmp_path: Path,
    mutate_contract,
    actual_indices: tuple[int, ...],
    match: str,
) -> None:
    contract = _contract()
    mutate_contract(contract)
    net_path = tmp_path / "net.xml"
    _write_net(net_path, contract, actual_indices=actual_indices)
    with pytest.raises(ValueError, match=match):
        compiled_tls_link_orders(contract, net_path)


def test_compiled_tls_order_rejects_connection_identity_mismatch(
    tmp_path: Path,
) -> None:
    contract = _contract()
    mismatched = copy.deepcopy(contract)
    mismatched["controllers"][0]["controlled_links"][0]["to_edge_id"] = "wrong"
    net_path = tmp_path / "net.xml"
    _write_net(net_path, contract)
    with pytest.raises(ValueError, match="无法一一匹配"):
        compiled_tls_link_orders(mismatched, net_path)


def test_traci_validation_uses_identity_and_rejects_duplicate_alternatives() -> None:
    links = _contract()["controllers"][0]["controlled_links"]
    actual = [
        [(link["from_lane_id"], link["to_lane_id"], f":via_{index}")]
        for index, link in enumerate(links)
    ]
    validate_traci_controlled_links(TLS_ID, links, actual)

    wrong = copy.deepcopy(actual)
    wrong[0] = wrong[1]
    with pytest.raises(ValueError, match="连接身份不一致"):
        validate_traci_controlled_links(TLS_ID, links, wrong)

    duplicate = copy.deepcopy(actual)
    duplicate[0].append(duplicate[0][0])
    with pytest.raises(ValueError, match="恰好对应一个连接"):
        validate_traci_controlled_links(TLS_ID, links, duplicate)
