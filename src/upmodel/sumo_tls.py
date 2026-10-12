"""Strict SUMO traffic-light connection ordering helpers."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ConnectionIdentity = tuple[str, str, int, str, int]


@dataclass(frozen=True)
class TlsLinkOrder:
    """A controller's contract links expressed in compiled SUMO link order."""

    links: tuple[dict[str, Any], ...]
    contract_to_actual: tuple[int, ...]


def _contract_identity(tls_id: str, link: dict[str, Any]) -> ConnectionIdentity:
    return (
        tls_id,
        str(link["from_edge_id"]),
        int(link["from_lane_index"]),
        str(link["to_edge_id"]),
        int(link["to_lane_index"]),
    )


def _xml_identity(connection: ET.Element) -> ConnectionIdentity:
    return (
        str(connection.attrib["tl"]),
        str(connection.attrib["from"]),
        int(connection.attrib["fromLane"]),
        str(connection.attrib["to"]),
        int(connection.attrib["toLane"]),
    )


def _indexed_contract_links(controller: dict[str, Any]) -> list[dict[str, Any]]:
    tls_id = str(controller["sumo_tl_id"])
    links = list(controller.get("controlled_links", []))
    indices = [int(link["link_index"]) for link in links]
    expected = list(range(len(links)))
    if sorted(indices) != expected:
        raise ValueError(
            f"{tls_id}: contract link_index 必须唯一且连续为 0..{len(links) - 1}，"
            f"实际={indices}"
        )
    ordered = sorted(links, key=lambda link: int(link["link_index"]))
    identities = [_contract_identity(tls_id, link) for link in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{tls_id}: contract 包含重复 controlled-link 连接身份")
    return ordered


def compiled_tls_link_orders(
    contract: dict[str, Any], net_path: str | Path
) -> dict[str, TlsLinkOrder]:
    """Build strict contract-to-compiled link permutations for every TLS."""
    controllers = list(contract.get("controllers", []))
    tls_ids = [str(controller["sumo_tl_id"]) for controller in controllers]
    if len(set(tls_ids)) != len(tls_ids):
        raise ValueError("traffic_signals.json 包含重复 sumo_tl_id")

    root = ET.parse(net_path).getroot()
    xml_by_tls: dict[str, list[ET.Element]] = {}
    for connection in root.findall("connection"):
        tls_id = connection.get("tl")
        if tls_id is not None:
            xml_by_tls.setdefault(str(tls_id), []).append(connection)

    orders: dict[str, TlsLinkOrder] = {}
    for controller in controllers:
        tls_id = str(controller["sumo_tl_id"])
        contract_links = _indexed_contract_links(controller)
        actual_connections = xml_by_tls.get(tls_id, [])
        actual_indices: list[int] = []
        actual_by_identity: dict[ConnectionIdentity, int] = {}
        for connection in actual_connections:
            if connection.get("linkIndex") is None:
                raise ValueError(f"{tls_id}: compiled connection 缺少 linkIndex")
            actual_index = int(connection.attrib["linkIndex"])
            identity = _xml_identity(connection)
            if identity in actual_by_identity:
                raise ValueError(
                    f"{tls_id}: compiled net 包含重复 controlled-link 连接身份"
                )
            actual_by_identity[identity] = actual_index
            actual_indices.append(actual_index)

        expected_indices = list(range(len(contract_links)))
        if sorted(actual_indices) != expected_indices:
            raise ValueError(
                f"{tls_id}: compiled linkIndex 必须唯一且连续为 "
                f"0..{len(contract_links) - 1}，实际={actual_indices}"
            )

        contract_identities = [
            _contract_identity(tls_id, link) for link in contract_links
        ]
        missing = [
            identity
            for identity in contract_identities
            if identity not in actual_by_identity
        ]
        extra = [
            identity
            for identity in actual_by_identity
            if identity not in set(contract_identities)
        ]
        if missing or extra:
            raise ValueError(
                f"{tls_id}: contract 与 compiled controlled-links 无法一一匹配，"
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

        contract_to_actual = tuple(
            actual_by_identity[identity] for identity in contract_identities
        )
        links_by_actual: list[dict[str, Any] | None] = [None] * len(contract_links)
        for contract_index, actual_index in enumerate(contract_to_actual):
            link = copy.deepcopy(contract_links[contract_index])
            link["link_index"] = actual_index
            links_by_actual[actual_index] = link
        if any(link is None for link in links_by_actual):
            raise ValueError(f"{tls_id}: controlled-link permutation 不是严格双射")
        orders[tls_id] = TlsLinkOrder(
            links=tuple(link for link in links_by_actual if link is not None),
            contract_to_actual=contract_to_actual,
        )
    return orders


def permute_tls_state(
    state: str, contract_to_actual: tuple[int, ...], *, tls_id: str
) -> str:
    """Move each contract-position state character to its compiled linkIndex."""
    if len(state) != len(contract_to_actual):
        raise ValueError(
            f"{tls_id}: TLS state 长度={len(state)}，"
            f"controlled-links={len(contract_to_actual)}"
        )
    result = [""] * len(state)
    for contract_index, actual_index in enumerate(contract_to_actual):
        result[actual_index] = state[contract_index]
    if any(not character for character in result):
        raise ValueError(f"{tls_id}: TLS state permutation 不是严格双射")
    return "".join(result)


def adapt_controller_to_compiled(
    controller: dict[str, Any], order: TlsLinkOrder
) -> dict[str, Any]:
    """Copy a controller and adapt all positional fields to compiled order."""
    result = copy.deepcopy(controller)
    tls_id = str(result["sumo_tl_id"])
    result["controlled_links"] = [copy.deepcopy(link) for link in order.links]
    program = result["program"]

    for key, value in list(program.items()):
        if key.endswith("_state") and isinstance(value, str):
            program[key] = permute_tls_state(
                value, order.contract_to_actual, tls_id=tls_id
            )

    for action in program.get("actions", []):
        for key, value in list(action.items()):
            if key.endswith("_state") and isinstance(value, str):
                action[key] = permute_tls_state(
                    value, order.contract_to_actual, tls_id=tls_id
                )
            elif key.endswith("_link_indices") and isinstance(value, list):
                action[key] = sorted(
                    order.contract_to_actual[int(index)] for index in value
                )
    return result


def validate_traci_controlled_links(
    tls_id: str,
    expected_links: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    controlled_links: Any,
) -> None:
    """Validate live TraCI link indices against saved compiled-order metadata."""
    actual = list(controlled_links)
    if len(actual) != len(expected_links):
        raise ValueError(
            f"{tls_id}: TraCI links={len(actual)}，metadata={len(expected_links)}"
        )
    for link_index, (expected, alternatives) in enumerate(
        zip(expected_links, actual, strict=True)
    ):
        choices = list(alternatives)
        if len(choices) != 1 or len(choices[0]) < 2:
            raise ValueError(
                f"{tls_id}: TraCI linkIndex={link_index} 必须恰好对应一个连接，"
                f"实际={choices}"
            )
        incoming, outgoing = str(choices[0][0]), str(choices[0][1])
        expected_pair = (
            str(expected["from_lane_id"]),
            str(expected["to_lane_id"]),
        )
        if (
            int(expected["link_index"]) != link_index
            or (
                incoming,
                outgoing,
            )
            != expected_pair
        ):
            raise ValueError(
                f"{tls_id}: TraCI linkIndex={link_index} 连接身份不一致，"
                f"actual={(incoming, outgoing)}，expected={expected_pair}"
            )
