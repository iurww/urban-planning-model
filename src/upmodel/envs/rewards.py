"""Reward functions matching the data-generation objectives and constraints."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from upmodel.road_metrics import (
    DEFAULT_ROAD_BUDGET_FRAC,
    DEFAULT_SPACING_MAX_M,
    DEFAULT_SPACING_MIN_M,
    road_budget,
    road_structure_metrics,
    road_traffic_metrics,
)
from upmodel.state import UrbanGraph
from upmodel.tasks import LAND_ACTIONS


LAND_QUOTA_LOWER = (0.08, 0.10, 0.20, 0.08, 0.08, 0.08)
LAND_QUOTA_UPPER = (0.30, 0.30, 0.50, 0.20, 0.20, 0.30)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    metrics: dict[str, float | bool]


class LandReward:
    SERVICE_TYPES = ("sch", "hos", "grn", "off", "com")

    def __init__(
        self,
        graph: UrbanGraph,
        *,
        service_radius_m: float = 500.0,
        ecology_buffer_m: float = 300.0,
    ) -> None:
        parcels = graph.raw_topology["parcels"]
        self.areas = [float(item["area_m2"]) for item in parcels]
        self.polygons = [
            Polygon(item["geometry"][0], holes=item["geometry"][1:] or None)
            for item in parcels
        ]
        centroids = [polygon.centroid for polygon in self.polygons]
        radius = float(service_radius_m)
        self.service_neighbors = [
            {
                other
                for other, target in enumerate(centroids)
                if index != other and source.distance(target) <= radius
            }
            for index, source in enumerate(centroids)
        ]
        buffer_distance = float(ecology_buffer_m)
        self.ecology_overlap: list[list[float]] = []
        buffers = [polygon.buffer(buffer_distance) for polygon in self.polygons]
        for residential in self.polygons:
            self.ecology_overlap.append(
                [float(residential.intersection(buffer).area) for buffer in buffers]
            )

    def evaluate(self, actions: list[int]) -> RewardResult:
        if len(actions) != len(self.areas) or any(action < 0 for action in actions):
            raise ValueError("land reward requires one assigned action per parcel")
        labels = [LAND_ACTIONS[action] for action in actions]
        area_by_type = [
            sum(area for area, action in zip(self.areas, actions) if action == class_index)
            for class_index in range(len(LAND_ACTIONS))
        ]
        total_area = sum(self.areas)
        tolerance = total_area * 1.0e-9
        feasible = all(
            LAND_QUOTA_LOWER[index] * total_area - tolerance
            <= area_by_type[index]
            <= LAND_QUOTA_UPPER[index] * total_area + tolerance
            for index in range(len(LAND_ACTIONS))
        )
        residential = [index for index, label in enumerate(labels) if label == "res"]
        residential_area = sum(self.areas[index] for index in residential)
        service_total = 0.0
        for index in residential:
            neighbors = self.service_neighbors[index]
            service_total += sum(
                any(labels[other] == service_type for other in neighbors) / 5.0
                for service_type in self.SERVICE_TYPES
            )
        service = service_total / len(residential) if residential else 0.0
        ecology_area = 0.0
        for index in residential:
            covered = sum(
                self.ecology_overlap[index][green]
                for green, label in enumerate(labels)
                if label == "grn"
            )
            ecology_area += min(covered, self.areas[index])
        ecology = (
            min(1.0, ecology_area / residential_area)
            if residential_area > 0.0
            else 0.0
        )
        reward = service + ecology if feasible else 0.0
        return RewardResult(
            reward=float(reward),
            metrics={
                "Service": float(service),
                "Ecology": float(ecology),
                "r_L": float(reward),
                "residential_area": float(residential_area),
                "feasible": feasible,
            },
        )


class RoadReward:
    def __init__(
        self,
        graph: UrbanGraph,
        *,
        budget_frac: float = DEFAULT_ROAD_BUDGET_FRAC,
        spacing_min_m: float = DEFAULT_SPACING_MIN_M,
        spacing_max_m: float = DEFAULT_SPACING_MAX_M,
    ) -> None:
        topology = graph.raw_topology
        intersections = topology["intersections"]
        intersection_index = {
            str(item["id"]): index for index, item in enumerate(intersections)
        }
        self.endpoints = [
            (
                intersection_index[str(item["from_intersection_id"])],
                intersection_index[str(item["to_intersection_id"])],
            )
            for item in topology["roads"]
        ]
        self.dead_end_nodes = {
            index
            for index, item in enumerate(intersections)
            if item.get("kind", "junction") == "junction"
        }
        if not self.dead_end_nodes:
            self.dead_end_nodes = set(range(len(intersections)))
        self.lengths = [float(item["length_m"]) for item in topology["roads"]]
        self.budget_frac = float(budget_frac)
        self.budget = road_budget(len(self.endpoints), self.budget_frac)
        self.spacing_min_m = float(spacing_min_m)
        self.spacing_max_m = float(spacing_max_m)
        polygons = [
            Polygon(item["geometry"][0], holes=item["geometry"][1:] or None)
            for item in topology["parcels"]
        ]
        bounds = unary_union(polygons).bounds
        bbox_area = max((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), 1.0)
        parcel_area = sum(float(item["area_m2"]) for item in topology["parcels"])
        self.community_area = max(float(bbox_area), float(parcel_area), 1.0)

    def evaluate(self, actions: list[int]) -> RewardResult:
        if len(actions) != len(self.endpoints) or any(action not in {0, 1} for action in actions):
            raise ValueError("road reward requires one binary action per road")
        selected = {index for index, action in enumerate(actions) if action == 1}
        metrics = road_traffic_metrics(
            self.endpoints,
            self.lengths,
            selected,
            self.community_area,
            budget=self.budget,
            spacing_min_m=self.spacing_min_m,
            spacing_max_m=self.spacing_max_m,
            dead_end_nodes=self.dead_end_nodes,
        )
        structure = road_structure_metrics(
            self.endpoints,
            selected,
            interior_nodes=self.dead_end_nodes,
        )
        metrics.update(structure)
        metrics["feasible"] = bool(
            metrics["feasible"] and structure["structurally_feasible"]
        )
        return RewardResult(
            reward=(
                float(metrics["Traffic"])
                if structure["structurally_feasible"]
                else 0.0
            ),
            metrics=dict(metrics),
        )

    def deletion_mask(
        self, actions: list[int], *, already_decided: torch.Tensor
    ) -> torch.Tensor:
        """Mask deletions that would irreversibly violate the road contract."""
        if len(actions) != len(self.endpoints) or already_decided.shape != (
            len(actions),
        ):
            raise ValueError("road deletion mask does not align with roads")
        mask = torch.zeros(len(actions), dtype=torch.bool)
        adjacency: dict[int, list[tuple[int, int]]] = {}
        degree: dict[int, int] = {}
        source_degree: dict[int, int] = {}
        for edge, (left, right) in enumerate(self.endpoints):
            if left < 0 or right < 0 or left == right:
                continue
            source_degree[left] = source_degree.get(left, 0) + 1
            source_degree[right] = source_degree.get(right, 0) + 1
            if actions[edge] != 1:
                continue
            adjacency.setdefault(left, []).append((right, edge))
            adjacency.setdefault(right, []).append((left, edge))
            degree[left] = degree.get(left, 0) + 1
            degree[right] = degree.get(right, 0) + 1

        # Edge-id Tarjan handles parallel roads correctly: only an edge whose
        # removal disconnects its current source component is a bridge.
        discovery: dict[int, int] = {}
        low: dict[int, int] = {}
        bridges: set[int] = set()
        clock = 0

        def visit(node: int, parent_edge: int = -1) -> None:
            nonlocal clock
            discovery[node] = low[node] = clock
            clock += 1
            for neighbor, edge in adjacency.get(node, []):
                if edge == parent_edge:
                    continue
                if neighbor not in discovery:
                    visit(neighbor, edge)
                    low[node] = min(low[node], low[neighbor])
                    if low[neighbor] > discovery[node]:
                        bridges.add(edge)
                else:
                    low[node] = min(low[node], discovery[neighbor])

        for node in adjacency:
            if node not in discovery:
                visit(node)

        for edge, action in enumerate(actions):
            if action != 1 or bool(already_decided[edge]) or edge in bridges:
                continue
            left, right = self.endpoints[edge]
            if left < 0 or right < 0 or left == right:
                mask[edge] = True
                continue
            left_minimum = (
                min(2, source_degree[left]) if left in self.dead_end_nodes else 1
            )
            right_minimum = (
                min(2, source_degree[right]) if right in self.dead_end_nodes else 1
            )
            mask[edge] = (
                degree[left] - 1 >= left_minimum
                and degree[right] - 1 >= right_minimum
            )
        return mask

    def action_mask(self, actions: list[int], *, remaining_including_current: int) -> tuple[bool, bool]:
        """Return legality of ``keep`` and ``upgrade`` under the paper budget."""
        if len(actions) != len(self.endpoints):
            raise ValueError("road action mask requires one slot per road")
        selected = sum(action == 1 for action in actions)
        keep = not (selected == 0 and remaining_including_current == 1)
        upgrade = selected < self.budget
        return keep, upgrade


class PoiReward:
    def __init__(self, graph: UrbanGraph, *, num_poi: int = 5) -> None:
        topology = graph.raw_topology
        polygons = [
            Polygon(item["geometry"][0], holes=item["geometry"][1:] or None)
            for item in topology["parcels"]
        ]
        roads = [LineString(item["geometry"]) for item in topology["roads"]]
        intersections = [Point(item["point"]) for item in topology["intersections"]]
        road_union = unary_union(roads)
        intersection_union = unary_union(intersections)
        areas = [float(item["area_m2"]) for item in topology["parcels"]]
        max_area = max(areas, default=1.0)
        self.weights = []
        for polygon, area in zip(polygons, areas):
            area_normalized = area / max(max_area, 1.0e-9)
            distance_road = float(polygon.distance(road_union))
            distance_intersection = float(polygon.distance(intersection_union))
            weight = (
                area**0.5
                * (
                    0.5 * _exp(-0.01 * distance_road)
                    + 0.5 * _exp(-0.015 * distance_intersection)
                )
                * (0.6 + 0.4 * area_normalized)
            )
            self.weights.append(float(weight))
        self.num_poi = min(int(num_poi), len(self.weights))

    def evaluate(self, actions: list[int]) -> RewardResult:
        if len(actions) != len(self.weights) or any(action not in {0, 1} for action in actions):
            raise ValueError("POI reward requires one binary action per parcel")
        selected = [index for index, action in enumerate(actions) if action == 1]
        feasible = len(selected) == self.num_poi
        utility = sum(self.weights[index] for index in selected)
        maximum = max(self.weights, default=0.0)
        denominator = max(self.num_poi * maximum, 1.0e-12)
        reward = min(1.0, max(0.0, utility / denominator)) if feasible else 0.0
        return RewardResult(
            reward=float(reward),
            metrics={
                "U_POI": float(utility),
                "r_POI": float(reward),
                "selected": float(len(selected)),
                "feasible": feasible,
            },
        )


def _exp(value: float) -> float:
    import math

    return math.exp(value)
