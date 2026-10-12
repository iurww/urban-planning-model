"""Task identifiers and action vocabularies shared across the package."""

from __future__ import annotations

from enum import IntEnum


class Task(IntEnum):
    LAND = 0
    ROAD = 1
    POI = 2
    SIGNAL = 3

    @classmethod
    def from_name(cls, name: str) -> "Task":
        normalized = name.strip().lower()
        aliases = {
            "land": cls.LAND,
            "land_use": cls.LAND,
            "land_use_planning": cls.LAND,
            "road": cls.ROAD,
            "road_planning": cls.ROAD,
            "poi": cls.POI,
            "poi_placement": cls.POI,
            "signal": cls.SIGNAL,
            "signal_control": cls.SIGNAL,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown task: {name!r}") from exc

    @property
    def label(self) -> str:
        return ("land", "road", "poi", "signal")[int(self)]


LAND_ACTIONS = ("grn", "com", "res", "sch", "hos", "off")
LAND_TO_INDEX = {name: index for index, name in enumerate(LAND_ACTIONS)}
TRAINING_TASKS = (Task.LAND, Task.ROAD, Task.SIGNAL)
NUM_TASKS = len(Task)
NUM_ENTITY_TYPES = 3
MAX_ACTIONS = max(len(LAND_ACTIONS), 8)
