"""PBF primitives and projected tile data contract."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import osmium
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box, mapping, shape
from shapely.ops import transform

ROAD_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

NON_NETWORK_SERVICE = {"parking_aisle", "driveway", "alley", "drive-through", "emergency_access"}

ROAD_LEVEL = {
    "motorway": 0,
    "motorway_link": 1,
    "trunk": 1,
    "trunk_link": 2,
    "primary": 2,
    "primary_link": 3,
    "secondary": 3,
    "secondary_link": 4,
    "tertiary": 4,
    "tertiary_link": 5,
    "unclassified": 5,
    "residential": 6,
    "living_street": 7,
    "service": 8,
}

DEFAULT_LANES = {
    "motorway": 3,
    "motorway_link": 1,
    "trunk": 3,
    "trunk_link": 1,
    "primary": 4,
    "primary_link": 1,
    "secondary": 4,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 2,
    "residential": 2,
    "living_street": 1,
    "service": 1,
}

LANDUSE_KEYS = {"landuse", "amenity", "leisure", "natural"}


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def parse(cls, value: str) -> "BBox":
        parts = [float(item.strip()) for item in value.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be west,south,east,north")
        bbox = cls(*parts)
        if not (bbox.west < bbox.east and bbox.south < bbox.north):
            raise ValueError("bbox bounds are not ordered")
        return bbox

    @property
    def polygon(self) -> Polygon:
        return box(self.west, self.south, self.east, self.north)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    def expanded(self, degrees: float = 0.002) -> Polygon:
        return box(self.west - degrees, self.south - degrees, self.east + degrees, self.north + degrees)

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


@dataclass
class Projection:
    crs: CRS
    forward: Transformer
    inverse: Transformer

    @classmethod
    def for_bbox(cls, bbox: BBox) -> "Projection":
        lon, lat = bbox.center
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        crs = CRS.from_epsg(epsg)
        return cls(
            crs=crs,
            forward=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            inverse=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        )

    @classmethod
    def for_crs(cls, crs: str | CRS) -> "Projection":
        """Build projection transforms for an explicitly selected metric CRS."""
        metric_crs = CRS.from_user_input(crs)
        if not metric_crs.is_projected:
            raise ValueError(f"projection CRS must be projected, got {crs}")
        return cls(
            crs=metric_crs,
            forward=Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True),
            inverse=Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True),
        )

    def project(self, geometry: Any) -> Any:
        return transform(self.forward.transform, geometry)

    def unproject(self, geometry: Any) -> Any:
        return transform(self.inverse.transform, geometry)


@dataclass
class RawRoad:
    osm_id: int
    node_ids: list[int]
    coordinates: list[tuple[float, float]]
    tags: dict[str, str]

    @property
    def geometry(self) -> LineString:
        return LineString(self.coordinates)


@dataclass
class RawArea:
    osm_id: int
    geometry: Any
    tags: dict[str, str]


@dataclass
class RawSignal:
    osm_id: int
    coordinate: tuple[float, float]
    tags: dict[str, str]


class PBFExtractor(osmium.SimpleHandler):
    """Read only task-relevant OSM features within a small working extent."""

    def __init__(self, bbox: BBox, include_service: bool = True) -> None:
        super().__init__()
        self.bbox = bbox
        self.clip = bbox.polygon
        self.prefilter = bbox.expanded()
        self.include_service = include_service
        self.roads: dict[int, RawRoad] = {}
        self.areas: dict[int, RawArea] = {}
        self.signals: dict[int, RawSignal] = {}
        self.stats: Counter[str] = Counter()
        self._geojson_factory = osmium.geom.GeoJSONFactory()

    def node(self, node: osmium.osm.Node) -> None:
        tags = dict(node.tags)
        if tags.get("highway") != "traffic_signals":
            return
        if not node.location.valid():
            return
        point = Point(node.location.lon, node.location.lat)
        if not self.prefilter.covers(point):
            return
        self.signals[node.id] = RawSignal(node.id, (point.x, point.y), relevant_signal_tags(tags))
        self.stats["traffic_signal_nodes"] += 1

    def way(self, way: osmium.osm.Way) -> None:
        tags = dict(way.tags)
        highway = tags.get("highway")
        if highway not in ROAD_TYPES:
            return
        if highway == "service" and (
            not self.include_service or tags.get("service") in NON_NETWORK_SERVICE
        ):
            self.stats["excluded_service_ways"] += 1
            return
        if tags.get("area") == "yes" or tags.get("access") == "private" and highway == "service":
            self.stats["excluded_non_network_ways"] += 1
            return
        coordinates: list[tuple[float, float]] = []
        node_ids: list[int] = []
        for node_ref in way.nodes:
            if not node_ref.location.valid():
                return
            coordinates.append((node_ref.location.lon, node_ref.location.lat))
            node_ids.append(node_ref.ref)
        if len(coordinates) < 2:
            return
        try:
            line = LineString(coordinates)
        except Exception:
            self.stats["invalid_road_ways"] += 1
            return
        if line.is_empty or not line.is_valid or not line.intersects(self.prefilter):
            return
        self.roads[way.id] = RawRoad(way.id, node_ids, coordinates, relevant_road_tags(tags))
        self.stats["retained_road_ways"] += 1

    def area(self, area: osmium.osm.Area) -> None:
        tags = dict(area.tags)
        if not any(key in tags for key in LANDUSE_KEYS):
            return
        if tags.get("natural") in {"coastline", "cliff", "ridge"}:
            return
        try:
            geometry = shape(json.loads(self._geojson_factory.create_multipolygon(area)))
        except Exception:
            self.stats["invalid_landuse_areas"] += 1
            return
        if geometry.is_empty or not geometry.intersects(self.prefilter):
            return
        clipped = geometry.intersection(self.clip)
        if clipped.is_empty:
            return
        self.areas[area.orig_id()] = RawArea(area.orig_id(), clipped, relevant_landuse_tags(tags))
        self.stats["retained_landuse_areas"] += 1


def extract_pbf(path: Path, bbox: BBox, include_service: bool = True) -> PBFExtractor:
    handler = PBFExtractor(bbox, include_service=include_service)
    handler.apply_file(str(path), locations=True, idx="flex_mem")
    return handler


def relevant_road_tags(tags: dict[str, str]) -> dict[str, str]:
    keys = {
        "highway",
        "name",
        "name:en",
        "ref",
        "oneway",
        "junction",
        "lanes",
        "lanes:forward",
        "lanes:backward",
        "width",
        "maxspeed",
        "service",
        "access",
        "bridge",
        "tunnel",
        "layer",
        "turn:lanes",
        "turn:lanes:forward",
        "turn:lanes:backward",
        "destination",
    }
    return {key: value for key, value in tags.items() if key in keys}


def relevant_signal_tags(tags: dict[str, str]) -> dict[str, str]:
    keys = {
        "highway",
        "crossing",
        "crossing:signals",
        "traffic_signals",
        "traffic_signals:direction",
        "traffic_signals:sound",
        "direction",
        "button_operated",
        "name",
        "ref",
    }
    return {key: value for key, value in tags.items() if key in keys}


def relevant_landuse_tags(tags: dict[str, str]) -> dict[str, str]:
    keys = {
        "landuse",
        "amenity",
        "leisure",
        "natural",
        "name",
        "name:en",
        "operator",
        "access",
    }
    return {key: value for key, value in tags.items() if key in keys}


def classify_landuse(tags: dict[str, str]) -> str:
    landuse = tags.get("landuse", "")
    amenity = tags.get("amenity", "")
    leisure = tags.get("leisure", "")
    natural = tags.get("natural", "")
    if landuse in {"residential"}:
        return "residential"
    if landuse in {"commercial", "retail"}:
        return "commercial"
    if landuse in {"industrial", "construction", "brownfield", "landfill"}:
        return "industrial"
    if landuse in {"education", "institutional", "civic"} or amenity in {
        "school",
        "college",
        "university",
        "hospital",
        "clinic",
        "community_centre",
        "kindergarten",
        "public_building",
    }:
        return "public_service"
    if landuse in {"forest", "grass", "meadow", "recreation_ground", "village_green"}:
        return "green_open_space"
    if leisure in {"park", "garden", "nature_reserve", "playground", "pitch", "sports_centre"}:
        return "green_open_space"
    if natural in {"wood", "grassland", "scrub", "heath", "wetland"}:
        return "green_open_space"
    if natural == "water" or landuse in {"reservoir", "basin"}:
        return "water"
    if landuse in {"farmland", "farmyard", "orchard", "plant_nursery", "greenhouse_horticulture"}:
        return "agriculture"
    if amenity:
        return "public_service"
    return "other"


def safe_float(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.lower().replace("meters", "").replace("meter", "").replace("m", "").strip()
    try:
        return float(cleaned.split(";")[0])
    except ValueError:
        return None


def parse_lanes(tags: dict[str, str]) -> dict[str, Any]:
    def integer(key: str) -> int | None:
        value = tags.get(key)
        if not value:
            return None
        try:
            return max(1, int(float(value.split(";")[0])))
        except ValueError:
            return None

    highway = tags.get("highway", "residential")
    total = integer("lanes")
    forward = integer("lanes:forward")
    backward = integer("lanes:backward")
    oneway = tags.get("oneway") in {"yes", "1", "true"} or tags.get("junction") == "roundabout"
    inferred = False
    if total is None:
        if forward is not None or backward is not None:
            total = (forward or 0) + (backward or 0)
        else:
            total = DEFAULT_LANES.get(highway, 2)
            inferred = True
    if oneway:
        forward = forward or total
        backward = 0
    elif forward is None and backward is None:
        forward = max(1, math.ceil(total / 2))
        backward = max(1, total // 2)
    return {
        "total": total,
        "forward": forward,
        "backward": backward,
        "oneway": oneway,
        "inferred": inferred,
    }


def infer_road_width(tags: dict[str, str]) -> tuple[float, bool]:
    explicit = safe_float(tags.get("width"))
    if explicit and 1.5 <= explicit <= 80:
        return explicit, False
    lanes = parse_lanes(tags)["total"]
    highway = tags.get("highway", "residential")
    lane_width = 3.5 if ROAD_LEVEL.get(highway, 6) <= 3 else 3.2
    shoulder = 2.0 if highway in {"motorway", "trunk"} else 0.8
    return max(3.0, lanes * lane_width + shoulder), True


def parse_speed_kph(tags: dict[str, str]) -> tuple[float, bool]:
    raw = tags.get("maxspeed")
    if raw:
        value = raw.lower().strip()
        try:
            number = float(value.split()[0])
            if "mph" in value:
                number *= 1.60934
            if 5 <= number <= 160:
                return number, False
        except ValueError:
            pass
    defaults = {
        "motorway": 100,
        "motorway_link": 60,
        "trunk": 80,
        "trunk_link": 50,
        "primary": 60,
        "primary_link": 40,
        "secondary": 50,
        "secondary_link": 35,
        "tertiary": 40,
        "tertiary_link": 30,
        "unclassified": 35,
        "residential": 30,
        "living_street": 15,
        "service": 15,
    }
    return float(defaults.get(tags.get("highway", ""), 30)), True


def normalize_name(tags: dict[str, str]) -> str:
    value = tags.get("name") or tags.get("ref") or ""
    return "".join(character.lower() for character in value if character.isalnum())


def iter_polygons(geometry: Any) -> Iterator[Polygon]:
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from iter_polygons(part)


def iter_lines(geometry: Any) -> Iterator[LineString]:
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from iter_lines(part)


def geojson_feature(feature_id: str, geometry: Any, properties: dict[str, Any]) -> dict[str, Any]:
    clean = {key: json_safe(value) for key, value in properties.items()}
    clean["id"] = feature_id
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": mapping(geometry),
        "properties": clean,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            json_safe(payload),
            stream,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def write_geojson(path: Path, features: Iterable[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": list(features)}
    if metadata:
        payload["metadata"] = metadata
    write_json(path, payload, compact=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relation_counts(edges: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(edge["relation"] for edge in edges).items()))

@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    city: str
    crs: str
    size_m: float
    column: int
    row: int
    bounds_m: tuple[float, float, float, float]
    polygon_wgs84: Polygon
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_polygon(self) -> Polygon:
        return box(*self.bounds_m)

    @property
    def extraction_bbox(self) -> BBox:
        west, south, east, north = self.polygon_wgs84.bounds
        return BBox(west, south, east, north)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "city": self.city,
            "crs": self.crs,
            "size_m": self.size_m,
            "column": self.column,
            "row": self.row,
            "bounds_m": list(self.bounds_m),
            "polygon_wgs84": mapping(self.polygon_wgs84),
            "bbox_wgs84": list(self.extraction_bbox.as_list()),
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TileSpec":
        return cls(
            tile_id=payload["tile_id"],
            city=payload["city"],
            crs=payload["crs"],
            size_m=float(payload["size_m"]),
            column=int(payload["column"]),
            row=int(payload["row"]),
            bounds_m=tuple(float(value) for value in payload["bounds_m"]),
            polygon_wgs84=shape(payload["polygon_wgs84"]),
            stats=dict(payload.get("stats", {})),
        )
