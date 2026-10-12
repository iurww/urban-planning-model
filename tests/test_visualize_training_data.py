from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Circle

from scripts.visualize_training_data import (
    JUNCTION_LABEL_COLOR,
    JUNCTION_COLOR,
    SIGNAL_JUNCTION_COLOR,
    TOPOLOGY_ROAD_COLOR,
    _available_components,
    _discover_tile_dirs,
    _plot_topology,
    _selection_csv_path,
    _selection_for_html,
    _write_html,
    _write_root_index,
    render_tile,
)


def _write_topology(path: Path, tile_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "abstract-urban-topology-1.0",
                "metadata": {
                    "tile_id": tile_id,
                    "tile_width_m": 4000.0,
                    "tile_height_m": 4000.0,
                },
                "parcels": [
                    {
                        "id": "parcel/p0",
                        "geometry": [
                            [[0.0, 0.0], [4000.0, 0.0], [4000.0, 4000.0], [0.0, 4000.0]]
                        ],
                    }
                ],
                "roads": [],
                "intersections": [],
            }
        ),
        encoding="utf-8",
    )


class AdaptiveVisualizationTest(unittest.TestCase):
    def test_selection_csv_defaults_to_candidate_dataset_root(self) -> None:
        args = argparse.Namespace(selection_csv=None)
        self.assertEqual(
            _selection_csv_path(args, Path("datadoc/testdoc")),
            Path("datadoc/topology_selection.csv"),
        )
        args.selection_csv = Path("custom/selection.csv")
        self.assertEqual(
            _selection_csv_path(args, Path("datadoc/testdoc")),
            Path("custom/selection.csv"),
        )

    def test_topology_panel_labels_only_real_junctions_and_unifies_roads(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tile_dir = Path(raw)
            topology = {
                "parcels": [
                    {
                        "id": "parcel/p0000",
                        "geometry": [[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]],
                    }
                ],
                "roads": [
                    {"id": "road/r0000", "highway": "primary", "geometry": [[0.0, 25.0], [100.0, 25.0]]},
                    {
                        "id": "road/r0001",
                        "highway": "residential",
                        "geometry": [[25.0, 0.0], [25.0, 25.0], [25.0, 100.0]],
                    },
                ],
                "intersections": [
                    {"id": "intersection/i0001", "kind": "junction", "point": [50.0, 50.0]},
                    {"id": "intersection/i0002", "kind": "junction", "point": [75.0, 50.0]},
                    {"id": "intersection/i0003", "kind": "geometry_break", "point": [25.0, 50.0]},
                    {"id": "intersection/i0004", "kind": "boundary_portal", "point": [0.0, 50.0]},
                ],
            }
            (tile_dir / "traffic_signals.json").write_text(
                json.dumps({"signalized_intersection_ids": ["intersection/i0002"]}),
                encoding="utf-8",
            )
            figure, axis = plt.subplots()
            try:
                _plot_topology(axis, tile_dir, topology)
                labels = {text.get_text() for text in axis.texts}
                self.assertTrue({"p0000", "r0", "r1", "i1", "i2"}.issubset(labels))
                self.assertNotIn("i0001", labels)
                self.assertNotIn("i0002", labels)
                self.assertNotIn("i0003", labels)
                self.assertNotIn("i0004", labels)
                junction_labels = [
                    text for text in axis.texts if text.get_text() in {"i1", "i2"}
                ]
                self.assertEqual(
                    {to_hex(text.get_color()) for text in junction_labels},
                    {JUNCTION_LABEL_COLOR.lower()},
                )
                circles = [patch for patch in axis.patches if isinstance(patch, Circle)]
                self.assertEqual(len(circles), 4)
                self.assertEqual(
                    {to_hex(circle.get_edgecolor()) for circle in circles},
                    {
                        TOPOLOGY_ROAD_COLOR.lower(),
                        JUNCTION_COLOR.lower(),
                        SIGNAL_JUNCTION_COLOR.lower(),
                    },
                )
                road_circles = [
                    circle
                    for circle in circles
                    if to_hex(circle.get_edgecolor()) == TOPOLOGY_ROAD_COLOR.lower()
                ]
                self.assertEqual(
                    {tuple(circle.center) for circle in road_circles},
                    {(50.0, 25.0), (25.0, 50.0)},
                )
                road_collections = [
                    collection
                    for collection in axis.collections
                    if isinstance(collection, LineCollection)
                ]
                self.assertEqual(len(road_collections), 1)
                self.assertEqual(len(road_collections[0].get_colors()), 1)
                portal_collections = [
                    collection
                    for collection in axis.collections
                    if isinstance(collection, PathCollection)
                ]
                self.assertEqual(len(portal_collections), 1)
                self.assertEqual(portal_collections[0].get_offsets().tolist(), [[0.0, 50.0]])
            finally:
                plt.close(figure)

    def test_discovery_excludes_evidence_and_keeps_root_and_abstract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_topology(root / "solved" / "topology.json", "solved")
            _write_topology(root / "cleaned" / "abstract" / "topology.json", "cleaned")
            _write_topology(root / "cleaned" / "evidence" / "topology.json", "evidence")

            found = _discover_tile_dirs(root)

            self.assertEqual(
                {path.relative_to(root).as_posix() for path in found},
                {"solved", "cleaned/abstract"},
            )

    def test_discovery_enters_direct_training_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "final" / "tile" / "abstract"
            _write_topology(source / "topology.json", "tile")
            train = root / "train"
            train.mkdir()
            (train / "tile").symlink_to(source, target_is_directory=True)

            found = _discover_tile_dirs(train)

            self.assertEqual(found, [train / "tile"])

    def test_components_require_complete_artifact_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tile_dir = Path(raw)
            (tile_dir / "land_use_solution.csv").touch()
            self.assertEqual(_available_components(tile_dir), ["topology"])
            (tile_dir / "land_use_trajectory.json").touch()
            self.assertEqual(_available_components(tile_dir), ["topology", "land"])

    def test_topology_only_render_and_city_filter_html(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tile_dir = root / "testcity_4km_c1_r2" / "abstract"
            _write_topology(tile_dir / "topology.json", "testcity_4km_c1_r2")

            row = render_tile(tile_dir, dpi=30)
            output_dir = root / "visualizations"
            output_dir.mkdir()
            index = _write_html(
                [row],
                output_dir,
                [],
                initial_selection=[{"tile_id": "testcity_4km_c1_r2", "split": "test"}],
                selection_source="data/topology_selection.csv",
            )
            document = index.read_text(encoding="utf-8")

            self.assertEqual(row["components"], ["topology"])
            self.assertEqual(row["city"], "Testcity")
            self.assertTrue(Path(row["image"]).is_file())
            self.assertIn('id="city-filter"', document)
            self.assertIn('id="selected-only"', document)
            self.assertIn('class="split-filter" value="sft"', document)
            self.assertIn('class="split-filter" value="ppo"', document)
            self.assertIn('class="split-filter" value="test"', document)
            self.assertIn('id="selected-list"', document)
            self.assertIn('id="csv-editor"', document)
            self.assertIn('id="save-server"', document)
            self.assertIn('class="select-tile"', document)
            self.assertIn('data-city="Testcity"', document)
            self.assertIn('data-topology=', document)
            self.assertIn('/[",\\r\\n]/', document)
            self.assertIn('.split(/\\r?\\n/)', document)
            self.assertIn('lines.join("\\n") + "\\n"', document)
            self.assertIn("Testcity（1）", document)
            self.assertIn("../testcity_4km_c1_r2/abstract/visualization/overview.png", document)
            self.assertIn("未求解拓扑仅展示拓扑本身", document)
            self.assertIn("仅显示已选择", document)
            self.assertIn("显示 SFT", document)
            self.assertIn("显示 PPO", document)
            self.assertIn("显示 Test", document)
            self.assertIn('data-split-label', document)
            self.assertIn('item.classList.add("selection-added")', document)
            self.assertIn(
                "enabledSplits.has(selectedRow.split)",
                document,
            )
            self.assertIn(
                'row.split=row.split === "test" ? "sft" : "test"',
                document,
            )
            self.assertIn(
                'const initialSelection = [{"tile_id": "testcity_4km_c1_r2", "split": "test"}]',
                document,
            )
            self.assertIn(
                'const initialSelectionSource = "data/topology_selection.csv"',
                document,
            )

            test_index = _write_html([row], output_dir, [], default_split="test")
            self.assertIn('split: "test"', test_index.read_text(encoding="utf-8"))

    def test_static_selection_and_server_root_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selection_csv = root / "topology_selection.csv"
            selection_csv.write_text(
                "tile_id,city,topology_path,split\n"
                "known,Test,final/known/abstract/topology.json,test\n"
                "outside,Test,final/outside/abstract/topology.json,ppo\n",
                encoding="utf-8",
            )

            rows, ignored = _selection_for_html(selection_csv, {"known"})

            self.assertEqual(rows, [{"tile_id": "known", "split": "test"}])
            self.assertEqual(ignored, 1)
            visualization_index = root / "visualizations" / "index.html"
            visualization_index.parent.mkdir()
            visualization_index.touch()
            root_index = _write_root_index(root, visualization_index)
            self.assertEqual(root_index, root / "index.html")
            document = root_index.read_text(encoding="utf-8")
            self.assertIn("generated by visualize-training-data", document)
            self.assertIn("url=visualizations/index.html", document)


if __name__ == "__main__":
    unittest.main()
