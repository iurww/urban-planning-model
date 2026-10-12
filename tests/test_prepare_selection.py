from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.batch_solve import _iter_real_tile_dirs
from scripts.prepare_training_data import (
    _dataset_layout_paths,
    _matched_train_test_rows,
    _spatial_footprint_key,
    _structural_pass,
    build_parser,
    prepare_selection_links,
    prepare_training_selection,
    run,
)
from upmodel.data.topology import discover_tile_dirs


def _topology(path: Path, tile_id: str, parcels: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "abstract-urban-topology-1.0",
                "metadata": {"tile_id": tile_id},
                "parcels": [{"id": f"parcel/{index}"} for index in range(parcels)],
                "roads": [],
                "intersections": [],
            }
        ),
        encoding="utf-8",
    )


def _training_candidate(path: Path, tile_id: str, parcels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "abstract-urban-topology-1.1",
        "metadata": {
            "tile_id": tile_id,
            "source_crs": "EPSG:32650",
            "local_origin_in_source_crs": [
                float(sum(map(ord, tile_id)) * 4000),
                0.0,
            ],
            "tile_width_m": 4000.0,
            "tile_height_m": 4000.0,
        },
        "parcels": [
            {
                "id": f"parcel/{index}",
                "area_m2": 16_000_000.0 / parcels,
                "compactness": 0.8,
                "geometry": [
                    [
                        [0.0, 0.0],
                        [4000.0, 0.0],
                        [4000.0, 4000.0],
                        [0.0, 4000.0],
                    ]
                ],
            }
            for index in range(parcels)
        ],
        "roads": [
            {
                "id": "road/0",
                "length_m": 4000.0,
                "geometry": [[0.0, 2000.0], [4000.0, 2000.0]],
            }
        ],
        "intersections": [
            {
                "id": "intersection/0",
                "kind": "junction",
                "degree": 3,
                "point": [2000.0, 2000.0],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.with_name("traffic_signals.json").write_text(
        json.dumps(
            {
                "schema_version": "abstract-traffic-signals-2.0",
                "sumo_network": {},
                "controllers": [{"intersection_id": "intersection/0"}],
            }
        ),
        encoding="utf-8",
    )


def _matched_row(
    tile_id: str,
    *,
    entities: float,
    marker: int,
    city: str = "A",
    footprint_key: str | None = None,
) -> dict:
    row = {
        "tile_id": tile_id,
        "city": city,
        "topology_path": Path(f"/tmp/{tile_id}/topology.json"),
        "features": {
            "parcels": entities * 0.25,
            "roads": entities * 0.50,
            "junctions": entities * 0.25,
            "entities": entities,
            "mean_junction_degree": 3.0 + (marker % 3) * 0.1,
            "degree4_fraction": (marker % 5) / 5.0,
            "mean_parcel_compactness": 0.5 + (marker % 4) * 0.02,
            "mean_road_length_m": 200.0 + marker,
            "polyline_road_fraction": (marker % 7) / 7.0,
        },
    }
    if footprint_key is not None:
        row["footprint_key"] = footprint_key
    return row


class CsvSelectionLinkTest(unittest.TestCase):
    def test_default_outputs_are_siblings_under_candidate_dataset_root(self) -> None:
        args = build_parser().parse_args(
            ["--data-root", "datadoc/testdoc"]
        )

        self.assertEqual(
            _dataset_layout_paths(args),
            (
                Path("datadoc/testdoc"),
                Path("datadoc/topology_selection.csv"),
                Path("datadoc/sft"),
                Path("datadoc/ppo"),
                Path("datadoc/test"),
            ),
        )

    def test_parser_has_one_candidate_root_workflow(self) -> None:
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--data-root", option_strings)
        self.assertIn("--from-csv", option_strings)
        self.assertTrue(
            {
                "--source",
                "--output-root",
                "--target",
                "--rebalance-split",
                "--selection-data-root",
                "--reselect-removed",
            }.isdisjoint(option_strings)
        )

    def test_from_csv_only_synchronizes_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = root / "data" / "final"
            selection_csv = root / "manual_selection.csv"
            _topology(
                data_root / "selected_tile" / "abstract" / "topology.json",
                "selected_tile",
            )
            csv_content = (
                "tile_id,city,topology_path,split\n"
                "selected_tile,Manual,"
                "data/final/selected_tile/abstract/topology.json,ppo\n"
            )
            selection_csv.write_text(csv_content, encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--data-root",
                    str(data_root),
                    "--from-csv",
                    str(selection_csv),
                ]
            )

            with patch(
                "scripts.prepare_training_data.prepare_training_selection"
            ) as automatic_selection:
                run(args)

            automatic_selection.assert_not_called()
            self.assertEqual(selection_csv.read_text(encoding="utf-8"), csv_content)
            self.assertFalse(
                selection_csv.with_name(
                    "manual_selection.split_report.json"
                ).exists()
            )
            self.assertTrue((root / "data" / "ppo" / "selected_tile").is_symlink())
            self.assertFalse((root / "data" / "sft" / "selected_tile").exists())
            self.assertFalse((root / "data" / "test" / "selected_tile").exists())
            ppo_manifest = json.loads(
                (root / "data" / "ppo" / "selection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(ppo_manifest["count"], 1)
            self.assertEqual(
                Path(ppo_manifest["selection_csv"]),
                selection_csv.resolve(),
            )

    def test_structural_filter_accepts_current_polyline_and_parallel_road_contract(self) -> None:
        topology = {
            "metadata": {
                "validation": {
                    "all_roads_are_two_point_straight_segments": False,
                    "all_roads_are_valid_lines_or_polylines": True,
                    "all_road_endpoints_reference_intersections": True,
                    "geometry_breaks_stored_as_intersections": 0,
                    "duplicate_road_endpoint_pairs": 2,
                    "parcel_coverage_ratio": 1.0,
                    "parcel_overlap_area_m2": 0.0,
                }
            }
        }

        self.assertTrue(_structural_pass(topology, max_complex_parcels=2))

        topology["metadata"]["validation"][
            "geometry_breaks_stored_as_intersections"
        ] = 1
        self.assertFalse(_structural_pass(topology, max_complex_parcels=2))

    def test_prepare_filters_splits_and_links_without_copying_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = root / "datadoc" / "candidates"
            sft_root = root / "data" / "sft"
            ppo_root = root / "data" / "ppo"
            test_root = root / "data" / "test"
            selection_csv = root / "data" / "topology_selection.csv"
            for tile_id in ("tile_a", "tile_b", "tile_c"):
                _training_candidate(
                    data_root / tile_id / "abstract" / "topology.json",
                    tile_id,
                )
            _training_candidate(
                data_root / "too_large" / "abstract" / "topology.json",
                "too_large",
                parcels=3,
            )
            missing_signals = (
                data_root / "missing_signals" / "abstract" / "topology.json"
            )
            _training_candidate(missing_signals, "missing_signals")
            missing_signals.with_name("traffic_signals.json").unlink()
            selection_csv.parent.mkdir(parents=True)
            selection_csv.write_text(
                "tile_id,city,topology_path,split\n"
                "too_large,Manual,candidates/too_large/abstract/topology.json,test\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args([])
            args.data_root = data_root
            args.selection_csv = selection_csv
            args.link_output_root = sft_root
            args.ppo_link_output_root = ppo_root
            args.test_link_output_root = test_root
            args.train_count = 2
            args.test_count = 1
            args.min_parcels = 2
            args.max_parcels = 2
            args.min_roads = 1
            args.max_roads = 1
            args.min_intersections = 1
            args.max_intersections = 1
            args.min_junctions = 1
            args.max_junctions = 1

            with patch(
                "scripts.prepare_training_data._land_quota_feasible",
                return_value=(True, "feasible"),
            ):
                prepare_training_selection(args)

            rows = selection_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 5)
            self.assertEqual(
                rows[1],
                "too_large,Manual,candidates/too_large/abstract/topology.json,test",
            )
            self.assertEqual(sum(row.endswith(",sft") for row in rows[1:]), 2)
            self.assertEqual(sum(row.endswith(",ppo") for row in rows[1:]), 0)
            self.assertEqual(sum(row.endswith(",test") for row in rows[1:]), 2)
            self.assertEqual(len(list(sft_root.glob("tile_*"))), 2)
            self.assertEqual(len(list(ppo_root.glob("tile_*"))), 0)
            self.assertEqual(len(list(test_root.glob("tile_*"))), 1)
            self.assertTrue((test_root / "too_large").is_symlink())
            selected_links = (
                *sft_root.glob("tile_*"),
                *test_root.glob("tile_*"),
            )
            self.assertTrue(all(path.is_symlink() for path in selected_links))
            report = json.loads(
                selection_csv.with_name(
                    "topology_selection.split_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report["candidate_count"], 5)
            self.assertEqual(report["complexity_eligible_count"], 3)
            self.assertEqual(
                report["selection_update"],
                {
                    "policy": "append-only; preserve existing CSV rows and splits",
                    "existing_count": 1,
                    "generated_count": 3,
                    "added_count": 3,
                    "final_count": 4,
                    "final_split_counts": {"sft": 2, "ppo": 0, "test": 2},
                },
            )
            self.assertEqual(
                {
                    row["tile_id"]: row["reasons"]
                    for row in report["complexity_rejected"]
                },
                {
                    "missing_signals": ["traffic_signals_missing"],
                    "too_large": ["parcels"],
                },
            )
            self.assertFalse((root / "data" / "final").exists())

    def test_csv_selection_creates_and_synchronizes_directory_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = root / "data" / "final"
            sft_root = root / "data" / "sft"
            selection_csv = root / "data" / "topology_selection.csv"
            _topology(data_root / "new_tile" / "abstract" / "topology.json", "new_tile")
            _topology(data_root / "solved_tile" / "topology.json", "solved_tile")
            selection_csv.write_text(
                "tile_id,city,topology_path\n"
                "new_tile,New,data/final/new_tile/abstract/topology.json\n"
                "solved_tile,Solved,data/final/solved_tile/topology.json\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                selection_csv=selection_csv,
                data_root=data_root,
                link_output_root=sft_root,
                keep_unselected_links=False,
                dry_run=False,
            )

            prepare_selection_links(args)

            self.assertTrue((sft_root / "new_tile").is_symlink())
            self.assertTrue((sft_root / "solved_tile").is_symlink())
            self.assertEqual(
                (sft_root / "new_tile").resolve(),
                (data_root / "new_tile" / "abstract").resolve(),
            )
            self.assertEqual(
                {path.name for path in discover_tile_dirs(sft_root)},
                {"new_tile", "solved_tile"},
            )
            self.assertEqual(
                {path.name for path in _iter_real_tile_dirs(sft_root)},
                {"new_tile", "solved_tile"},
            )

            selection_csv.write_text(
                "tile_id,city,topology_path\n"
                "new_tile,New,data/final/new_tile/abstract/topology.json\n",
                encoding="utf-8",
            )
            prepare_selection_links(args)

            self.assertTrue((sft_root / "new_tile").is_symlink())
            self.assertFalse((sft_root / "solved_tile").exists())
            manifest = json.loads(
                (sft_root / "selection_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(manifest["removed"], 1)

    def test_matched_split_has_exact_counts_and_close_entity_sizes(self) -> None:
        rows = [
            _matched_row(
                f"tile_{index:02d}",
                entities=100.0 + index * 3.0,
                marker=index,
                city="A" if index < 20 else "B",
            )
            for index in range(40)
        ]
        selected, report = _matched_train_test_rows(
            rows, train_count=16, test_count=4
        )
        self.assertEqual(sum(row["split"] == "train" for row in selected), 16)
        self.assertEqual(sum(row["split"] == "test" for row in selected), 4)
        self.assertEqual(len(report["matched_groups"]), 4)
        for group in report["matched_groups"]:
            self.assertEqual(len(group["train"]), 4)
            self.assertLess(
                max(abs(value - group["test_entities"]) for value in group["train_entities"]),
                40.0,
            )

    def test_spatial_footprint_key_ignores_alias_and_city(self) -> None:
        metadata = {
            "source_crs": "EPSG:32651",
            "local_origin_in_source_crs": [264000.0, 3468000.0],
            "tile_width_m": 4000.0,
            "tile_height_m": 4000.0,
            "tile_id": "suzhou_alias",
            "city": "Suzhou",
        }
        alias = {"metadata": {**metadata, "tile_id": "wuxi_alias", "city": "Wuxi"}}
        different_crs = {
            "metadata": {**metadata, "source_crs": "EPSG:32650"}
        }

        self.assertEqual(
            _spatial_footprint_key({"metadata": metadata}),
            _spatial_footprint_key(alias),
        )
        self.assertNotEqual(
            _spatial_footprint_key({"metadata": metadata}),
            _spatial_footprint_key(different_crs),
        )
        self.assertIsNone(_spatial_footprint_key({"metadata": {"tile_id": "unknown"}}))

    def test_matched_split_keeps_size_two_expert_variants_atomic_in_train(self) -> None:
        rows = [
            _matched_row(
                f"single_{index:02d}",
                entities=100.0 + index * 2.0,
                marker=index,
                city="A" if index < 9 else "B",
            )
            for index in range(18)
        ]
        rows.extend(
            [
                _matched_row(
                    "expert_variant_old",
                    entities=119.0,
                    marker=30,
                    city="A",
                    footprint_key="footprint:shared",
                ),
                _matched_row(
                    "expert_variant_new",
                    entities=119.0,
                    marker=31,
                    city="A",
                    footprint_key="footprint:shared",
                ),
            ]
        )

        selected, report = _matched_train_test_rows(
            rows, train_count=16, test_count=4
        )
        split_by_id = {row["tile_id"]: row["split"] for row in selected}

        self.assertEqual(split_by_id["expert_variant_old"], "train")
        self.assertEqual(split_by_id["expert_variant_new"], "train")
        self.assertEqual(sum(row["split"] == "train" for row in selected), 16)
        self.assertEqual(sum(row["split"] == "test" for row in selected), 4)
        self.assertTrue(report["peer_capacity_balanced"])
        self.assertTrue(
            all(
                row.get("footprint_key") != "footprint:shared"
                for row in selected
                if row["split"] == "test"
            )
        )

        reversed_selected, reversed_report = _matched_train_test_rows(
            list(reversed(rows)), train_count=16, test_count=4
        )
        self.assertEqual(
            [(row["tile_id"], row["split"]) for row in selected],
            [(row["tile_id"], row["split"]) for row in reversed_selected],
        )
        self.assertEqual(report["matched_groups"], reversed_report["matched_groups"])

    def test_matched_split_reports_unreachable_atomic_train_count(self) -> None:
        rows = [
            _matched_row("test_singleton", entities=100.0, marker=0),
            _matched_row(
                "pair_a_1",
                entities=101.0,
                marker=1,
                footprint_key="footprint:pair-a",
            ),
            _matched_row(
                "pair_a_2",
                entities=101.0,
                marker=2,
                footprint_key="footprint:pair-a",
            ),
            _matched_row(
                "pair_b_1",
                entities=102.0,
                marker=3,
                footprint_key="footprint:pair-b",
            ),
            _matched_row(
                "pair_b_2",
                entities=102.0,
                marker=4,
                footprint_key="footprint:pair-b",
            ),
        ]

        with self.assertRaisesRegex(
            ValueError, "cannot fill train_count=3 atomically.*footprint group sizes"
        ):
            _matched_train_test_rows(rows, train_count=3, test_count=1)


if __name__ == "__main__":
    unittest.main()
