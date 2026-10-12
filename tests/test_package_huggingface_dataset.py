from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from package_huggingface_dataset import (
    FIXED_ZIP_TIMESTAMP,
    build_archive,
    collect_archive_files,
)


class PackageHuggingFaceDatasetTest(unittest.TestCase):
    def test_packages_only_selected_final_tiles_and_root_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = root / "data"
            selected = data_root / "final" / "selected" / "abstract"
            unselected = data_root / "final" / "unselected" / "abstract"
            selected.mkdir(parents=True)
            unselected.mkdir(parents=True)
            (selected / "topology.json").write_text("selected", encoding="utf-8")
            (selected / "signal_trajectory.json").write_text(
                "trajectory", encoding="utf-8"
            )
            (unselected / "topology.json").write_text(
                "unselected", encoding="utf-8"
            )
            (data_root / "metadata.json").write_text("{}", encoding="utf-8")
            (data_root / "ignored.txt").write_text("ignored", encoding="utf-8")
            selection_csv = data_root / "topology_selection.csv"
            selection_csv.write_text(
                "tile_id,city,topology_path,split\n"
                "selected,Test,final/selected/abstract/topology.json,sft\n",
                encoding="utf-8",
            )

            files, tile_count = collect_archive_files(
                data_root,
                selection_csv,
            )
            archive_paths = {item.archive_path for item in files}

            self.assertEqual(tile_count, 1)
            self.assertEqual(
                archive_paths,
                {
                    "data/final/selected/abstract/topology.json",
                    "data/final/selected/abstract/signal_trajectory.json",
                    "data/metadata.json",
                    "data/topology_selection.csv",
                },
            )

            output = root / "dataset.zip"
            build_archive(
                files,
                output,
                force=False,
                compression=zipfile.ZIP_STORED,
                compresslevel=None,
            )
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), archive_paths)
                self.assertTrue(
                    all(
                        info.date_time == FIXED_ZIP_TIMESTAMP
                        for info in archive.infolist()
                    )
                )
                self.assertEqual(
                    archive.read(
                        "data/final/selected/abstract/topology.json"
                    ),
                    b"selected",
                )

    def test_rejects_topology_path_outside_matching_final_tile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw) / "data"
            topology = data_root / "final" / "selected" / "abstract"
            topology.mkdir(parents=True)
            (topology / "topology.json").write_text("{}", encoding="utf-8")
            selection_csv = data_root / "topology_selection.csv"
            selection_csv.write_text(
                "tile_id,city,topology_path,split\n"
                "selected,Test,../outside/topology.json,sft\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不指向规范候选"):
                collect_archive_files(data_root, selection_csv)


if __name__ == "__main__":
    unittest.main()
