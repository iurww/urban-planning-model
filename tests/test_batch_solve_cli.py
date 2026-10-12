from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.batch_solve import add_subcommand


class BatchSolveCliTest(unittest.TestCase):
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_subcommand(parser.add_subparsers(dest="command", required=True))
        return parser

    def test_solve_all_real_has_one_optional_topology_root(self) -> None:
        parser = self._parser()

        defaults = parser.parse_args(["solve-all-real"])
        explicit = parser.parse_args(["solve-all-real", "datadoc/train"])

        self.assertEqual(defaults.topology_base, Path("data/final"))
        self.assertEqual(explicit.topology_base, Path("datadoc/train"))

    def test_batch_only_ranges_do_not_leak_into_solve_all_real(self) -> None:
        parser = self._parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        real_parser = subparsers.choices["solve-all-real"]

        self.assertEqual(
            sum(action.dest == "topology_base" for action in real_parser._actions),
            1,
        )
        self.assertNotIn("batch_from", {action.dest for action in real_parser._actions})
        self.assertNotIn("batch_to", {action.dest for action in real_parser._actions})

    def test_solve_road_real_is_not_registered(self) -> None:
        parser = self._parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

        self.assertNotIn("solve-road-real", subparsers.choices)


if __name__ == "__main__":
    unittest.main()
