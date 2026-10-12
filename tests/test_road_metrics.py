from __future__ import annotations

import unittest

from upmodel.road_metrics import (
    road_budget,
    road_structure_metrics,
    road_traffic_metrics,
)


class RoadMetricsTest(unittest.TestCase):
    def test_paper_formula_is_applied_term_by_term(self) -> None:
        metrics = road_traffic_metrics(
            [(0, 1), (2, 3)],
            [100.0, 700.0],
            {0, 1},
            16_000_000.0,
            budget=2,
        )
        self.assertAlmostEqual(float(metrics["Tdensity"]), 0.05)
        self.assertEqual(metrics["NCC"], 2.0)
        self.assertEqual(metrics["dead_ends"], 4.0)
        self.assertAlmostEqual(float(metrics["Tconnectivity"]), 1 / 2 + 1 / 5)
        self.assertEqual(metrics["too_long"], 1.0)
        self.assertEqual(metrics["too_short"], 0.0)
        self.assertAlmostEqual(float(metrics["Tspacing"]), 1 / 2 + 1)
        self.assertAlmostEqual(
            float(metrics["Traffic"]),
            (0.05 + (1 / 2 + 1 / 5) + (1 / 2 + 1)) / 3,
        )
        self.assertFalse(bool(metrics["connected"]))
        self.assertTrue(bool(metrics["feasible"]))

    def test_empty_plan_is_defined_as_zero(self) -> None:
        metrics = road_traffic_metrics([(0, 1)], [200.0], set(), 1_000_000.0, budget=1)
        self.assertEqual(metrics["Traffic"], 0.0)
        self.assertFalse(bool(metrics["feasible"]))

    def test_budget_fraction_is_explicit_and_default_supports_backbones(self) -> None:
        self.assertEqual(road_budget(100, 0.40), 40)
        self.assertEqual(road_budget(3, 0.40), 2)
        self.assertEqual(road_budget(100), 85)

    def test_omitted_source_nodes_make_a_plan_structurally_infeasible(self) -> None:
        metrics = road_structure_metrics(
            [(0, 1), (1, 2), (2, 0)],
            {0},
            interior_nodes={0, 1, 2},
        )
        self.assertEqual(metrics["road_node_coverage"], 2.0 / 3.0)
        self.assertFalse(metrics["connected_within_source_components"])
        self.assertFalse(metrics["structurally_feasible"])


if __name__ == "__main__":
    unittest.main()
