import unittest

from freecus.modal_cost import projection, rate
from freecus.modal_client import pilot_projection, accounted_usd


class ModalCostTests(unittest.TestCase):
    def test_parallelism_reduces_wall_time_without_halving_compute_cost(self):
        one = projection(750, 30, workers=1)
        two = projection(750, 30, workers=2)
        self.assertEqual(one["compute_usd"], two["compute_usd"])
        self.assertEqual(two["wall_seconds"], one["wall_seconds"] / 2)
        self.assertAlmostEqual(rate() * 3600, 3.104208)

    def test_pending_and_failed_jobs_reserve_budget(self):
        receipts = [dict(calls=[dict(budget_usd=1.25), dict(budget_usd=1.25, result=dict(estimated_compute_usd=0.3))])]
        self.assertEqual(accounted_usd(receipts), 1.55)

    def test_full_run_requires_successful_pilot(self):
        with self.assertRaises(ValueError):
            pilot_projection([], "dreambench")
        failed = [dict(mode="pilot", calls=[dict(result=dict(status="failed"))])]
        with self.assertRaises(ValueError):
            pilot_projection(failed, "dreambench")

    def test_pilot_separates_cold_model_loading_from_steady_images(self):
        result = dict(status="complete", elapsed_seconds=150,
                      events=[dict(phase="generated", seconds=90), dict(phase="generated", seconds=20),
                              dict(phase="generated", seconds=20)])
        receipts = [dict(mode="pilot", invocation_id="test", calls=[dict(result=result), dict(result=result)])]
        estimate = pilot_projection(receipts, "dreambench")
        self.assertEqual(estimate["seconds_per_image"], 20)
        self.assertEqual(estimate["measured_images"], 6)
        self.assertAlmostEqual(estimate["total_worker_seconds"], 750 * 20 + 2 * 110)

    def test_h100_projection_uses_h100_rate(self):
        result = dict(status="complete", elapsed_seconds=150,
                      events=[dict(phase="generated", seconds=90), dict(phase="generated", seconds=15)])
        receipts = [dict(mode="pilot", invocation_id="test", calls=[dict(result=result), dict(result=result)])]
        estimate = pilot_projection(receipts, "dreambench_plus", gpu="H100")
        self.assertEqual(estimate["gpu"], "H100")
        self.assertEqual(estimate["samples"], 1350)
        self.assertAlmostEqual(estimate["compute_usd"], estimate["total_worker_seconds"] * rate("H100"))


if __name__ == "__main__":
    unittest.main()
