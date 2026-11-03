import unittest
from dreamo.efficiency import choose_gpu, estimate_phase, peak_vram


class EfficiencyTests(unittest.TestCase):
    def test_hourly_cheaper_is_not_enough_if_slower_per_image(self):
        baseline = dict(compute_usd=10, peak_vram_gib=36)
        self.assertEqual(choose_gpu({"A100-80GB":baseline,"L40S":dict(compute_usd=11,peak_vram_gib=37)}),"A100-80GB")
        self.assertEqual(choose_gpu({"A100-80GB":baseline,"L40S":dict(compute_usd=8,peak_vram_gib=37)}),"L40S")
        self.assertEqual(choose_gpu({"A100-80GB":baseline,"L40S":dict(compute_usd=8,peak_vram_gib=44)}),"A100-80GB")

    def test_initialization_memory_is_included_in_admission(self):
        result = dict(status="complete", elapsed_seconds=50, initialization=dict(peak_reserved_gib=44),
                      timings=[dict(seconds=10,peak_allocated_gib=35),dict(seconds=10,peak_reserved_gib=36)])
        phase = dict(gpu="A100-80GB", calls=[dict(result=result),dict(result=result)])
        self.assertEqual(peak_vram(phase),44)
        estimate = estimate_phase(phase,750)
        self.assertEqual(estimate["seconds_per_image"],10)
        self.assertEqual(estimate["total_worker_seconds"],7560)


if __name__ == "__main__":
    unittest.main()
