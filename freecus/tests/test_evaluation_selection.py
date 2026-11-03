import unittest
from pathlib import Path

from freecus.evaluate import prepare, project_evaluator
from freecus.manifest import PROJECT


class EvaluationSelectionTests(unittest.TestCase):
    def test_no_style_excludes_exactly_twenty_references_and_180_cases(self):
        manifest = Path(__file__).resolve().parents[1] / "manifests/dreambench_plus_project_seed42.jsonl"
        plan = prepare(manifest, PROJECT, Path("unused"), project_evaluator(), "dbpp-no-style")
        self.assertEqual(len(plan["records"]), 1170)
        self.assertEqual(plan["selection"]["excluded_cases"], 180)
        self.assertEqual(len(plan["selection"]["excluded_reference_ids"]), 20)
        self.assertEqual(len({r["row"]["reference_id"] for r in plan["records"]}), 130)
        self.assertTrue(all(r["category"] != "style" for r in plan["records"]))
        self.assertEqual({r["category"] for r in plan["records"]}, {"animal", "human", "object"})

    def test_cannot_apply_dbpp_subset_to_dreambench(self):
        manifest = Path(__file__).resolve().parents[1] / "manifests/dreambench_project_seed42.jsonl"
        with self.assertRaises(ValueError):
            prepare(manifest, PROJECT, Path("unused"), project_evaluator(), "dbpp-no-style")


if __name__ == "__main__":
    unittest.main()
