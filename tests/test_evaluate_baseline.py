import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baseline_diff import metrics_from_counts
from evaluate_baseline import evaluate_split, select_examples


class BaselineEvaluationTests(unittest.TestCase):
    def test_pooled_metrics_include_false_alarms_on_blank_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory)
            for folder in ("A", "B", "label"):
                (split / folder).mkdir()

            pairs = {
                # One correct change, one false alarm, one missed change.
                "changed.png": ([255, 255, 0, 30], [255, 0, 255, 0]),
                # Two false alarms on a completely unchanged reference.
                "unchanged.png": ([255, 255, 0, 0], [0, 0, 0, 0]),
            }
            for name, (after, label) in pairs.items():
                Image.new("RGB", (4, 1)).save(split / "A" / name)
                Image.fromarray(np.array([after], dtype=np.uint8)).convert("RGB").save(split / "B" / name)
                Image.fromarray(np.array([label], dtype=np.uint8)).save(split / "label" / name)

            rows, totals = evaluate_split(split, threshold=30)

        self.assertEqual([row["sample"] for row in rows], ["changed.png", "unchanged.png"])
        self.assertEqual(rows[0]["F1"], 0.5)
        self.assertEqual(rows[1]["fp"], 2)
        self.assertEqual(rows[1]["Recall"], 0)
        self.assertEqual(totals, {"tp": 1, "fp": 3, "fn": 1})
        self.assertEqual(json.loads(json.dumps(totals)), totals)
        metrics = metrics_from_counts(**totals)
        self.assertEqual(metrics, {"Precision": 0.25, "Recall": 0.5, "F1": 1 / 3, "IoU": 0.2})

        examples = select_examples(rows)
        self.assertEqual(examples["false_alarms"]["sample"], "unchanged.png")
        self.assertEqual(examples["missed_changes"]["sample"], "changed.png")

    def test_perfect_unchanged_prediction_has_no_error_examples(self):
        rows = [{"sample": "unchanged.png", "tp": 0, "fp": 0, "fn": 0}]
        self.assertEqual(select_examples(rows), {"false_alarms": None, "missed_changes": None})
        self.assertEqual(metrics_from_counts(tp=0, fp=0, fn=0)["F1"], 0.0)


if __name__ == "__main__":
    unittest.main()
