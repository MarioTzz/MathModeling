"""97% 候选审计的训练折权重与推断隔离检查。"""
import unittest

import numpy as np
import pandas as pd

from 问题二_97目标核验 import (
    fit_candidate, four_class_scores, position_sublabels, subtype_class_weights,
)


class Q297AuditChecks(unittest.TestCase):
    def test_training_weights_use_only_the_given_fold(self):
        fold = np.asarray(["N", "N", "OR_C", "OR_O", "IR", "B"])
        weights = subtype_class_weights(fold)
        self.assertAlmostEqual(weights["N"], 6 / (5 * 2))
        self.assertAlmostEqual(weights["OR_C"], 4 * 6 / 5)
        self.assertAlmostEqual(weights["OR_O"], 6 / 5)
        with self.assertRaises(ValueError):
            subtype_class_weights(fold[fold != "OR_C"])

    def test_inference_ignores_labels_paths_and_split(self):
        labels = ["N", "N", "OR", "OR", "OR", "OR", "IR", "IR", "B", "B"]
        paths = ["n0", "n1", "path/Centered/or0", "path/Centered/or1",
                 "path/Opposite/or2", "path/Orthogonal/or3", "ir0", "ir1",
                 "b0", "b1"]
        frame = pd.DataFrame({
            "file_id": paths, "label": labels, "split": ["train"] * 10,
            "signal_a": np.arange(10, dtype=float),
            "signal_b": np.arange(10, dtype=float)[::-1],
        })
        self.assertEqual(set(position_sublabels(frame)),
                         {"N", "OR_C", "OR_O", "IR", "B"})
        model = fit_candidate(frame, ["signal_a", "signal_b"], position_subtypes=True)
        original = four_class_scores(model, frame)
        misleading = frame.copy()
        misleading["label"] = ["B"] * len(frame)
        misleading["file_id"] = ["changed"] * len(frame)
        misleading["split"] = ["test"] * len(frame)
        np.testing.assert_allclose(four_class_scores(model, misleading), original)
        np.testing.assert_allclose(original.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
