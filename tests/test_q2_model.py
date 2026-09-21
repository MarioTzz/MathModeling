"""问题二模型的权重、文件汇总和 SVM 数值一致性测试。"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from utils.q1_model import FEATURES, LABELS
from utils.q2_model import (
    aggregate_predictions, audit_weights, classification_metrics,
    fit_file_equal_scaler, fit_ovr, reconstruct_scores, sample_weights,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/q2.json").read_text(encoding="utf-8"))


def synthetic_frame():
    rows = []
    rng = np.random.default_rng(7)
    for class_index, label in enumerate(LABELS):
        for file_index, windows in enumerate([2, 5]):
            file_id = f"{label}-{file_index}"
            for window_index in range(windows):
                values = rng.normal(scale=0.03, size=len(FEATURES))
                values[0] += 2.0 * class_index
                rows.append({"file_id": file_id, "window_index": window_index,
                             "label": label, "split": "train",
                             **dict(zip(FEATURES, values))})
    return pd.DataFrame(rows)


class Q2Checks(unittest.TestCase):
    def test_weight_identities(self):
        frame = synthetic_frame()
        ids = frame.file_id.unique()
        for scheme in ["window_uniform", "class_balanced", "file_class_balanced"]:
            self.assertAlmostEqual(sample_weights(frame, ids, scheme).sum(), 1.0)
        by_file, by_class = audit_weights(frame, ids, "file_class_balanced")
        np.testing.assert_allclose(by_class.to_numpy(), np.repeat(0.25, 4))
        for _, group in by_file.groupby("label"):
            np.testing.assert_allclose(group.sample_weight, group.sample_weight.iloc[0])

    def test_file_equal_scaler_ignores_window_count(self):
        frame = synthetic_frame()
        ids = frame.file_id.unique()
        scaler = fit_file_equal_scaler(frame, ids, [FEATURES[0]])
        file_means = frame.groupby("file_id")[FEATURES[0]].mean()
        self.assertAlmostEqual(scaler["mean"][0], file_means.mean(), places=12)

    def test_explicit_ovr_and_score_reconstruction(self):
        frame = synthetic_frame()
        ids = frame.file_id.unique()
        selected = [FEATURES[0], FEATURES[1]]
        scaler = fit_file_equal_scaler(frame, ids, selected)
        model = fit_ovr(frame, ids, selected, scaler, kernel="rbf", C=64,
                        gamma=.125, weighting="file_class_balanced", config=CONFIG)
        self.assertEqual(set(model.estimators), set(LABELS))
        direct = model.decision_function(frame)
        rebuilt = reconstruct_scores(model, frame)
        np.testing.assert_allclose(direct, rebuilt, rtol=1e-10, atol=1e-10)
        predictions, _ = aggregate_predictions(model, frame)
        metrics, detail, confusion = classification_metrics(
            predictions.true_label, predictions.prediction)
        self.assertEqual(metrics["n_files"], 8)
        self.assertEqual(confusion.values.sum(), 8)
        self.assertEqual(set(detail.label), set(LABELS))


if __name__ == "__main__":
    unittest.main()
