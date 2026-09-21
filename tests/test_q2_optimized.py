"""优化模型的物理口径、文件级聚合和可移植树状态。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.q2_optimized import (
    aggregate_file_medians, bearing_fault_orders, feature_sets, fit_extra_trees,
    reconstruct_scores, save_model, scores_from_export, tune_extra_trees,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/q2.json").read_text(encoding="utf-8"))["optimized"]


class OptimizedQ2Checks(unittest.TestCase):
    def test_title_bsf_and_spin_half_have_distinct_names_and_values(self):
        orders = bearing_fault_orders(CONFIG["bearing_geometry"])
        d = 0.3126
        pitch = 1.537
        self.assertAlmostEqual(orders["bsf"], pitch/d * (1 - (d/pitch)**2))
        self.assertAlmostEqual(orders["bsf"], 2*orders["spin_half"])
        families = feature_sets(CONFIG)
        self.assertTrue(all("spin_half" not in name
                            for name in families["multiband_title_bsf"]))
        self.assertTrue(all("bsf_" not in name
                            for name in families["multiband_spin_half"]))

    def test_file_median_does_not_duplicate_long_record(self):
        windows = pd.DataFrame({
            "file_id": ["A", "A", "A", "B"], "label": ["N"]*3+["OR"],
            "split": ["train"]*4, "signal": [1., 2., 100., 4.],
        })
        table = aggregate_file_medians(windows, ["signal"])
        self.assertEqual(len(table), 2)
        self.assertEqual(table.set_index("file_id").loc["A", "signal__median"], 2.)

    def test_exported_tree_scores_reconstruct_without_pickle(self):
        frame = pd.DataFrame({
            "file_id": list("abcdefgh"), "label": ["N", "N", "OR", "OR",
                                                    "IR", "IR", "B", "B"],
            "split": ["train"]*8,
            "a": [0., .1, 1., 1.1, 2., 2.1, 3., 3.1],
            "b": [3., 2.9, 2., 1.9, 1., .9, 0., -.1],
        })
        params = {"min_samples_leaf": 1, "max_features": "sqrt"}
        local = dict(CONFIG, n_estimators=11)
        model = fit_extra_trees(frame, ["a", "b"], params, local)
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "model.json"
            arrays_path = Path(directory) / "model_arrays.npz"
            save_model(model, json_path, arrays_path)
            self.assertTrue(np.allclose(model.scores(frame),
                                        scores_from_export(json_path, arrays_path, frame),
                                        atol=1e-14))
            # 在树的真实根阈值两侧构造高精度探针，验证 sklearn 的 float32 路由。
            probes = []
            for estimator in model.estimator.estimators_:
                feature_index = int(estimator.tree_.feature[0])
                if feature_index < 0:
                    continue
                threshold = float(estimator.tree_.threshold[0])
                for value in [np.nextafter(threshold, -np.inf), threshold,
                              np.nextafter(threshold, np.inf), threshold + 1e-8]:
                    probe = frame.iloc[0].copy()
                    probe[model.features[feature_index]] = value
                    probes.append(probe)
            self.assertGreater(len(probes), 0)
            boundary = pd.DataFrame(probes)
            expected = model.scores(boundary)
            self.assertTrue(np.allclose(expected, reconstruct_scores(model, boundary),
                                        atol=1e-14))
            self.assertTrue(np.allclose(expected,
                                        scores_from_export(json_path, arrays_path, boundary),
                                        atol=1e-14))

    def test_tuning_tie_uses_configured_candidate_order(self):
        frame = pd.DataFrame({
            "file_id": list("abcdefgh"), "label": ["N", "N", "OR", "OR",
                                                    "IR", "IR", "B", "B"],
            "split": ["train"]*8, "constant": [1.]*8,
        })
        local = dict(CONFIG, n_estimators=11, cv_seeds=[42],
                     min_samples_leaf_grid=[1],
                     max_features_grid=["sqrt", "0.25"])
        parameters, _, summary = tune_extra_trees(frame, ["constant"], local, 2)
        self.assertEqual(summary.accuracy_mean.nunique(), 1)
        self.assertEqual(parameters["max_features"], "sqrt")


if __name__ == "__main__":
    unittest.main()
