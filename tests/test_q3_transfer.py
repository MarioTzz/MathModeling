"""问题三迁移模型的文件级合同与目标信息隔离。"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from utils.q3_transfer import (
    CLASS_ORDER, fit_alignment, fit_robust_scale, fit_transfer, ordered_scores,
    select_alpha, shrunk_covariance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/q3.json").read_text(encoding="utf-8"))


class Q3TransferChecks(unittest.TestCase):
    def sample_tables(self):
        features = CONFIG["features"]
        source_values = np.arange(8 * len(features), dtype=float).reshape(8, -1)
        source_values += np.arange(8)[:, None] ** 2 / 3
        target_values = source_values[:4] * .3 + 4
        source = pd.DataFrame(source_values, columns=features)
        target = pd.DataFrame(target_values, columns=features)
        source["file_id"] = [f"source/{i}" for i in range(8)]
        target["file_id"] = [f"target/{i}" for i in range(4)]
        source["label"] = ["N", "N", "OR", "OR", "IR", "IR", "B", "B"]
        target["label"] = [""] * 4
        return source, target

    def test_alignment_identity_and_target_metadata_isolation(self):
        source, target = self.sample_tables()
        scale, alignment, model = fit_transfer(source, target, CONFIG, 0.0, CLASS_ORDER)
        original_source = scale.transform(source)
        np.testing.assert_array_equal(alignment.transform_source(original_source),
                                      original_source)
        expected = ordered_scores(model, scale.transform(target), CLASS_ORDER)
        altered = target.copy()
        altered["file_id"] = ["N", "B", "IR", "OR"]
        altered["label"] = ["B", "N", "OR", "IR"]
        np.testing.assert_allclose(ordered_scores(model, scale.transform(altered),
                                                   CLASS_ORDER), expected)
        np.testing.assert_allclose(expected.sum(axis=1), 1.0)

    def test_shrinkage_positive_with_small_target_sample(self):
        source, target = self.sample_tables()
        scale = fit_robust_scale(source, CONFIG["features"])
        cov, rho, epsilon = shrunk_covariance(scale.transform(target)[:, 3:])
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        self.assertGreaterEqual(rho, 0)
        self.assertLessEqual(rho, 1)
        self.assertGreater(epsilon, 0)
        mapping = fit_alignment(scale.transform(source), scale.transform(target),
                                CONFIG["features"], CONFIG["blocks"], .25)
        self.assertTrue(np.isfinite(mapping.transform_source(scale.transform(source))).all())

    def test_proxy_selection_prioritizes_worst_direction_then_lower_alpha(self):
        proxy = pd.DataFrame([
            {"alpha": 0.0, "direction": "12_to_48", "macro_f1": .9},
            {"alpha": 0.0, "direction": "48_to_12", "macro_f1": .5},
            {"alpha": .25, "direction": "12_to_48", "macro_f1": .7},
            {"alpha": .25, "direction": "48_to_12", "macro_f1": .7},
            {"alpha": .5, "direction": "12_to_48", "macro_f1": .7},
            {"alpha": .5, "direction": "48_to_12", "macro_f1": .7},
        ])
        self.assertEqual(select_alpha(proxy), .25)


if __name__ == "__main__":
    unittest.main()
