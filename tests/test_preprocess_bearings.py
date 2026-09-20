"""Regression checks for time, labels, scaling, and preservation of signal information."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.preprocess_bearings import (
    parse_metadata, split_files, make_windows, resample_signal, fit_scaler,
    apply_scaler, save_mat, loadmat,
)


class PreprocessingTests(unittest.TestCase):
    def test_target_letters_are_not_labels_or_precise_speed_measurements(self):
        for name in ("B", "N"):
            m = parse_metadata(Path(f"目标域数据集/{name}.mat"), {name: np.ones((20, 1))})
            self.assertEqual(m["label_code"], -1)
            self.assertFalse(m["label_known"])
            self.assertEqual(m["rpm_source"], "problem_approximate")
            self.assertIsNone(m["fault_size_mm"])

    def test_filename_speed_fallback_has_provenance_and_conflicts_fail(self):
        p = Path("源域数据集/12kHz_DE_data/B/0028/B028_0_(1797rpm).mat")
        m = parse_metadata(p, {"X1_DE_time": np.arange(20.).reshape(-1, 1)})
        self.assertEqual(m["rpm"], 1797)
        self.assertEqual(m["rpm_source"], "filename")
        self.assertAlmostEqual(m["fault_size_mm"], .7112)
        with self.assertRaises(ValueError):
            parse_metadata(p, {"X1RPM": np.array([[1772]])})

    def test_resampling_preserves_duration_and_suppresses_aliasing(self):
        for fs in (32000, 48000):
            t = np.arange(fs) / fs
            x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 9000 * t)
            y = resample_signal(x, fs)
            self.assertEqual(len(y), 12000)
            reference = np.sin(2 * np.pi * 1000 * np.arange(12000) / 12000)
            # Direct decimation aliases the 9 kHz component to 3 kHz and fails this.
            self.assertLess(np.sqrt(np.mean((y[100:-100] - reference[100:-100])**2)), .01)

    def test_window_normalization_retains_inverse_and_tail(self):
        x = np.r_[np.tile([1., 3.], 6000), np.array([7., 8., 9.])]
        w = make_windows(x)
        self.assertEqual(w["unit_rms"].shape, (1, 12000))
        self.assertEqual(w["tail_samples"], 3)
        np.testing.assert_allclose(w["unit_rms"].mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.mean(w["unit_rms"]**2, axis=1), 1)
        np.testing.assert_allclose(w["unit_rms"][0] * w["rms"][0] + w["means"][0], x[:12000])

    def test_zero_energy_window_is_explicit_and_not_divided_by_zero(self):
        w = make_windows(np.full(12000, 5.))
        self.assertFalse(w["valid"][0])
        self.assertTrue(np.isfinite(w["unit_rms"]).all())
        self.assertEqual(w["rms"][0], 0)

    def test_scaler_ignores_test_and_target_and_handles_constant_feature(self):
        x = np.array([[1., 7.], [3., 7.], [100., 2.], [1000., 4.]])
        train = np.array([True, True, False, False])
        p = fit_scaler(x, train)
        np.testing.assert_allclose(p["mean"], [2., 7.])
        np.testing.assert_allclose(p["scale"], [1., 1.])
        x[2:] = -99999
        self.assertEqual(p, fit_scaler(x, train))
        np.testing.assert_allclose(apply_scaler(x, p)[:2], [[-1., 0.], [1., 0.]])

    def test_file_group_split_is_disjoint_and_preserves_rare_normal(self):
        rows = [dict(file_id=f"source/{c}/{i}", domain="source", label_code=c)
                for c in range(4) for i in range(4)]
        rows += [dict(file_id="target/N", domain="target", label_code=-1)]
        s = split_files(rows, seed=42)
        self.assertEqual(s["target/N"], "target")
        self.assertEqual(set(s[f"source/0/{i}"] for i in range(4)), {"train", "validation", "test"})
        self.assertEqual(s, split_files(list(reversed(rows)), seed=42))
        self.assertEqual(sum(v == "train" for k, v in s.items() if k.startswith("source/0/")), 2)

    def test_mat_output_is_deterministic_and_float64_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/"a.mat", Path(d)/"b.mat"
            signal = np.random.default_rng(4).normal(size=(12, 1))
            save_mat(a, {"signal": signal})
            save_mat(b, {"signal": signal})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            np.testing.assert_array_equal(loadmat(a)["signal"], signal)


if __name__ == "__main__":
    unittest.main()
