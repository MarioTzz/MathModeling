"""有已知答案的机制、边界及信息隔离检查；不训练后续分类器。"""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from utils.q1_model import (
    Record, FEATURES, prepare_record, window_features, order_spectrum,
    fit_window_scaler, choose_variant,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/q1.json").read_text())


def simulated_record(fs=12000, gain=1.0, seconds=4):
    time = np.arange(round(fs * seconds)) / fs
    # 2 kHz 载波、60 Hz 调制，在20 Hz转频下主峰应为3阶。
    values = gain * (1 + .6 * np.cos(2 * np.pi * 60 * time)) * np.cos(2 * np.pi * 2000 * time)
    metadata = dict(file_id="simulated", domain="source", split="train", label="IR", label_code=2)
    return Record(metadata, values - values.mean(), fs, 20)


class Q1Checks(unittest.TestCase):
    def test_known_modulation_and_cross_sampling_rates(self):
        medians = []
        for fs in [12000, 32000, 48000]:
            record = simulated_record(fs)
            envelope = prepare_record(record, [1500, 3000], CONFIG)
            features, _ = window_features(record, envelope, 20, CONFIG)
            self.assertTrue(np.allclose(features.env_peak_order, 3, atol=.1))
            medians.append(features[FEATURES].median().to_numpy())
        np.testing.assert_allclose(np.array(medians)[:, 6:], [medians[0][6:]] * 3,
                                   atol=.002, rtol=.002)

    def test_gain_after_full_filter_chain(self):
        reference = None
        for gain in [.5, 1, 2]:
            record = simulated_record(gain=gain)
            envelope = prepare_record(record, [1500, 3000], CONFIG)
            frame, _ = window_features(record, envelope, 20, CONFIG)
            values = frame[FEATURES].to_numpy()
            values[:, :3] /= gain
            if reference is None:
                reference = values
            np.testing.assert_allclose(values, reference, rtol=1e-8, atol=1e-8)

    def test_empty_window_and_constant_envelope_rejected(self):
        record = simulated_record(seconds=.2)
        envelope = prepare_record(record, [1500, 3000], CONFIG)
        with self.assertRaises(ValueError):
            window_features(record, envelope, 20, CONFIG)
        with self.assertRaises(ValueError):
            order_spectrum(np.ones(2000), 20, CONFIG)

    def test_file_weighted_scaler_and_target_isolation(self):
        frame = pd.DataFrame({
            "file_id": ["a", "b", "b", "b", "target"],
            "split": ["train"] * 4 + ["target"], "domain": ["source"] * 4 + ["target"],
            "feature": [0, 2, 2, 2, 1e9],
        })
        scaler = fit_window_scaler(frame, ["feature"])
        np.testing.assert_allclose(scaler["mean"], [1.0], atol=1e-14)
        np.testing.assert_allclose(scaler["std"], [1.0], atol=1e-14)
        frame.loc[4, "feature"] = -1e18
        self.assertEqual(scaler, fit_window_scaler(frame, ["feature"]))

    def test_candidate_choice_ignores_nontraining_labels_and_values(self):
        rng = np.random.default_rng(11)
        metadata, rows = [], []
        for split in ["train", "test", "target"]:
            for label_index, label in enumerate(["N", "OR", "IR", "B"]):
                for index in range(2):
                    file_id = f"{split}-{label}-{index}"
                    metadata.append(dict(file_id=file_id, label=label, split=split))
                    rows.append(dict(file_id=file_id, **dict(zip(
                        FEATURES, rng.normal(size=len(FEATURES)) + label_index))))
        metadata, frame = pd.DataFrame(metadata), pd.DataFrame(rows)
        candidates = {"first": frame, "second": frame.assign(ac_rms=frame.ac_rms * 2)}
        before = choose_variant(candidates, metadata, CONFIG)[0:2]
        metadata.loc[metadata.split.ne("train"), "label"] = "changed"
        changed = {}
        for name, table in candidates.items():
            changed[name] = table.copy()
            changed[name].loc[~table.file_id.str.startswith("train"), FEATURES] = 1e30
        after = choose_variant(changed, metadata, CONFIG)[0:2]
        self.assertEqual(before[0], after[0])
        pd.testing.assert_frame_equal(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
