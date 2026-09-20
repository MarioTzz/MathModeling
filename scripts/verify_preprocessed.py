"""Read every original and output to verify preservation, indexing, and no split leakage."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    files = read_csv(output / "files.csv")
    channels = read_csv(output / "channels.csv")
    windows = read_csv(output / "windows.csv")
    scaled = read_csv(output / "features_standardized.csv")
    scaler = json.loads((output / "scaler.json").read_text())
    summary = json.loads((output / "processing_summary.json").read_text())
    check(sha(ROOT / "scripts/preprocess_bearings.py") == manifest["script_sha256"], "Pipeline code hash changed")
    check(sha(ROOT / "requirements-preprocessing.txt") == manifest["requirements_sha256"], "Requirements changed")
    for relative, expected in manifest["output_hashes"].items():
        check(sha(output / relative) == expected, f"Output hash mismatch {relative}")
    for relative, expected in manifest["input_hashes"].items():
        check(sha(ROOT / "数据集" / relative) == expected, f"Input hash mismatch {relative}")
    check(len(files) == len({r["file_id"] for r in files}) == summary["files"], "Nonunique file IDs")
    check(len(windows) == len({r["window_id"] for r in windows}) == summary["windows"], "Nonunique window IDs")
    check(len(channels) == summary["channels"], "Channel count mismatch")
    by_file = {r["file_id"]:r for r in files}
    check(set(by_file) == set(r["file_id"] for r in windows), "Window/file coverage mismatch")
    signal_count = sample_count = 0
    max_native_error = max_window_error = max_resample_error = 0.
    for row in files:
        original = loadmat(ROOT / "数据集" / row["relative_path"])
        native = loadmat(output / row["native_path"])
        common = loadmat(output / row["common_path"])
        for name, raw in original.items():
            if name.startswith("__"):
                continue
            if "RPM" in name.upper():
                np.testing.assert_array_equal(native[name], raw)
                continue
            signal_count += 1
            sample_count += raw.size
            transformed = native[name]
            check(transformed.dtype == np.float64 and transformed.shape == raw.shape, f"Native shape/dtype {row['file_id']} {name}")
            check(np.isfinite(transformed).all(), "Nonfinite native output")
            mean = float(native[name + "_mean"].item())
            check(abs(mean-float(raw.mean())) < 1e-14, "Incorrect stored mean")
            recovered = transformed + mean
            error = float(np.max(np.abs(recovered-raw)))
            tolerance = 8*np.finfo(float).eps*max(1, float(np.max(np.abs(raw))))
            check(error <= tolerance, f"Native reconstruction {row['file_id']} {name}")
            check(abs(float(transformed.mean())) <= tolerance, "Native mean not removed")
            max_native_error = max(max_native_error, error)
        raw = original[row["primary_variable"]].ravel()
        fs = int(row["nominal_fs_hz"])
        gcd = math.gcd(fs, 12000)
        up, down = 12000//gcd, fs//gcd
        expected = raw-raw.mean()
        if fs != 12000:
            expected = resample_poly(expected, up, down, window=("kaiser",5.), padtype="line")
        y = common["signal_resampled"].ravel()
        check(len(y) == math.ceil(len(raw)*up/down) == int(row["resampled_samples"]), "Resampling length")
        err = float(np.max(np.abs(y-expected)))
        check(err < 1e-12, "Resampling values differ")
        max_resample_error = max(max_resample_error, err)
        count = len(y)//12000
        unit = common["windows_unit_rms"]
        means = common["window_means"].ravel()
        rms = common["window_ac_rms"].ravel()
        check(unit.shape == (count,12000), "Window layout mismatch")
        check((common["window_valid"].ravel() == 1).all(), "Invalid window")
        check(int(row["tail_samples"]) == len(y)%12000, "Tail accounting mismatch")
        np.testing.assert_allclose(unit.mean(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(np.mean(unit**2,axis=1), 1, atol=1e-12)
        restored = (unit*rms[:,None] + means[:,None]).ravel()
        err = float(np.max(np.abs(restored-y[:count*12000])))
        check(err < 1e-11, "Window inverse mismatch")
        max_window_error = max(max_window_error, err)
        check(int(common["label_code"].item()) == int(row["label_code"]), "MAT/CSV label mismatch")
        check(int(common["label_known"].item()) == (row["domain"] == "source"), "MAT label-known mismatch")
        related = [w for w in windows if w["file_id"] == row["file_id"]]
        check(len(related) == count, "Window row coverage")
        for i,w in enumerate(related):
            check(w["split"] == row["split"] and w["group_id"] == row["group_id"] == row["file_id"], "Group leakage")
            check(int(w["start_sample"]) == 12000*i and int(w["end_sample_exclusive"]) == 12000*(i+1), "Window indices")
            check(int(w["label_code"]) == int(row["label_code"]), "Window label mismatch")
            ac = y[i*12000:(i+1)*12000] - means[i]
            expected_features = [np.sqrt(np.mean(ac**2)), np.mean(np.abs(ac)), np.ptp(ac),
                                 np.max(np.abs(unit[i])), np.mean(unit[i]**4), np.mean(unit[i]**3)]
            np.testing.assert_allclose([float(w[f]) for f in scaler["features"]], expected_features, rtol=1e-12, atol=1e-12)
        if row["domain"] == "target":
            check(row["label_code"] == "-1" and row["label_known"] == "False" and row["label"] == "" and row["split"] == "target", "Target false label/split")
            check(row["rpm_source"] == "problem_approximate" and row["rpm_is_approximate"] == "True", "Target speed precision")
        else:
            check(row["split"] in ("train","validation","test"), "Source split")
        for name in ("DE","FE","BA"):
            check((row["has_"+name] == "True") == any(k.endswith("_"+name+"_time") for k in original), "Missing channel mask")
    check(signal_count == summary["channels"] and sample_count == summary["native_samples"], "Signal/sample coverage")
    features = np.array([[float(w[f]) for f in scaler["features"]] for w in windows])
    mask = np.array([w["domain"] == "source" and w["split"] == "train" for w in windows])
    train = features[mask]
    np.testing.assert_allclose(scaler["mean"], train.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler["std"], train.std(axis=0), rtol=1e-12, atol=1e-12)
    check(scaler["fit_file_ids"] == sorted({w["file_id"] for w in windows if w["split"] == "train"}), "Wrong scaler fit files")
    check(scaler["n_fit_windows"] == len(train), "Wrong scaler fit count")
    check([w["window_id"] for w in windows] == [w["window_id"] for w in scaled], "Feature row order")
    observed = np.array([[float(w[f]) for f in scaler["features"]] for w in scaled])
    np.testing.assert_allclose(observed, (features-np.array(scaler["mean"]))/np.array(scaler["scale"]), rtol=1e-12, atol=1e-12)
    check(np.isfinite(observed).all(), "Nonfinite standardized features")
    for row in channels:
        check(row["split"] == by_file[row["file_id"]]["split"], "Channel split leakage")
    split_counts = {s:dict(Counter(r["label"] for r in files if r["split"] == s)) for s in ("train","validation","test")}
    check(split_counts == summary["source_class_by_split"], "Split summary mismatch")
    return {"status": "PASS", "checked_original_files": len(files), "checked_payload_hashes": len(manifest["output_hashes"]),
            "checked_channels": signal_count, "checked_native_samples": sample_count, "checked_windows": len(windows),
            "max_native_reconstruction_abs_error": max_native_error,
            "max_window_reconstruction_abs_error": max_window_error, "max_resampling_abs_error": max_resample_error,
            "file_group_leakage": False, "source_train_only_scaling": True,
            "target_labels_unknown": True, "source_class_by_split": split_counts,
            "pipeline_sha256": manifest["script_sha256"], "verifier_sha256": sha(Path(__file__)),
            "data_manifest_sha256": sha(output / "manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "预处理数据集")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.dataset)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
