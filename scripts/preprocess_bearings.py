"""Reproducible, non-destructive bearing preprocessing; no classifier is trained.

Run from the project root: python scripts/preprocess_bearings.py
Dependencies: requirements-preprocessing.txt. Output must be a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np
import scipy
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"N": 0, "OR": 1, "IR": 2, "B": 3}
FEATURES = ["ac_rms", "mean_abs", "peak_to_peak", "crest_factor", "kurtosis_pearson", "skewness"]
PARAMETERS = {
    "seed": 42, "common_fs_hz": 12000, "window_seconds": 1,
    "window_overlap": 0, "source_channel": "DE", "float_dtype": "float64",
    "resample_filter": ["kaiser", 5.0], "resample_padtype": "line",
    "split": "class-stratified by original file; floor(60%) train, floor(20%) validation, remainder test; min 1 validation/test",
    "feature_scaling": "z-score; source training windows only; population std ddof=0; zero std -> 1",
    "quality_thresholds": {"abs_mean_over_raw_rms": .1, "block_ac_rms_ratio": 2,
                           "kurtosis_pearson": 20, "extreme_plateau_min_run": 2},
    "quality_action": "flag_only; no deletion, clipping or interpolation",
}
# Previously audited same-file DE/FE relationships. Revalidated on actual values below.
KNOWN_OVERLAPS = {
    "48kHz_DE_data/B/0014/B014_0.mat": (0, 1068, 190932),
    "48kHz_DE_data/B/0021/B021_0.mat": (0, 33686, 158314),
    "48kHz_DE_data/IR/0021/IR021_0.mat": (14598, 0, 177402),
    "48kHz_DE_data/OR/Centered/0014/OR014@6_0.mat": (42870, 0, 149130),
    "48kHz_DE_data/OR/Centered/0021/OR021@6_0.mat": (0, 13318, 178682),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_mat(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, fields, do_compression=True, oned_as="column")
    # MAT v5 has an informational creation-time header; make it deterministic.
    with path.open("r+b") as f:
        f.write(b"MATLAB 5.0 MAT-file, Bearing preprocessing v1; deterministic float64 output".ljust(116, b" "))


def parse_metadata(relative_path, mat):
    p = Path(relative_path)
    source = p.parts[0] == "源域数据集"
    group = p.parts[1] if source else "target"
    label = ("N" if group == "48kHz_Normal_data" else p.parts[2]) if source else None
    if source and label not in LABELS:
        raise ValueError(f"Unrecognized class: {p}")
    rpm_fields = {k: float(np.asarray(v).squeeze()) for k, v in mat.items()
                  if not k.startswith("__") and "RPM" in k.upper() and np.asarray(v).size == 1}
    name_rpm = re.search(r"\((\d+)rpm\)", p.name)
    name_rpm = float(name_rpm.group(1)) if name_rpm else None
    if len(set(rpm_fields.values())) > 1 or (name_rpm is not None and any(v != name_rpm for v in rpm_fields.values())):
        raise ValueError(f"Conflicting RPM evidence: {p}")
    rpm = (next(iter(rpm_fields.values())) if rpm_fields else name_rpm) if source else 600.
    if rpm is None or rpm <= 0:
        raise ValueError(f"No valid RPM evidence: {p}")
    load = re.search(r"_(\d)(?:_|$)", p.stem) if source else None
    size = re.match(r"(?:OR|IR|B)(\d{3})", p.stem) if source else None
    size_inch = int(size.group(1)) / 1000 if size else None
    position = re.search(r"@(\d+)", p.stem) if source else None
    domain = "source" if source else "target"
    return {
        "file_id": domain + "/" + "/".join(p.parts[1:]), "relative_path": p.as_posix(),
        "domain": domain, "dataset_group": group, "label": label,
        "label_code": LABELS[label] if source else -1, "label_known": source,
        "nominal_fs_hz": (12000 if group.startswith("12kHz") else 48000) if source else 32000,
        "fs_source": "directory_nominal_not_channel_verified" if source else "problem_statement",
        "rpm": rpm, "rpm_source": ("field" if rpm_fields else "filename") if source else "problem_approximate",
        "rotation_hz": rpm / 60, "rpm_is_approximate": not source,
        "load_hp": int(load.group(1)) if load else None,
        "fault_bearing_location": ("FE" if "FE_data" in group else "DE") if source and label != "N" else None,
        "fault_size_inch": size_inch, "fault_size_mm": size_inch * 25.4 if size else None,
        "outer_race_clock_position": int(position.group(1)) if position else None,
        "amplitude_unit": "unknown_original_scale",
    }


def get_signals(mat):
    signals = {}
    for key, value in mat.items():
        if key.startswith("__") or "RPM" in key.upper():
            continue
        a = np.asarray(value)
        if a.ndim != 2 or a.shape[1] != 1 or a.size < 2 or not np.issubdtype(a.dtype, np.number):
            raise ValueError(f"Unexpected signal {key}: {a.shape}, {a.dtype}")
        x = a[:, 0].astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite signal {key}; no implicit imputation is allowed")
        signals[key] = x
    if not signals:
        raise ValueError("No signals found")
    return signals


def channel_name(name):
    match = re.search(r"_(DE|FE|BA)_time$", name)
    return match.group(1) if match else "target_single"


def split_files(rows, seed):
    rng = np.random.default_rng(seed)
    result = {r["file_id"]: "target" for r in rows if r["domain"] == "target"}
    for label in range(4):
        ids = sorted(r["file_id"] for r in rows if r["domain"] == "source" and r["label_code"] == label)
        if len(ids) < 3:
            raise ValueError(f"Need at least three independent files in class {label}")
        rng.shuffle(ids)
        n = len(ids)
        nv = max(1, int(.2 * n))
        nt = min(n - nv - 1, max(1, int(.6 * n)))
        for i, file_id in enumerate(ids):
            result[file_id] = "train" if i < nt else "validation" if i < nt + nv else "test"
    return result


def resample_signal(x, source_fs):
    if source_fs not in (12000, 32000, 48000):
        raise ValueError(f"Unsupported sampling rate: {source_fs}")
    gcd = math.gcd(source_fs, 12000)
    up, down = 12000 // gcd, source_fs // gcd
    return x.copy() if up == down else resample_poly(x, up, down, window=("kaiser", 5.0), padtype="line")


def make_windows(x):
    n = len(x) // 12000
    w = x[:n * 12000].reshape(n, 12000)
    means = w.mean(axis=1)
    ac = w - means[:, None]
    rms = np.sqrt(np.mean(ac * ac, axis=1))
    valid = rms > 0
    unit = ac / np.where(valid, rms, 1)[:, None]
    return {"unit_rms": unit, "means": means, "rms": rms, "valid": valid,
            "tail_samples": len(x) - n * 12000, "ac": ac}


def fit_scaler(x, train_mask):
    train = x[np.asarray(train_mask, dtype=bool)]
    if not len(train) or not np.isfinite(train).all():
        raise ValueError("Scaler needs finite source training rows")
    std = train.std(axis=0, ddof=0)
    return {"mean": train.mean(axis=0).tolist(), "std": std.tolist(),
            "scale": np.where(std == 0, 1, std).tolist(), "constant_columns": np.flatnonzero(std == 0).tolist(),
            "n_fit_windows": len(train)}


def apply_scaler(x, params):
    return (x - np.asarray(params["mean"])) / np.asarray(params["scale"])


def longest_true(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(max(ends - starts)) if len(starts) else 0


def quality(x):
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    ac_rms = float(np.sqrt(np.mean(ac * ac)))
    if ac_rms == 0:
        raise ValueError("Constant signal encountered; requires explicit quality policy")
    b = [float(np.std(a)) for a in np.array_split(x, 8)]
    ratio = max(b) / min(b) if min(b) else None
    kurtosis = float(np.mean((ac / ac_rms)**4))
    extremum = (x == x.min()) | (x == x.max())
    adjacent = np.r_[False, np.diff(x) == 0] | np.r_[np.diff(x) == 0, False]
    plateau = extremum & adjacent
    return {"n_samples": len(x), "raw_mean": float(x.mean()), "raw_rms": rms, "ac_rms": ac_rms,
            "min": float(x.min()), "max": float(x.max()), "kurtosis_pearson": kurtosis,
            "outside_3sigma": int(np.count_nonzero(np.abs(ac) > 3 * ac_rms)),
            "extrema_count": int(extremum.sum()), "extreme_plateau_points": int(plateau.sum()),
            "max_extreme_run": max(longest_true(x == x.min()), longest_true(x == x.max())),
            "block_ac_rms_ratio": ratio, "dc_flag": bool(abs(x.mean()) / rms > .1),
            "energy_variation_flag": bool(ratio is None or ratio > 2),
            "high_kurtosis_flag": kurtosis > 20, "suspected_clipping_flag": bool(plateau.any())}, plateau


def validate_overlap(relative_path, signals):
    key = "/".join(Path(relative_path).parts[1:])
    if key not in KNOWN_OVERLAPS:
        return None
    de = next(v for k, v in signals.items() if channel_name(k) == "DE")
    fe = next(v for k, v in signals.items() if channel_name(k) == "FE")
    ds, fs, n = KNOWN_OVERLAPS[key]
    a, b = de[ds:ds+n], fe[fs:fs+n]
    if len(a) != n or len(b) != n:
        raise ValueError(f"Audited overlap no longer available: {relative_path}")
    az, bz = a-a.mean(), b-b.mean()
    slope = float(np.dot(az, bz) / np.dot(az, az))
    residual = float(np.std(bz - slope * az) / np.std(bz))
    if residual > 1e-12:
        raise ValueError(f"Audited overlap changed: {relative_path}")
    return {"relative_path": relative_path, "de_start": ds, "fe_start": fs, "overlap_samples": n,
            "fe_over_de_slope": slope, "affine_residual_std_ratio": residual}


def run(output, smoke=False):
    output = Path(output).resolve()
    source_dir = ROOT / "数据集"
    if output == source_dir or source_dir in output.parents or output == ROOT or ROOT not in output.parents:
        raise ValueError("Output must be a new project subdirectory outside original data")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}; choose a new output directory")
    paths = sorted(source_dir.rglob("*.mat"))
    rows = [dict(parse_metadata(p.relative_to(source_dir), loadmat(p)), input_sha256=digest(p)) for p in paths]
    if smoke:
        keep = set()
        for c in range(4):
            keep.update(r["file_id"] for r in rows if r["label_code"] == c and
                        (r["dataset_group"] == "48kHz_Normal_data" or "_0.mat" in r["relative_path"]))
        selected = []
        for c in range(4):
            selected.extend([r for r in rows if r["label_code"] == c and r["file_id"] in keep][:3])
        # Include 48 kHz overlap, missing RPM/channel, and target B/N label traps.
        extras = {"源域数据集/48kHz_DE_data/B/0014/B014_0.mat",
                  "源域数据集/12kHz_DE_data/B/0028/B028_0_(1797rpm).mat", "目标域数据集/B.mat", "目标域数据集/N.mat"}
        rows = sorted({r["file_id"]: r for r in selected + [r for r in rows if r["relative_path"] in extras]}.values(), key=lambda r:r["file_id"])
    if not rows:
        raise ValueError("No MAT inputs")
    splits = split_files(rows, PARAMETERS["seed"])
    output.mkdir(parents=True)
    channels, windows, overlaps = [], [], []
    max_reconstruction_error = 0.
    for row in rows:
        path = source_dir / row["relative_path"]
        mat = loadmat(path)
        signals = get_signals(mat)
        row["split"] = splits[row["file_id"]]
        row["group_id"] = row["file_id"]
        native = {k:v for k, v in mat.items() if not k.startswith("__") and "RPM" in k.upper()}
        native["rpm_metadata"] = row["rpm"]
        row["native_path"] = "native_centered/" + row["relative_path"]
        row["common_path"] = "common_12k/" + row["relative_path"]
        overlap = validate_overlap(row["relative_path"], signals)
        if overlap:
            overlaps.append(overlap)
        primary_name = next(k for k in signals if channel_name(k) == ("DE" if row["domain"] == "source" else "target_single"))
        primary_plateau = None
        for name, x in signals.items():
            stat, plateau = quality(x)
            ch = channel_name(name)
            centered = x - stat["raw_mean"]
            native[name] = centered[:, None]
            native[name + "_mean"] = stat["raw_mean"]
            native[name + "_raw_rms"] = stat["raw_rms"]
            error = float(np.max(np.abs(centered + stat["raw_mean"] - x)))
            max_reconstruction_error = max(max_reconstruction_error, error)
            channels.append({"file_id": row["file_id"], "variable": name, "channel": ch,
                             "split": row["split"], "nominal_fs_hz": row["nominal_fs_hz"],
                             "fs_status": row["fs_source"], "affine_overlap_flag": overlap is not None and ch in ("DE", "FE"),
                             "primary_common_input": name == primary_name, **stat})
            if name == primary_name:
                primary_plateau = plateau
        for ch in ("DE", "FE", "BA"):
            row["has_" + ch] = any(channel_name(k) == ch for k in signals)
        row["primary_variable"] = primary_name
        save_mat(output / row["native_path"], native)
        x = signals[primary_name]
        fs = row["nominal_fs_hz"]
        y = resample_signal(x - x.mean(), fs)
        w = make_windows(y)
        row.update(original_samples=len(x), duration_seconds=len(x)/fs,
                   resampled_samples=len(y), resampled_duration_seconds=len(y)/12000,
                   n_windows=len(w["rms"]), tail_samples=w["tail_samples"],
                   tail_seconds=w["tail_samples"]/12000, resample_up=12000//math.gcd(fs,12000),
                   resample_down=fs//math.gcd(fs,12000))
        save_mat(output / row["common_path"], {
            "signal_resampled": y[:, None], "windows_unit_rms": w["unit_rms"],
            "window_means": w["means"], "window_ac_rms": w["rms"], "window_valid": w["valid"].astype(np.uint8),
            "window_start": np.arange(len(w["rms"])) * 12000,
            "original_mean": x.mean(), "original_raw_rms": np.sqrt(np.mean(x*x)),
            "fs_hz": 12000, "original_nominal_fs_hz": fs, "label_code": row["label_code"],
            "label_known": int(row["label_known"]), "tail_samples": w["tail_samples"],
        })
        for i, ac in enumerate(w["ac"]):
            rms = float(w["rms"][i])
            if not w["valid"][i]:
                raise ValueError(f"Zero energy window needs separate policy: {path}, {i}")
            unit = w["unit_rms"][i]
            start, end = i*12000, (i+1)*12000
            raw_start, raw_end = i*fs, min((i+1)*fs, len(x))
            windows.append({"window_id": row["file_id"] + f"::w{i:04d}", "file_id": row["file_id"],
                            "group_id": row["group_id"], "domain": row["domain"], "split": row["split"],
                            "label_code": row["label_code"], "label_known": row["label_known"],
                            "window_index": i, "start_sample": start, "end_sample_exclusive": end,
                            "start_seconds": i, "end_seconds": i+1,
                            "window_mean_before_center": float(w["means"][i]),
                            "suspected_clipping_points_original": int(primary_plateau[raw_start:raw_end].sum()),
                            "filter_boundary_flag": fs != 12000 and (start < 10 or end > len(y)-10),
                            "file_equal_weight": 1/len(w["rms"]),
                            "ac_rms": rms, "mean_abs": float(np.mean(np.abs(ac))),
                            "peak_to_peak": float(np.ptp(ac)), "crest_factor": float(np.max(np.abs(unit))),
                            "kurtosis_pearson": float(np.mean(unit**4)), "skewness": float(np.mean(unit**3))})
    features = np.array([[r[k] for k in FEATURES] for r in windows])
    train_mask = np.array([r["domain"] == "source" and r["split"] == "train" for r in windows])
    scaler = fit_scaler(features, train_mask)
    scaler.update(features=FEATURES, ddof=0, fit_domain="source", fit_split="train",
                  fit_file_ids=sorted({r["file_id"] for r in windows if r["split"] == "train"}),
                  weighting="each training window equally weighted; file_equal_weight is supplied separately")
    standardized = apply_scaler(features, scaler)
    scaled_rows = [{"window_id": r["window_id"], **{f: float(v) for f,v in zip(FEATURES, z)}} for r,z in zip(windows, standardized)]
    write_csv(output / "files.csv", rows)
    write_csv(output / "channels.csv", channels)
    write_csv(output / "windows.csv", windows)
    write_csv(output / "features_standardized.csv", scaled_rows)
    write_json(output / "scaler.json", scaler)
    write_json(output / "channel_overlap_evidence.json", overlaps)
    write_json(output / "parameters.json", {**PARAMETERS, "label_mapping": LABELS, "target_unknown_code": -1,
                                           "smoke": smoke, "features": FEATURES})
    unchanged = all(digest(source_dir / r["relative_path"]) == r["input_sha256"] for r in rows)
    if not unchanged:
        raise RuntimeError("Original input changed during processing")
    summary = {"files": len(rows), "source_files": sum(r["domain"] == "source" for r in rows),
               "target_files": sum(r["domain"] == "target" for r in rows), "channels": len(channels),
               "native_samples": sum(r["n_samples"] for r in channels), "windows": len(windows),
               "windows_by_split": dict(Counter(r["split"] for r in windows)),
               "files_by_split": dict(Counter(r["split"] for r in rows)),
               "source_class_by_split": {s: dict(Counter(r["label"] for r in rows if r["split"]==s)) for s in ("train", "validation", "test")},
               "rpm_from_filename": sum(r["rpm_source"] == "filename" for r in rows),
               "validated_overlap_pairs": len(overlaps), "tail_samples_common": sum(r["tail_samples"] for r in rows),
               "max_native_reconstruction_abs_error": max_reconstruction_error,
               "original_hashes_unchanged": unchanged, "interpolated_points": 0, "deleted_native_points": 0,
               "standardized_all_finite": bool(np.isfinite(standardized).all()),
               "channels_by_quality_flag": {k:sum(r[k] for r in channels) for k in ("dc_flag", "energy_variation_flag", "high_kurtosis_flag", "suspected_clipping_flag")},
               "windows_touching_suspected_clipping": sum(r["suspected_clipping_points_original"] > 0 for r in windows),
               "windows_touching_filter_boundary": sum(r["filter_boundary_flag"] for r in windows)}
    write_json(output / "processing_summary.json", summary)
    generated = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {"version": 1, "input_hashes": {r["relative_path"]:r["input_sha256"] for r in rows},
                "output_hashes": {p.relative_to(output).as_posix():digest(p) for p in generated},
                "script_sha256": digest(Path(__file__)), "requirements_sha256": digest(ROOT / "requirements-preprocessing.txt"),
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "parameters": PARAMETERS, "reproduce_command": "python scripts/preprocess_bearings.py --output 预处理数据集_复现" + (" --smoke" if smoke else ""),
                "manifest_scope": "All generated payloads; manifest itself and hand-written README are excluded"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "预处理数据集")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke)
