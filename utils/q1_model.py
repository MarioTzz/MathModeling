"""问题一数值模型；公式编号与《问题一_数学模型与求解说明》一致。

本模块不包含分类器、域适应或目标域标签推断。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

FEATURES = [
    "ac_rms", "mean_abs", "peak_to_peak", "crest_factor",
    "kurtosis_pearson", "skewness", "env_entropy", "env_centroid_order",
    "env_spread_order", "env_peak_order", "env_harmonic_ratio", "env_low_order_ratio",
]
FEATURE_ZH = [
    "交流均方根", "绝对均值", "峰峰值", "峰值因子", "Pearson 峭度", "偏度",
    "包络谱熵", "阶次质心", "阶次离散度", "主峰阶次", "谐波集中度", "低阶能量比",
]
FEATURE_BLOCKS = dict(zip(FEATURES, ["amplitude"] * 3 + ["impulse_shape"] * 3 + ["envelope"] * 6))
LABELS = ["N", "OR", "IR", "B"]
CORE_GROUPS = ["12kHz_DE_data", "48kHz_DE_data", "48kHz_Normal_data"]
META = ["file_id", "domain", "split", "label", "label_code", "window_index",
        "start_seconds", "end_seconds", "actual_revolutions", "rpm_used"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def numeric_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"不支持的 JSON 对象: {type(item).__name__}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=numeric_scalar),
                    encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


@dataclass
class Record:
    metadata: dict
    centered: np.ndarray
    fs: int
    rotation_hz: float

    @property
    def file_id(self):
        return self.metadata["file_id"]


def read_inventory(root: Path, smoke=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1.01) 继承文件划分，筛选故障轴承体系，不按预测好坏挑数据。"""
    inventory = pd.read_csv(root / "预处理数据集/files.csv", keep_default_na=False)
    if inventory.file_id.duplicated().any():
        raise ValueError("文件清单存在重复 file_id")
    selected = inventory.dataset_group.isin(CORE_GROUPS) | inventory.domain.eq("target")
    inventory["q1_selected"] = selected
    inventory["q1_reason"] = np.where(selected, "core_DE_or_target", "FE_bearing_extension")
    chosen = inventory[selected].copy()
    if smoke:
        training = chosen[chosen.split.eq("train")]
        ids = []
        for label in LABELS:
            ids.extend(training[training.label.eq(label)].sort_values("file_id").file_id.head(2))
        ids.append(chosen.loc[chosen.duration_seconds.idxmin(), "file_id"])
        ids.extend(chosen[chosen.domain.eq("target")].file_id.head(2))
        chosen = chosen[chosen.file_id.isin(ids)].copy()
    return inventory, chosen.sort_values("file_id").reset_index(drop=True)


def load_record(root: Path, metadata: dict) -> Record:
    """(1.02–1.03) 核对原始哈希、长度、数值、采样率及转速。"""
    path = root / "数据集" / metadata["relative_path"]
    if sha256(path) != metadata["input_sha256"]:
        raise ValueError(f"原始输入哈希发生变化: {path}")
    values = loadmat(path, variable_names=[metadata["primary_variable"]])
    raw = np.asarray(values[metadata["primary_variable"]], dtype=float).squeeze()
    if raw.ndim != 1 or raw.size != int(metadata["original_samples"]):
        raise ValueError(f"数组形状与清单不符: {path}")
    if not np.isfinite(raw).all() or np.ptp(raw) == 0:
        raise ValueError(f"非有限值或恒定信号: {path}")
    fs, rpm = int(metadata["nominal_fs_hz"]), float(metadata["rpm"])
    if fs <= 0 or not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(f"采样率或转速非法: {path}")
    return Record(metadata, raw - raw.mean(), fs, rpm / 60)


def filter_taps(fs: int, band, config) -> np.ndarray:
    """(1.04) 对称 FIR；中心频率单位增益。"""
    if not 0 < band[0] < band[1] < fs / 2:
        raise ValueError(f"频带越界: {band}, fs={fs}")
    half = math.ceil(config["fir_half_seconds"] * fs)
    return signal.firwin(2 * half + 1, band, pass_zero=False, fs=fs,
                         window=("kaiser", config["kaiser_beta"]), scale=True)


def prepare_record(record: Record, band, config, return_carrier=False):
    """(1.04–1.06) 原采样率解调后降低包络采样率。"""
    padding = math.ceil(config["reflect_seconds"] * record.fs)
    extended = np.pad(record.centered, (padding, padding), mode="reflect")
    filtered = signal.fftconvolve(extended, filter_taps(record.fs, band, config), mode="same")
    native_envelope = np.abs(signal.hilbert(filtered))[padding:-padding]
    envelope_fs = config["envelope_fs_hz"]
    divisor = math.gcd(record.fs, envelope_fs)
    envelope = signal.resample_poly(native_envelope, envelope_fs // divisor,
                                   record.fs // divisor, window=("kaiser", 5.0),
                                   padtype="line")
    if not np.isfinite(envelope).all():
        raise ValueError(f"解调结果存在非有限值: {record.file_id}")
    if return_carrier:
        return envelope, filtered[padding:-padding], native_envelope
    return envelope


def order_spectrum(envelope_window, rotation_hz: float, config):
    """(1.10–1.11) 对称 Hann、无补零 FFT、共同阶次网格及质量归一化。"""
    centered = envelope_window - np.mean(envelope_window)
    power = abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequency = np.fft.rfftfreq(len(centered), 1 / config["envelope_fs_hz"])
    orders = np.round(np.arange(config["order_min"],
                               config["order_max"] + config["order_step"] / 2,
                               config["order_step"]), 10)
    if rotation_hz * orders[-1] > frequency[-1]:
        raise ValueError("包络采样率不足以覆盖设定阶次")
    interpolated = np.interp(rotation_hz * orders, frequency, power)
    total = interpolated.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("包络阶次谱退化；不以补零或均匀分布代替")
    return orders, interpolated / total


def features_one_window(raw, envelope, rotation_hz, config) -> np.ndarray:
    """(1.12–1.23) 十二个量，顺序由 FEATURES 固定。"""
    centered = raw - raw.mean()
    rms = np.sqrt(np.mean(centered ** 2))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("原始窗口为恒定信号")
    relative = centered / rms
    orders, mass = order_spectrum(envelope, rotation_hz, config)
    positive = mass > 0
    entropy = -np.sum(mass[positive] * np.log(mass[positive])) / np.log(len(mass))
    centroid = np.dot(orders, mass)
    spread = np.sqrt(np.dot((orders - centroid) ** 2, mass))
    search = (orders >= config["peak_search_min"]) & (orders <= config["peak_search_max"])
    peak = orders[search][np.argmax(mass[search])]
    resolution = max(config["order_step"],
                     config["envelope_fs_hz"] / (rotation_hz * len(envelope)))
    harmonic_orders = np.arange(1, int(config["order_max"] / peak) + 1) * peak
    # 邻域取并集，边界保留浮点容差，防止重复计算同一个谱格点。
    harmonic_mask = np.any(abs(orders[:, None] - harmonic_orders) <= resolution + 1e-12,
                           axis=1)
    values = np.array([
        rms, np.mean(abs(centered)), np.ptp(centered), np.max(abs(relative)),
        np.mean(relative ** 4), np.mean(relative ** 3), entropy, centroid, spread,
        peak, mass[harmonic_mask].sum(), mass[orders <= config["low_order_cutoff"]].sum(),
    ])
    if not np.isfinite(values).all():
        raise ValueError("特征出现非有限值")
    if any(values[k] < -1e-12 or values[k] > 1 + 1e-12 for k in [6, 10, 11]):
        raise ValueError("归一化特征越过 [0,1]")
    return values


def window_features(record: Record, envelope, revolutions, config,
                    rpm_change=0.0, edge_guard=None):
    """(1.07–1.23) revolutions=None 表示一秒基线，时间索引映射回原信号。"""
    rotation = record.rotation_hz * (1 + rpm_change)
    if rotation <= 0:
        raise ValueError("扰动后转速非正")
    fs = config["envelope_fs_hz"]
    duration = config["baseline_seconds"] if revolutions is None else revolutions / rotation
    length = round(fs * duration)
    guard_seconds = config["edge_guard_seconds"] if edge_guard is None else edge_guard
    guard = math.ceil(guard_seconds * fs)
    count = (len(envelope) - 2 * guard) // length
    if count < 1:
        raise ValueError(f"无完整窗口: {record.file_id}, R={revolutions}, delta={rpm_change}")
    rows = []
    for window_index in range(count):
        start, end = guard + window_index * length, guard + (window_index + 1) * length
        raw_start, raw_end = round(start * record.fs / fs), round(end * record.fs / fs)
        if raw_end > len(record.centered):
            raise ValueError("窗口时间映射越过原始记录")
        values = features_one_window(record.centered[raw_start:raw_end],
                                     envelope[start:end], rotation, config)
        metadata = {name: record.metadata[name] for name in META[:5]}
        metadata.update(window_index=window_index, start_seconds=start / fs,
                        end_seconds=end / fs, actual_revolutions=rotation * length / fs,
                        rpm_used=60 * rotation)
        rows.append(metadata | dict(zip(FEATURES, values)))
    coverage = {
        "file_id": record.file_id, "windows": count, "window_seconds": length / fs,
        "actual_revolutions": rotation * length / fs, "edge_guard_seconds": guard_seconds,
        "tail_seconds": (len(envelope) - 2 * guard - count * length) / fs,
        "used_seconds": count * length / fs,
    }
    return pd.DataFrame(rows), coverage


def file_medians(frame):
    """(1.24) 每文件一行，以原文件作为独立统计单位。"""
    return frame.groupby("file_id", sort=True)[FEATURES].median()


def fisher_scores(features: pd.DataFrame, labels, config):
    """(1.25) 类别平衡的类间 / 类内比，常数特征记零。"""
    values = features.to_numpy(float)
    labels = np.asarray(labels)
    if set(labels) != set(LABELS):
        raise ValueError("Fisher 筛选的当前文件子集没有覆盖四类")
    std = values.std(axis=0)
    active = std > config["numeric_epsilon"]
    normalized = np.zeros_like(values)
    normalized[:, active] = (values[:, active] - values[:, active].mean(axis=0)) / std[active]
    class_means = np.array([normalized[labels == label].mean(axis=0) for label in LABELS])
    class_vars = np.array([normalized[labels == label].var(axis=0) for label in LABELS])
    scores = class_means.var(axis=0) / (class_vars.mean(axis=0) + config["numeric_epsilon"])
    scores[~active] = 0
    return pd.Series(scores, index=features.columns)


def select_nonredundant(features, ranking, config):
    """(1.26) 排名固定后贪心剔除高相关列；常数列没有筛选资格。"""
    correlation = features.corr(method="spearman").fillna(0)
    selected = []
    for feature in ranking:
        if features[feature].std(ddof=0) <= config["numeric_epsilon"]:
            continue
        if all(abs(correlation.loc[feature, prior]) <= config["correlation_limit"]
               for prior in selected):
            selected.append(feature)
        if len(selected) == config["max_features"]:
            break
    return selected


def make_folds(training_metadata, config):
    rng = np.random.default_rng(config["seed"])
    assignment = {}
    for label in LABELS:
        ids = sorted(training_metadata[training_metadata.label.eq(label)].file_id)
        if len(ids) < 2:
            raise ValueError(f"类别 {label} 不足以进行训练内二折")
        for position, file_id in enumerate(rng.permutation(ids)):
            assignment[file_id] = position % 2
    return pd.Series(assignment, name="inner_fold").sort_index()


def choose_variant(candidates: dict, metadata, config):
    """(1.27) 仅训练文件内二折选择，不调用任何分类器。"""
    training = metadata[metadata.split.eq("train")].set_index("file_id", drop=False)
    folds = make_folds(training, config)
    rows = []
    for name, frame in candidates.items():
        medians = file_medians(frame).loc[training.index]
        for fold in [0, 1]:
            fit_ids = folds[folds.ne(fold)].index
            hold_ids = folds[folds.eq(fold)].index
            fit = medians.loc[fit_ids]
            ranking = fisher_scores(fit, training.loc[fit_ids, "label"], config).sort_values(
                ascending=False, kind="stable").index
            selected = select_nonredundant(fit, ranking, config)
            if not selected:
                raise ValueError("内折特征选择为空")
            scores = fisher_scores(medians.loc[hold_ids], training.loc[hold_ids, "label"], config)
            rows.append({"variant": name, "fold": fold, "score": float(np.log1p(
                scores[selected]).mean()), "features": "|".join(selected),
                "fit_files": len(fit_ids), "hold_files": len(hold_ids)})
    table = pd.DataFrame(rows)
    mean_scores = table.groupby("variant", sort=False).score.mean()
    winner = mean_scores.idxmax()
    return winner, table, folds


def stable_selection(frame, metadata, config):
    """(1.28–1.29) 分层文件重采样，而非把相关窗口当成独立重复。"""
    training = metadata[metadata.split.eq("train")].set_index("file_id")
    medians = file_medians(frame).loc[training.index]
    rng = np.random.default_rng(config["seed"])
    counts = pd.Series(0, index=FEATURES)
    for _ in range(config["bootstrap_repeats"]):
        ids = []
        for label in LABELS:
            group = training[training.label.eq(label)].index.to_numpy()
            ids.extend(rng.choice(group, len(group), replace=True))
        sample = medians.loc[ids]
        scores = fisher_scores(sample, training.loc[ids, "label"], config)
        selected = select_nonredundant(sample, scores.sort_values(
            ascending=False, kind="stable").index, config)
        counts.loc[selected] += 1
    frequency = counts / config["bootstrap_repeats"]
    scores = fisher_scores(medians, training.label, config)
    table = pd.DataFrame({"feature": FEATURES, "block": [FEATURE_BLOCKS[x] for x in FEATURES],
                          "selection_frequency": frequency.values, "fisher_score": scores.values})
    table = table.sort_values(["selection_frequency", "fisher_score"],
                              ascending=[False, False], kind="stable")
    alternatives = {}
    for threshold in sorted(set([0.5, config["stability_threshold"], 0.7])):
        ranking = table[table.selection_frequency.ge(threshold)].feature
        alternatives[str(threshold)] = select_nonredundant(medians, ranking, config)
    selected = alternatives[str(config["stability_threshold"])]
    if not selected:
        raise ValueError("稳定筛选后没有特征，请回到特征逻辑，不自动降低阈值")
    table["selected"] = table.feature.isin(selected)
    return selected, table, alternatives


def fit_window_scaler(frame, selected):
    """(1.30–1.31) 源训练文件等权，每文件内部的窗口均分权重。"""
    training = frame[frame.split.eq("train") & frame.domain.eq("source")]
    counts = training.groupby("file_id").file_id.transform("size").to_numpy()
    file_count = training.file_id.nunique()
    if file_count < 1:
        raise ValueError("没有训练文件")
    weights = 1 / (file_count * counts)
    values = training[selected].to_numpy(float)
    mean = np.sum(values * weights[:, None], axis=0)
    std = np.sqrt(np.sum((values - mean) ** 2 * weights[:, None], axis=0))
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("选中特征训练尺度为零或非法")
    return {"features": list(selected), "mean": mean.tolist(), "std": std.tolist(),
            "fit_file_ids": sorted(training.file_id.unique().tolist()),
            "weighting": "equal_file_then_equal_window", "weight_sum": float(weights.sum())}


def apply_scaler(frame, scaler):
    selected = scaler["features"]
    scaled = frame[META].copy()
    scaled[selected] = (frame[selected].to_numpy() - scaler["mean"]) / scaler["std"]
    return scaled


def file_descriptors(scaled, selected, config):
    """(1.32–1.33) 同一组特征的文件中位数/IQR，源文件拟合运输坐标。"""
    grouped = scaled.groupby("file_id", sort=True)
    medians = grouped[selected].median().add_suffix("__median")
    spread = (grouped[selected].quantile(.75, interpolation="linear")
              - grouped[selected].quantile(.25, interpolation="linear")).add_suffix("__iqr")
    # 每个特征紧接其 median/IQR，方便后续按机理块构造成本。
    columns = [f"{feature}__{stat}" for feature in selected for stat in ["median", "iqr"]]
    descriptors = medians.join(spread)[columns]
    metadata = grouped[META[1:5]].first()
    training = metadata.domain.eq("source") & metadata.split.eq("train")
    center, std = descriptors[training].mean(), descriptors[training].std(ddof=0)
    active = std.gt(config["numeric_epsilon"])
    kept = list(std[active].index)
    if not kept:
        raise ValueError("全部文件描述量都是常数")
    normalized = (descriptors[kept] - center[kept]) / std[kept]
    scaler = {"features": kept, "dropped_constant": list(std[~active].index),
              "mean": center[kept].tolist(), "std": std[kept].tolist(),
              "fit_file_ids": list(metadata[training].index)}
    return (metadata.join(descriptors).reset_index(),
            metadata.join(normalized).reset_index(), scaler)


def window_admission(metadata, config):
    rows = []
    for row in metadata.to_dict("records"):
        envelope_count = math.ceil(row["original_samples"] *
                                   config["envelope_fs_hz"] / row["nominal_fs_hz"])
        for revolutions in [*config["revolutions"], 40]:
            for delta in config["rpm_relative_changes"]:
                for guard_seconds in config["edge_guard_checks_seconds"]:
                    length = round(config["envelope_fs_hz"] * revolutions /
                                   (row["rotation_hz"] * (1 + delta)))
                    guard = math.ceil(config["envelope_fs_hz"] * guard_seconds)
                    count = (envelope_count - 2 * guard) // length
                    rows.append({"file_id": row["file_id"], "revolutions": revolutions,
                                 "rpm_change": delta, "guard_seconds": guard_seconds,
                                 "windows": count, "admitted": count >= 1})
    table = pd.DataFrame(rows)
    if not table[table.revolutions.isin(config["revolutions"])].admitted.all():
        raise ValueError("当前候选转数违反完整窗口准入要求")
    return table


def run_sensitivity(records, envelopes, revolutions, selected, config, main_scaler,
                    baseline_scaler, main_frame, baseline_frame):
    """(1.34–1.35) 转速/裁边只变换数据，所有拟合参数冻结。"""
    routes = {"equal_revolutions": (revolutions, main_scaler, main_frame),
              "one_second": (None, baseline_scaler, baseline_frame)}
    rows, edge_rows, gain_rows = [], [], []
    for route, (turns, own_scaler, original) in routes.items():
        base = file_medians(original)[selected]
        for record in records:
            envelope = envelopes[record.file_id]
            if record.metadata["domain"] == "target":
                for delta in config["rpm_relative_changes"]:
                    changed, coverage = window_features(record, envelope, turns, config,
                                                        rpm_change=delta)
                    difference = changed[selected].median().to_numpy() - base.loc[record.file_id].to_numpy()
                    rows.append({"route": route, "file_id": record.file_id, "rpm_change": delta,
                                 "windows": coverage["windows"],
                                 "own_scale_drift": float(np.sqrt(np.mean((difference / own_scaler["std"]) ** 2))),
                                 "common_scale_drift": float(np.sqrt(np.mean((difference / main_scaler["std"]) ** 2)))})
            for guard in config["edge_guard_checks_seconds"]:
                changed, coverage = window_features(record, envelope, turns, config, edge_guard=guard)
                difference = changed[selected].median().to_numpy() - base.loc[record.file_id].to_numpy()
                edge_rows.append({"route": route, "file_id": record.file_id,
                                  "domain": record.metadata["domain"], "guard_seconds": guard,
                                  "windows": coverage["windows"],
                                  "common_scale_drift": float(np.sqrt(np.mean((difference / main_scaler["std"]) ** 2)))})
    # 确定性代表文件：每类首个训练文件和首个目标文件。
    representatives = []
    for label in LABELS + [""]:
        group = [r for r in records if r.metadata["label"] == label
                 and (r.metadata["split"] == "train" or r.metadata["domain"] == "target")]
        if group:
            representatives.append(group[0])
    for record in representatives:
        envelope = envelopes[record.file_id]
        reference, _ = window_features(record, envelope, revolutions, config)
        for gain in config["gain_checks"]:
            adjusted = Record(record.metadata, record.centered * gain, record.fs, record.rotation_hz)
            changed, _ = window_features(adjusted, envelope * gain, revolutions, config)
            expected = reference[FEATURES].to_numpy().copy()
            expected[:, :3] *= gain
            error = np.max(abs(changed[FEATURES].to_numpy() - expected) / np.maximum(1, abs(expected)))
            if error > 1e-8:
                raise ValueError(f"增益齐次性失败: {record.file_id}, {error}")
            gain_rows.append({"file_id": record.file_id, "gain": gain,
                              "max_scaled_absolute_error": float(error), "tolerance": 1e-8})
    return pd.DataFrame(rows), pd.DataFrame(edge_rows), pd.DataFrame(gain_rows)


def filter_diagnostics(config):
    rows = []
    for fs in [12000, 32000, 48000]:
        for band in config["bands_hz"]:
            taps = filter_taps(fs, band, config)
            f, response = signal.freqz(taps, worN=32768, fs=fs)
            center = abs(signal.freqz(taps, worN=[np.pi * sum(band) / fs])[1][0])
            # 报告距通带边界至少300Hz的阻带，不把过渡带误称阻带。
            stop = (f <= max(0, band[0] - 300)) | (f >= band[1] + 300)
            stop_db = float(20 * np.log10(np.maximum(abs(response[stop]).max(), 1e-15)))
            rows.append({"fs_hz": fs, "low_hz": band[0], "high_hz": band[1],
                         "taps": len(taps), "center_gain": float(center),
                         "stopband_max_db_300hz_margin": stop_db})
            if abs(center - 1) > 1e-10 or stop_db > -40:
                raise ValueError("FIR 频响未达到本实现的基本数值核验要求")
    return pd.DataFrame(rows)
