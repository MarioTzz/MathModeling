"""问题二优化模型：多共振频带物理阶次特征与文件级极端随机树。

本模块只在文件级划分上拟合与评价。窗口用于形成同一文件的稳健中位数，
不作为独立训练/测试样本。模型状态以 JSON + NPZ 保存，不使用 pickle。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from utils.q1_model import (
    FEATURES, LABELS, Record, features_one_window, load_record, order_spectrum,
    prepare_record,
)
from utils.q2_model import classification_metrics, deterministic_noise_seed


META_COLUMNS = ["file_id", "label", "split", "window_index", "start_seconds",
                "end_seconds", "actual_revolutions", "rpm_used"]


def bearing_fault_orders(geometry: dict) -> dict[str, float]:
    """由滚动体数、滚动体直径、节径和接触角计算四类特征阶次。"""
    count = float(geometry["rolling_elements"])
    ball = float(geometry["ball_diameter_inch"])
    pitch = float(geometry["pitch_diameter_inch"])
    angle = math.radians(float(geometry["contact_angle_deg"]))
    ratio = ball * math.cos(angle) / pitch
    if not 0 < ratio < 1 or count <= 0:
        raise ValueError("轴承几何参数非法")
    title_bsf = pitch / ball * (1 - ratio ** 2)
    return {
        "ftf": 0.5 * (1 - ratio),
        "bpfo": count / 2 * (1 - ratio),
        "bpfi": count / 2 * (1 + ratio),
        "bsf": title_bsf,
        # 自旋半值单列候选；不与题面 BSF 混称。
        "spin_half": title_bsf / 2,
    }


def physics_feature_names(config: dict) -> list[str]:
    names = list(FEATURES)
    for low, high in config["resonance_bands_hz"]:
        prefix = f"b{low:g}_{high:g}__"
        for family in ["ftf", "bpfo", "bpfi", "bsf", "spin_half"]:
            names.extend([f"{prefix}{family}_hm{width:g}"
                          for width in config["order_half_widths"]])
            names.extend([f"{prefix}{family}_fund", f"{prefix}{family}_peak"])
        for family in ["bpfo", "bpfi", "bsf", "spin_half"]:
            names.append(f"{prefix}{family}_contrast")
        names.extend([f"{prefix}envelope_cv", f"{prefix}envelope_kurtosis",
                      f"{prefix}carrier_ratio"])
    if len(names) != len(set(names)):
        raise ValueError("物理特征名重复")
    return names


def feature_sets(config: dict) -> dict[str, list[str]]:
    all_names = physics_feature_names(config)
    chosen_low, chosen_high = config["single_band_hz"]
    chosen_prefix = f"b{chosen_low:g}_{chosen_high:g}__"
    title_names = [name for name in all_names if "spin_half" not in name]
    spin_names = [name for name in all_names if "bsf_" not in name]
    return {
        "q1_base": list(FEATURES),
        "single_band_physics": list(FEATURES) + [
            name for name in title_names if name.startswith(chosen_prefix)],
        "multiband_title_bsf": title_names,
        "multiband_spin_half": spin_names,
        "multiband_both": all_names,
    }


def _physics_features(raw: np.ndarray, envelope: np.ndarray, carrier: np.ndarray,
                      rotation_hz: float, q1_parameters: dict, config: dict) -> dict:
    orders, mass = order_spectrum(envelope, rotation_hz, q1_parameters)
    fault_orders = bearing_fault_orders(config["bearing_geometry"])
    values: dict[str, float] = {}
    for family, fundamental in fault_orders.items():
        harmonics = np.arange(1, int(q1_parameters["order_max"] / fundamental) + 1) \
            * fundamental
        for width in config["order_half_widths"]:
            mask = np.any(np.abs(orders[:, None] - harmonics[None, :])
                          <= float(width) + 1e-12, axis=1)
            values[f"{family}_hm{width:g}"] = float(mass[mask].sum())
        fundamental_mask = np.abs(orders - fundamental) <= config["fundamental_half_width"]
        values[f"{family}_fund"] = float(mass[fundamental_mask].sum())
        values[f"{family}_peak"] = float(mass[fundamental_mask].max())
    for family in ["bpfo", "bpfi", "bsf", "spin_half"]:
        fundamental = fault_orders[family]
        harmonics = np.arange(1, int(q1_parameters["order_max"] / fundamental) + 1) \
            * fundamental
        line = np.any(np.abs(orders[:, None] - harmonics[None, :])
                      <= config["fundamental_half_width"], axis=1)
        nearby = np.any(np.abs(orders[:, None] - harmonics[None, :])
                        <= config["contrast_half_width"], axis=1)
        background = nearby & ~line
        if not background.any():
            raise ValueError(f"{family} 对比度没有背景谱格点")
        values[f"{family}_contrast"] = float(
            mass[line].mean() / (mass[background].mean() + config["numeric_epsilon"]))
    centered_envelope = envelope - envelope.mean()
    envelope_rms = float(np.sqrt(np.mean(centered_envelope ** 2)))
    raw_centered = raw - raw.mean()
    raw_rms = float(np.sqrt(np.mean(raw_centered ** 2)))
    carrier_rms = float(np.sqrt(np.mean(carrier ** 2)))
    values["envelope_cv"] = envelope_rms / (
        float(envelope.mean()) + config["numeric_epsilon"])
    values["envelope_kurtosis"] = float(np.mean(
        (centered_envelope / (envelope_rms + config["numeric_epsilon"])) ** 4))
    values["carrier_ratio"] = carrier_rms / (raw_rms + config["numeric_epsilon"])
    if not np.isfinite(list(values.values())).all():
        raise ValueError("物理阶次特征出现非有限值")
    return values


def extract_multiband_windows(root: Path, metadata: pd.DataFrame,
                              q1_parameters: dict, config: dict, *,
                              file_ids=None, noise_relative_rms: float = 0.0,
                              amplitude_gain: float = 1.0) -> pd.DataFrame:
    """从原始 MAT 完整重算多频带20转窗口特征。"""
    if noise_relative_rms < 0 or amplitude_gain <= 0:
        raise ValueError("扰动参数非法")
    table = metadata[metadata.domain.eq("source")].copy()
    if file_ids is not None:
        table = table[table.file_id.isin(set(file_ids))]
    bands = [tuple(map(float, band)) for band in config["resonance_bands_hz"]]
    rows = []
    envelope_fs = float(q1_parameters["envelope_fs_hz"])
    guard = math.ceil(float(q1_parameters["edge_guard_seconds"]) * envelope_fs)
    revolutions = float(config["window_revolutions"])
    for metadata_row in table.sort_values("file_id").to_dict("records"):
        record = load_record(root, metadata_row)
        centered = record.centered.copy()
        if noise_relative_rms > 0:
            rng = np.random.default_rng(deterministic_noise_seed(
                int(config["seed"]), record.file_id, noise_relative_rms))
            direction = rng.normal(size=len(centered))
            direction /= np.sqrt(np.mean(direction ** 2))
            centered = centered + noise_relative_rms * np.sqrt(np.mean(centered ** 2)) \
                * direction
        centered *= amplitude_gain
        changed_record = Record(record.metadata, centered, record.fs, record.rotation_hz)
        prepared = {
            band: prepare_record(changed_record, band, q1_parameters, return_carrier=True)[:2]
            for band in bands
        }
        length = round(envelope_fs * revolutions / record.rotation_hz)
        first_envelope = next(iter(prepared.values()))[0]
        count = (len(first_envelope) - 2 * guard) // length
        if count < 1:
            raise ValueError(f"文件没有完整20转窗口: {record.file_id}")
        for window_index in range(count):
            start = guard + window_index * length
            end = start + length
            raw_start = round(start * record.fs / envelope_fs)
            raw_end = round(end * record.fs / envelope_fs)
            raw = changed_record.centered[raw_start:raw_end]
            chosen_band = tuple(map(float, config["single_band_hz"]))
            base = features_one_window(
                raw, prepared[chosen_band][0][start:end], record.rotation_hz, q1_parameters)
            row = {
                "file_id": record.file_id, "label": metadata_row["label"],
                "split": metadata_row["split"], "window_index": window_index,
                "start_seconds": start / envelope_fs, "end_seconds": end / envelope_fs,
                "actual_revolutions": record.rotation_hz * length / envelope_fs,
                "rpm_used": record.rotation_hz * 60,
            }
            row.update(dict(zip(FEATURES, base)))
            for band, (envelope, carrier) in prepared.items():
                prefix = f"b{band[0]:g}_{band[1]:g}__"
                physics = _physics_features(
                    raw, envelope[start:end], carrier[raw_start:raw_end],
                    record.rotation_hz, q1_parameters, config)
                row.update({prefix + name: value for name, value in physics.items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    expected = set(META_COLUMNS + physics_feature_names(config))
    if set(frame.columns) != expected or frame[["file_id", "window_index"]].duplicated().any():
        raise ValueError("多频带窗口表字段或主键不符合合同")
    if not np.isfinite(frame[physics_feature_names(config)].to_numpy()).all():
        raise ValueError("多频带窗口表含非有限值")
    return frame.sort_values(["file_id", "window_index"]).reset_index(drop=True)


def aggregate_file_medians(windows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """每个原文件一行；窗口特征以中位数汇总。"""
    grouped = windows.groupby("file_id", sort=True)
    meta = grouped[["label", "split"]].first()
    medians = grouped[features].median().add_suffix("__median")
    result = meta.join(medians).reset_index()
    if result.file_id.duplicated().any() or not np.isfinite(
            result.filter(like="__median").to_numpy()).all():
        raise ValueError("文件级中位数表非法")
    return result


def median_columns(features: list[str]) -> list[str]:
    return [f"{name}__median" for name in features]


@dataclass
class ExtraTreesFileModel:
    estimator: ExtraTreesClassifier
    features: list[str]
    labels: list[str]
    parameters: dict

    def scores(self, file_table: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(file_table[self.features])
        by_class = {label: raw[:, index]
                    for index, label in enumerate(self.estimator.classes_)}
        scores = np.column_stack([by_class[label] for label in self.labels])
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1):
            raise ValueError("极端随机树分数非法")
        return scores

    def predict(self, file_table: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.labels)[np.argmax(self.scores(file_table), axis=1)]


def fit_extra_trees(file_table: pd.DataFrame, features: list[str], parameters: dict,
                    config: dict) -> ExtraTreesFileModel:
    maximum = parameters["max_features"]
    if maximum != "sqrt":
        maximum = float(maximum)
    estimator = ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=maximum, class_weight="balanced",
        random_state=int(config["seed"]), n_jobs=1,
    )
    estimator.fit(file_table[features], file_table.label)
    if set(estimator.classes_) != set(LABELS):
        raise ValueError("模型拟合集合没有覆盖四类")
    stored = dict(parameters)
    stored["max_features"] = maximum
    return ExtraTreesFileModel(estimator, list(features), list(LABELS), stored)


def prediction_table(model: ExtraTreesFileModel, file_table: pd.DataFrame) -> pd.DataFrame:
    scores = model.scores(file_table)
    predicted = np.asarray(model.labels)[np.argmax(scores, axis=1)]
    ordered = np.sort(scores, axis=1)
    result = file_table[["file_id", "label", "split"]].rename(
        columns={"label": "true_label"}).copy()
    result["prediction"] = predicted
    result["margin"] = ordered[:, -1] - ordered[:, -2]
    for index, label in enumerate(model.labels):
        result[f"score_{label}"] = scores[:, index]
    return result


def tune_extra_trees(file_table: pd.DataFrame, features: list[str], config: dict,
                     n_splits: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """重复分层文件交叉验证；每个文件始终只有一行。"""
    rows = []
    for leaf in config["min_samples_leaf_grid"]:
        for maximum in config["max_features_grid"]:
            for seed in config["cv_seeds"]:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                            random_state=int(seed))
                predictions = np.empty(len(file_table), dtype="<U2")
                for fold, (fit_index, hold_index) in enumerate(
                        splitter.split(file_table[features], file_table.label)):
                    parameters = {"min_samples_leaf": int(leaf), "max_features": maximum}
                    model = fit_extra_trees(file_table.iloc[fit_index], features,
                                            parameters, config)
                    predictions[hold_index] = model.predict(file_table.iloc[hold_index])
                metrics, _, _ = classification_metrics(file_table.label, predictions)
                rows.append({"min_samples_leaf": int(leaf), "max_features": maximum,
                             "cv_seed": int(seed), "folds": n_splits, **metrics})
    detailed = pd.DataFrame(rows)
    summary = detailed.groupby(["min_samples_leaf", "max_features"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_min=("accuracy", "min"),
        macro_f1_mean=("macro_f1", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        normal_recall_mean=("normal_recall", "mean"),
    )
    order = {str(value): index for index, value in enumerate(config["max_features_grid"])}
    summary["max_features_order"] = summary.max_features.map(order)
    if summary.max_features_order.isna().any():
        raise ValueError("调参结果中出现配置网格外的候选特征数")
    best = summary.sort_values(
        ["accuracy_mean", "macro_f1_mean", "balanced_accuracy_mean", "accuracy_min",
         "min_samples_leaf", "max_features_order"],
        ascending=[False, False, False, False, True, True], kind="stable").iloc[0]
    maximum = best.max_features if best.max_features == "sqrt" else float(best.max_features)
    parameters = {"min_samples_leaf": int(best.min_samples_leaf),
                  "max_features": maximum}
    return parameters, detailed, summary


def full_cv_audit(file_table: pd.DataFrame, features: list[str], parameters: dict,
                  config: dict) -> pd.DataFrame:
    rows = []
    for seed in config["cv_seeds"]:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
        predictions = np.empty(len(file_table), dtype="<U2")
        for fit_index, hold_index in splitter.split(file_table[features], file_table.label):
            model = fit_extra_trees(file_table.iloc[fit_index], features, parameters, config)
            predictions[hold_index] = model.predict(file_table.iloc[hold_index])
        metrics, _, _ = classification_metrics(file_table.label, predictions)
        rows.append({"cv_seed": int(seed), "folds": 3, **metrics})
    return pd.DataFrame(rows)


def feature_importance_table(model: ExtraTreesFileModel) -> pd.DataFrame:
    table = pd.DataFrame({"feature": model.features,
                          "importance": model.estimator.feature_importances_})
    table["band"] = table.feature.str.extract(r"^(b[0-9_]+)__", expand=False).fillna(
        "q1_base")
    table["physical_family"] = np.select(
        [table.feature.str.contains("spin_half"), table.feature.str.contains("bpfo"),
         table.feature.str.contains("bpfi"),
         table.feature.str.contains("bsf"), table.feature.str.contains("ftf")],
        ["spin_half", "BPFO", "BPFI", "title_BSF", "FTF"], default="statistical")
    return table.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


def model_state(model: ExtraTreesFileModel) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = {
        "model_type": "ExtraTreesClassifier_file_median",
        "labels": model.labels, "features": model.features,
        "parameters": model.parameters | {
            "n_estimators": len(model.estimator.estimators_),
            "class_weight": "balanced", "random_state": model.estimator.random_state,
        },
        "estimator_classes": model.estimator.classes_.tolist(),
        "score_semantics": "mean_leaf_class_fraction_not_calibrated_probability",
    }
    arrays: dict[str, np.ndarray] = {}
    for index, tree in enumerate(model.estimator.estimators_):
        prefix = f"tree_{index:04d}__"
        arrays[prefix + "children_left"] = tree.tree_.children_left
        arrays[prefix + "children_right"] = tree.tree_.children_right
        arrays[prefix + "feature"] = tree.tree_.feature
        arrays[prefix + "threshold"] = tree.tree_.threshold
        arrays[prefix + "value"] = tree.tree_.value[:, 0, :]
    return metadata, arrays


def save_model(model: ExtraTreesFileModel, json_path: Path, arrays_path: Path) -> None:
    metadata, arrays = model_state(model)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(arrays_path, **arrays)


def reconstruct_scores(model: ExtraTreesFileModel, file_table: pd.DataFrame) -> np.ndarray:
    """不调用 sklearn.predict_proba，从树节点状态独立重建四类分数。"""
    # sklearn Tree.predict(_proba) 首先把输入转换为 float32；阈值附近必须同路由。
    x = file_table[model.features].to_numpy(dtype=np.float32)
    classes = list(model.estimator.classes_)
    accumulated = np.zeros((len(x), len(classes)), dtype=float)
    for estimator in model.estimator.estimators_:
        tree = estimator.tree_
        for row_index, row in enumerate(x):
            node = 0
            while tree.children_left[node] != tree.children_right[node]:
                feature = tree.feature[node]
                node = tree.children_left[node] if row[feature] <= tree.threshold[node] \
                    else tree.children_right[node]
            value = tree.value[node, 0, :].astype(float)
            accumulated[row_index] += value / value.sum()
    accumulated /= len(model.estimator.estimators_)
    by_class = {label: accumulated[:, index] for index, label in enumerate(classes)}
    return np.column_stack([by_class[label] for label in model.labels])


def scores_from_export(json_path: Path, arrays_path: Path,
                       file_table: pd.DataFrame) -> np.ndarray:
    """仅凭 JSON/NPZ 模型状态计算分数，便于跨进程复用与验收。"""
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    x = file_table[metadata["features"]].to_numpy(dtype=np.float32)
    labels = metadata["labels"]
    classes = metadata["estimator_classes"]
    accumulated = np.zeros((len(x), len(classes)), dtype=float)
    count = int(metadata["parameters"]["n_estimators"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for index in range(count):
            prefix = f"tree_{index:04d}__"
            left = arrays[prefix + "children_left"]
            right = arrays[prefix + "children_right"]
            feature = arrays[prefix + "feature"]
            threshold = arrays[prefix + "threshold"]
            values = arrays[prefix + "value"]
            for row_index, row in enumerate(x):
                node = 0
                while left[node] != right[node]:
                    node = left[node] if row[feature[node]] <= threshold[node] \
                        else right[node]
                value = values[node].astype(float)
                accumulated[row_index] += value / value.sum()
    accumulated /= count
    by_class = {label: accumulated[:, index] for index, label in enumerate(classes)}
    return np.column_stack([by_class[label] for label in labels])
