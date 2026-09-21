"""问题二核心数值模型；公式编号对应《问题二_数学模型与求解说明》。

所有选择、尺度和参数函数都以显式文件集合为输入，避免把验证、测试或目标文件
误用到拟合步骤。多分类采用四个显式 OvR SVC，不使用 sklearn 的内部 OvO 多分类。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

from utils.q1_model import (
    FEATURES, FEATURE_BLOCKS, LABELS, file_medians, fisher_scores,
    select_nonredundant,
)

SCORE_COLUMNS = [f"score_{label}" for label in LABELS]
AMPLITUDE_FEATURES = ["ac_rms", "mean_abs", "peak_to_peak"]


def stable_select_for_files(frame: pd.DataFrame, file_ids, config: dict):
    """式(2.01)—(2.04)：只在给定文件内做分层文件重采样与稳定筛选。"""
    file_ids = sorted(set(file_ids))
    subset = frame[frame.file_id.isin(file_ids)]
    metadata = subset.groupby("file_id", sort=True)["label"].first().to_frame()
    if set(metadata.index) != set(file_ids):
        raise ValueError("特征表没有覆盖全部拟合文件")
    if set(metadata.label) != set(LABELS):
        raise ValueError("拟合文件没有覆盖四个类别")
    medians = file_medians(subset).loc[metadata.index]
    rng = np.random.default_rng(config["seed"])
    counts = pd.Series(0, index=FEATURES, dtype=int)
    for _ in range(config["bootstrap_repeats"]):
        sampled_ids = []
        for label in LABELS:
            group = metadata.index[metadata.label.eq(label)].to_numpy()
            sampled_ids.extend(rng.choice(group, len(group), replace=True))
        sampled = medians.loc[sampled_ids]
        sampled_labels = metadata.loc[sampled_ids, "label"]
        scores = fisher_scores(sampled, sampled_labels, config)
        chosen = select_nonredundant(
            sampled, scores.sort_values(ascending=False, kind="stable").index, config)
        counts.loc[chosen] += 1
    frequency = counts / config["bootstrap_repeats"]
    full_scores = fisher_scores(medians, metadata.label, config)
    table = pd.DataFrame({
        "feature": FEATURES,
        "block": [FEATURE_BLOCKS[name] for name in FEATURES],
        "selection_frequency": frequency.values,
        "fisher_score": full_scores.values,
    }).sort_values(["selection_frequency", "fisher_score"],
                   ascending=[False, False], kind="stable")
    eligible = table.loc[
        table.selection_frequency.ge(config["stability_threshold"]), "feature"]
    selected = select_nonredundant(medians, eligible, config)
    if not selected:
        raise ValueError("稳定筛选为空；不自动降低阈值")
    table["selected"] = table.feature.isin(selected)
    return selected, table.reset_index(drop=True)


def fit_file_equal_scaler(frame: pd.DataFrame, file_ids, selected, epsilon=1e-12):
    """式(2.05)—(2.06)：每文件总权重相同的尺度拟合。"""
    ids = sorted(set(file_ids))
    fit = frame[frame.file_id.isin(ids)].copy()
    if fit.file_id.nunique() != len(ids):
        raise ValueError("尺度拟合文件不完整")
    counts = fit.groupby("file_id").file_id.transform("size").to_numpy(float)
    weights = 1.0 / (len(ids) * counts)
    values = fit[list(selected)].to_numpy(float)
    mean = np.sum(weights[:, None] * values, axis=0)
    std = np.sqrt(np.sum(weights[:, None] * (values - mean) ** 2, axis=0))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("尺度权重和不为1")
    if np.any(std <= epsilon) or not np.isfinite(std).all():
        raise ValueError("尺度拟合出现常数或非法特征")
    return {
        "features": list(selected), "mean": mean.tolist(), "std": std.tolist(),
        "fit_file_ids": ids, "weighting": "equal_file_then_equal_window",
        "weight_sum": float(weights.sum()),
    }


def transform_features(frame: pd.DataFrame, scaler: dict):
    """式(2.07)：应用冻结尺度，保留行索引以便文件汇总。"""
    features = scaler["features"]
    values = (frame[features].to_numpy(float) - np.asarray(scaler["mean"])) \
        / np.asarray(scaler["std"])
    if not np.isfinite(values).all():
        raise ValueError("标准化后出现非有限值")
    return values


def sample_weights(frame: pd.DataFrame, file_ids, scheme: str):
    """式(2.08)—(2.11)：三种总和均为1的损失权重。"""
    fit = frame[frame.file_id.isin(set(file_ids))]
    if fit.empty:
        raise ValueError("没有训练窗口")
    if scheme == "window_uniform":
        weights = np.full(len(fit), 1.0 / len(fit))
    elif scheme == "class_balanced":
        class_windows = fit.groupby("label").file_id.transform("size").to_numpy(float)
        weights = 1.0 / (len(LABELS) * class_windows)
    elif scheme == "file_class_balanced":
        class_files = fit.groupby("label").file_id.transform("nunique").to_numpy(float)
        file_windows = fit.groupby("file_id").file_id.transform("size").to_numpy(float)
        weights = 1.0 / (len(LABELS) * class_files * file_windows)
    else:
        raise ValueError(f"未知权重方案: {scheme}")
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError(f"{scheme} 权重和不为1")
    return weights


def audit_weights(frame: pd.DataFrame, file_ids, scheme: str):
    fit = frame[frame.file_id.isin(set(file_ids))].copy()
    fit["sample_weight"] = sample_weights(frame, file_ids, scheme)
    by_file = fit.groupby(["label", "file_id"], sort=True).sample_weight.sum().reset_index()
    by_class = fit.groupby("label", sort=False).sample_weight.sum().reindex(LABELS)
    return by_file, by_class


def make_group_folds(metadata: pd.DataFrame, file_ids, seed: int):
    """式(2.21)：每类排序、固定种子置换、按奇偶交替分折。"""
    table = metadata[metadata.file_id.isin(set(file_ids))].drop_duplicates("file_id")
    table = table.set_index("file_id")
    rng = np.random.default_rng(seed)
    assignment = {}
    for label in LABELS:
        ids = np.asarray(sorted(table.index[table.label.eq(label)]))
        if len(ids) < 2:
            raise ValueError(f"类别 {label} 独立文件少于2，无法二折")
        for position, file_id in enumerate(rng.permutation(ids)):
            assignment[file_id] = position % 2
    return pd.Series(assignment, name="inner_fold").sort_index()


@dataclass
class OVRModel:
    labels: list
    kernel: str
    C: float
    gamma: float
    weighting: str
    scaler: dict
    estimators: dict
    selected_features: list
    fit_seconds: float

    def decision_function(self, frame: pd.DataFrame):
        x = transform_features(frame, self.scaler)
        scores = np.column_stack([
            self.estimators[label].decision_function(x) for label in self.labels
        ])
        if scores.shape != (len(frame), len(self.labels)) or not np.isfinite(scores).all():
            raise ValueError("决策分数形状或数值非法")
        return scores


def fit_ovr(frame: pd.DataFrame, file_ids, selected, scaler, *, kernel: str,
            C: float, gamma: float, weighting: str, config: dict):
    """式(2.12)—(2.16)：训练四个显式 OvR SVC。"""
    fit = frame[frame.file_id.isin(set(file_ids))].copy()
    x = transform_features(fit, scaler)
    weights = sample_weights(frame, file_ids, weighting)
    started = time.perf_counter()
    estimators = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for label in LABELS:
            target = np.where(fit.label.to_numpy() == label, 1, -1)
            kwargs = dict(
                C=float(C), kernel=kernel, class_weight=None,
                tol=float(config["solver_tolerance"]), max_iter=int(config["max_iter"]),
                shrinking=True, cache_size=512,
            )
            if kernel == "rbf":
                kwargs["gamma"] = float(gamma)
            estimator = SVC(**kwargs)
            estimator.fit(x, target, sample_weight=weights)
            if estimator.fit_status_ != 0:
                raise RuntimeError(f"{label} SVC 未收敛")
            estimators[label] = estimator
    return OVRModel(
        labels=list(LABELS), kernel=kernel, C=float(C), gamma=float(gamma),
        weighting=weighting, scaler=scaler, estimators=estimators,
        selected_features=list(selected), fit_seconds=time.perf_counter() - started,
    )


def aggregate_predictions(model: OVRModel, frame: pd.DataFrame):
    """式(2.17)—(2.20)：窗口分数转为文件级均值、间隔和投票比例。"""
    scores = model.decision_function(frame)
    windows = frame[["file_id", "window_index", "label", "split"]].copy()
    windows[SCORE_COLUMNS] = scores
    windows["window_prediction"] = np.asarray(LABELS)[np.argmax(scores, axis=1)]
    rows = []
    for file_id, group in windows.groupby("file_id", sort=True):
        mean_scores = group[SCORE_COLUMNS].mean().to_numpy()
        median_scores = group[SCORE_COLUMNS].median().to_numpy()
        mean_index = int(np.argmax(mean_scores))
        median_index = int(np.argmax(median_scores))
        votes = group.window_prediction.value_counts().reindex(LABELS, fill_value=0).to_numpy()
        vote_index = int(np.argmax(votes))
        sorted_scores = np.sort(mean_scores)
        row = {
            "file_id": file_id, "true_label": group.label.iloc[0],
            "split": group.split.iloc[0], "windows": len(group),
            "prediction": LABELS[mean_index],
            "median_prediction": LABELS[median_index],
            "majority_prediction": LABELS[vote_index],
            "margin": float(sorted_scores[-1] - sorted_scores[-2]),
            "vote_fraction": float(votes[mean_index] / len(group)),
        }
        row.update({column: float(value) for column, value in zip(SCORE_COLUMNS, mean_scores)})
        rows.append(row)
    return pd.DataFrame(rows), windows


def classification_metrics(true_labels, predicted_labels):
    """式(2.23)—(2.25)：全部以文件为单位。"""
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    confusion = pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=int)
    for true, predicted in zip(true_labels, predicted_labels):
        confusion.loc[true, predicted] += 1
    per_class = []
    for label in LABELS:
        tp = int(confusion.loc[label, label])
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "precision": precision, "recall": recall,
                          "f1": f1, "support": int(confusion.loc[label].sum())})
    detail = pd.DataFrame(per_class)
    return {
        "accuracy": float(np.mean(true_labels == predicted_labels)),
        "macro_f1": float(detail.f1.mean()),
        "balanced_accuracy": float(detail.recall.mean()),
        "normal_recall": float(detail.loc[detail.label.eq("N"), "recall"].iloc[0]),
        "n_files": int(len(true_labels)),
    }, detail, confusion


def build_cv_cache(frame: pd.DataFrame, metadata: pd.DataFrame, file_ids, config: dict):
    """为一个表示和开发集合缓存两折处理；各折独立筛选与拟合尺度。"""
    folds = make_group_folds(metadata, file_ids, config["seed"])
    cache = []
    for hold_fold in [0, 1]:
        fit_ids = list(folds.index[folds.ne(hold_fold)])
        hold_ids = list(folds.index[folds.eq(hold_fold)])
        selected, selection = stable_select_for_files(frame, fit_ids, config)
        scaler = fit_file_equal_scaler(frame, fit_ids, selected, config["numeric_epsilon"])
        cache.append({"fold": hold_fold, "fit_ids": fit_ids, "hold_ids": hold_ids,
                      "selected": selected, "selection": selection, "scaler": scaler})
    return folds, cache


def tune_model(frame: pd.DataFrame, cv_cache, model_spec: dict, config: dict):
    """式(2.22)、(2.27)：相同折上的有限网格，按预定规则选择。"""
    gamma_grid = config["gamma_grid"] if model_spec["kernel"] == "rbf" \
        else [config["linear_gamma_placeholder"]]
    rows = []
    for C in config["c_grid"]:
        for gamma in gamma_grid:
            for fold in cv_cache:
                model = fit_ovr(
                    frame, fold["fit_ids"], fold["selected"], fold["scaler"],
                    kernel=model_spec["kernel"], C=C, gamma=gamma,
                    weighting=model_spec["weighting"], config=config)
                hold = frame[frame.file_id.isin(fold["hold_ids"])]
                predictions, _ = aggregate_predictions(model, hold)
                metrics, _, _ = classification_metrics(
                    predictions.true_label, predictions.prediction)
                rows.append({"model": model_spec["name"], "fold": fold["fold"],
                             "C": C, "gamma": gamma, "features": "|".join(fold["selected"]),
                             "fit_files": len(fold["fit_ids"]),
                             "hold_files": len(fold["hold_ids"]),
                             "fit_seconds": model.fit_seconds, **metrics})
    results = pd.DataFrame(rows)
    summary = results.groupby(["C", "gamma"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        normal_recall=("normal_recall", "mean"),
        fit_seconds=("fit_seconds", "sum"),
    )
    best = summary.sort_values(
        ["macro_f1", "balanced_accuracy", "C", "gamma"],
        ascending=[False, False, True, True], kind="stable").iloc[0]
    return {"C": float(best.C), "gamma": float(best.gamma)}, results, summary


def fit_pipeline(frame: pd.DataFrame, file_ids, model_spec: dict, parameters: dict,
                 config: dict):
    selected, selection = stable_select_for_files(frame, file_ids, config)
    scaler = fit_file_equal_scaler(frame, file_ids, selected, config["numeric_epsilon"])
    model = fit_ovr(frame, file_ids, selected, scaler, kernel=model_spec["kernel"],
                    C=parameters["C"], gamma=parameters["gamma"],
                    weighting=model_spec["weighting"], config=config)
    return model, selection


def model_state(model: OVRModel):
    """把可复算的支持向量状态拆成 JSON 元数据和 NPZ 数组。"""
    metadata = {
        "labels": model.labels, "kernel": model.kernel, "C": model.C,
        "gamma": model.gamma, "weighting": model.weighting,
        "selected_features": model.selected_features, "scaler": model.scaler,
        "fit_seconds": model.fit_seconds, "score_type": "native_svm_margin_not_probability",
        "estimators": {},
    }
    arrays = {}
    for label, estimator in model.estimators.items():
        metadata["estimators"][label] = {
            "intercept": float(estimator.intercept_[0]),
            "n_support": int(len(estimator.support_)),
            "classes": estimator.classes_.tolist(),
        }
        arrays[f"{label}__support_vectors"] = estimator.support_vectors_
        arrays[f"{label}__dual_coef"] = estimator.dual_coef_[0]
    return metadata, arrays


def save_model(model: OVRModel, json_path: Path, arrays_path: Path):
    metadata, arrays = model_state(model)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(arrays_path, **arrays)


def reconstruct_scores(model: OVRModel, frame: pd.DataFrame):
    """独立用支持向量/对偶系数重建分数，作为实现一致性检查。"""
    x = transform_features(frame, model.scaler)
    columns = []
    for label in LABELS:
        estimator = model.estimators[label]
        sv = estimator.support_vectors_
        if model.kernel == "rbf":
            squared = ((x[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2)
            kernel = np.exp(-model.gamma * squared)
        else:
            kernel = x @ sv.T
        columns.append(kernel @ estimator.dual_coef_[0] + estimator.intercept_[0])
    return np.column_stack(columns)


def deterministic_noise_seed(seed: int, file_id: str, level: float):
    digest = hashlib.sha256(f"{seed}|{file_id}|{level:.8g}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def amplitude_perturbation(frame: pd.DataFrame, gain: float):
    changed = frame.copy()
    changed[AMPLITUDE_FEATURES] = changed[AMPLITUDE_FEATURES] * float(gain)
    return changed
