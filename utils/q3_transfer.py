"""问题三文件级部分 CORAL 与极端随机树迁移诊断。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score


CLASS_ORDER = ["N", "OR", "IR", "B"]
PROXY_ORDER = ["OR", "IR", "B"]


def file_median_table(windows: pd.DataFrame, metadata: pd.DataFrame,
                      features: list[str]) -> pd.DataFrame:
    """每个 MAT 文件一行；目录、标签和转速只保留为元数据。"""
    if windows[["file_id", "window_index"]].duplicated().any():
        raise ValueError("窗口主键重复")
    if set(windows.file_id) != set(metadata.file_id):
        raise ValueError("窗口表与文件表覆盖不一致")
    if not np.isfinite(windows[features].to_numpy(float)).all():
        raise ValueError("迁移特征含非有限值")
    medians = windows.groupby("file_id", sort=True)[features].median()
    counts = windows.groupby("file_id", sort=True).size().rename("n_windows")
    meta = metadata[["file_id", "domain", "split", "label", "dataset_group",
                     "nominal_fs_hz", "rpm", "rpm_is_approximate"]].set_index("file_id")
    result = meta.join(medians, how="inner").join(counts).reset_index()
    if result.file_id.duplicated().any() or len(result) != len(metadata):
        raise ValueError("文件中位数没有一文件一行")
    return result.sort_values("file_id").reset_index(drop=True)


@dataclass(frozen=True)
class RobustScale:
    columns: tuple[str, ...]
    median: np.ndarray
    divisor: np.ndarray
    zero_iqr: tuple[str, ...]

    def transform(self, files: pd.DataFrame) -> np.ndarray:
        values = files[list(self.columns)].to_numpy(float)
        result = (values - self.median) / self.divisor
        if not np.isfinite(result).all():
            raise ValueError("鲁棒缩放后存在非有限值")
        return result


def fit_robust_scale(files: pd.DataFrame, features: list[str]) -> RobustScale:
    values = files[features].to_numpy(float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("训练文件不足或特征非法")
    median = np.median(values, axis=0)
    q75, q25 = np.quantile(values, [.75, .25], axis=0)
    iqr = q75 - q25
    zero = iqr <= 0
    return RobustScale(tuple(features), median, np.where(zero, 1.0, iqr),
                       tuple(np.asarray(features)[zero]))


def matrix_power_symmetric(matrix: np.ndarray, power: float) -> np.ndarray:
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("协方差矩阵不对称")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() <= 0 or not np.isfinite(eigenvalues).all():
        raise ValueError("收缩协方差非正定")
    return (eigenvectors * eigenvalues ** power) @ eigenvectors.T


def shrunk_covariance(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """式(3.06)—(3.07)：文件等权，Ledoit–Wolf 收缩及数值正定修正。"""
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("协方差输入文件不足或非法")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(values)
    rho = float(LedoitWolf(assume_centered=False).fit(values).shrinkage_)
    dimension = values.shape[1]
    average_variance = float(np.trace(covariance) / dimension)
    epsilon = 1e-8 * max(1.0, average_variance)
    shrunk = ((1 - rho) * covariance + rho * average_variance * np.eye(dimension)
              + epsilon * np.eye(dimension))
    if np.linalg.eigvalsh(shrunk).min() <= 0:
        raise ValueError("协方差修正后仍非正定")
    return shrunk, rho, epsilon


@dataclass(frozen=True)
class BlockMap:
    indices: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    matrix: np.ndarray
    source_shrinkage: float
    target_shrinkage: float


@dataclass(frozen=True)
class Alignment:
    alpha: float
    blocks: dict[str, BlockMap]

    def transform_source(self, values: np.ndarray) -> np.ndarray:
        transformed = values.copy()
        for block in self.blocks.values():
            x = values[:, block.indices]
            full = (x - block.source_mean) @ block.matrix + block.target_mean
            transformed[:, block.indices] = (1 - self.alpha) * x + self.alpha * full
        if not np.isfinite(transformed).all():
            raise ValueError("对齐后的源特征非法")
        return transformed


def fit_alignment(source: np.ndarray, target: np.ndarray, features: list[str],
                  blocks: dict[str, list[str]], alpha: float) -> Alignment:
    if not 0 <= alpha <= 1:
        raise ValueError("对齐强度必须在[0,1]")
    if sorted(sum(blocks.values(), [])) != sorted(features):
        raise ValueError("机理块未恰好覆盖迁移特征")
    mappings = {}
    for name, names in blocks.items():
        indices = np.asarray([features.index(feature) for feature in names], dtype=int)
        source_values, target_values = source[:, indices], target[:, indices]
        source_cov, rho_source, _ = shrunk_covariance(source_values)
        target_cov, rho_target, _ = shrunk_covariance(target_values)
        matrix = (matrix_power_symmetric(source_cov, -.5)
                  @ matrix_power_symmetric(target_cov, .5))
        mappings[name] = BlockMap(indices, source_values.mean(axis=0),
                                  target_values.mean(axis=0), matrix,
                                  rho_source, rho_target)
    return Alignment(alpha, mappings)


def fit_classifier(values: np.ndarray, labels: np.ndarray,
                   config: dict) -> ExtraTreesClassifier:
    counts = pd.Series(labels).value_counts()
    if len(counts) < 3 or counts.min() < 1:
        raise ValueError("训练类别不完整")
    model = ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"], class_weight="balanced",
        random_state=int(config["seed"]), n_jobs=1,
    )
    model.fit(values, labels)
    return model


def ordered_scores(model: ExtraTreesClassifier, values: np.ndarray,
                   classes: list[str]) -> np.ndarray:
    if set(model.classes_) != set(classes):
        raise ValueError("模型类别不符合当前任务")
    raw = model.predict_proba(values)
    index = {label: i for i, label in enumerate(model.classes_)}
    scores = raw[:, [index[label] for label in classes]]
    if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("模型分数非法")
    return scores


def predict_frame(model: ExtraTreesClassifier, values: np.ndarray,
                  files: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    scores = ordered_scores(model, values, classes)
    order = np.sort(scores, axis=1)
    prediction = np.asarray(classes)[scores.argmax(axis=1)]
    frame = files[["file_id", "domain", "split", "label"]].copy()
    frame = frame.rename(columns={"label": "true_label"})
    frame["prediction"] = prediction
    frame["margin"] = order[:, -1] - order[:, -2]
    for col, label in enumerate(classes):
        frame[f"score_{label}"] = scores[:, col]
    return frame


def fit_transfer(source_files: pd.DataFrame, target_files: pd.DataFrame,
                 config: dict, alpha: float, classes: list[str]):
    features = config["features"]
    scale = fit_robust_scale(source_files, features)
    source = scale.transform(source_files)
    target = scale.transform(target_files)
    alignment = fit_alignment(source, target, features, config["blocks"], alpha)
    model = fit_classifier(alignment.transform_source(source),
                           source_files.label.to_numpy(), config)
    if set(model.classes_) != set(classes):
        raise ValueError("分类器缺少拟合类别")
    return scale, alignment, model


def proxy_metrics(files: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for name, source_group, hold_group in [
        ("12_to_48", "12kHz_DE_data", "48kHz_DE_data"),
        ("48_to_12", "48kHz_DE_data", "12kHz_DE_data"),
    ]:
        source = files[files.split.eq("train") & files.dataset_group.eq(source_group)]
        held = files[files.split.eq("validation") & files.dataset_group.eq(hold_group)]
        if set(source.label) != set(PROXY_ORDER) or set(held.label) != set(PROXY_ORDER):
            raise ValueError("代理任务未覆盖三类故障")
        for alpha in config["alpha_candidates"]:
            scale, _, model = fit_transfer(source, held, config, alpha, PROXY_ORDER)
            prediction = predict_frame(model, scale.transform(held), held, PROXY_ORDER)
            rows.append({"direction": name, "alpha": alpha,
                         "train_files": len(source), "hold_files": len(held),
                         "accuracy": accuracy_score(held.label, prediction.prediction),
                         "balanced_accuracy": balanced_accuracy_score(
                             held.label, prediction.prediction),
                         "macro_f1": f1_score(held.label, prediction.prediction,
                                              labels=PROXY_ORDER, average="macro",
                                              zero_division=0)})
    return pd.DataFrame(rows)


def select_alpha(proxy: pd.DataFrame) -> float:
    candidates = []
    for alpha, group in proxy.groupby("alpha", sort=True):
        if set(group.direction) != {"12_to_48", "48_to_12"}:
            raise ValueError("代理方向不完整")
        candidates.append((float(group.macro_f1.min()),
                           float(group.macro_f1.mean()), -float(alpha), float(alpha)))
    return max(candidates)[-1]
