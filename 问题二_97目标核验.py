"""问题二 97% 目标复核：开发集候选与固定测试的文件级对照。

这个程序导出实验审计，不替换已验收的 20 转三频带主模型。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from utils.q2_model import classification_metrics
from utils.q2_optimized import (
    aggregate_file_medians, extract_multiband_windows, feature_sets, median_columns,
)


ROOT = Path(__file__).resolve().parent
CLASS_ORDER = ["N", "OR", "IR", "B"]
CV_SEEDS = [11, 29, 42, 71, 97]
OR_CENTER_WEIGHT = 4.0


def position_sublabels(files: pd.DataFrame) -> np.ndarray:
    """仅从拟合折的已知源域标签构造位置子类。"""
    labels = []
    for row in files[["file_id", "label"]].itertuples(index=False):
        if row.label != "OR":
            labels.append(row.label)
        elif "/Centered/" in row.file_id:
            labels.append("OR_C")
        elif "/Opposite/" in row.file_id or "/Orthogonal/" in row.file_id:
            labels.append("OR_O")
        else:
            raise ValueError(f"外圈训练位置无法识别: {row.file_id}")
    return np.asarray(labels)


def subtype_class_weights(training_sublabels: np.ndarray) -> dict[str, float]:
    counts = pd.Series(training_sublabels).value_counts()
    expected = {"N", "OR_C", "OR_O", "IR", "B"}
    if set(counts.index) != expected:
        raise ValueError(f"拟合折未包含五个位置子类: {set(counts.index)}")
    weights = {label: len(training_sublabels) / (5 * int(count))
               for label, count in counts.items()}
    weights["OR_C"] *= OR_CENTER_WEIGHT
    return weights


def fit_candidate(files: pd.DataFrame, features: list[str], *, position_subtypes: bool):
    if position_subtypes:
        fit_labels = position_sublabels(files)
        class_weight = subtype_class_weights(fit_labels)
    else:
        fit_labels = files.label.to_numpy()
        class_weight = "balanced"
    estimator = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=1, max_features="sqrt",
        class_weight=class_weight, random_state=42, n_jobs=1,
    )
    estimator.fit(files[features], fit_labels)
    return estimator


def four_class_scores(estimator: ExtraTreesClassifier,
                      files: pd.DataFrame) -> np.ndarray:
    """推断只读信号特征；两个 OR 训练子类在输出端合并。"""
    raw = estimator.predict_proba(files[estimator.feature_names_in_])
    model_classes = estimator.classes_.tolist()
    scores = np.column_stack([
        raw[:, [index for index, name in enumerate(model_classes)
                if name == label or (label == "OR" and name in {"OR_C", "OR_O"})]].sum(axis=1)
        for label in CLASS_ORDER
    ])
    if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("候选的四类分数非法")
    return scores


def prediction_table(estimator: ExtraTreesClassifier,
                     files: pd.DataFrame) -> pd.DataFrame:
    scores = four_class_scores(estimator, files)
    predicted = np.asarray(CLASS_ORDER)[scores.argmax(axis=1)]
    result = files[["file_id", "label", "split"]].rename(
        columns={"label": "true_label"}).copy()
    result["prediction"] = predicted
    for index, label in enumerate(CLASS_ORDER):
        result[f"score_{label}"] = scores[:, index]
    return result


def repeated_cv(files: pd.DataFrame, features: list[str], *,
                position_subtypes: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    prediction_rows = []
    for seed in CV_SEEDS:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        predictions = np.empty(len(files), dtype=object)
        for fit_index, hold_index in splitter.split(files[features], files.label):
            estimator = fit_candidate(files.iloc[fit_index], features,
                                      position_subtypes=position_subtypes)
            scores = four_class_scores(estimator, files.iloc[hold_index])
            predictions[hold_index] = np.asarray(CLASS_ORDER)[scores.argmax(axis=1)]
        metrics, _, _ = classification_metrics(files.label, predictions)
        summary_rows.append({"seed": seed, "folds": 3,
                             "model": "position_subtypes" if position_subtypes else "four_classes",
                             **metrics})
        prediction_rows.extend({"seed": seed,
                                "model": "position_subtypes" if position_subtypes else "four_classes",
                                "file_id": file_id, "true_label": truth,
                                "prediction": prediction}
                               for file_id, truth, prediction in zip(
                                   files.file_id, files.label, predictions))
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def prepare_output(output: Path, overwrite: bool) -> None:
    path = output.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("输出目录必须位于项目内")
    marker = path / ".q2-97-generated"
    if path.exists() and any(path.iterdir()) and (not overwrite or not marker.exists()):
        raise FileExistsError(f"非空输出目录不可覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("Generated by 问题二_97目标核验.py; original data are read-only.\n",
                      encoding="utf-8")


def smoke_file_ids(metadata: pd.DataFrame) -> set[str]:
    train = metadata[metadata.split.eq("train")].sort_values("file_id")
    validation = metadata[metadata.split.eq("validation")].sort_values("file_id")
    selected = []
    for label in ["N", "IR", "B"]:
        selected.append(train[train.label.eq(label)].iloc[0].file_id)
    selected.append(train[(train.label.eq("OR")) &
                          train.file_id.str.contains("/Centered/")].iloc[0].file_id)
    selected.append(train[(train.label.eq("OR")) &
                          ~train.file_id.str.contains("/Centered/")].iloc[0].file_id)
    for label in CLASS_ORDER:
        selected.append(validation[validation.label.eq(label)].iloc[0].file_id)
    if len(selected) != len(set(selected)):
        raise ValueError("烟雾测试文件选择重复")
    return set(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二97%文件级目标的可复现核验")
    parser.add_argument("--output-root", type=Path, default=Path("results/q2_97_audit"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    output = (ROOT / args.output_root).resolve()
    prepare_output(output, args.overwrite)

    q2 = json.loads((ROOT / "configs/q2.json").read_text(encoding="utf-8"))["optimized"]
    q1 = json.loads((ROOT / "results/q1/summary.json").read_text(encoding="utf-8"))
    config = deepcopy(q2)
    config["window_revolutions"] = 30
    config["resonance_bands_hz"] = [q2["single_band_hz"]]
    metadata = pd.read_csv(ROOT / "模型适配数据集/q1/files.csv", keep_default_na=False)
    source = metadata[metadata.domain.eq("source")].copy()
    if args.smoke:
        selected_ids = smoke_file_ids(source)
        source = source[source.file_id.isin(selected_ids)].copy()
    elif [int(source.split.eq(part).sum()) for part in ["train", "validation", "test"]] != [70, 23, 23]:
        raise ValueError("源域原有文件划分与70/23/23不一致")

    windows = extract_multiband_windows(ROOT, source, q1["parameters"], config)
    families = feature_sets(config)
    files = aggregate_file_medians(windows, families["multiband_both"])
    features = median_columns(families["single_band_physics"])
    if len(features) != 38 or files.file_id.nunique() != len(source):
        raise ValueError("候选特征维度或源文件覆盖错误")
    write_csv(windows, output / "source_window_features_30turn.csv")
    write_csv(files, output / "source_file_medians_30turn.csv")

    train = files[files.split.eq("train")].copy()
    validation = files[files.split.eq("validation")].copy()
    development = files[files.split.isin(["train", "validation"])].copy()
    if args.smoke:
        estimator = fit_candidate(train, features, position_subtypes=True)
        held = prediction_table(estimator, validation)
        write_csv(held, output / "smoke_predictions.csv")
        summary = {"mode": "smoke", "fit_files": len(train), "hold_files": len(validation),
                   "n_features": len(features), "n_windows": len(windows),
                   "correct_files": int((held.true_label == held.prediction).sum()),
                   "runtime_seconds": time.perf_counter() - started}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    test = files[files.split.eq("test")].copy()
    dev_cv_parts = []
    dev_oof_parts = []
    validation_parts = []
    test_parts = []
    for subtypes in [False, True]:
        name = "position_subtypes" if subtypes else "four_classes"
        cv, oof = repeated_cv(development, features, position_subtypes=subtypes)
        dev_cv_parts.append(cv)
        dev_oof_parts.append(oof)
        model70 = fit_candidate(train, features, position_subtypes=subtypes)
        val_predictions = prediction_table(model70, validation)
        val_predictions.insert(0, "model", name)
        validation_parts.append(val_predictions)
        model93 = fit_candidate(development, features, position_subtypes=subtypes)
        test_predictions = prediction_table(model93, test)
        test_predictions.insert(0, "model", name)
        test_parts.append(test_predictions)
    dev_cv = pd.concat(dev_cv_parts, ignore_index=True)
    dev_oof = pd.concat(dev_oof_parts, ignore_index=True)
    validation_predictions = pd.concat(validation_parts, ignore_index=True)
    test_predictions = pd.concat(test_parts, ignore_index=True)
    write_csv(dev_cv, output / "development_cv_detailed.csv")
    write_csv(dev_oof, output / "development_oof_predictions.csv")
    write_csv(validation_predictions, output / "validation_predictions.csv")
    write_csv(test_predictions, output / "test_predictions.csv")

    # 保持探索审计时的行序：先 93 个开发文件，再 23 个固定测试文件。
    # StratifiedKFold 的随机置换依赖行位置；改为全体 file_id 排序会改变折分。
    full_cv_files = pd.concat([development, test], ignore_index=True)
    full_cv, full_oof = repeated_cv(full_cv_files, features, position_subtypes=True)
    write_csv(full_cv, output / "full_source_cv_detailed.csv")
    write_csv(full_oof, output / "full_source_oof_predictions.csv")
    baseline = json.loads((ROOT / "results/q2_optimized/summary.json").read_text(
        encoding="utf-8"))
    compared = {}
    for name in ["four_classes", "position_subtypes"]:
        detail = dev_cv[dev_cv.model.eq(name)]
        val = validation_predictions[validation_predictions.model.eq(name)]
        tst = test_predictions[test_predictions.model.eq(name)]
        compared[name] = {
            "development_cv_accuracy_mean": float(detail.accuracy.mean()),
            "development_cv_accuracy_min": float(detail.accuracy.min()),
            "development_cv_macro_f1_mean": float(detail.macro_f1.mean()),
            "validation_correct": int((val.true_label == val.prediction).sum()),
            "validation_files": len(val),
            "fixed_test_correct": int((tst.true_label == tst.prediction).sum()),
            "fixed_test_files": len(tst),
            "fixed_test_accuracy": float((tst.true_label == tst.prediction).mean()),
            "fixed_test_errors": tst.loc[tst.true_label.ne(tst.prediction),
                                           ["file_id", "true_label", "prediction"]].to_dict("records"),
        }
    summary = {
        "mode": "audit", "source_files": len(files), "train_files": len(train),
        "validation_files": len(validation), "development_files": len(development),
        "test_files": len(test), "target_files_used": 0,
        "n_windows": len(windows), "n_features": len(features),
        "window_revolutions": 30, "band_hz": config["resonance_bands_hz"][0],
        "n_estimators": 500, "or_center_weight": OR_CENTER_WEIGHT,
        "cv_seeds": CV_SEEDS, "cv_stratification": "original_four_classes",
        "baseline_20turn_threeband_test_correct": baseline["test_correct_files"],
        "baseline_20turn_threeband_test_accuracy": baseline["test_main_metrics"]["accuracy"],
        "candidates": compared,
        "subtype_full_source_cv_accuracy_mean": float(full_cv.accuracy.mean()),
        "required_test_accuracy": 0.97,
        "minimum_correct_for_strictly_above_97_percent": 23,
        "candidate_meets_fixed_test_target": compared["position_subtypes"]["fixed_test_correct"] == 23,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    if compared["four_classes"]["fixed_test_correct"] != 21 or \
            compared["position_subtypes"]["fixed_test_correct"] != 21:
        raise ValueError("探索冻结结果变化；请核对数据、参数和代码逻辑")
    print(json.dumps({"test_correct": {name: value["fixed_test_correct"]
                                       for name, value in compared.items()},
                      "subtype_development_cv_mean":
                      compared["position_subtypes"]["development_cv_accuracy_mean"],
                      "subtype_full_source_cv_mean":
                      summary["subtype_full_source_cv_accuracy_mean"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
