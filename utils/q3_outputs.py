"""问题三的模型状态、迁移诊断和有证据边界的结果文本。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.q1_model import save_csv, write_json
from utils.q2_optimized import ExtraTreesFileModel, save_model, scores_from_export
from utils.q3_transfer import CLASS_ORDER, ordered_scores


def alignment_diagnostics(source: np.ndarray, target: np.ndarray, alignment) -> pd.DataFrame:
    """仅量化分布位置/协方差距离，不把距离下降当成准确率。"""
    mapped = alignment.transform_source(source)
    rows = []
    for name, block in alignment.blocks.items():
        indices = block.indices
        tgt = target[:, indices]
        for stage, src in [("before", source[:, indices]),
                           ("after", mapped[:, indices])]:
            rows.append({
                "block": name, "stage": stage,
                "mean_distance_l2": float(np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0))),
                "cov_distance_frobenius": float(np.linalg.norm(
                    np.cov(src, rowvar=False, bias=True)
                    - np.cov(tgt, rowvar=False, bias=True), ord="fro")),
                "source_files": len(src), "target_files": len(tgt),
            })
    return pd.DataFrame(rows)


def export_transfer_model(output: Path, config: dict, q1: dict, scale,
                          alignment, model, target_scaled: np.ndarray) -> None:
    """树状态及完整变换分开导出，回读验证四类分数。"""
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    fitted = ExtraTreesFileModel(model, list(config["features"]), CLASS_ORDER, {
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"], "alignment_alpha": alignment.alpha,
    })
    json_path, npz_path = model_dir / "trees.json", model_dir / "trees.npz"
    save_model(fitted, json_path, npz_path)
    interface = {
        "schema": "q3-file-median-partial-coral-v1",
        "input": "Q1 20-revolution window table, one median per MAT file",
        "chosen_band_hz": q1["chosen_band_hz"],
        "chosen_revolutions": q1["chosen_revolutions"],
        "features": config["features"], "labels": CLASS_ORDER,
        "source_fit_median": scale.median.tolist(),
        "source_fit_iqr_or_one": scale.divisor.tolist(),
        "alpha": alignment.alpha,
        "blocks": {name: {"indices": block.indices.tolist(),
                           "source_mean": block.source_mean.tolist(),
                           "target_mean": block.target_mean.tolist(),
                           "source_to_target_matrix": block.matrix.tolist()}
                   for name, block in alignment.blocks.items()},
        "inference": "Scale file medians with source parameters. Apply alignment only to source training files; score target file medians directly with exported trees.",
        "score_semantics": "uncalibrated tree-vote fraction",
        "target_truth_available": False,
    }
    write_json(model_dir / "model_interface.json", interface)
    table = pd.DataFrame(target_scaled, columns=config["features"])
    exported = scores_from_export(json_path, npz_path, table)
    live = ordered_scores(model, target_scaled, CLASS_ORDER)
    if not np.allclose(exported, live, atol=1e-12):
        raise ValueError("导出的树状态与内存模型分数不一致")
    write_json(model_dir / "export_validation.json", {
        "target_files_checked": len(target_scaled),
        "max_abs_score_difference": float(np.max(np.abs(exported - live))),
        "passed": True,
    })


def write_result_report(output: Path, summary: dict, proxy: pd.DataFrame,
                        target: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    one = proxy[proxy.direction.eq("12_to_48")].set_index("alpha")
    two = proxy[proxy.direction.eq("48_to_12")].set_index("alpha")
    lines = ["# 问题三：无标签目标域诊断实算结果", "",
             "## 数据与模型", "",
             "源域 116 个文件，固定开发集 93 个、测试集 23 个；目标域 16 个文件，"
             "每文件 3 个 20 转窗口，共 48 个窗口。目标域无真实类别。"
             "每文件对 9 个无量纲特征取中位数，以开发集拟合鲁棒缩放，"
             "按冲击形态和包络阶次两块做部分 CORAL 对齐，再训练 500 棵极端随机树。", "",
             "## 代理域选择与源域对照", "",
             "| 对齐强度 α | 12 kHz→48 kHz 宏 F1 | 48 kHz→12 kHz 宏 F1 |",
             "|---:|---:|---:|"]
    for alpha in sorted(one.index):
        lines.append(f"| {alpha:g} | {one.loc[alpha, 'macro_f1']:.3f} | "
                     f"{two.loc[alpha, 'macro_f1']:.3f} |")
    base = summary["source_baseline_metrics"]
    aligned = summary["source_aligned_metrics"]
    lines.extend(["", f"按最差方向宏 F1、两方向平均宏 F1、较小 α 的预设顺序选得 "
                  f"α={summary['alpha_selected']:g}。代理域只覆盖 OR/IR/B 三类，"
                  "且九特征设计曾参考开发集验证表现，因此该比较属于开发阶段证据。", "",
                  f"固定源域测试：无对齐 {base['correct']}/{base['files']}，"
                  f"对齐 {aligned['correct']}/{aligned['files']}；对齐后准确率"
                  f" {aligned['accuracy']:.2%}，平衡准确率"
                  f" {aligned['balanced_accuracy']:.2%}，宏 F1"
                  f" {aligned['macro_f1']:.3f}。源域固定测试在前期研究中已经查看，"
                  "不是全新盲测。问题二源域主线的 22/23 不能移作本模型结果。", "",
                  "## 目标域逐文件预测", "",
                  "| 文件 | 预测 | 前两类分数差 | 无对齐 | 目标留一 | 540 rpm | 660 rpm | 复核 |",
                  "|---|---|---:|---|---|---|---|---|"])
    for row in target.itertuples(index=False):
        lines.append(f"| {Path(row.file_id).stem} | {row.prediction} | {row.margin:.3f} | "
                     f"{row.source_baseline_prediction} | {row.loo_prediction} | "
                     f"{row.rpm_540_prediction} | {row.rpm_660_prediction} | "
                     f"{'是' if row.review_flag else '否'} |")
    counts = summary["target_predicted_counts"]
    lines.extend(["", f"预测计数：IR {counts.get('IR', 0)}，B {counts.get('B', 0)}，"
                  f"OR {counts.get('OR', 0)}，N {counts.get('N', 0)}。"
                  f"复核标记 {summary['target_review_files']}/16；该标记指不同"
                  "模型分支、目标留一或转速扰动预测不一致，不是人工核验后的错误标签。"
                  "类别分数是树投票比例，未做概率校准。", "",
                  "## 分布对齐诊断与适用边界", "",
                  "| 特征块 | 阶段 | 均值距离 L2 | 协方差距离 Frobenius |",
                  "|---|---|---:|---:|"])
    for row in diagnostics.itertuples(index=False):
        lines.append(f"| {row.block} | {row.stage} | {row.mean_distance_l2:.3f} | "
                     f"{row.cov_distance_frobenius:.3f} |")
    lines.extend(["", "这些距离仅说明特征分布的变化，不证明故障分类更准确。"
                  "目标域没有真值，无法计算目标准确率；题面转速约 600 rpm，"
                  "540/660 rpm 重算只检验这一近似的敏感性。"
                  "16 个目标文件的标签仍为预测结果，需优先复核带标记文件。", ""])
    (output / "问题三_结果报告.md").write_text("\n".join(lines), encoding="utf-8")
