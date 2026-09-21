"""问题二优化模型的可编辑 SVG 与 300 DPI PNG 图件。"""
from __future__ import annotations

import html
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from utils.figure_export import export_figure
from utils import plot_style
from utils.visual_qa import audit_layout
from utils.q1_model import LABELS, write_json
from utils.q2_optimized import bearing_fault_orders


COLORS = {"N": "#0072B2", "OR": "#E69F00", "IR": "#009E73", "B": "#CC79A7"}


def _figure(height=3.4):
    return plt.subplots(figsize=(7.2, height), layout="constrained")


def _save(fig, path: Path, contracts: list, title: str, unit: str, source: str,
          caption: str):
    issues = [(level, message) for level, message in audit_layout(fig)
              if level in ("WARN", "FAIL")]
    design = plot_style.audit_design(fig)
    legacy = plot_style.audit_layout(fig)
    if issues or design or legacy:
        raise ValueError(f"{path.name} 图表预检失败: {issues}; {design}; {legacy}")
    export_figure(fig, str(path), formats=["svg", "png"], dpi=300,
                  size_inches=tuple(fig.get_size_inches()), tight=False,
                  grayscale_preview=False)
    plot_style._save_grayscale_preview(path.with_suffix(".png"), 300)
    contracts.append({"figure": path.name, "question": "q2", "goal": title,
                      "unit_of_analysis": unit, "evidence": source,
                      "caption": caption, "dpi": 300})
    plt.close(fig)


def _box(ax, x, y, text, color, width=.36, height=.12, faded=False):
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         facecolor=color + ("0D" if faded else "1A"),
                         edgecolor=color, linewidth=1.1,
                         linestyle="--" if faded else "-")
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.1, color="#222222")


def _arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1, color="#667085",
                                linestyle="--" if dashed else "-"))


def _flow_canvas(height=4.5):
    fig, ax = _figure(height)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    return fig, ax


def make_optimized_figures(*, data_dir: Path, figure_dir: Path, result_dir: Path, summary: dict,
                           predictions: pd.DataFrame, importance: pd.DataFrame,
                           validation: pd.DataFrame, confusion: pd.DataFrame,
                           robustness: pd.DataFrame):
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ["Noto Sans CJK SC", "Source Han Sans SC",
                                   "Microsoft YaHei", "PingFang SC", "Heiti TC"]
                 if name in available), None)
    if font is None:
        raise RuntimeError("中文图缺少字体")
    plot_style.apply_publication_style(language="en", width="double")
    plt.rcParams["font.sans-serif"] = [font, "Arial", "DejaVu Sans"]
    contracts = []
    files = pd.read_csv(data_dir / "source_multiband_file_medians.csv")
    windows = pd.read_csv(data_dir / "source_multiband_window_features.csv")
    config = __import__("json").loads(
        (Path(__file__).resolve().parents[1] / "configs/q2.json").read_text())["optimized"]

    # 原始图：只呈现未参与拟合的文件结构与物理量。
    fig, ax = _figure(3.1)
    counts = files[files.split.eq("train")].label.value_counts().reindex(LABELS, fill_value=0)
    bars = ax.bar(LABELS, counts, color=[COLORS[x] for x in LABELS])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(xlabel="源域状态", ylabel="独立训练文件数（个）", ylim=(0, max(counts)*1.2))
    _save(fig, figure_dir / "raw_q2_class_files", contracts, "训练类不平衡",
          "70个源训练文件", "模型适配数据集/q1/files.csv", "N类只有2个训练文件。")

    fig, ax = _figure(3.2)
    count = windows.groupby("file_id").size().to_numpy()
    ax.hist(count, bins=np.arange(count.min()-.5, count.max()+1.5),
            color="#0072B2", edgecolor="white")
    ax.set(xlabel="单文件20转窗口数（个）", ylabel="源文件数（个）")
    _save(fig, figure_dir / "raw_q2_windows_per_file", contracts, "窗口不是独立样本",
          "116个源文件", "source_multiband_window_features.csv", "每文件提取多个相关窗口。")

    fig, ax = _figure(3.2)
    orders = bearing_fault_orders(config["bearing_geometry"])
    ax.bar(list(orders), list(orders.values()),
           color=["#6B7280", COLORS["OR"], COLORS["IR"], COLORS["B"], "#56B4E9"])
    for index, value in enumerate(orders.values()):
        ax.text(index, value+.1, f"{value:.2f}", ha="center", fontsize=8)
    ax.set(xlabel="轴承故障阶次族", ylabel="特征频率 / 转频（阶）",
           ylim=(0, max(orders.values())*1.2))
    _save(fig, figure_dir / "raw_q2_fault_orders", contracts, "题面BSF与自旋半值分列",
          "SKF6205 DE几何", "原题附件1表1与表2", "几何阶次不由测试标签拟合；spin_half是BSF的一半。")

    # 过程图：调参、特征家族、跨频带及可复用状态。
    cv = pd.read_csv(result_dir / "final_cv_summary.csv")
    fig, ax = _figure(3.5)
    grid = cv.pivot(index="min_samples_leaf", columns="max_features",
                    values="accuracy_mean").sort_index()
    matrix = grid.to_numpy(float)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                   facecolor=plt.get_cmap("cividis")(matrix[row,col]),
                                   edgecolor="white", linewidth=.8))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row,col]:.3f}", ha="center", va="center",
                    color="white" if matrix[row,col] < .6 else "black", fontsize=7)
    ax.set(xlim=(-.5, len(grid.columns)-.5), ylim=(len(grid.index)-.5, -.5),
           xticks=range(len(grid.columns)), xticklabels=grid.columns,
           yticks=range(len(grid.index)), yticklabels=grid.index,
           xlabel="每次分裂候选特征", ylabel="叶节点最小文件数")
    _save(fig, figure_dir / "process_q2_parameter_surface", contracts, "93文件内部选择树参数",
          "93开发文件重复三折", "final_cv_summary.csv", "颜色及数值为文件级平均准确率。")

    fig, ax = _figure(3.3)
    ordered = validation.set_index("feature_family").loc[
        config["feature_family_order"]]
    positions = np.arange(len(ordered))
    ax.plot(positions, ordered.accuracy, marker="o", label="准确率", color="#0072B2")
    ax.plot(positions, ordered.macro_f1, marker="s", label="宏F1", color="#E69F00")
    labels = ["原12特征", "单频带题面BSF", "三频带题面BSF", "三频带自旋半值", "三频带两者"]
    ax.set(xticks=positions, xticklabels=labels,
           xlabel="特征家族", ylabel="23验证文件指标", ylim=(.75, 1.02))
    ax.tick_params(axis="x", rotation=12, labelsize=7)
    ax.legend(loc="lower right")
    _save(fig, figure_dir / "process_q2_feature_family", contracts, "验证集冻结特征家族",
          "23个验证文件", "feature_family_validation_metrics.csv",
          "每种家族的树参数只用70个训练文件选取。")

    fig, ax = _figure(3.6)
    top = importance.head(12).iloc[::-1]
    ax.barh(range(len(top)), top.importance, color="#009E73")
    ax.set(yticks=range(len(top)), yticklabels=top.feature,
           xlabel="树不纯度下降的相对份额", ylabel="选中特征")
    ax.tick_params(axis="y", labelsize=6.8)
    _save(fig, figure_dir / "process_q2_feature_importance", contracts, "主要分裂特征",
          "93开发文件模型", "feature_importances.csv",
          "不纯度重要性为训练内描述，非因果贡献。")

    fig, ax = _figure(3.2)
    band = importance.groupby("band").importance.sum().sort_values(ascending=False)
    bars = ax.bar(range(len(band)), band, color=["#0072B2", "#E69F00",
                                                   "#009E73", "#CC79A7"][:len(band)])
    ax.set(xticks=range(len(band)), xticklabels=band.index, xlabel="特征所属频带",
           ylabel="不纯度重要性总和")
    ax.tick_params(axis="x", rotation=12)
    _save(fig, figure_dir / "process_q2_band_contribution", contracts, "三频带均可进入树分裂",
          "93开发文件模型", "feature_importances.csv",
          "q1_base是1500—3000 Hz原统计特征；其余为物理阶次及包络量。")

    # 结果图：测试只评价冻结模型，图注显式声明复用过的测试集。
    fig, ax = _figure(3.5)
    matrix = confusion.loc[LABELS, LABELS].to_numpy()
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                   facecolor=plt.get_cmap("Blues")(
                                       matrix[row,col]/max(1,matrix.max())),
                                   edgecolor="white", linewidth=1))
    for row in range(4):
        for col in range(4):
            ax.text(col, row, str(matrix[row,col]), ha="center", va="center",
                    color="white" if matrix[row,col] > matrix.max()/2 else "#222222")
    ax.set(xlim=(-.5,3.5), ylim=(3.5,-.5),
           xticks=range(4), xticklabels=LABELS, yticks=range(4), yticklabels=LABELS,
           xlabel="预测类别", ylabel="真实类别")
    _save(fig, figure_dir / "result_q2_confusion", contracts, "23文件中22个正确",
          "23个原有测试文件", "confusion_test.csv",
          "唯一错误为48kHz外圈故障判成滚动体故障；测试集此前已查看。")

    fig, ax = _figure(3.3)
    values = [summary["legacy_rbf_test_metrics"]["accuracy"],
              summary["test_main_metrics"]["accuracy"],
              summary["full_source_cv_audit"]["accuracy_mean"]]
    names = ["原RBF\n旧测试", "三频带树\n旧测试", "三频带树\n五组CV均值"]
    bars = ax.bar(range(3), values, color=["#AAB2BD", "#009E73", "#0072B2"])
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.axhline(.95, color="#D55E00", linestyle="--", label="新增95%门槛")
    ax.set(xticks=range(3), xticklabels=names, ylabel="文件级准确率", ylim=(0, 1.08))
    ax.legend(loc="upper left")
    _save(fig, figure_dir / "result_q2_accuracy", contracts, "优化模型达到观察测试门槛",
          "原23文件测试及116文件重复CV", "summary.json / full_source_cv_audit.csv",
          "测试集曾被查看，柱形为观察结果，不构成全新盲测保证。")

    fig, ax = _figure(3.5)
    ordered = predictions.sort_values("margin").reset_index(drop=True)
    correct = ordered.prediction.eq(ordered.true_label)
    ax.scatter(ordered.margin, np.arange(len(ordered)),
               c=np.where(correct, "#009E73", "#D55E00"), s=30)
    for index, row in ordered.iterrows():
        if not correct.iloc[index]:
            ax.annotate(f"{row.true_label}→{row.prediction}",
                        (row.margin, index), xytext=(5, 0),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="最高树分数与次高分数之差", ylabel="测试文件（按间隔排序）",
           yticks=[])
    _save(fig, figure_dir / "result_q2_file_margins", contracts, "错误文件间隔可追踪",
          "23个原有测试文件", "source_test_file_scores.csv",
          "树分数不是校准的正确概率。")

    per_class = pd.read_csv(result_dir / "per_class_test.csv").set_index("label").loc[LABELS]
    fig, ax = _figure(3.1)
    bars = ax.bar(LABELS, per_class.recall, color=[COLORS[label] for label in LABELS])
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set(xlabel="源域状态", ylabel="文件级召回", ylim=(0, 1.14))
    _save(fig, figure_dir / "result_q2_per_class", contracts, "分别检查四类召回",
          "23个原有测试文件", "per_class_test.csv",
          "正常类测试仅1个文件，召回缺乏精细分辨率。")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    for ax, kind, xlabel in [(axes[0], "amplitude_gain", "幅值增益"),
                             (axes[1], "white_noise", "噪声相对RMS")]:
        group = robustness[robustness.perturbation.eq(kind)].sort_values("level")
        ax.plot(group.level, group.accuracy, marker="o", label="准确率", color="#0072B2")
        ax.plot(group.level, group.prediction_retention, marker="s", linestyle="--",
                label="预测保持率", color="#E69F00")
        ax.set(xlabel=xlabel, ylabel="文件级指标", ylim=(0, 1.08))
        ax.legend(loc="lower left", fontsize=7)
    plot_style.add_panel_labels(axes)
    _save(fig, figure_dir / "result_q2_robustness", contracts, "原始信号扰动后完整重算",
          "23个原有测试文件", "robustness_perturbations.csv",
          "增益与噪声都从MAT重新提取三频带特征。")

    fig, ax = _flow_canvas(4.2)
    _box(ax, .5, .87, "问题一 · 原始信号与固定文件划分", "#0072B2", width=.56)
    _box(ax, .5, .62, "问题二 · 三频带故障阶次 + 文件级极端随机树", "#009E73", width=.68)
    _arrow(ax, (.5,.81), (.5,.69))
    _box(ax, .26, .34, "问题三 · 迁移诊断\n待用户确认", "#E69F00", faded=True)
    _box(ax, .74, .34, "问题四 · 决策解释\n待问题三完成", "#CC79A7", faded=True)
    _arrow(ax, (.4,.55), (.29,.42), True)
    _arrow(ax, (.6,.55), (.71,.42), True)
    ax.text(.5, .08, "问题二只评价源域；目标域无标签未参与拟合。", ha="center", fontsize=8)
    _save(fig, figure_dir / "flow_overall_model", contracts, "四问流程与交付边界",
          "研究阶段", "问题一和问题二已实算", "虚线表示后续计划。")

    fig, ax = _flow_canvas(4.7)
    _box(ax, .24, .87, "原始MAT + SKF6205几何\n3个共振频带", "#0072B2", width=.4)
    _box(ax, .76, .87, "文件划分70/23/23\n转速换算成阶次", "#6B7280", width=.4)
    _box(ax, .5, .66, "20转窗口物理特征 → 文件中位数", "#009E73", width=.56)
    _arrow(ax, (.3,.8), (.42,.73)); _arrow(ax, (.7,.8), (.58,.73))
    _box(ax, .25, .44, "训练内调参 + 验证集\n选择特征家族", "#E69F00", width=.4)
    _box(ax, .75, .44, "93开发文件重调与拟合\n500棵极端随机树", "#0072B2", width=.4)
    _arrow(ax, (.45,.59), (.28,.51)); _arrow(ax, (.45,.44), (.55,.44))
    _box(ax, .5, .20, "23测试文件评价 + 扰动重算\n导出模型、图表与哈希清单", "#CC79A7", width=.66)
    _arrow(ax, (.75,.37), (.58,.28))
    _save(fig, figure_dir / "flow_q2_model", contracts, "优化模型完整求解路径",
          "原始文件", "问题二_求解.py", "各折以文件为单位；测试结果不进入拟合。")

    write_json(result_dir / "figure_contracts.json", contracts)
    lines = ["# 问题二优化图表清单", "", "| 图件 | 核心结论 | 统计单位 |",
             "|---|---|---|"]
    for item in contracts:
        lines.append(f"| `{item['figure']}` | {item['goal']} | {item['unit_of_analysis']} |")
    (result_dir / "图表清单.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cards = []
    for item in contracts:
        relative = Path(os.path.relpath(figure_dir / (item["figure"]+".png"),
                                        result_dir)).as_posix()
        cards.append("<figure><img src='"+html.escape(relative)+"'><figcaption>"+
                     html.escape(item["caption"])+"</figcaption></figure>")
    panel = """<!doctype html><meta charset='utf-8'><title>问题二优化图表</title>
    <style>body{font-family:Arial,'PingFang SC',sans-serif;margin:24px;color:#222}
    main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
    figure{margin:0;border:1px solid #ddd;padding:12px;background:#fff}img{width:100%}
    figcaption{font-size:13px;line-height:1.5;margin-top:8px}</style>
    <h1>问题二优化模型图表</h1><main>"""
    (result_dir / "图表面板.html").write_text(panel+"".join(cards)+"</main>",
                                               encoding="utf-8")
    return contracts
