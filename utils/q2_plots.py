"""问题二出版级图表：真实数据、模型过程、冻结测试结果和流程图。"""
from __future__ import annotations

import html
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .figure_export import export_figure
from . import plot_style
from .visual_qa import audit_layout
from .q1_model import FEATURES, FEATURE_ZH, LABELS, file_medians, write_json
from .q2_model import transform_features

COLORS = {"N": "#0072B2", "OR": "#E69F00", "IR": "#009E73", "B": "#CC79A7"}
MODEL_NAMES = {
    "rbf_unweighted": "RBF·窗口均权",
    "rbf_class_balanced": "RBF·类别均衡",
    "rbf_file_class_balanced": "RBF·双重均衡",
    "linear_file_class_balanced": "线性·双重均衡",
}
REP_NAMES = {"equal_revolutions": "20 转", "one_second": "1 秒"}
FEATURE_NAMES = dict(zip(FEATURES, FEATURE_ZH))


def panels(rows=1, cols=1, height=3.2, width_ratios=None):
    gridspec = {"width_ratios": width_ratios} if width_ratios else None
    return plt.subplots(rows, cols, figsize=(7.2, height), layout="constrained",
                        squeeze=False, gridspec_kw=gridspec)


def save(fig, stem: Path, contracts, goal, unit, evidence, caption):
    issues = audit_layout(fig)
    significant = [(level, msg) for level, msg in issues if level in ["WARN", "FAIL"]]
    design = plot_style.audit_design(fig)
    legacy = plot_style.audit_layout(fig)
    if significant or design or legacy:
        raise ValueError(f"{stem.name} 图表预检失败: {significant}; {design}; {legacy}")
    export_figure(fig, str(stem), formats=["svg", "png"], dpi=300,
                  size_inches=tuple(fig.get_size_inches()), tight=False,
                  grayscale_preview=False)
    plot_style._save_grayscale_preview(stem.with_suffix(".png"), 300)
    contracts.append({
        "figure": stem.name, "question": "q2", "goal": goal,
        "unit_of_analysis": unit, "evidence": evidence, "caption": caption,
        "width_inches": float(fig.get_size_inches()[0]),
        "height_inches": float(fig.get_size_inches()[1]), "dpi": 300,
        "layout_issues": significant, "design_issues": design,
    })
    plt.close(fig)


def box(ax, x, y, width, height, text, color, faded=False, shape="round"):
    style = "round,pad=0.012,rounding_size=0.016" if shape == "round" else "square,pad=0.01"
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                           boxstyle=style, facecolor=color + ("0D" if faded else "1A"),
                           edgecolor=color, linewidth=1.05,
                           linestyle="--" if faded else "-")
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.2,
            color="#222222", linespacing=1.5)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.0, color="#667085",
                                linestyle="--" if dashed else "-"))


def make_flows(directory, contracts, selected_representation, features):
    fig, axes = panels(height=4.3)
    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    box(ax, .5, .88, .58, .12, "问题一 · 特征表征\n已完成并固定输入接口", "#0072B2")
    box(ax, .5, .66, .58, .13, "问题二 · 源域诊断\n双重均衡 RBF-SVM + 线性对照（本次完成）", "#009E73")
    arrow(ax, (.5, .82), (.5, .74))
    box(ax, .28, .41, .40, .15, "问题三 · 推荐主线\n分块部分 CORAL\n待用户确认", "#0072B2", True)
    box(ax, .72, .41, .40, .15, "问题三 · 冲奖对照\n非平衡最优传输\n待用户确认", "#E69F00", True)
    arrow(ax, (.42, .585), (.28, .50), True)
    arrow(ax, (.58, .585), (.72, .50), True)
    box(ax, .5, .17, .72, .12, "问题四 · 解释迁移后的实际决策\n待问题三模型冻结", "#CC79A7", True)
    arrow(ax, (.28, .325), (.39, .23), True)
    arrow(ax, (.72, .325), (.61, .23), True)
    ax.text(.5, .04, "实线为已完成阶段；虚线为后续计划。问题二未输出目标域标签。",
            ha="center", fontsize=7.5, color="#555555")
    save(fig, directory / "flow_overall_model", contracts, "四问依赖及当前完成边界",
         "研究阶段", "问题一与问题二真实产物 / 后续模型合同",
         "问题二只冻结源域分类器；问题三、四仍等待用户确认。")

    fig, axes = panels(height=4.8)
    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    box(ax, .22, .88, .35, .10, "两套 Q1 未标准化窗口表\n20 转 / 1 秒", "#0072B2")
    box(ax, .78, .88, .35, .10, "固定文件划分\n70 训练 / 23 验证 / 23 测试", "#6B7280")
    box(ax, .50, .69, .68, .12, "训练内二折：每折重做稳定筛选与文件等权尺度\n有限网格选择 C、γ", "#009E73")
    arrow(ax, (.28, .82), (.42, .755))
    arrow(ax, (.72, .82), (.58, .755))
    box(ax, .25, .49, .40, .12, "70 文件模型 → 验证集\n冻结表示：" + REP_NAMES[selected_representation], "#E69F00")
    box(ax, .75, .49, .40, .12, "93 开发文件重新调参\n四项权重 / 核函数消融", "#0072B2")
    arrow(ax, (.50, .63), (.25, .56))
    arrow(ax, (.25, .43), (.60, .49))
    box(ax, .50, .29, .68, .12, "冻结主模型 → 23 个测试文件一次评价\n文件平均分数 + 中位数 / 多数投票对照", "#CC79A7")
    arrow(ax, (.75, .43), (.58, .35))
    box(ax, .50, .10, .68, .10,
        f"导出：{len(features)} 维输入、OvR 支持向量、文件分数、消融与扰动结果", "#009E73")
    arrow(ax, (.50, .23), (.50, .15))
    save(fig, directory / "flow_q2_model", contracts, "问题二选择、重训和封存测试流程",
         "独立源文件与相关窗口", "问题二_求解.py / utils/q2_model.py",
         "Q1 频带固定；Q2 每个拟合折重做特征筛选和尺度。测试结果不回流。")


def make_q2_figures(*, metadata, frames, selected_representation, validation_metrics,
                    representation_cv, final_cv, parameters, ablation, aggregations,
                    predictions, window_scores, selection, by_file, by_class,
                    robustness, confusion, model, figure_dir, result_dir, config):
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ["Noto Sans CJK SC", "Source Han Sans SC",
                                   "Microsoft YaHei", "PingFang SC", "Heiti TC"]
                 if name in available), None)
    if font is None:
        raise RuntimeError("正式中文图需要中文字体")
    plot_style.apply_publication_style(language="en", width="double")
    plt.rcParams["font.sans-serif"] = [font, "Arial", "DejaVu Sans"]
    contracts = []
    source_meta = metadata[metadata.domain.eq("source")]
    source_frame = frames[selected_representation]

    # 原始1：文件数与窗口数同时显示，柱形从零开始。
    fig, axes = panels(cols=2, height=3.2)
    train_meta = source_meta[source_meta.split.eq("train")]
    file_counts = train_meta.label.value_counts().reindex(LABELS, fill_value=0)
    window_counts = source_frame[source_frame.split.eq("train")].label.value_counts().reindex(LABELS, fill_value=0)
    for ax, values, ylabel, title in [
        (axes[0, 0], file_counts, "独立训练文件数（个）", "文件层"),
        (axes[0, 1], window_counts, "训练窗口数（个）", "窗口层")]:
        bars = ax.bar(LABELS, values, color=[COLORS[x] for x in LABELS])
        ax.bar_label(bars, fontsize=7, padding=2)
        ax.set(xlabel="源域工作状态", ylabel=ylabel, title=title, ylim=(0, max(values)*1.18))
    plot_style.add_panel_labels(axes.flat)
    save(fig, figure_dir / "raw_q2_class_imbalance", contracts,
         "正常类稀缺且类别窗口数不平衡", "70个训练文件及其相关窗口",
         "files.csv / 选定表示的原始窗口表",
         "左图按原文件计数，右图按窗口计数；窗口不能作为独立样本量。")

    # 原始2：两种表示每文件窗口数的经验累积分布。
    fig, axes = panels(height=3.0)
    ax = axes[0, 0]
    for representation, frame in frames.items():
        counts = frame[frame.domain.eq("source")].groupby("file_id").size().sort_values().to_numpy()
        y = np.arange(1, len(counts)+1) / len(counts)
        ax.step(counts, y, where="post", label=REP_NAMES[representation],
                color="#0072B2" if representation == "equal_revolutions" else "#E69F00",
                linewidth=1.5)
    ax.set(xlabel="每文件完整窗口数（个）", ylabel="源文件经验累积比例", ylim=(0, 1.03))
    ax.legend(loc="lower right")
    save(fig, figure_dir / "raw_q2_windows_per_file", contracts,
         "两种表示产生不同数量的相关窗口", "116个源文件",
         "两张 Q1 候选窗口表",
         "经验累积分布按文件计算；1 秒表示的窗口数整体更少。")

    # 原始3：冻结尺度下的文件中位数；正常文件逐点可见。
    dev = source_frame[source_frame.split.isin(["train", "validation"])].copy()
    transformed = transform_features(dev, model.scaler)
    scaled = dev[["file_id", "label"]].copy()
    scaled[model.selected_features] = transformed
    med = scaled.groupby("file_id", sort=True).agg(
        {"label": "first", **{name: "median" for name in model.selected_features}})
    x_feature, y_feature = model.selected_features[:2]
    fig, axes = panels(height=3.4)
    ax = axes[0, 0]
    for label in LABELS:
        group = med[med.label.eq(label)]
        ax.scatter(group[x_feature], group[y_feature], s=25, alpha=.72,
                   color=COLORS[label], label=f"{label} (n={len(group)})",
                   marker={"N":"o","OR":"s","IR":"^","B":"D"}[label])
    ax.set(xlabel=f"{FEATURE_NAMES[x_feature]}（文件中位数，训练尺度）",
           ylabel=f"{FEATURE_NAMES[y_feature]}（文件中位数，训练尺度）")
    ax.legend(ncols=2, loc="best")
    save(fig, figure_dir / "raw_q2_feature_space", contracts,
         "选中特征存在类间结构，也保留重叠", "93个开发文件",
         "冻结主模型尺度下的文件特征中位数",
         "每点为一个开发文件；正常类只有3点，图中不绘制分布估计。")

    # 过程1：验证集表示选择。
    fig, axes = panels(height=3.2)
    ax = axes[0, 0]
    metric_specs = [("macro_f1", "宏 F1", "o"),
                    ("balanced_accuracy", "平衡准确率", "s"),
                    ("accuracy", "准确率", "^")]
    x = np.arange(2)
    ordered = validation_metrics.set_index("representation").loc[
        ["equal_revolutions", "one_second"]]
    for metric, label, marker in metric_specs:
        ax.plot(x, ordered[metric], marker=marker, label=label, linewidth=1.2)
        for position, value in zip(x, ordered[metric]):
            ax.text(position, value+.012, f"{value:.3f}", ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=["20 转", "1 秒"], ylabel="验证文件指标",
           xlabel="候选表示", ylim=(0.72, 1.0))
    ax.legend(loc="lower right")
    save(fig, figure_dir / "process_q2_representation_selection", contracts,
         "验证集选择 1 秒表示", "23个验证文件",
         "representation_validation_metrics.csv",
         "两条路线均由70个训练文件拟合；表示按宏F1、平衡准确率和正常召回依次选择。")

    # 过程2：主模型开发内折参数面。
    grid = final_cv[final_cv.model.eq(config["main_model"])].pivot(
        index="C", columns="gamma", values="macro_f1").sort_index()
    fig, axes = panels(height=3.6)
    ax = axes[0, 0]
    color_map = plt.get_cmap("cividis")
    normalizer = Normalize(vmin=0, vmax=1)
    for row in range(len(grid.index)):
        for col in range(len(grid.columns)):
            ax.add_patch(Rectangle((col, row), 1, 1,
                                   facecolor=color_map(normalizer(grid.iloc[row, col])),
                                   edgecolor="white", linewidth=.7))
    ax.set(xlim=(0, len(grid.columns)), ylim=(0, len(grid.index)))
    ax.set(xticks=np.arange(len(grid.columns))+.5,
           xticklabels=[f"{v:g}" for v in grid.columns],
           yticks=np.arange(len(grid.index))+.5,
           yticklabels=[f"{v:g}" for v in grid.index], xlabel="$\\gamma$", ylabel="$C$")
    for row in range(len(grid.index)):
        for col in range(len(grid.columns)):
            ax.text(col+.5, row+.5, f"{grid.iloc[row,col]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if grid.iloc[row,col] < .5 else "black")
    save(fig, figure_dir / "process_q2_parameter_surface", contracts,
         "主模型参数由93文件开发内折选择", "93个开发文件的文件分组二折",
         "final_cv_summary.csv",
         "每格为两折文件级宏F1平均；网格在运行前固定。")

    # 过程3：双重权重的类总和与文件总和。
    fig, axes = panels(cols=2, height=3.3)
    bars = axes[0, 0].bar(LABELS, by_class.reindex(LABELS),
                          color=[COLORS[x] for x in LABELS])
    axes[0, 0].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axes[0, 0].axhline(.25, color="#333333", linestyle="--", linewidth=.8)
    axes[0, 0].set(xlabel="类别", ylabel="类别总损失权重", ylim=(0, .30), title="类别等权")
    rng = np.random.default_rng(config["seed"])
    for position, label in enumerate(LABELS):
        values = by_file.loc[by_file.label.eq(label), "sample_weight"]
        axes[0, 1].scatter(position+rng.uniform(-.10,.10,len(values)), values,
                           color=COLORS[label], s=15, alpha=.7)
    axes[0, 1].set(xticks=range(4), xticklabels=LABELS, xlabel="类别",
                   ylabel="单个文件总损失权重", title="类内文件等权")
    plot_style.add_panel_labels(axes.flat)
    save(fig, figure_dir / "process_q2_weight_audit", contracts,
         "损失先平衡类别，再平衡类内文件", "93个开发文件及其窗口",
         "sample_weights_by_class.csv / sample_weights_by_file.csv",
         "左图每类总权重严格为0.25；右图同类文件总权重相同。")

    # 过程4：稳定特征选择。
    fig, axes = panels(height=3.9)
    ax = axes[0, 0]
    ordered_selection = selection.sort_values("selection_frequency")
    bars = ax.barh([FEATURE_NAMES[x] for x in ordered_selection.feature],
                   ordered_selection.selection_frequency,
                   color=["#0072B2" if keep else "#AAB2BD"
                          for keep in ordered_selection.selected])
    for bar, keep in zip(bars, ordered_selection.selected):
        if not keep:
            bar.set_hatch("//")
    ax.axvline(config["stability_threshold"], color="#E69F00", linestyle="--",
               label="预设阈值 0.6")
    ax.set(xlim=(0, 1.04), xlabel="分层文件重采样入选比例", ylabel="候选特征")
    ax.legend(loc="lower right")
    save(fig, figure_dir / "process_q2_feature_selection", contracts,
         "最终93文件模型保留7项稳定特征", "93个开发文件，100次文件重采样",
         "selected_features_final.csv",
         "蓝色为最终保留，灰色斜纹为未保留；稳定比例不是概率。")

    # 结果1：混淆矩阵，数字直接可读，不加冗余色条。
    fig, axes = panels(height=3.5)
    ax = axes[0, 0]
    values = confusion.loc[LABELS, LABELS].to_numpy()
    color_map = plt.get_cmap("Blues")
    normalizer = Normalize(vmin=0, vmax=max(1, values.max()))
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                   facecolor=color_map(normalizer(values[row,col])),
                                   edgecolor="white", linewidth=1.0))
            ax.text(col, row, str(values[row,col]), ha="center", va="center",
                    color="white" if values[row,col] > values.max()/2 else "#222222",
                    fontsize=9)
    ax.set(xticks=range(4), xticklabels=LABELS, yticks=range(4), yticklabels=LABELS,
           xlabel="预测类别", ylabel="真实类别", xlim=(-.5, 3.5), ylim=(3.5, -.5))
    save(fig, figure_dir / "result_q2_confusion", contracts,
         "主模型在测试集漏判唯一正常文件", "23个封存测试文件",
         "confusion_test_rbf_file_class_balanced.csv",
         "行为真实类别、列为预测类别；正常文件被判为B，OR另有2个判为B，IR有1个判为OR。")

    # 结果2：消融指标点图，避免少量估计用柱状隐藏差异。
    fig, axes = panels(height=4.0)
    ax = axes[0, 0]
    ordered_models = list(MODEL_NAMES)
    offsets = {"accuracy": -.18, "macro_f1": 0, "balanced_accuracy": .18}
    labels_metric = {"accuracy":"准确率", "macro_f1":"宏 F1",
                     "balanced_accuracy":"平衡准确率"}
    markers = {"accuracy":"o", "macro_f1":"s", "balanced_accuracy":"^"}
    model_positions = np.arange(len(ordered_models))
    table = ablation.set_index("model").loc[ordered_models]
    for metric, offset in offsets.items():
        ax.scatter(model_positions+offset, table[metric], label=labels_metric[metric],
                   marker=markers[metric], s=32)
    ax.set(xticks=model_positions, xticklabels=[MODEL_NAMES[x] for x in ordered_models],
           ylabel="测试文件指标", xlabel="冻结方案", ylim=(0.55, .92))
    ax.tick_params(axis="x", rotation=12)
    ax.legend(ncols=3, loc="upper center")
    save(fig, figure_dir / "result_q2_ablation", contracts,
         "线性对照的类别均衡指标高于预注册 RBF 主模型", "同一组23个测试文件",
         "ablation_test_metrics.csv",
         "四个方案均在93个开发文件内独立调参；测试不参与选择。三种RBF方案的文件预测相同。")

    # 结果3：逐文件间隔，错误与正确双编码。
    ordered_pred = predictions.sort_values(["true_label", "margin"],
                                           key=lambda s: s.map({k:i for i,k in enumerate(LABELS)})
                                           if s.name == "true_label" else s).reset_index(drop=True)
    correct = ordered_pred.prediction.eq(ordered_pred.true_label)
    fig, axes = panels(height=4.2)
    ax = axes[0, 0]
    y = np.arange(len(ordered_pred))
    ax.scatter(ordered_pred.margin, y, c=np.where(correct, "#009E73", "#D55E00"),
               marker="o", s=28)
    for index, row in ordered_pred.iterrows():
        ax.text(row.margin+.03, index, f"{row.true_label}→{row.prediction}", va="center", fontsize=6.5)
    ax.set(xlabel="文件诊断间隔（最大分数－次大分数）", ylabel="测试文件（按真实类别与间隔排序）",
           yticks=[])
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.text(.98, .04, "绿色：正确　橙色：错误", transform=ax.transAxes,
            ha="right", fontsize=7)
    save(fig, figure_dir / "result_q2_file_margins", contracts,
         "错误文件并非都具有小间隔", "23个封存测试文件",
         "source_test_file_scores.csv",
         "间隔是原生SVM分数差，不是正确概率；标签显示真实类别→预测类别。")

    # 结果4：增益与噪声扰动分开显示。
    fig, axes = panels(cols=2, height=3.3)
    for ax, perturbation, xlabel in [
        (axes[0,0], "amplitude_gain", "幅值增益"),
        (axes[0,1], "white_noise", "噪声相对 RMS")]:
        group = robustness[robustness.perturbation.eq(perturbation)].sort_values("level")
        ax.plot(group.level, group.macro_f1, color="#0072B2", marker="o", label="宏 F1")
        ax.plot(group.level, group.prediction_retention, color="#E69F00", marker="s",
                linestyle="--", label="预测保持率")
        ax.set(xlabel=xlabel, ylabel="文件级指标", ylim=(0, 1.04))
        ax.legend(loc="lower left")
    plot_style.add_panel_labels(axes.flat)
    save(fig, figure_dir / "result_q2_robustness", contracts,
         "模型耐受所测白噪声，但对二倍增益敏感", "23个封存测试文件",
         "robustness_perturbations.csv",
         "噪声回到原始信号重算；增益按问题一齐次关系重算特征。预测保持率相对未扰动主模型。")

    # 结果5：预注册主汇总与两项鲁棒对照。
    main_agg = aggregations[aggregations.model.eq(config["main_model"])].copy()
    order_agg = ["mean_score", "median_score", "majority_vote"]
    main_agg = main_agg.set_index("aggregation").loc[order_agg]
    fig, axes = panels(height=3.2)
    ax = axes[0, 0]
    positions = np.arange(3)
    for metric, offset, marker, label in [
        ("macro_f1", -.12, "s", "宏 F1"),
        ("balanced_accuracy", .12, "^", "平衡准确率")]:
        ax.scatter(positions+offset, main_agg[metric], s=38, marker=marker, label=label)
    ax.set(xticks=positions, xticklabels=["窗口分数均值\n（主规则）", "窗口分数中位数", "窗口多数投票"],
           xlabel="文件汇总规则", ylabel="测试文件指标", ylim=(0.55, .98))
    ax.legend(loc="upper left")
    save(fig, figure_dir / "result_q2_aggregation", contracts,
         "多数投票在测试集改善了正常类，但只作事后对照", "23个封存测试文件",
         "aggregation_robustness.csv",
         "主报告保持预注册的窗口分数均值；看过测试结果后不改成多数投票。")

    make_flows(figure_dir, contracts, selected_representation, model.selected_features)
    write_json(result_dir / "figure_contracts.json", contracts)
    lines = ["# 问题二图表清单", "", "| 图文件 | 类别 | 核心结论 | 统计单位 |", "|---|---|---|---|"]
    for item in contracts:
        category = "流程" if item["figure"].startswith("flow_") else item["figure"].split("_")[0]
        lines.append(f"| `{item['figure']}` | {category} | {item['goal']} | {item['unit_of_analysis']} |")
    (result_dir / "图表清单.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    items = []
    for item in contracts:
        png = Path(os.path.relpath(figure_dir / f"{item['figure']}.png", result_dir)).as_posix()
        items.append(f"<figure><img src='{html.escape(png)}'><figcaption>"
                     f"{html.escape(item['figure'])}：{html.escape(item['caption'])}"
                     "</figcaption></figure>")
    document = """<!doctype html><meta charset='utf-8'><title>问题二图表面板</title>
    <style>body{font-family:Arial,'PingFang SC',sans-serif;margin:24px;color:#222}
    main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
    figure{margin:0;border:1px solid #ddd;padding:12px;background:#fff}img{width:100%}
    figcaption{font-size:13px;line-height:1.5;margin-top:8px}</style><h1>问题二图表面板</h1><main>"""
    (result_dir / "图表面板.html").write_text(document+"".join(items)+"</main>", encoding="utf-8")
    return contracts
