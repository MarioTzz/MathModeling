"""问题三数据、计算过程与无标签结果图；矢量 SVG + 300 DPI PNG。"""
from __future__ import annotations

import html
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from utils.figure_export import export_figure
from utils import plot_style
from utils.visual_qa import audit_layout
from utils.q1_model import write_json
from utils.q3_transfer import CLASS_ORDER


COLORS = {"N": "#0072B2", "OR": "#E69F00", "IR": "#009E73", "B": "#CC79A7"}
INK = "#263238"


def _new(height: float = 3.5):
    return plt.subplots(figsize=(7.2, height), layout="constrained")


def _finish(fig, stem: Path, contracts: list[dict], goal: str,
            unit: str, source: str, caption: str):
    problems = [item for item in audit_layout(fig) if item[0] in {"WARN", "FAIL"}]
    design = plot_style.audit_design(fig)
    legacy = plot_style.audit_layout(fig)
    if problems or design or legacy:
        raise ValueError(f"{stem.name} 排版预检失败: {problems}; {design}; {legacy}")
    export_figure(fig, str(stem), formats=["svg", "png"], dpi=300,
                  size_inches=tuple(fig.get_size_inches()), tight=False)
    plot_style._save_grayscale_preview(stem.with_suffix(".png"), 300)
    contracts.append({"figure": stem.name, "question": "q3", "goal": goal,
                      "unit_of_analysis": unit, "evidence": source,
                      "caption": caption, "dpi": 300})
    plt.close(fig)


def _flow_box(ax, x, y, label, color, width=.27, height=.19):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                 boxstyle="round,pad=.015,rounding_size=.025",
                 facecolor=color+"1A", edgecolor=color, linewidth=1.3))
    ax.text(x, y, label, ha="center", va="center", fontsize=9, color=INK)


def _flow_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.2, color="#697780"))


def _flow_figure():
    fig, ax = _new(3.4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    return fig, ax


def make_q3_figures(figure_dir: Path, result_dir: Path, files: pd.DataFrame,
                    proxy: pd.DataFrame, diagnostics: pd.DataFrame,
                    target: pd.DataFrame, summary: dict) -> None:
    plot_style.apply_publication_style(language="en", width="double")
    figure_dir.mkdir(parents=True, exist_ok=True)
    contracts: list[dict] = []
    source = files[files.domain.eq("source")]
    tgt = files[files.domain.eq("target")]
    letters = [Path(value).stem for value in target.file_id]

    # 原始数据图：文件是独立统计单位，窗口与目标文件编号只说明数据结构。
    fig, ax = _new()
    counts = [len(source[source.split.eq(part)]) for part in ["train", "validation", "test"]]
    counts.append(len(tgt))
    bars = ax.bar(["Train", "Validation", "Test", "Target"], counts,
                  color=["#0072B2", "#6F91AD", "#A2B7C8", "#CC79A7"])
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Files", title="File-level data partition", ylim=(0, max(counts)*1.18))
    _finish(fig, figure_dir/"raw_q3_files", contracts, "源域划分与目标域规模",
            "MAT 文件", "Q1 files.csv", "源域 70/23/23；目标域 16 文件无真值。")

    fig, ax = _new()
    for index, (domain, color) in enumerate([("source", "#0072B2"), ("target", "#CC79A7")]):
        vals = files[files.domain.eq(domain)].n_windows.to_numpy()
        ax.scatter(np.full(len(vals), index)+np.linspace(-.12, .12, len(vals)), vals,
                   s=15, alpha=.7, color=color, label=domain)
    ax.set(xticks=[0, 1], xticklabels=["Source", "Target"],
           ylabel="20-revolution windows per file", title="Window coverage by file",
           ylim=(0, max(files.n_windows)*1.13))
    _finish(fig, figure_dir/"raw_q3_windows", contracts, "窗口覆盖与文件聚合依据",
            "MAT 文件", "Q1 window_features_all.csv", "目标域每文件 3 个完整窗口。")

    fig, ax = _new()
    for domain, color in [("source", "#0072B2"), ("target", "#CC79A7")]:
        group = files[files.domain.eq(domain)]
        ax.scatter(group.crest_factor, group.env_entropy, s=22, alpha=.7,
                   color=color, label=f"{domain} (n={len(group)})")
    ax.set(xlabel="Crest factor (unitless)", ylabel="Envelope entropy (unitless)",
           title="Two shared shape features")
    ax.legend()
    _finish(fig, figure_dir/"raw_q3_shared_features", contracts,
            "源目标共享特征的原始位置", "MAT 文件",
            "Q1 file median features", "二维散点用于显示分布位置，不代表类别可分性。")

    # 过程图：代理任务、对齐距离和特征位置变化均按实际计算值作图。
    fig, ax = _new()
    for direction, color, label in [("12_to_48", "#0072B2", "12→48 kHz"),
                                    ("48_to_12", "#E69F00", "48→12 kHz")]:
        part = proxy[proxy.direction.eq(direction)].sort_values("alpha")
        ax.plot(part.alpha, part.macro_f1, marker="o", color=color, label=label)
    ax.axvline(summary["alpha_selected"], color="#555555", linestyle="--", linewidth=.9)
    ax.set(xlabel="Partial alignment strength α", ylabel="Macro F1",
           title="Proxy-domain F1", xticks=sorted(proxy.alpha.unique()),
           ylim=(0, 1))
    ax.legend()
    _finish(fig, figure_dir/"process_q3_proxy", contracts, "双向代理任务选 α",
            "保留源域文件", "proxy_metrics.csv", "仅覆盖 OR、IR、B 三类故障。")

    fig, ax = _new()
    wide = diagnostics.pivot(index="block", columns="stage", values="cov_distance_frobenius")
    wide = wide.reindex(["impulse_shape", "envelope_order"])
    x = np.arange(len(wide))
    ax.bar(x-.18, wide["before"], .36, color="#0072B2", label="Before")
    ax.bar(x+.18, wide["after"], .36, color="#CC79A7", label="After")
    ax.set(xticks=x, xticklabels=["Impulse shape", "Envelope order"],
           ylabel="Covariance distance (Frobenius)",
           title="Covariance gap")
    ax.legend()
    _finish(fig, figure_dir/"process_q3_covariance", contracts,
            "协方差对齐前后距离", "特征块", "alignment_diagnostics.csv",
            "距离变化只是迁移过程诊断，并非目标分类准确率。")

    fig, ax = _new()
    wide = diagnostics.pivot(index="block", columns="stage", values="mean_distance_l2")
    wide = wide.reindex(["impulse_shape", "envelope_order"])
    x = np.arange(len(wide))
    ax.bar(x-.18, wide["before"], .36, color="#0072B2", label="Before")
    ax.bar(x+.18, wide["after"], .36, color="#009E73", label="After")
    ax.set(xticks=x, xticklabels=["Impulse shape", "Envelope order"],
           ylabel="Mean distance (L2)", title="Mean gap")
    ax.legend()
    _finish(fig, figure_dir/"process_q3_mean", contracts,
            "均值对齐前后距离", "特征块", "alignment_diagnostics.csv",
            "仅依据 93 个开发源文件和 16 个无标签目标文件计算。")

    # 结果图：投票分数、决策间隔、扰动一致性；不标示不存在的目标真值。
    fig, ax = _new(5.0)
    matrix = target[[f"score_{c}" for c in CLASS_ORDER]].to_numpy()
    levels = np.linspace(0, 1, 11)
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap("Blues")(np.linspace(.08, .94, 10)))
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    mesh = ax.pcolormesh(np.arange(5), np.arange(17), matrix, cmap=cmap,
                         norm=norm, shading="flat")
    ax.set(xticks=np.arange(4)+.5, xticklabels=CLASS_ORDER,
           yticks=np.arange(16)+.5, yticklabels=letters,
           ylim=(16, 0), xlabel="Predicted class score",
           ylabel="Target file", title="Target tree-vote scores")
    fig.colorbar(mesh, ax=ax, label="Vote fraction", boundaries=levels)
    _finish(fig, figure_dir/"result_q3_scores", contracts, "16文件四类模型分数",
            "目标 MAT 文件", "target_file_labels.csv", "分数未经概率校准，颜色不能当成真实类别概率。")

    fig, ax = _new(4.2)
    x = np.arange(len(target))
    ax.bar(x, target.margin, color=[COLORS[c] for c in target.prediction])
    ax.set(xticks=x, xticklabels=letters, ylabel="Top-two vote gap",
           xlabel="Target file", title="Decision margins")
    _finish(fig, figure_dir/"result_q3_margins", contracts, "逐文件前两类票差",
            "目标 MAT 文件", "target_file_labels.csv", "票差越小，两个最高分数越接近；不是误差概率。")

    fig, ax = _new(4.2)
    categories = ["No alignment", "Leave-one-out", "540 rpm", "660 rpm"]
    agree = np.column_stack([
        target.branch_agreement.astype(int), target.loo_agreement.astype(int),
        target.prediction.eq(target.rpm_540_prediction).astype(int),
        target.prediction.eq(target.rpm_660_prediction).astype(int)])
    mesh = ax.pcolormesh(np.arange(5), np.arange(17), agree,
                         cmap=matplotlib.colors.ListedColormap(["#E69F00", "#009E73"]),
                         vmin=0, vmax=1, shading="flat")
    ax.set(xticks=np.arange(4)+.5, xticklabels=categories,
           yticks=np.arange(16)+.5, yticklabels=letters,
           ylim=(16, 0), ylabel="Target file",
           title="Prediction stability")
    ax.tick_params(axis="x", labelrotation=15)
    fig.colorbar(mesh, ax=ax, ticks=[0, 1], label="0 changed / 1 same")
    _finish(fig, figure_dir/"result_q3_stability", contracts,
            "分支与转速扰动稳定性", "目标 MAT 文件", "target_file_labels.csv",
            "橙色表示与基准预测不一致；8 文件需人工复核。")

    fig, ax = _flow_figure()
    _flow_box(ax, .17, .55, "Q1\nshared features", "#0072B2")
    _flow_box(ax, .50, .55, "Q2\nsource classifier", "#009E73")
    _flow_box(ax, .83, .55, "Q3\ntransfer labels", "#CC79A7")
    _flow_arrow(ax, (.31,.55), (.35,.55)); _flow_arrow(ax, (.64,.55), (.68,.55))
    ax.text(.5, .15, "Q4 decision explanation awaits confirmation of Q3",
            ha="center", fontsize=9, color="#53616A")
    _finish(fig, figure_dir/"flow_overall_model", contracts,
            "四问递进关系", "建模阶段", "题目要求与现有模型",
            "第三问使用共享无量纲表征，第四问尚未执行。")

    fig, ax = _flow_figure()
    nodes = [(.13, "20-revolution\nfile medians", "#0072B2"),
             (.37, "Source-only\nrobust scale", "#009E73"),
             (.62, "Two-block\npartial CORAL", "#E69F00"),
             (.87, "500-tree\nExtraTrees", "#CC79A7")]
    for xpos, label, color in nodes:
        _flow_box(ax, xpos, .63, label, color, width=.20)
    for xpos in [.235,.49,.74]:
        _flow_arrow(ax, (xpos,.63), (xpos+.02,.63))
    ax.text(.5, .20, "Proxy selection → fixed source test → A–P labels + stability checks",
            ha="center", fontsize=8.6, color="#53616A")
    _finish(fig, figure_dir/"flow_q3_model", contracts,
            "问题三可执行求解流程", "MAT 文件", "问题三_求解.py",
            "缩放仅由源域开发集拟合，目标域不提供监督标签。")

    write_json(result_dir/"figure_contracts.json", contracts)
    lines = ["# 问题三图表清单", "", "| 图件 | 论证目标 | 统计单位 |",
             "|---|---|---|"]
    for row in contracts:
        lines.append(f"| `{row['figure']}` | {row['goal']} | {row['unit_of_analysis']} |")
    (result_dir/"图表清单.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cards = []
    for row in contracts:
        relative = Path(os.path.relpath(figure_dir/(row["figure"]+".png"),
                                        result_dir)).as_posix()
        cards.append("<figure><img src='"+html.escape(relative)+"'><figcaption>"+
                     html.escape(row["caption"])+"</figcaption></figure>")
    (result_dir/"图表面板.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>问题三图表</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#263238}"
        "main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}"
        "figure{margin:0;border:1px solid #ddd;padding:12px;background:white}"
        "img{width:100%}figcaption{font-size:13px;line-height:1.5;margin-top:8px}</style>"
        "<h1>问题三：无标签迁移诊断</h1><main>"+"".join(cards)+"</main>", encoding="utf-8")
