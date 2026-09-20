"""问题一图表：所有数据来自实算；SVG、300 DPI PNG、灰度及语义清单。"""
from __future__ import annotations
import html
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from . import plot_style
from .figure_export import export_figure
from .visual_qa import audit_layout
from .q1_model import (LABELS, FEATURES, FEATURE_ZH, prepare_record, order_spectrum,
                       file_medians, save_csv, write_json)

COLORS = {"N": "#0072B2", "OR": "#E69F00", "IR": "#009E73", "B": "#CC79A7",
          "target": "#6B7280"}
NAMES = dict(zip(FEATURES, FEATURE_ZH))
MARKERS = {"N": "o", "OR": "s", "IR": "^", "B": "D", "target": "x"}


def save(fig, stem, contracts, goal, unit, evidence, caption):
    """专用审计器+样式审计，实际尺寸导出，灰度不覆盖彩图。"""
    issues = audit_layout(fig)
    significant = [(level, message) for level, message in issues if level in ["WARN", "FAIL"]]
    design = plot_style.audit_design(fig)
    legacy = plot_style.audit_layout(fig)
    if significant or design or legacy:
        raise ValueError(f"{stem.name} 图表预检失败: {significant}; {design}; {legacy}")
    export_figure(fig, str(stem), formats=["svg", "png"], dpi=300,
                  size_inches=tuple(fig.get_size_inches()), tight=False, grayscale_preview=False)
    plot_style._save_grayscale_preview(stem.with_suffix(".png"), 300)
    contracts.append({"figure": stem.name, "question": "q1", "goal": goal,
                      "unit_of_analysis": unit, "evidence": evidence,
                      "caption": caption, "width_inches": float(fig.get_size_inches()[0]),
                      "height_inches": float(fig.get_size_inches()[1]), "dpi": 300,
                      "layout_issues": significant, "design_issues": design})
    plt.close(fig)


def panels(rows=1, cols=1, height=3.2):
    return plt.subplots(rows, cols, figsize=(7.2, height), layout="constrained", squeeze=False)


def box(ax, x, y, width, height, text, color, faded=False):
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                          boxstyle="round,pad=0.012,rounding_size=0.016",
                          facecolor=color + ("10" if faded else "20"),
                          edgecolor=color, linewidth=1.1, linestyle="--" if faded else "-")
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#222222", linespacing=1.55)


def arrow(ax, start, end, color="#667085", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.1, color=color,
                                linestyle="--" if dashed else "-"))


def flows(directory, contracts, source_count, target_count, training_count, feature_count):
    fig, axes = panels(height=4.35)
    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    box(ax, .5, .88, .62, .13, "问题一 · 共用特征表征\n等转数包络 + 文件稳定筛选（本次完成）", "#0072B2")
    box(ax, .5, .66, .62, .13, "问题二 · 源域诊断\n文件与类别平衡 RBF-SVM（待确认）", "#009E73", True)
    arrow(ax, (.5, .80), (.5, .74))
    box(ax, .25, .42, .41, .15, "问题三 · 推荐主线\n机理分块部分 CORAL\n待确认后实施", "#0072B2", True)
    box(ax, .75, .42, .41, .15, "问题三 · 优先冲奖对照\n机制分组非平衡最优传输\n待确认后实施", "#E69F00", True)
    arrow(ax, (.38, .58), (.25, .51), dashed=True)
    arrow(ax, (.62, .58), (.75, .51), dashed=True)
    box(ax, .5, .18, .79, .13, "问题四 · 迁移诊断解释\n推荐主线：机理约束积分梯度；运输对照：质量与成本分解（待确认）", "#CC79A7", True)
    arrow(ax, (.25, .33), (.37, .255), dashed=True)
    arrow(ax, (.75, .33), (.63, .255), dashed=True)
    ax.text(.5, .045, "每一问完成并提交 GitHub 后，等待用户确认再进入下一问。",
            ha="center", va="center", fontsize=8, color="#555555")
    save(fig, directory / "flow_overall_model", contracts, "已完成与计划阶段的真实边界",
         "研究阶段", "本次授权与创新建模方案",
         "实线框为本次问题一；虚线框为后续计划。两条路线共享问题一输入，本阶段不执行分类或运输。")
    fig, axes = panels(height=4.55)
    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.axis("off")
    box(ax, .25, .88, .40, .11, f"源域核心 {source_count} 文件\n继承原文件划分", "#0072B2")
    box(ax, .75, .88, .40, .11, f"目标域 {target_count} 文件\n标签未知 · 近似转速", "#6B7280")
    box(ax, .5, .69, .84, .105, "原采样率共振解调 → 2 kHz 包络 → 等转数切窗与阶次谱", "#009E73")
    arrow(ax, (.25, .81), (.35, .75))
    arrow(ax, (.75, .81), (.65, .75))
    box(ax, .24, .45, .39, .18,
        f"仅源训练 {training_count} 文件拟合\n二折选择频带与转数\n文件重采样筛选稳定特征", "#E69F00")
    box(ax, .76, .45, .39, .18,
        f"冻结参数后变换全部文件\n12 个候选量 → {feature_count} 个选中量\n使用训练文件等权尺度", "#0072B2")
    arrow(ax, (.3, .63), (.24, .555))
    arrow(ax, (.452, .45), (.548, .45))
    box(ax, .24, .19, .39, .115, "窗口级特征出口\n推荐诊断主线", "#0072B2")
    box(ax, .76, .19, .39, .115, "文件中位数与四分位距\n最优传输对照输入", "#CC79A7")
    arrow(ax, (.67, .345), (.33, .26))
    arrow(ax, (.76, .345), (.76, .26))
    ax.text(.5, .045, "一秒基线同步导出  ·  转速 / 裁边 / 增益检验  ·  保存参数与输入哈希",
            ha="center", va="center", fontsize=8, color="#555555")
    save(fig, directory / "flow_q1_model", contracts, "问题一公式到实际数据出口",
         "文件与相关窗口", "问题一_求解.py / utils/q1_model.py",
         "参数拟合与数据变换分开；原文件划分先于切窗。基线与主线冻结同一频带和特征集合。")


def make_figures(records, metadata, raw, scaled, selection, scores, sensitivity,
                 band, revolutions, envelopes, selected, config, directory, result_dir):
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei",
                 "PingFang SC", "Lantinghei SC", "Heiti TC", "STHeiti", "PingFang HK"]
    font = next((name for name in preferred if name in available), None)
    if font is None:
        raise RuntimeError("正式中文图需要中文字体，请安装 Noto Sans CJK SC 后重试")
    style = plot_style.apply_publication_style(language="en", width="double")
    plt.rcParams["font.sans-serif"] = [font, "Arial", "DejaVu Sans"]
    style["font"] = font
    contracts = []
    source = metadata[metadata.domain.eq("source")]
    train_meta = metadata[metadata.split.eq("train")]
    rng = np.random.default_rng(config["seed"])

    # 原始1：数量是计数，采用零起点柱形；不将窗口重复视为样本。
    fig, axes = panels(height=3.1)
    ax = axes[0, 0]
    positions = np.arange(4)
    for index, (split, title, color, hatch) in enumerate([
        ("train", "训练", "#0072B2", ""), ("validation", "验证", "#E69F00", "//"),
        ("test", "测试", "#009E73", "..")]):
        counts = source[source.split.eq(split)].label.value_counts().reindex(LABELS, fill_value=0)
        bars = ax.bar(positions + (index - 1) * .24, counts, width=.22,
                      label=title, color=color, hatch=hatch)
        ax.bar_label(bars, fontsize=7, padding=2)
    ax.set(xticks=positions, xticklabels=LABELS, xlabel="源域工作状态",
           ylabel="独立原文件数（个）", ylim=(0, 40))
    ax.legend(ncols=3, loc="upper right")
    save(fig, directory / "raw_q1_class_split", contracts, "正常状态独立文件稀缺",
         "116个源文件", "files.csv / split / label",
         "每柱是独立原文件数，核心源域训练/验证/测试为70/23/23；正常类为2/1/1。")

    # 原始2：真实采样条件，类别用形状和颜色双重编码。
    fig, axes = panels(cols=2, height=3.0)
    for label in LABELS + ["target"]:
        group = metadata[metadata.domain.eq("target") if label == "target" else
                         (metadata.domain.eq("source") & metadata.label.eq(label))]
        for ax, xcol, xlabel in [
            (axes[0, 0], "duration_seconds", "采集时长（s）"),
            (axes[0, 1], "nominal_fs_hz", "名义采样率（kHz）")]:
            x = group[xcol].to_numpy() / (1000 if xcol == "nominal_fs_hz" else 1)
            ax.scatter(x, group.rpm, s=20, color=COLORS[label], marker=MARKERS[label],
                       alpha=.65, label="目标域" if label == "target" else label)
            ax.set(xlabel=xlabel, ylabel="平均转速（rpm）")
    axes[0, 0].legend(loc="center right", fontsize=7)
    axes[0, 1].set_xticks([12, 32, 48])
    plot_style.add_panel_labels(axes.flat)
    save(fig, directory / "raw_q1_acquisition", contracts, "源目标转速与采样条件不同",
         "132个文件，重叠点未增加抖动", "files.csv",
         "目标转速600 rpm来自题面近似值。点可重叠；右图仅显示主通道名义采样率，不推断其他通道的时钟。")

    representatives = [next(record for record in records
                            if record.metadata["split"] == "train"
                            and record.metadata["label"] == label) for label in LABELS]
    representatives += [next(record for record in records if record.metadata["domain"] == "target")]
    rep_rows = []
    fig, axes = panels(rows=5, height=6.0)
    for ax, record in zip(axes.flat, representatives):
        start = round(.1 * record.fs)
        end = round(.3 * record.fs)
        label = record.metadata["label"] or "target"
        ax.plot(np.arange(start, end) / record.fs, record.centered[start:end],
                color=COLORS[label], linewidth=.65)
        file_name = record.file_id.split("/")[-1]
        ax.set_title(f"{label if label != 'target' else '目标'} · {file_name}", fontsize=8)
        ax.set(xlabel="时间（s）", ylabel="原始尺度振动")
        rep_rows.append({"file_id": record.file_id, "label": record.metadata["label"],
                         "rule": "first_training_file_per_class_or_first_target"})
    plot_style.add_panel_labels(axes.flat)
    save_csv(pd.DataFrame(rep_rows), result_dir / "representative_files.csv")
    save(fig, directory / "raw_q1_waveforms", contracts, "代表波形的冲击形状和幅值",
         "每类首个训练文件及首个目标文件", "representative_files.csv / 原始MAT",
         "各文件截取0.1—0.3秒；子图纵轴独立，未做幅值归一。单位未知，不标为g。")

    # 过程1：实际计算的候选留出分数；热力块保持矢量。
    matrix = scores.pivot(index="variant", columns="fold", values="score").reindex(scores.variant.drop_duplicates())
    fig, axes = panels(height=3.3)
    ax = axes[0, 0]
    mesh = ax.pcolormesh(np.arange(3), np.arange(len(matrix) + 1), matrix.values,
                        cmap="cividis", shading="flat")
    ax.set(xticks=[.5, 1.5], xticklabels=["内折 1", "内折 2"],
           yticks=np.arange(len(matrix)) + .5,
           yticklabels=[name.replace("Hz_R", " Hz / ") + " 转" for name in matrix.index],
           xlabel="仅源训练文件的内折留出", ylabel="候选频带 / 转数")
    ax.invert_yaxis()
    colorbar = fig.colorbar(mesh, ax=ax, label="平均 log(1 + Fisher 分数)")
    colorbar.solids.set_rasterized(False)
    mid = (matrix.values.min() + matrix.values.max()) / 2
    for row in range(len(matrix)):
        for col in range(2):
            ax.text(col + .5, row + .5, f"{matrix.iloc[row, col]:.3f}",
                    ha="center", va="center", color="white" if matrix.iloc[row, col] < mid else "black")
    save(fig, directory / "process_q1_candidate_scores", contracts, "频带和转数由训练内统计选定",
         "70个源训练文件二折", "candidate_scores.csv / training_inner_folds.csv",
         "每格为当前留出折上所选特征的平均log(1+Fisher分数)。正常类每折只有1个文件；分数不是准确率。")

    # 过程2：首个外圈训练文件的实际信号处理路径。
    record = representatives[1]
    envelope, carrier, native_envelope = prepare_record(record, band, config, return_carrier=True)
    fig, axes = panels(rows=3, height=5.2)
    take = (np.arange(len(carrier)) / record.fs >= .1) & (np.arange(len(carrier)) / record.fs <= .2)
    native_time = np.arange(len(carrier))[take] / record.fs
    axes[0, 0].plot(native_time, carrier[take], color="#0072B2", lw=.7)
    axes[0, 0].set(xlabel="时间（s）", ylabel="带通振动（原始尺度）", title="共振频带")
    axes[1, 0].plot(native_time, native_envelope[take], color="#E69F00", lw=1)
    axes[1, 0].set(xlabel="时间（s）", ylabel="包络（原始尺度）", title="希尔伯特包络")
    guard = math_ceil(config["edge_guard_seconds"] * config["envelope_fs_hz"])
    length = round(config["envelope_fs_hz"] * revolutions / record.rotation_hz)
    orders, mass = order_spectrum(envelope[guard:guard + length], record.rotation_hz, config)
    axes[2, 0].plot(orders, mass, color="#009E73")
    bpfo = 9 / 2 * (1 - .3126 / 1.537)
    axes[2, 0].axvline(bpfo, color="#CC79A7", linestyle="--", label="源域 BPFO 参照")
    axes[2, 0].set(xlabel="阶次（频率 / 转频）", ylabel="归一化谱质量", title="包络阶次谱")
    axes[2, 0].legend(loc="upper right")
    plot_style.add_panel_labels(axes.flat)
    save(fig, directory / "process_q1_demodulation", contracts, "先解调再降采样和换算阶次",
         record.file_id, "原始MAT / prepare_record / order_spectrum",
         f"上两图为0.1—0.2秒，同一外圈训练文件；下图为首个完整{revolutions}转窗。虚线只用于SKF6205源域机理参照。")

    # 过程3：稳定性为文件重采样频率，直接展示全部12项。
    fig, axes = panels(height=3.9)
    ax = axes[0, 0]
    ordered = selection.sort_values("selection_frequency")
    bars = ax.barh([NAMES[x] for x in ordered.feature], ordered.selection_frequency,
                   color=["#0072B2" if keep else "#AAB2BD" for keep in ordered.selected])
    for bar, keep in zip(bars, ordered.selected):
        if not keep:
            bar.set_hatch("//")
    ax.axvline(config["stability_threshold"], color="#E69F00", linestyle="--", label="预设入选率阈值")
    ax.set(xlim=(0, 1.04), xlabel="分层文件重采样入选比例", ylabel="候选特征")
    ax.legend(loc="lower right")
    save(fig, directory / "process_q1_stability", contracts, "特征需要稳定且不高度冗余",
         "70个源训练文件，100次文件重采样", "feature_selection.csv",
         "蓝色为最终保留，灰色斜纹为未保留；最终还需满足Spearman绝对相关≤0.9。比例不是置信概率。")

    # 结果1：全部选中特征，以原文件中位数作点；不画小样本均值柱。
    medians = scaled.groupby("file_id")[selected].median()
    meta = metadata.set_index("file_id")
    training = medians.loc[train_meta.file_id]
    ncols, nrows = 2, int(np.ceil(len(selected) / 2))
    fig, axes = panels(nrows, ncols, height=max(3, 1.8 * nrows))
    for ax, feature in zip(axes.flat, selected):
        for index, label in enumerate(LABELS):
            ids = train_meta[train_meta.label.eq(label)].file_id
            values = training.loc[ids, feature]
            ax.scatter(index + rng.uniform(-.12, .12, len(values)), values,
                       color=COLORS[label], marker=MARKERS[label], s=11, alpha=.72)
            ax.plot([index - .2, index + .2], [values.median()] * 2, color="#222222", lw=1.2)
        counts = train_meta.label.value_counts()
        ax.set(xticks=range(4), xticklabels=[f"{c}\nn={counts[c]}" for c in LABELS],
               ylabel="标准化文件中位数", title=NAMES[feature])
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    plot_style.add_panel_labels(list(axes.flat)[:len(selected)])
    save(fig, directory / "result_q1_feature_distributions", contracts, "各类特征分布及正常类样本限制",
         "70个源训练文件，每点1文件", "window_features_selected_scaled.csv",
         "点为文件内窗口的特征中位数，短横线为各类别文件中位数；横向抖动仅帮助阅读。N仅2文件，不估计其总体区间。")

    # 结果2：显示选中特征全部相关，连续配色有明确colorbar。
    corr = training.corr(method="spearman")
    save_csv(corr.rename_axis("feature").reset_index(), result_dir / "selected_feature_correlations.csv")
    fig, axes = panels(height=4.25)
    ax = axes[0, 0]
    size = len(selected)
    mesh = ax.pcolormesh(np.arange(size + 1), np.arange(size + 1), corr.values,
                        cmap="RdBu_r", vmin=-1, vmax=1, shading="flat")
    short = [NAMES[x] for x in selected]
    ax.set(xticks=np.arange(size) + .5, xticklabels=short,
           yticks=np.arange(size) + .5, yticklabels=short)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.invert_yaxis()
    colorbar = fig.colorbar(mesh, ax=ax, label="Spearman 相关系数")
    colorbar.solids.set_rasterized(False)
    for row in range(size):
        for col in range(size):
            value = corr.iloc[row, col]
            ax.text(col + .5, row + .5, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > .65 else "#222222", fontsize=7)
    save(fig, directory / "result_q1_correlations", contracts, "保留特征的文件级冗余程度",
         "70个源训练文件中位数", "selected_feature_correlations.csv",
         "只在源训练文件上计算；对角线为1。相关性小并不自动说明存在分类增益。")

    # 结果3：压力变量连续，用线+不同标记；区间明确为文件IQR。
    fig, axes = panels(height=3.2)
    ax = axes[0, 0]
    for route, name, color, marker in [
        ("equal_revolutions", f"等转数（{revolutions}转）", "#0072B2", "o"),
        ("one_second", "固定一秒", "#E69F00", "s")]:
        group = sensitivity[sensitivity.route.eq(route)].groupby("rpm_change").common_scale_drift
        median = group.median()
        low, high = group.quantile(.25), group.quantile(.75)
        x = median.index.to_numpy() * 100
        ax.plot(x, median, marker=marker, color=color, label=name)
        ax.fill_between(x, low, high, color=color, alpha=.16)
    ax.set(xlabel="目标平均转速相对扰动（%）", ylabel="统一训练尺度下的特征漂移",
           ylim=(0, None), xticks=[-10, -5, 0, 5, 10])
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncols=2)
    save(fig, directory / "result_q1_rpm_sensitivity", contracts, "两种窗口对转速近似误差的响应",
         "16个目标文件，每个情景16个文件漂移", "rpm_sensitivity.csv / common_scale_drift",
         "线为16文件中位数，阴影为25%—75%分位区间；两方案用相同主线尺度比较。0扰动为自身基准；漂移不是诊断错误率。")

    # 额外源域机理证据：四类代表完整窗口的谱质量平均，保留真实峰偏差。
    fig, axes = panels(rows=4, height=5.7)
    mechanism_rows = []
    theoretical = {"OR": bpfo, "IR": 9 / 2 * (1 + .3126 / 1.537),
                   "B": 1.537 / .3126 * (1 - (.3126 / 1.537) ** 2)}
    for ax, record in zip(axes.flat, representatives[:4]):
        label = record.metadata["label"]
        envelope = envelopes[record.file_id]
        length = round(config["envelope_fs_hz"] * revolutions / record.rotation_hz)
        count = (len(envelope) - 2 * guard) // length
        masses = []
        for index in range(count):
            start = guard + index * length
            orders, mass = order_spectrum(envelope[start:start + length], record.rotation_hz, config)
            masses.append(mass)
        mean_mass = np.mean(masses, axis=0)
        ax.plot(orders, mean_mass, color=COLORS[label], lw=1)
        expected = theoretical.get(label)
        if expected is not None:
            ax.axvline(expected, color="#555555", ls="--", label="题面理论参照")
            ax.legend(loc="upper right")
        search = (orders >= 1) & (orders <= 10)
        observed = orders[search][np.argmax(mean_mass[search])]
        mechanism_rows.append({"file_id": record.file_id, "label": label,
                               "windows": count, "dominant_order_1_to_10": observed,
                               "theoretical_order": expected,
                               "absolute_difference": None if expected is None else abs(observed - expected)})
        ax.set(xlabel="阶次（频率 / 转频）", ylabel="平均归一化谱质量",
               title=f"{label} · {count} 个完整窗")
    plot_style.add_panel_labels(axes.flat)
    save_csv(pd.DataFrame(mechanism_rows), result_dir / "source_mechanism_checks.csv")
    save(fig, directory / "result_q1_source_mechanism", contracts, "源域代表谱峰与题面机理的关系",
         "每类首个训练文件，完整窗谱质量平均", "source_mechanism_checks.csv / representative_files.csv",
         "各窗先单位质量归一，再取文件内平均。虚线是题面故障基阶，不要求最大峰必然出现在该位置；正常类不画故障线。")
    flows(directory, contracts, len(source), int(metadata.domain.eq("target").sum()),
          len(train_meta), len(selected))
    write_json(result_dir / "figure_contracts.json", {"style": style, "figures": contracts})
    captions = "# 问题一图表清单\n\n每图均保存 SVG、300 DPI PNG及独立灰度预览。SVG文字可编辑。\n\n"
    html_items = []
    relative_figures = os.path.relpath(directory, result_dir).replace(os.sep, "/")
    for item in contracts:
        captions += f"## {item['figure']}\n\n{item['caption']}\n\n![{item['goal']}]({relative_figures}/{item['figure']}.png)\n\n"
        html_items.append(f"<article><h2>{html.escape(item['goal'])}</h2>"
                          f"<img src='{item['figure']}.svg'><p>{html.escape(item['caption'])}</p></article>")
    (result_dir / "图表清单.md").write_text(captions.rstrip() + "\n", encoding="utf-8")
    (directory / "图表面板.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>问题一图表</title>"
        "<style>body{font:16px sans-serif;background:#f4f6f8;color:#202832;max-width:1100px;margin:30px auto}"
        "article{background:white;padding:28px;margin:22px 0;border-radius:12px}img{width:100%}"
        "h2{font-size:20px}p{line-height:1.7;color:#566273}</style>"
        "<h1>问题一 · 数据、特征与稳健性</h1>" + "".join(html_items), encoding="utf-8")


def math_ceil(value):
    return int(np.ceil(value))
