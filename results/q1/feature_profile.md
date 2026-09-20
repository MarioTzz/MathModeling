# Data profile: <DataFrame>

**Shape:** 132 rows × 16 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `file_id` | text | 132 | 0 |  |
| `domain` | categorical | 132 | 0 | 2 levels: source(116), target(16); min_group_n=16 |
| `label` | categorical | 132 | 0 | 5 levels: OR(56), B(28), IR(28), (16), N(4); min_group_n=4 |
| `split` | categorical | 132 | 0 | 4 levels: train(70), validation(23), test(23), target(16); min_group_n=16 |
| `ac_rms` | continuous | 132 | 0 | mean=1.75, sd=4.99, range=[0.0631, 31.6], skew=4.64 (highly skewed); outliers=17 (IQR); -> log axis |
| `mean_abs` | continuous | 132 | 0 | mean=1.11, sd=2.86, range=[0.0506, 18], skew=4.07 (highly skewed); outliers=18 (IQR); -> log axis |
| `peak_to_peak` | continuous | 132 | 0 | mean=30.4, sd=113, range=[0.488, 827], skew=5.41 (highly skewed); outliers=13 (IQR); -> log axis |
| `crest_factor` | continuous | 132 | 0 | mean=6.61, sd=2.62, range=[3.6, 16.6], skew=0.98 (moderately skewed); outliers=1 (IQR) |
| `kurtosis_pearson` | continuous | 132 | 0 | mean=9.81, sd=8.74, range=[2.75, 40.7], skew=1.51 (highly skewed); outliers=9 (IQR) |
| `skewness` | continuous | 132 | 0 | mean=0.0164, sd=0.131, range=[-0.45, 0.337], skew=-1.06 (highly skewed); outliers=19 (IQR) |
| `env_entropy` | continuous | 132 | 0 | mean=0.655, sd=0.173, range=[0.283, 0.925], skew=-0.21 (approximately symmetric) |
| `env_centroid_order` | continuous | 132 | 0 | mean=5.12, sd=1.8, range=[2.44, 10.3], skew=0.86 (moderately skewed); outliers=2 (IQR) |
| `env_spread_order` | continuous | 132 | 0 | mean=3.68, sd=1.02, range=[1.83, 6.06], skew=0.47 (approximately symmetric) |
| `env_peak_order` | continuous | 132 | 0 | mean=3.04, sd=1.86, range=[1, 8.3], skew=0.90 (moderately skewed); outliers=7 (IQR) |
| `env_harmonic_ratio` | continuous | 132 | 0 | mean=0.312, sd=0.193, range=[0.0427, 0.838], skew=1.18 (highly skewed); outliers=8 (IQR) |
| `env_low_order_ratio` | continuous | 132 | 0 | mean=0.29, sd=0.151, range=[0.0495, 0.615], skew=0.39 (approximately symmetric) |

## Group structure
- Grouped by: `domain`, `label`
- Number of groups: 5
- Group size: min=4, median=28, max=56
- **WARN**: at least one group has n<10 — use box/violin + stripplot rather than mean-only bar chart.

## Correlations (Pearson, sorted by |r|)
- `ac_rms` ↔ `mean_abs` : r = 0.982 (very strong)
- `ac_rms` ↔ `peak_to_peak` : r = 0.965 (very strong)
- `mean_abs` ↔ `peak_to_peak` : r = 0.909 (very strong)
- `crest_factor` ↔ `kurtosis_pearson` : r = 0.905 (very strong)
- `env_entropy` ↔ `env_harmonic_ratio` : r = -0.832 (very strong)
- `env_centroid_order` ↔ `env_spread_order` : r = 0.824 (very strong)
- `env_entropy` ↔ `env_spread_order` : r = 0.773 (very strong)
- `env_centroid_order` ↔ `env_low_order_ratio` : r = -0.729 (very strong)
- `env_peak_order` ↔ `env_low_order_ratio` : r = -0.591 (strong)
- `crest_factor` ↔ `env_low_order_ratio` : r = 0.502 (strong)
- ... +56 more pairs

## Warnings
- 列 'label' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。

## Chart suggestions (preliminary)
- 分类 vs 连续，小样本（每组 n<10）→ **箱线图/小提琴图 + stripplot 叠加原始点**；**避免**只画均值柱状图，会掩盖分布。
- ≥3 个连续变量 → 相关性热力图（['ac_rms', 'mean_abs', 'peak_to_peak', 'crest_factor', 'kurtosis_pearson']）或 pairplot 散点矩阵
- 分类维度组合数 = 40（domain, label, split 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。
- ac_rms 跨数个量级（0.0631 ~ 31.6）→ 用对数 y 轴
- mean_abs 跨数个量级（0.0506 ~ 18）→ 用对数 y 轴
- peak_to_peak 跨数个量级（0.488 ~ 827）→ 用对数 y 轴
- kurtosis_pearson 高度偏态（skew=1.51）→ 考虑对数变换或小提琴图代替均值柱图
- skewness 高度偏态（skew=-1.06）→ 考虑对数变换或小提琴图代替均值柱图
- env_harmonic_ratio 高度偏态（skew=1.18）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。