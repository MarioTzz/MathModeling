# 高速列车轴承预处理数据集

依据[数据质量检查与预处理建议](../数据质量检查与预处理建议.md)，本次对全部 177 个 MAT 文件进行了可复现预处理，另存于本目录。原始 `数据集/` 未修改。处理日期为 2026-09-20；当前完成数据准备，尚未训练诊断模型或预测目标域类别。

## 1. 数据交付与实际数量

| 内容 | 实际结果 |
|---|---|
| 原始文件覆盖 | 源域 161 个，目标域 16 个，共 177 个 |
| 原采样率版本 | 177 个 MAT，427 条信号，54,870,522 个采样值 |
| 12 kHz 对照版本 | 177 个 MAT，177 条主信号，共 14,390,548 点 |
| 一秒完整窗口 | 源域 1,068 个，目标域 128 个，共 1,196 个 |
| 不足一秒的尾部 | 7 个文件共 38,548 个 12 kHz 采样值，仍保留在连续信号中 |
| 转速补全 | 10 个源域文件从文件名补入元数据，记录来源 |
| 通道关联 | 5 对已知 DE/FE 时移、比例对应关系重新验证成立 |
| 数值填补与异常点删除 | 均为 0；没有补零、插值、截断冲击或恢复疑似削顶 |

本目录包含 354 个处理后 MAT 文件，两种版本各 177 个，数据约 400 MiB。单文件最大 2,219,419 字节。保留全部文件不等于认定全部信号适合任何后续模型，质量标记用于选择对照实验。

```text
预处理数据集/
├── native_centered/           # 原采样率、原通道、原长度的去均值数据
│   ├── 源域数据集/            # 沿用完整目录层级
│   └── 目标域数据集/
├── common_12k/                # DE/目标单通道，连续信号及单位 RMS 窗口
│   ├── 源域数据集/
│   └── 目标域数据集/
├── files.csv                 # 文件、标签、分组、通道可用性、单位和处理参数
├── channels.csv              # 原始通道统计与质量标记
├── windows.csv               # 窗口索引、划分、幅值/形状特征与审计信息
├── features_standardized.csv # 六列候选特征的训练集 z-score 变换
├── scaler.json               # 标准化均值、标准差及拟合文件范围
├── channel_overlap_evidence.json
├── parameters.json
├── processing_summary.json
├── manifest.json             # 177 个输入及 362 个生成载荷的 SHA-256
└── README.md
```

## 2. 已实施的处理及影响

### 2.1 原采样率去均值版本

`native_centered/` 保留所有原始振动变量名、列向量形状和 float64 精度，每条信号执行 `x_centered = x - mean(x)`。这一步移除整段直流分量，不作去趋势或平滑。原变量名现在指向去均值后的数据；同文件新增 `<原变量名>_mean` 和 `<原变量名>_raw_rms`，分别保存原始均值和原始均方根（Root Mean Square，RMS）。

原始 RPM 字段原样保留；`rpm_metadata` 为字段或文件名解析得到的转速，目标域为题面约 600 rpm。精确/近似属性及来源以 `files.csv` 为准。源域缺失的风扇端（FE）和基座（BA）通道不生成替代变量，已有通道保持完整。

将去均值信号加回保存的均值，可以在浮点误差范围内恢复原值；全量核验最大绝对误差为 `1.4210854715202004e-14`。去均值会改变包含直流的 RMS，因此原始均值、原始 RMS 和交流 RMS 分别存储。

### 2.2 12 kHz 单通道对照版本

每个源域文件采用其驱动端（DE）测点，目标域采用文件内的单条信号。源域某些文件的故障发生在 FE 轴承，但 DE 仍是观测测点；这两种位置在元数据中分别表达。DE 覆盖 161 个源域文件，作为统一输入的候选，不构成已验证的迁移最优通道选择。

整段去均值后，以 `scipy.signal.resample_poly` 进行多相抗混叠重采样：48 kHz→12 kHz 为 1/4，32 kHz→12 kHz 为 3/8，12 kHz 数据保持采样点不变。参数固定为 `window=('kaiser', 5.0)`、`padtype='line'`。输出长度为 `ceil(原长度 × up / down)`，取整造成的采样覆盖时长偏差小于一个 12 kHz 采样间隔。没有把不同记录拉伸到同样总长度。

12 kHz 的奈奎斯特频率为 6 kHz，低通滤波在截止附近存在过渡带。该版本会损失原高采样率数据中的高频成分，尤其可能影响后续高频共振与包络分析；保留原采样率版本供对照。源域采样率沿用目录名义值，其逐通道时钟尚未另行核实；辅助通道不参与本次重采样。目标域 32 kHz 来源于题面。共同采样率没有消除转速、测点和轴承差异。

对重采样滤波的首尾影响，按当前滤波器的约 10 个输出采样点半支撑范围标记接触边界的窗口，共 138 个。这些窗口保留，后续可比较保留与屏蔽的结果。该标记是滤波支撑范围检查，不表示已经检测到伪影。

### 2.3 一秒切窗与波形归一化

一秒、无重叠的窗口是本版本的处理参数，不是题目给定的条件。每窗 12,000 点；所有起止索引从 0 开始，终点不包含在内。一个窗口内再次减去该窗均值，然后除以该窗交流 RMS，生成 `windows_unit_rms`。矩阵形状为“窗口数 × 12,000”，每行一个窗口。

这种逐窗变换只使用本窗信号，不估计源域或目标域总体分布。它突出波形形状，同时移除窗间能量差异。保留以下字段以支持还原和幅值对照：

| `common_12k/*.mat` 字段 | 含义 |
|---|---|
| `signal_resampled` | 整段去均值后重采样的连续列向量，含不足一窗的尾部；重采样后未再强制整段均值为零 |
| `windows_unit_rms` | 逐窗去均值、单位交流 RMS 的波形矩阵 |
| `window_means`、`window_ac_rms` | 对连续重采样信号切窗得到的均值与交流 RMS |
| `window_start` | 各窗口在 12 kHz 连续序列中的起点，0-based |
| `window_valid` | 窗口交流 RMS 是否大于零；本批全部为 1 |
| `original_mean`、`original_raw_rms` | 主通道重采样前的原始幅值信息 |
| `original_nominal_fs_hz`、`fs_hz` | 原名义采样率与实际输出网格 12,000 Hz |
| `label_code`、`label_known` | 类别编码及是否具有已知标签 |
| `tail_samples` | 未进入完整窗口的尾部点数，仍存在于 `signal_resampled` |

还原第 `i` 个重采样窗口的公式为 `windows_unit_rms[i] * window_ac_rms[i] + window_means[i]`。这只还原重采样窗口，不能逆推出降采样去掉的高频信息。全量窗口还原最大绝对误差为 `1.4210854715202004e-14`。

### 2.4 编码、单位与新增变量

标签固定为 N=0、OR=1、IR=2、B=3；目标域统一为 -1，`label_known=False`，其类别字符串留空。目标文件 `B.mat` 和 `N.mat` 的字母仍是编号，不代表类别。

`file_id` 包含域及完整相对路径，`group_id=file_id`，解决跨目录同名文件问题。`rpm_source` 区分内部字段、文件名和题面近似值；`rotation_hz=rpm/60`。故障尺寸同时保存英寸和毫米，毫米值按 25.4 换算；正常及目标域没有人为填写故障尺寸。载荷保留马力。振动幅值标为原始尺度、物理单位未知，没有实施 g 与 m/s² 换算。

标签、路径、故障尺寸、载荷、通道可用性和质量标记均属于标签或审计元数据，不加入六列特征白名单。`files.csv` 的源域 `has_DE/has_FE/has_BA` 表示通道是否存在；目标单通道测点未知，因此三个位置标记均为 False，不表示目标信号缺失。

### 2.5 异常与相关通道仅标记

筛查阈值是本流水线的复核规则，不作为故障类别或坏点判决：

| 标记 | 规则 | 命中通道数 |
|---|---|---:|
| 直流偏置 | `abs(mean) / raw_rms > 0.1` | 2 |
| 分段能量变化 | 原序列等分 8 块，各自去均值的 RMS 最大/最小比 >2；零分母另行标记 | 4 |
| 高峰度 | Pearson 峰度 >20，未减 3 | 32 |
| 疑似削顶/量化平台 | 全局最大值或最小值连续出现至少 2 点 | 4 |

相应原始平台采样点落入 7 个主通道窗口，在 `windows.csv` 中记录数量；其他通道的候选在 `channels.csv` 中保留。标记按原始主通道的同一时间区间映射，不把重采样后的平滑波形作为量程证据。高峰度与大幅冲击可能是诊断信息，均完整保留。

5 对已知 DE/FE 时移、比例对应关系按照前次审计起点重新核对，仿射残差比例不超过 `1e-12` 才接受原标记；实际证据写入 `channel_overlap_evidence.json`。两通道均留在原采样率版本并保持同一文件分组；共同采样率版本仅取 DE，因此没有将这两个通道重复作为独立样本。此次没有扩展为全部源域跨文件任意时移的穷举查重。

## 3. 数据划分与特征标准化

固定随机种子 42，以原始文件为分组，按源域类别分层。每类训练数取约 60% 向下取整，验证数约 20% 向下取整，其余为测试，至少保留各一个验证与测试文件。然后在各文件内部切窗，同文件全部通道和窗口保持同一划分。

| 划分 | N | OR | IR | B | 文件合计 | 窗口数 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 2 | 46 | 24 | 24 | 96 | 637 |
| 验证 | 1 | 15 | 8 | 8 | 32 | 220 |
| 测试 | 1 | 16 | 8 | 8 | 33 | 211 |
| 无标签目标域 | 未知 | 未知 | 未知 | 未知 | 16 | 128 |

这是源域按文件留出的候选协议，未专门按载荷、故障尺寸或采集批次隔离，不能据此声称对新工况已具备泛化能力。正常类只有 4 个独立文件，测试中只有 1 个；窗口增加不会增加独立正常记录数。后续发现跨文件关联时应合并分组并重新划分。

`windows.csv` 提供六列候选特征，均在**去窗均值、未除 RMS**的重采样窗口 `a` 上定义；令 `r=sqrt(mean(a²))`：

| 列名 | 定义 |
|---|---|
| `ac_rms` | `r` |
| `mean_abs` | `mean(abs(a))` |
| `peak_to_peak` | `max(a)-min(a)` |
| `crest_factor` | `max(abs(a))/r` |
| `kurtosis_pearson` | `mean((a/r)^4)` |
| `skewness` | `mean((a/r)^3)` |

`features_standardized.csv` 仅包含 `window_id` 和六列变换后的数值。逐列执行 `z=(feature-training_mean)/training_std`，参数仅由 637 个源域训练窗口拟合，标准差使用总体口径 `ddof=0`；若为零，以 1 作为除数并记录退化列。本批无退化列。验证、测试和目标域使用同一组参数，不重新拟合，也不裁剪超范围值。

当前 scaler 对训练窗口等权，所以长文件参与的窗口更多。另提供 `file_equal_weight=1/该文件窗口数`，便于后续在训练损失或文件级汇总中令每个文件总权重相同；该权重没有偷偷用于本次 scaler。无量纲特征也执行标准化，因为它们的数值尺度仍不同。

本次未实施 Min–Max、稳健缩放、对数变换、样本扩增和类别重采样；它们是依赖模型的对照选项。重新划分或进行交叉验证时，应在每个新训练折上重新拟合 scaler，不能直接复用当前预计算的标准化表。目标域没有真值，本次没有把预处理核验转换成分类准确率。

## 4. 读取示例

Python 从项目根目录读取。仅将 `scaler.json` 列出的六个特征作为此示例的候选输入，避免把标签元数据混入模型：

```python
from pathlib import Path
from scipy.io import loadmat
import csv
import json
import numpy as np

root = Path('预处理数据集')
with (root / 'files.csv').open(encoding='utf-8-sig', newline='') as f:
    files = list(csv.DictReader(f))
record = next(r for r in files if r['split'] == 'train')
data = loadmat(root / record['common_path'])
normalized_windows = data['windows_unit_rms']  # shape: (n_windows, 12000)
amplitude_windows = (
    normalized_windows * data['window_ac_rms'].reshape(-1, 1)
    + data['window_means'].reshape(-1, 1)
)

with (root / 'windows.csv').open(encoding='utf-8-sig', newline='') as f:
    index = {r['window_id']: r for r in csv.DictReader(f)}
with (root / 'features_standardized.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
columns = json.loads((root / 'scaler.json').read_text())['features']
train_rows = [r for r in rows if index[r['window_id']]['split'] == 'train']
X_train = np.array([[float(r[c]) for c in columns] for r in train_rows])
y_train = np.array([int(index[r['window_id']]['label_code']) for r in train_rows])
```

MAT 文件也可用 MATLAB `load` 打开；MAT 数组使用 MATLAB 的行列索引，存储的 `window_start` 数值仍是 0-based，映射到 MATLAB 下标时需要加 1。

## 5. 复现与验收

生成代码为 [preprocess_bearings.py](../scripts/preprocess_bearings.py)，核验代码为 [verify_preprocessed.py](../scripts/verify_preprocessed.py)。本次环境为 Python 3.12.14、NumPy 2.3.5、SciPy 1.18.1，依赖固定在 [requirements-preprocessing.txt](../requirements-preprocessing.txt)。

配置好 Python 3.12 环境后，从项目根目录执行：

```bash
python -m pip install -r requirements-preprocessing.txt
python -m unittest discover -s tests -v
python scripts/preprocess_bearings.py --output 预处理数据集_复现
python scripts/verify_preprocessed.py --dataset 预处理数据集_复现
```

处理脚本要求输出目录尚不存在，以免覆盖已有产物；再次运行请选择另一个新目录。默认输出为 `预处理数据集/`。`--smoke` 可先处理固定的 16 个真实文件进行最小检查，其规模不代表正式全量结果。

`manifest.json` 记录所有输入哈希、362 个生成载荷的哈希、代码哈希、版本与参数；不含其自身及人工编写的 README。MAT 的说明性时间戳头已固定，在相同环境下可进行输出文件逐字节复现比较。跨平台浮点库或压缩库改变时应先比较数值，不以字节不同直接判为研究结果不同。

全量核验见 [预处理核验.json](../results/预处理核验.json)，项目运行清单见 [复现清单.json](../results/复现清单.json)。核验逐个读取原始与输出文件，检查哈希、54,870,522 个原采样率值的可还原性、1,196 个窗口的索引和归一化、全部特征定义、训练集拟合范围与未知目标标签。具体的独立验收范围与结果见 [预处理独立验收.md](../results/预处理独立验收.md)。

上述核验确认处理实现与记录相符。12 kHz、单通道、一秒窗口和幅值缩放对最终诊断效果的净影响，仍需在后续建模阶段通过同一划分下的对照实验评价。
