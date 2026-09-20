# 问题一模型适配数据集

由项目根目录的 问题一_求解.py 生成，原始与既有预处理文件保持只读。正式结果对应116个源域文件与16个无标签目标文件。数据来源、公式和处理依据见根目录的数学模型说明；可运行命令见问题一运行说明。

## 数据表

| 文件 | 内容 | 后续用途 |
|---|---|---|
| source_selection.csv | 原177文件及选中/未选中理由 | 审计源域筛选，不作为模型特征 |
| files.csv | 本次132文件的来源、采样、转速和继承划分 | 按文件分组和溯源 |
| window_features_all.csv | 20转主线的全部12候选特征，未作训练标准化 | 问题二内折重新筛选与拟合 |
| window_features_selected_scaled.csv | 主线8个选中特征，使用源训练70文件拟合尺度 | 冻结后的推荐主线输入 |
| baseline_1s_features_all.csv | 一秒对照的全部12候选特征 | 公平基线与折内处理 |
| baseline_1s_selected_scaled.csv | 同一组8特征，一秒源训练尺度 | 基线诊断输入 |
| uot_file_descriptors_unscaled.csv | 标准化主线窗特征的文件中位数/IQR，尚未进行第二次文件尺度拟合 | 检查文件级统计 |
| uot_file_descriptors_scaled.csv | 上表用源训练文件等权再次标准化的16维描述量 | 问题三非平衡最优传输对照输入 |
| window_coverage.csv、baseline_1s_coverage.csv | 每文件完整窗数、真实转数、已用与尾段时间 | 窗口覆盖与截尾审计 |
| feature_dictionary.json | 特征名、中文名、公式号、单位、机理块与入选标记 | 后续显式选择特征列 |

上述“unscaled”仅指文件描述量未作第二次缩放；其基础窗口特征已经经过训练尺度标准化。

## 必须保留的读取规则

1. CSV使用UTF-8带BOM，可用 pandas.read_csv(..., keep_default_na=False)；目标类别为空字符串，label_code=-1，不把文件编号B/N当成已知类别。
2. 只有数据字典中明确的特征列可以进入模型。file_id、label、label_code、split、时间索引、转速、文件目录等为元数据；标签只能作源域监督目标，不作为输入变量。
3. 窗口键为 file_id + window_index。索引从0开始，start_seconds包含端点，end_seconds不包含端点；完整窗不重叠。
4. 同一文件的所有窗始终属于同一split。source的train/validation/test继承既有划分，target保留未知标签。切窗与文件重采样不增加独立文件数。
5. 数值标准化参数保存在 results/q1/ 的 window_scaler.json、baseline_window_scaler.json、file_descriptor_scaler.json。后两种数据出口均不拟合目标标签。
6. 当前未标准化候选表只含已选频带和窗口长度，可用于该表征条件下重新筛选特征及拟合尺度。若问题二需要在新的训练折内重新选择频带或转数，必须从原始MAT重新计算六个候选组合；不能仅凭本表完成。当前冻结表用于开发层最终拟合后的变换，不能直接充当无泄漏的内折验证输入。
7. 四分位距采用线性分位数插值；仅一窗的文件IQR为0，不补成非零。源训练常数描述列会被删除并记录。
8. 原始振动单位未知，前三个候选特征保留原始尺度。不同采样带宽、传感器位置和增益造成的差异由后续迁移对照评估。

当前目录只提供特征与来源信息，不包含源域预测、目标类别预测或运输矩阵。
