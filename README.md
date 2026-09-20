# MathModeling

2025 年中国研究生数学建模竞赛 E 题「高速列车轴承智能故障诊断问题」的建模分析资料。

## 项目入口

- [竞赛题目原文](E题：高速列车轴承智能故障诊断问题.pdf)：2025 年中国研究生数学建模竞赛 E 题 PDF。
- [使用指南](使用指南.md)：所用数学建模技能的工作流程与使用边界。

仓库当前包含题目原文、数据集和建模工作流程；尚未包含可用于评估诊断性能的训练结果或目标域预测标签。

## 数据与本地文件

本地数据目录为 `数据集/`，包含 161 个源域文件和 16 个目标域文件。原始数据未纳入 Git 版本管理；获取方式见题目原文。本仓库不能单独复现尚未实施的诊断实验。

`.analysis-work/` 中的临时依赖、审计中间文件和排版校样不提交。下载的第三方技能副本 `math-modeling-skill/` 也不提交，其来源为 [XiaoMaColtAI/math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill)。

资料用于学习与研究，不作为可直接提交的竞赛作品，也不用于实际列车安全决策。

## 写作技能协作

项目级协作规则见 [AGENTS.md](AGENTS.md)。现有 `math-modeling-skill/` 负责模型、实验依据和阶段质检；`skills/` 下的 `nature-writing`、`nature-polishing`、`humanizer-zh-academic`、`stss` 分别负责结构、学术表达、中文可读性与防御性表达审查，另含 `nature-shared` 共享依赖。

技能来源、固定版本、许可证和使用方法见 [本地写作技能说明](skills/README.md)。项目通过 `.agents/skills/` 的相对链接发现这些技能；中文研究写作保留中文，并保持术语、数据和证据边界一致。
