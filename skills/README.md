# 本项目写作技能

安装日期：2026-09-20。四个独立写作技能与现有 `../math-modeling-skill/` 配合使用；`nature-shared` 为必要依赖，不是第五个独立写作流程。

## 来源与固定版本

| 本地目录 | 上游路径与固定提交 | 许可证 |
|---|---|---|
| `nature-writing/` | [Yuan1z0825/nature-skills · skills/nature-writing](https://github.com/Yuan1z0825/nature-skills/tree/9cecfef6ac683fa59d7d15d2e22f98fa71dacaf5/skills/nature-writing) | [Apache-2.0](licenses/nature-skills-LICENSE) |
| `nature-polishing/` | [Yuan1z0825/nature-skills · skills/nature-polishing](https://github.com/Yuan1z0825/nature-skills/tree/9cecfef6ac683fa59d7d15d2e22f98fa71dacaf5/skills/nature-polishing) | 同上 |
| `nature-shared/` | [Yuan1z0825/nature-skills · skills/nature-shared](https://github.com/Yuan1z0825/nature-skills/tree/9cecfef6ac683fa59d7d15d2e22f98fa71dacaf5/skills/nature-shared) | 同上 |
| `humanizer-zh-academic/` | [redbaronyyyyy-eng/humanizer-zh-academic · 仓库根目录](https://github.com/redbaronyyyyy-eng/humanizer-zh-academic/tree/50e11af64eb4817f75b8be7d609b99d7cd80d801) | [MIT](humanizer-zh-academic/LICENSE) |
| `stss/` | [lennney/stop-that-shit · skills/stss](https://github.com/lennney/stop-that-shit/tree/7f3dc86f268437c3d2f18ba026311d9ab40d0db9/skills/stss) | [MIT](licenses/stop-that-shit-LICENSE) |

技能使用安装工具按固定提交下载，保留原有 `SKILL.md`、manifest、agents 配置及配套参考、脚本和模板。Nature 及 stss 仓库根目录的许可证另存于 `licenses/`；humanizer 的许可证随目录保留。未安装这些仓库的其他技能、插件、hooks 或后台组件，也未运行其附带脚本。

项目适配集中在 [AGENTS.md](../AGENTS.md)，不改写第三方技能正文。`math-modeling-skill/` 是已有下载副本，其入口名为 `math-modeling`，保持原有内容与 Git 忽略策略。

## 协作方式

数学建模技能负责模型、计算证据、阶段门禁与最终交付；四个写作技能依次负责结构、学术表达、中文自然度和防御性表达审查。中文任务默认交付中文，保留数据与术语，不套用 Nature 投稿格式，也不把语言检查当成模型验证或 AI 检测。

项目根目录的 `.agents/skills/` 用相对符号链接注册六个入口：四个写作技能、`nature-shared` 依赖及 `math-modeling`。实际文件分别留在本目录及 `../math-modeling-skill/`，没有复制到用户全局技能目录。此布局依据 [Codex 技能发现说明](https://developers.openai.com/codex/skills/)。

安装后下一轮对话即可使用；若技能列表未刷新，重启 Codex 后在本项目重试。也可以明确指定入口，例如：

> 使用 math-modeling 和 AGENTS.md 约定的四个写作技能，依据真实实验结果起草方法章节，中文输出。

> 按 AGENTS.md 润色题目分析报告的指定段落，保留技术含义、数值、符号和引用。

仅克隆 Git 仓库时，受忽略规则影响，`math-modeling-skill/` 和数据集不会随仓库恢复；先按根目录 README 的来源重新下载数学建模技能，才能使用其对应链接。项目级发现入口只解决加载位置，不代表所有写作分支已做行为测试。

## 后续维护

升级前确认上游仓库、提交与依赖变化；在临时目录下载并比较，保留许可证，更新本表，再检查入口链接和 manifest 引用。中文输出、术语与证据约束继续以 `AGENTS.md` 为本项目约定。
