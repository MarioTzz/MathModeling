# 问题一的可复现辅助工具

`plot_style.py`、`repro_manifest.py` 复制自本地 `math-modeling-skill/references/roles/编程手/scripts/`；`figure_export.py`（原名 `export_figure.py`）、`visual_qa.py`、`profile_data.py` 原样复制自 `math-modeling-skill/tools/figure/scripts/`。上游为 [XiaoMaColtAI/math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill)。副本用于独立复现；原技能文件未修改。具体文件哈希随每次运行清单保存。

本题绘图调用原始导出器，关闭自动裁边以保持最终尺寸；灰度预览单独生成于 `_qa/`，不改写彩色 PNG。布局检查同时使用样式工具与专用检查器。

`repro_manifest.py` 仅调用 `build_manifest`，不使用其面向技能目录的命令行路径解析；将副本中的技能根目录定位改为当前工具目录，防止浅目录安装时导入失败。
