"""问题三输入哈希、环境与确定性结果哈希清单。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = (ROOT / "math-modeling-skill/references/roles/编程手/scripts/"
                 "repro_manifest.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_q3_manifest(output: Path, config: dict, summary: dict) -> None:
    tooling = runpy.run_path(str(MANIFEST_TOOL))
    metadata = pd.read_csv(ROOT / "模型适配数据集/q1/files.csv", keep_default_na=False)
    target_raw = [ROOT / "数据集" / relative for relative in
                  metadata[metadata.domain.eq("target")].relative_path]
    fixed = [
        "模型适配数据集/q1/files.csv",
        "模型适配数据集/q1/window_features_all.csv",
        "results/q1/summary.json", "configs/q3.json",
        "问题三_求解.py", "utils/q3_transfer.py", "utils/q3_outputs.py",
        "utils/q3_plots.py", "utils/q3_manifest.py", "utils/q1_model.py",
        "utils/q2_optimized.py", "utils/q2_model.py", "utils/figure_export.py",
        "utils/plot_style.py", "utils/visual_qa.py",
        "math-modeling-skill/references/roles/编程手/scripts/repro_manifest.py",
        "math-modeling-skill/tools/figure/scripts/style_constants.py",
        "math-modeling-skill/VERSION",
    ]
    inputs = [ROOT / item for item in fixed] + target_raw
    command = (
        "LOKY_MAX_CPU_COUNT=8 PYTHONDONTWRITEBYTECODE=1 "
        "MPLCONFIGDIR=.analysis-work/matplotlib "
        "XDG_CACHE_HOME=.analysis-work/font-cache "
        "PYTHONPATH=.analysis-work/q2-deps:.analysis-work/q1-deps:.analysis-work/python-deps:. "
        "/Users/mariozz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 "
        "问题三_求解.py --overwrite"
    )
    manifest = tooling["build_manifest"](
        inputs, int(config["seed"]),
        {"alpha_candidates": config["alpha_candidates"],
         "alpha_selected": summary["alpha_selected"],
         "n_estimators": config["n_estimators"],
         "target_rpm_sensitivity": config["target_rpm_sensitivity"]},
        command, ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"])
    for item in manifest["input_files"]:
        item["relative_path"] = Path(item["path"]).relative_to(ROOT).as_posix()
    deterministic = [
        "proxy_metrics.csv", "file_medians_nine_features.csv",
        "source_test_predictions.csv", "target_file_labels.csv",
        "rpm_sensitivity_predictions.csv", "target_leave_one_out.csv",
        "alignment_diagnostics.csv", "model/trees.json", "model/trees.npz",
        "model/model_interface.json", "model/export_validation.json",
        "问题三_结果报告.md", "figure_contracts.json", "图表清单.md",
        "图表面板.html",
    ]
    manifest["deterministic_output_files"] = [
        {"relative_path": item, "sha256": _sha256(output / item),
         "bytes": (output / item).stat().st_size}
        for item in deterministic
    ]
    manifest["summary_semantic_values"] = {key: value for key, value in summary.items()
                                           if key != "elapsed_seconds"}
    manifest["figure_stems"] = sorted(path.stem for path in (ROOT/"figures/q3").glob("*.svg"))
    (output / "复现清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
