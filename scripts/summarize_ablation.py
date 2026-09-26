from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TASKS = ("growth_stage", "health_level", "maturity_level", "abnormal_alert")

TASK_DISPLAY_NAMES = {
    "growth_stage": "生长阶段",
    "health_level": "健康等级",
    "maturity_level": "成熟度等级",
    "abnormal_alert": "异常报警",
}

DEFAULT_EXPERIMENTS = (
    ("real_data_exp001", "multimodal", "第一次完整多模态训练"),
    ("exp002_image_only", "image_only", "只使用图片分支"),
    ("exp003_env_only", "env_only", "只使用环境时序分支"),
    ("exp004_multimodal_retry", "multimodal", "同条件多模态复训"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation experiment results.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--output", default="experiments/ablation_summary.md")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def metric(report: dict[str, Any], task: str, name: str) -> float:
    return float(report["metrics"][task][name])


def abnormal_counts(error_analysis_path: Path) -> tuple[int, int]:
    false_positive = 0
    false_negative = 0
    if not error_analysis_path.exists() or error_analysis_path.stat().st_size == 0:
        return false_positive, false_negative

    with error_analysis_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("abnormal_error_type") == "false_positive":
                false_positive += 1
            elif row.get("abnormal_error_type") == "false_negative":
                false_negative += 1
    return false_positive, false_negative


def build_rows(outputs_dir: Path, experiments_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_name, input_mode, note in DEFAULT_EXPERIMENTS:
        report_path = outputs_dir / experiment_name / "evaluation_report.json"
        report = read_json(report_path)
        error_analysis_path = experiments_dir / experiment_name / "error_analysis.csv"
        false_positive, false_negative = abnormal_counts(error_analysis_path)

        rows.append(
            {
                "experiment_name": experiment_name,
                "input_mode": report.get("input_mode", input_mode),
                "note": note,
                "samples": int(report["samples"]),
                "checkpoint_epoch": report.get("checkpoint_epoch", ""),
                "total_loss": float(report["losses"]["total_loss"]),
                "growth_stage_f1": metric(report, "growth_stage", "macro_f1"),
                "health_level_f1": metric(report, "health_level", "macro_f1"),
                "maturity_level_f1": metric(report, "maturity_level", "macro_f1"),
                "abnormal_alert_f1": metric(report, "abnormal_alert", "macro_f1"),
                "abnormal_positive_f1": metric(report, "abnormal_alert", "positive_f1"),
                "error_samples": count_csv_rows(error_analysis_path),
                "abnormal_false_positive": false_positive,
                "abnormal_false_negative": false_negative,
                "report_path": report_path,
                "error_analysis_path": error_analysis_path,
            }
        )
    return rows


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_experiment_summary(row: dict[str, Any], experiments_dir: Path) -> None:
    experiment_name = row["experiment_name"]
    path = experiments_dir / experiment_name / "experiment_summary.md"
    lines = [
        f"# {experiment_name} Experiment Summary",
        "",
        "## Purpose",
        "",
        f"Ablation experiment: {row['note']}.",
        "",
        "## Archived Files",
        "",
        f"- `checkpoints/{experiment_name}/best_model.pth`",
        f"- `checkpoints/{experiment_name}/last_model.pth`",
        f"- `checkpoints/{experiment_name}/effective_config.yaml`",
        f"- `outputs/{experiment_name}/evaluation_report.json`",
        f"- `outputs/{experiment_name}/predictions.csv`",
        f"- `outputs/{experiment_name}/confusion_matrix.csv`",
        f"- `experiments/{experiment_name}/train_command.txt`",
        f"- `experiments/{experiment_name}/train.log`",
        f"- `experiments/{experiment_name}/error_analysis.csv`",
        f"- `experiments/{experiment_name}/error_summary.md`",
        "",
        "## Main Result",
        "",
        f"- Input mode: `{row['input_mode']}`",
        f"- Test samples: {row['samples']}",
        f"- Best checkpoint epoch: {row['checkpoint_epoch']}",
        f"- Total loss: {fmt(row['total_loss'])}",
        f"- Growth stage macro F1: {fmt(row['growth_stage_f1'])}",
        f"- Health level macro F1: {fmt(row['health_level_f1'])}",
        f"- Maturity level macro F1: {fmt(row['maturity_level_f1'])}",
        f"- Abnormal alert macro F1: {fmt(row['abnormal_alert_f1'])}",
        f"- Error samples: {row['error_samples']}",
        f"- Abnormal false positives: {row['abnormal_false_positive']}",
        f"- Abnormal false negatives: {row['abnormal_false_negative']}",
        "",
        "## Dataset Note",
        "",
        "This experiment uses `dataset_fused`, a fused dataset built from public watermelon-related image datasets, weak-supervised fusion labels, and real watermelon-field weather / soil water time series. It should be treated as an engineering and demonstration dataset, not as a single-site manually annotated greenhouse dataset.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    first_multimodal = next(row for row in rows if row["experiment_name"] == "real_data_exp001")
    image_only = next(row for row in rows if row["experiment_name"] == "exp002_image_only")
    env_only = next(row for row in rows if row["experiment_name"] == "exp003_env_only")
    multimodal_retry = next(row for row in rows if row["experiment_name"] == "exp004_multimodal_retry")

    lines = [
        "# Ablation Experiment Summary",
        "",
        "## Purpose",
        "",
        "Verify whether the multimodal model is better than using image data or environment data alone.",
        "",
        "## Result Table",
        "",
        "| 实验 | 输入模态 | Total Loss | 生长阶段 F1 | 健康等级 F1 | 成熟度 F1 | 异常报警 F1 | 错误样本 | 异常误报 | 异常漏报 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['experiment_name']}` | "
            f"`{row['input_mode']}` | "
            f"{fmt(row['total_loss'])} | "
            f"{fmt(row['growth_stage_f1'])} | "
            f"{fmt(row['health_level_f1'])} | "
            f"{fmt(row['maturity_level_f1'])} | "
            f"{fmt(row['abnormal_alert_f1'])} | "
            f"{row['error_samples']} | "
            f"{row['abnormal_false_positive']} | "
            f"{row['abnormal_false_negative']} |"
        )

    lines.extend(
        [
            "",
            "## Comparison Note",
            "",
            "`exp002_image_only`, `exp003_env_only`, and `exp004_multimodal_retry` are the formal ablation runs generated with the same current training and evaluation scripts. `real_data_exp001` is kept as the archived first run reference. Use macro F1 and error counts as the main comparison indicators; total loss is secondary because the archived first report was produced before the latest evaluation-script update.",
            "",
            "## Key Findings",
            "",
            f"1. `exp004_multimodal_retry` is clearly better than `exp002_image_only` on growth stage: macro F1 {fmt(multimodal_retry['growth_stage_f1'])} vs {fmt(image_only['growth_stage_f1'])}.",
            f"2. `exp003_env_only` performs well on growth stage with macro F1 {fmt(env_only['growth_stage_f1'])}, but is weak on health level, maturity level, and abnormal alert.",
            f"3. `exp002_image_only` performs well on health, maturity, and abnormal alert, but loses much more growth-stage information than multimodal training.",
            f"4. `real_data_exp001` and `exp004_multimodal_retry` have the same sample-level error count ({first_multimodal['error_samples']} and {multimodal_retry['error_samples']}), showing that the multimodal result is repeatable under the current setup.",
            "5. The ablation supports keeping the multimodal design: image features and environment features provide complementary information.",
            "",
            "## Project Conclusion",
            "",
            "For the current fused dataset, the model should remain multimodal. Environment data contributes strongly to growth-stage recognition, while image data is necessary for health, maturity, and abnormal-alert performance. The result can be used as evidence for the project design, with the caveat that `dataset_fused` uses weak labels and mixed public data sources.",
            "",
            "## Next Step",
            "",
            "Run a stricter source-aware or group-aware split experiment to check whether the model is learning robust watermelon features or shortcuts from source datasets.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    experiments_dir = Path(args.experiments_dir)
    output_path = Path(args.output)

    rows = build_rows(outputs_dir=outputs_dir, experiments_dir=experiments_dir)
    for row in rows:
        if row["experiment_name"] != "real_data_exp001":
            write_experiment_summary(row, experiments_dir)
    write_ablation_summary(rows, output_path)
    print(f"saved_ablation_summary={output_path}")


if __name__ == "__main__":
    main()
