from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASKS = ("growth_stage", "health_level", "maturity_level", "abnormal_alert")

TASK_DISPLAY_NAMES = {
    "growth_stage": "生长阶段",
    "health_level": "健康等级",
    "maturity_level": "成熟度等级",
    "abnormal_alert": "生长异常预警",
}

CLASS_NAMES = {
    "growth_stage": {
        0: "发芽期",
        1: "伸蔓期",
        2: "开花期",
        3: "坐果期",
        4: "膨大期",
        5: "成熟期",
    },
    "health_level": {
        0: "正常",
        1: "轻微异常",
        2: "中度异常",
        3: "严重异常",
    },
    "maturity_level": {
        0: "未成熟",
        1: "接近成熟",
        2: "成熟",
        3: "过熟",
    },
    "abnormal_alert": {
        0: "无异常",
        1: "有异常",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze wrong predictions for one watermelon experiment."
    )
    parser.add_argument("--experiment-name", default="real_data_exp001")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions.csv. Defaults to outputs/<experiment-name>/predictions.csv.",
    )
    parser.add_argument(
        "--manifest",
        default="dataset_fused/fusion_manifest.csv",
        help="Path to fusion_manifest.csv for source metadata.",
    )
    parser.add_argument(
        "--evaluation-report",
        default=None,
        help="Optional evaluation_report.json. Defaults to outputs/<experiment-name>/evaluation_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to experiments/<experiment-name>.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset_fused",
        help="Dataset directory used to build absolute image paths in the CSV.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: str, field_name: str, image_id: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer for {field_name} in {image_id}: {value}") from exc


def class_name(task: str, value: int) -> str:
    return CLASS_NAMES.get(task, {}).get(value, str(value))


def confusion_key(task: str, true_value: int, pred_value: int) -> str:
    return (
        f"{TASK_DISPLAY_NAMES[task]}: "
        f"{true_value}-{class_name(task, true_value)} -> "
        f"{pred_value}-{class_name(task, pred_value)}"
    )


def load_manifest_by_image_id(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    manifest: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row.get("image_id", "")
        if image_id:
            manifest[image_id] = row
    return manifest


def build_error_rows(
    predictions: list[dict[str, str]],
    manifest_by_id: dict[str, dict[str, str]],
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []
    total_rows = 0
    any_task_error_count = 0
    task_error_counts: Counter[str] = Counter()
    confusion_counts: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    source_error_counts: Counter[str] = Counter()
    source_total_counts: Counter[str] = Counter()
    fusion_group_error_counts: Counter[str] = Counter()
    fusion_group_total_counts: Counter[str] = Counter()
    source_task_error_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fusion_group_task_error_counts: dict[str, Counter[str]] = defaultdict(Counter)
    error_combination_counts: Counter[str] = Counter()
    abnormal_false_positive = 0
    abnormal_false_negative = 0
    missing_manifest_count = 0

    for row in predictions:
        total_rows += 1
        image_id = row.get("image_id", "")
        manifest = manifest_by_id.get(image_id, {})
        if not manifest:
            missing_manifest_count += 1

        source_dataset = manifest.get("source_dataset", "UNKNOWN")
        source_class = manifest.get("source_class", "UNKNOWN")
        fusion_group = manifest.get("fusion_group", "UNKNOWN")
        source_total_counts[source_dataset] += 1
        fusion_group_total_counts[fusion_group] += 1

        wrong_tasks = []
        task_values: dict[str, tuple[int, int]] = {}
        image_path = manifest.get("image_path", "")
        absolute_image_path = str((dataset_dir / image_path).resolve()) if image_path else ""

        output_row: dict[str, Any] = {
            "image_id": image_id,
            "capture_time": row.get("capture_time", manifest.get("capture_time", "")),
            "image_path": image_path,
            "absolute_image_path": absolute_image_path,
            "source_dataset": source_dataset,
            "source_class": source_class,
            "fusion_group": fusion_group,
            "split": manifest.get("split", ""),
            "source_path": manifest.get("source_path", ""),
            "label_note": manifest.get("label_note", ""),
        }

        for task in TASKS:
            true_value = to_int(row[f"{task}_true"], f"{task}_true", image_id)
            pred_value = to_int(row[f"{task}_pred"], f"{task}_pred", image_id)
            task_values[task] = (true_value, pred_value)

            is_wrong = true_value != pred_value
            output_row[f"{task}_true"] = true_value
            output_row[f"{task}_true_name"] = class_name(task, true_value)
            output_row[f"{task}_pred"] = pred_value
            output_row[f"{task}_pred_name"] = class_name(task, pred_value)
            output_row[f"{task}_wrong"] = int(is_wrong)

            if is_wrong:
                wrong_tasks.append(task)
                task_error_counts[task] += 1
                confusion_counts[task][confusion_key(task, true_value, pred_value)] += 1
                source_task_error_counts[source_dataset][task] += 1
                fusion_group_task_error_counts[fusion_group][task] += 1

        abnormal_true, abnormal_pred = task_values["abnormal_alert"]
        abnormal_error_type = ""
        if abnormal_true == 0 and abnormal_pred == 1:
            abnormal_error_type = "false_positive"
            abnormal_false_positive += 1
        elif abnormal_true == 1 and abnormal_pred == 0:
            abnormal_error_type = "false_negative"
            abnormal_false_negative += 1

        if wrong_tasks:
            any_task_error_count += 1
            source_error_counts[source_dataset] += 1
            fusion_group_error_counts[fusion_group] += 1
            error_combination_counts[" + ".join(wrong_tasks)] += 1
            output_row["wrong_tasks"] = ";".join(wrong_tasks)
            output_row["wrong_task_names"] = ";".join(TASK_DISPLAY_NAMES[task] for task in wrong_tasks)
            output_row["wrong_task_count"] = len(wrong_tasks)
            output_row["abnormal_error_type"] = abnormal_error_type
            error_rows.append(output_row)

    summary = {
        "total_rows": total_rows,
        "any_task_error_count": any_task_error_count,
        "task_error_counts": task_error_counts,
        "confusion_counts": confusion_counts,
        "source_error_counts": source_error_counts,
        "source_total_counts": source_total_counts,
        "fusion_group_error_counts": fusion_group_error_counts,
        "fusion_group_total_counts": fusion_group_total_counts,
        "source_task_error_counts": source_task_error_counts,
        "fusion_group_task_error_counts": fusion_group_task_error_counts,
        "error_combination_counts": error_combination_counts,
        "abnormal_false_positive": abnormal_false_positive,
        "abnormal_false_negative": abnormal_false_negative,
        "missing_manifest_count": missing_manifest_count,
    }
    return error_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_counter_table(counter: Counter[str], total_counter: Counter[str] | None = None) -> list[str]:
    lines = ["| 项目 | 错误数 | 总数 | 错误率 |", "| --- | ---: | ---: | ---: |"]
    for key, count in counter.most_common():
        total = total_counter[key] if total_counter is not None else 0
        rate = count / total if total else 0.0
        lines.append(f"| {key} | {count} | {total if total_counter is not None else '-'} | {rate:.2%} |")
    if len(lines) == 2:
        lines.append("| 无 | 0 | 0 | 0.00% |")
    return lines


def format_task_table(counter: Counter[str], total_rows: int) -> list[str]:
    lines = ["| 任务 | 错误数 | 样本数 | 错误率 |", "| --- | ---: | ---: | ---: |"]
    for task in TASKS:
        count = counter[task]
        lines.append(
            f"| {TASK_DISPLAY_NAMES[task]} (`{task}`) | {count} | {total_rows} | {count / total_rows:.2%} |"
            if total_rows
            else f"| {TASK_DISPLAY_NAMES[task]} (`{task}`) | {count} | 0 | 0.00% |"
        )
    return lines


def format_confusion_sections(confusion_counts: dict[str, Counter[str]]) -> list[str]:
    lines: list[str] = []
    for task in TASKS:
        lines.append(f"### {TASK_DISPLAY_NAMES[task]}")
        lines.append("")
        lines.append("| 混淆方向 | 数量 |")
        lines.append("| --- | ---: |")
        for label, count in confusion_counts[task].most_common(10):
            lines.append(f"| {label} | {count} |")
        if not confusion_counts[task]:
            lines.append("| 无 | 0 |")
        lines.append("")
    return lines


def format_task_breakdown_by_group(
    title: str,
    group_error_counts: Counter[str],
    group_task_error_counts: dict[str, Counter[str]],
    limit: int = 10,
) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| 分组 | 总错误样本数 | 生长阶段 | 健康等级 | 成熟度 | 异常预警 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for group, count in group_error_counts.most_common(limit):
        task_counts = group_task_error_counts[group]
        lines.append(
            "| "
            f"{group} | {count} | "
            f"{task_counts['growth_stage']} | "
            f"{task_counts['health_level']} | "
            f"{task_counts['maturity_level']} | "
            f"{task_counts['abnormal_alert']} |"
        )
    if not group_error_counts:
        lines.append("| 无 | 0 | 0 | 0 | 0 | 0 |")
    lines.append("")
    return lines


def write_summary(
    path: Path,
    experiment_name: str,
    predictions_path: Path,
    manifest_path: Path,
    error_csv_path: Path,
    summary: dict[str, Any],
    evaluation_report: dict[str, Any] | None,
) -> None:
    total_rows = int(summary["total_rows"])
    any_errors = int(summary["any_task_error_count"])
    overall_error_rate = any_errors / total_rows if total_rows else 0.0
    task_error_counts: Counter[str] = summary["task_error_counts"]

    most_error_task = max(TASKS, key=lambda task: task_error_counts[task])
    if task_error_counts[most_error_task] == 0:
        most_error_text = "无"
    else:
        most_error_text = (
            f"{TASK_DISPLAY_NAMES[most_error_task]} "
            f"({task_error_counts[most_error_task]} 次任务级错误)"
        )

    lines = [
        f"# {experiment_name} Error Analysis",
        "",
        "## 输入与输出",
        "",
        f"- 预测文件: `{predictions_path}`",
        f"- 数据来源清单: `{manifest_path}`",
        f"- 错误样本明细: `{error_csv_path}`",
        "",
        "## 总体结论",
        "",
        f"- 测试样本数: {total_rows}",
        f"- 至少一个任务预测错误的样本数: {any_errors}",
        f"- 样本级错误率: {overall_error_rate:.2%}",
        f"- 错误最多的任务: {most_error_text}",
        f"- 异常报警误报 false positive: {summary['abnormal_false_positive']}",
        f"- 异常报警漏报 false negative: {summary['abnormal_false_negative']}",
        f"- 未匹配到 fusion_manifest 的样本数: {summary['missing_manifest_count']}",
        "",
    ]

    if evaluation_report:
        lines.extend([
            "## 原评估指标摘要",
            "",
            f"- checkpoint epoch: {evaluation_report.get('checkpoint_epoch', '')}",
            f"- device: {evaluation_report.get('device', '')}",
            f"- total loss: {evaluation_report.get('losses', {}).get('total_loss', 0.0):.4f}",
            "",
        ])
        metrics = evaluation_report.get("metrics", {})
        lines.extend(["| 任务 | Accuracy | Macro F1 |", "| --- | ---: | ---: |"])
        for task in TASKS:
            task_metrics = metrics.get(task, {})
            lines.append(
                f"| {TASK_DISPLAY_NAMES[task]} | "
                f"{float(task_metrics.get('accuracy', 0.0)):.4f} | "
                f"{float(task_metrics.get('macro_f1', 0.0)):.4f} |"
            )
        lines.append("")

    lines.extend(["## 各任务错误数", ""])
    lines.extend(format_task_table(task_error_counts, total_rows))
    lines.append("")

    lines.extend(["## 错误组合", ""])
    lines.extend(["| 错误任务组合 | 样本数 |", "| --- | ---: |"])
    for combination, count in summary["error_combination_counts"].most_common():
        display = " + ".join(TASK_DISPLAY_NAMES.get(task, task) for task in combination.split(" + "))
        lines.append(f"| {display} | {count} |")
    if not summary["error_combination_counts"]:
        lines.append("| 无 | 0 |")
    lines.append("")

    lines.extend(["## 主要混淆方向", ""])
    lines.extend(format_confusion_sections(summary["confusion_counts"]))

    lines.extend(["## 按数据来源统计", ""])
    lines.extend(format_counter_table(summary["source_error_counts"], summary["source_total_counts"]))
    lines.append("")

    lines.extend(["## 按融合分组统计", ""])
    lines.extend(format_counter_table(summary["fusion_group_error_counts"], summary["fusion_group_total_counts"]))
    lines.append("")

    lines.extend(
        format_task_breakdown_by_group(
            title="数据来源中的任务错误拆分",
            group_error_counts=summary["source_error_counts"],
            group_task_error_counts=summary["source_task_error_counts"],
        )
    )
    lines.extend(
        format_task_breakdown_by_group(
            title="融合分组中的任务错误拆分",
            group_error_counts=summary["fusion_group_error_counts"],
            group_task_error_counts=summary["fusion_group_task_error_counts"],
        )
    )

    lines.extend([
        "## 下一步建议",
        "",
        "1. 优先人工查看 `error_analysis.csv` 中健康等级和异常预警相关错误样本。",
        "2. 对错误集中的数据来源或融合分组，检查原始标签映射是否过于简单。",
        "3. 重点复查异常报警漏报样本，因为漏报比误报更影响监测系统安全性。",
        "4. 下一次实验建议做 source-aware split 或 group-aware split，验证模型是否依赖数据来源特征。",
        "5. 在真实大棚人工标注数据到位后，把这次结果作为基线或预训练结果，不直接作为最终结论。",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    predictions_path = Path(args.predictions) if args.predictions else Path("outputs") / experiment_name / "predictions.csv"
    manifest_path = Path(args.manifest)
    evaluation_report_path = (
        Path(args.evaluation_report)
        if args.evaluation_report
        else Path("outputs") / experiment_name / "evaluation_report.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / experiment_name
    dataset_dir = Path(args.dataset_dir)

    predictions = read_csv(predictions_path)
    manifest_by_id = load_manifest_by_image_id(manifest_path)
    evaluation_report = read_json(evaluation_report_path)

    error_rows, summary = build_error_rows(
        predictions=predictions,
        manifest_by_id=manifest_by_id,
        dataset_dir=dataset_dir,
    )

    error_csv_path = output_dir / "error_analysis.csv"
    summary_path = output_dir / "error_summary.md"
    write_csv(error_csv_path, error_rows)
    write_summary(
        path=summary_path,
        experiment_name=experiment_name,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        error_csv_path=error_csv_path,
        summary=summary,
        evaluation_report=evaluation_report,
    )

    print(f"experiment={experiment_name}")
    print(f"samples={summary['total_rows']}")
    print(f"error_samples={summary['any_task_error_count']}")
    print(f"abnormal_false_positive={summary['abnormal_false_positive']}")
    print(f"abnormal_false_negative={summary['abnormal_false_negative']}")
    print(f"saved_error_analysis={error_csv_path}")
    print(f"saved_error_summary={summary_path}")


if __name__ == "__main__":
    main()
