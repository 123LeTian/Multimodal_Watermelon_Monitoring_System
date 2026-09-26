from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

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

SOURCE_RECOMMENDATIONS = {
    "watermelon_disease_recognition": {
        "growth_stage": False,
        "health_level": True,
        "maturity_level": False,
        "abnormal_alert": True,
        "growth_stage_confidence": "low",
        "health_level_confidence": "high",
        "maturity_level_confidence": "low",
        "abnormal_alert_confidence": "high",
        "reason": (
            "原始任务是西瓜叶片健康/病害识别，health_level 和 abnormal_alert "
            "与人工类别一致；growth_stage 来自环境/融合规则先验，maturity_level=0 "
            "只是叶片样本的占位推断。"
        ),
    },
    "kurdistan_watermelon_disease": {
        "growth_stage": False,
        "health_level": True,
        "maturity_level": False,
        "abnormal_alert": True,
        "growth_stage_confidence": "low",
        "health_level_confidence": "high",
        "maturity_level_confidence": "low",
        "abnormal_alert_confidence": "high",
        "reason": (
            "原始任务覆盖健康、病害、虫害和缺素叶片类别，适合训练健康等级和异常预警；"
            "生长阶段是规则/环境时间先验，成熟度是非果实样本占位标签。"
        ),
    },
    "watermelon_ripe_semiripe_unripe": {
        "growth_stage": False,
        "health_level": False,
        "maturity_level": True,
        "abnormal_alert": False,
        "growth_stage_confidence": "low",
        "health_level_confidence": "low",
        "maturity_level_confidence": "high",
        "abnormal_alert_confidence": "low",
        "reason": (
            "原始任务是果实成熟度/未成熟度识别，适合 maturity_level；"
            "health_level=0 和 abnormal_alert=0 多为健康假设，growth_stage 由成熟度映射得到。"
        ),
    },
    "watermelon_ripe_unripe": {
        "growth_stage": False,
        "health_level": False,
        "maturity_level": True,
        "abnormal_alert": False,
        "growth_stage_confidence": "low",
        "health_level_confidence": "low",
        "maturity_level_confidence": "high",
        "abnormal_alert_confidence": "low",
        "reason": (
            "原始任务是 ripe/unripe 二分类，适合训练 maturity_level 的未成熟/成熟边界；"
            "健康和异常标签是非异常假设，生长阶段是弱映射。"
        ),
    },
    "fruq_db": {
        "growth_stage": False,
        "health_level": True,
        "maturity_level": True,
        "abnormal_alert": True,
        "growth_stage_confidence": "low",
        "health_level_confidence": "medium",
        "maturity_level_confidence": "medium",
        "abnormal_alert_confidence": "medium",
        "reason": (
            "原始任务是通用水果 fresh/mild/rotten 质量标签，可作为果实质量异常、过熟/腐烂补充；"
            "不是西瓜专属大棚标注，health_level 和 maturity_level 应降权使用，growth_stage=5 是规则推断。"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fused weak labels and recommend source/task validity masks."
    )
    parser.add_argument("--dataset-dir", default="dataset_fused")
    parser.add_argument("--exp005-output-dir", default="outputs/exp005_source_holdout")
    parser.add_argument("--exp005-experiment-dir", default="experiments/exp005_source_holdout")
    parser.add_argument("--output-dir", default="experiments/exp006_label_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str) -> int:
    return int(value)


def pct(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def class_label(task: str, value: str | int) -> str:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int_value}-{CLASS_NAMES[task].get(int_value, str(int_value))}"


def compact_counter(counter: Counter[str | int], task: str | None = None, limit: int | None = None) -> str:
    items = counter.most_common(limit)
    if not items:
        return "无"
    parts = []
    for key, count in items:
        label = class_label(task, key) if task else str(key)
        parts.append(f"{label}:{count}")
    return "; ".join(parts)


def split_summary(source_dataset: str, split_rows: list[dict[str, str]], source_by_id: dict[str, str]) -> str:
    counter: Counter[str] = Counter()
    for row in split_rows:
        if source_by_id.get(row["image_id"]) == source_dataset:
            counter[row["split"]] += 1
    return compact_counter(counter)


def validate_label_alignment(
    manifest_rows: list[dict[str, str]], label_rows: list[dict[str, str]]
) -> dict[str, Any]:
    labels_by_id = {row["image_id"]: row for row in label_rows}
    missing_in_labels = []
    mismatches: list[str] = []

    for row in manifest_rows:
        image_id = row["image_id"]
        label_row = labels_by_id.get(image_id)
        if not label_row:
            missing_in_labels.append(image_id)
            continue
        for field in ("image_path", "capture_time", *TASKS):
            if str(row[field]) != str(label_row[field]):
                mismatches.append(f"{image_id}:{field}")

    manifest_ids = {row["image_id"] for row in manifest_rows}
    extra_label_ids = sorted(set(labels_by_id) - manifest_ids)
    return {
        "manifest_rows": len(manifest_rows),
        "label_rows": len(label_rows),
        "missing_in_labels": len(missing_in_labels),
        "extra_label_ids": len(extra_label_ids),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }


def build_source_distribution_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_source[row["source_dataset"]].append(row)

    for source_dataset in sorted(by_source):
        source_rows = by_source[source_dataset]
        source_total = len(source_rows)
        distribution_fields = ("source_class", "fusion_group", *TASKS)
        for field in distribution_fields:
            counts: Counter[str] = Counter(row[field] for row in source_rows)
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
                rows.append(
                    {
                        "source_dataset": source_dataset,
                        "sample_count": source_total,
                        "distribution_field": field,
                        "label_value": value,
                        "label_name": class_label(field, value) if field in TASKS else value,
                        "count": count,
                        "percent_within_source": pct(count, source_total),
                    }
                )
    return rows


def build_fusion_group_distribution_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_group[row["fusion_group"]].append(row)

    for fusion_group in sorted(by_group):
        group_rows = by_group[fusion_group]
        group_total = len(group_rows)
        for task in TASKS:
            counts: Counter[str] = Counter(row[task] for row in group_rows)
            for value, count in sorted(counts.items(), key=lambda item: (int(item[0]), item[0])):
                rows.append(
                    {
                        "fusion_group": fusion_group,
                        "sample_count": group_total,
                        "task": task,
                        "task_name": TASK_DISPLAY_NAMES[task],
                        "label_value": value,
                        "label_name": class_label(task, value),
                        "count": count,
                        "percent_within_fusion_group": pct(count, group_total),
                    }
                )
    return rows


def build_recommendation_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_source[row["source_dataset"]].append(row)

    for source_dataset in sorted(by_source):
        source_rows = by_source[source_dataset]
        rec = SOURCE_RECOMMENDATIONS.get(
            source_dataset,
            {
                "growth_stage": False,
                "health_level": False,
                "maturity_level": False,
                "abnormal_alert": False,
                "growth_stage_confidence": "unknown",
                "health_level_confidence": "unknown",
                "maturity_level_confidence": "unknown",
                "abnormal_alert_confidence": "unknown",
                "reason": "未知来源，需人工确认原始任务后再作为监督信号。",
            },
        )
        groups = Counter(row["fusion_group"] for row in source_rows)
        classes = Counter(row["source_class"] for row in source_rows)
        rows.append(
            {
                "source_dataset": source_dataset,
                "sample_count": len(source_rows),
                "recommended_growth_stage_valid": rec["growth_stage"],
                "recommended_health_level_valid": rec["health_level"],
                "recommended_maturity_level_valid": rec["maturity_level"],
                "recommended_abnormal_alert_valid": rec["abnormal_alert"],
                "growth_stage_confidence": rec["growth_stage_confidence"],
                "health_level_confidence": rec["health_level_confidence"],
                "maturity_level_confidence": rec["maturity_level_confidence"],
                "abnormal_alert_confidence": rec["abnormal_alert_confidence"],
                "dominant_source_classes": compact_counter(classes, limit=5),
                "fusion_groups": compact_counter(groups),
                "reason": rec["reason"],
            }
        )
    return rows


def exp005_task_errors(evaluation_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metrics = evaluation_report.get("metrics", {})
    total_samples = int(evaluation_report.get("samples", 0))
    for task in TASKS:
        task_metrics = metrics.get(task, {})
        support = task_metrics.get("support", [])
        confusion = task_metrics.get("confusion_matrix", [])
        sample_count = sum(int(value) for value in support) if support else total_samples
        correct = 0
        for index, row in enumerate(confusion):
            if index < len(row):
                correct += int(row[index])
        errors = max(sample_count - correct, 0)
        rows.append(
            {
                "task": task,
                "task_name": TASK_DISPLAY_NAMES[task],
                "sample_count": sample_count,
                "error_count": errors,
                "error_rate": pct(errors, sample_count),
                "accuracy": task_metrics.get("accuracy", 0.0),
                "macro_f1": task_metrics.get("macro_f1", 0.0),
                "support": support,
            }
        )
    return rows


def extract_error_summary_facts(error_summary_text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    patterns = {
        "sample_error_rate": r"样本级错误率:\s*([0-9.]+%)",
        "most_wrong_task": r"错误最多的任务:\s*(.+)",
        "abnormal_false_positive": r"异常报警误报 false positive:\s*(\d+)",
        "abnormal_false_negative": r"异常报警漏报 false negative:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, error_summary_text)
        if match:
            facts[key] = match.group(1).strip()
    return facts


def source_overview_rows(
    manifest_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    quality_report: dict[str, Any],
) -> list[dict[str, str]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_source[row["source_dataset"]].append(row)
    source_by_id = {row["image_id"]: row["source_dataset"] for row in manifest_rows}
    roles = quality_report.get("sources", {})

    rows = []
    for source_dataset in sorted(by_source):
        source_rows = by_source[source_dataset]
        rows.append(
            {
                "source_dataset": source_dataset,
                "sample_count": str(len(source_rows)),
                "source_aware_split": split_summary(source_dataset, split_rows, source_by_id),
                "source_classes": compact_counter(Counter(row["source_class"] for row in source_rows), limit=6),
                "fusion_groups": compact_counter(Counter(row["fusion_group"] for row in source_rows)),
                "role": roles.get(source_dataset, {}).get("role", ""),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def label_distribution_summary(
    manifest_rows: list[dict[str, str]], group_field: str
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[row[group_field]].append(row)

    summaries = []
    for key in sorted(grouped):
        rows = grouped[key]
        summary = {
            group_field: key,
            "sample_count": str(len(rows)),
        }
        if group_field == "source_dataset":
            summary["source_classes"] = compact_counter(
                Counter(row["source_class"] for row in rows), limit=5
            )
            summary["fusion_groups"] = compact_counter(Counter(row["fusion_group"] for row in rows))
        for task in TASKS:
            summary[task] = compact_counter(Counter(row[task] for row in rows), task=task)
        summaries.append(summary)
    return summaries


def bool_text(value: Any) -> str:
    return "是" if value in (True, "True", "true", "1", 1) else "否"


def build_report(
    manifest_rows: list[dict[str, str]],
    label_alignment: dict[str, Any],
    split_rows: list[dict[str, str]],
    quality_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    error_summary_text: str,
    recommendation_rows: list[dict[str, Any]],
) -> str:
    overview = source_overview_rows(manifest_rows, split_rows, quality_report)
    source_summaries = label_distribution_summary(manifest_rows, "source_dataset")
    group_summaries = label_distribution_summary(manifest_rows, "fusion_group")
    exp005_errors = exp005_task_errors(evaluation_report)
    error_facts = extract_error_summary_facts(error_summary_text)
    heldout_sources = sorted(
        {
            row["source_dataset"]
            for row in manifest_rows
            if any(split["image_id"] == row["image_id"] and split["split"] == "test" for split in split_rows)
        }
    )
    source_by_id = {row["image_id"]: row["source_dataset"] for row in manifest_rows}
    group_by_id = {row["image_id"]: row["fusion_group"] for row in manifest_rows}
    heldout_group_counts: Counter[str] = Counter()
    for split in split_rows:
        if split["split"] == "test":
            heldout_group_counts[group_by_id.get(split["image_id"], "UNKNOWN")] += 1

    lines: list[str] = [
        "# exp006_label_audit 标签任务有效性审计",
        "",
        "## 本次分析目的",
        "",
        (
            "本次实验不训练模型，只审计 `dataset_fused` 的弱监督融合标签是否适合作为多任务训练目标。"
            "重点是把原始数据集真实承担的任务，与融合规则补出的辅助标签区分开，"
            "为后续 masked multitask training 设计 loss mask 和任务权重。"
        ),
        "",
        "## 输入与一致性检查",
        "",
        f"- fusion_manifest 样本数: {label_alignment['manifest_rows']}",
        f"- labels.csv 样本数: {label_alignment['label_rows']}",
        f"- manifest 中缺失 labels 的样本数: {label_alignment['missing_in_labels']}",
        f"- labels 中额外样本数: {label_alignment['extra_label_ids']}",
        f"- manifest 与 labels 字段不一致数量: {label_alignment['mismatch_count']}",
        f"- source-aware split: {compact_counter(Counter(row['split'] for row in split_rows))}",
        "",
        "## 数据来源概况",
        "",
    ]

    lines.extend(
        markdown_table(
            ["source_dataset", "样本数", "source-aware split", "主要 source_class", "fusion_group", "原始角色"],
            [
                [
                    row["source_dataset"],
                    row["sample_count"],
                    row["source_aware_split"],
                    row["source_classes"],
                    row["fusion_groups"],
                    row["role"],
                ]
                for row in overview
            ],
        )
    )
    lines.extend(["", "## 各 source_dataset 的标签分布总结", ""])
    lines.extend(
        markdown_table(
            [
                "source_dataset",
                "样本数",
                "source_class",
                "fusion_group",
                "growth_stage",
                "health_level",
                "maturity_level",
                "abnormal_alert",
            ],
            [
                [
                    row["source_dataset"],
                    row["sample_count"],
                    row["source_classes"],
                    row["fusion_groups"],
                    row["growth_stage"],
                    row["health_level"],
                    row["maturity_level"],
                    row["abnormal_alert"],
                ]
                for row in source_summaries
            ],
        )
    )
    lines.extend(["", "## 各 fusion_group 的标签分布总结", ""])
    lines.extend(
        markdown_table(
            ["fusion_group", "样本数", "growth_stage", "health_level", "maturity_level", "abnormal_alert"],
            [
                [
                    row["fusion_group"],
                    row["sample_count"],
                    row["growth_stage"],
                    row["health_level"],
                    row["maturity_level"],
                    row["abnormal_alert"],
                ]
                for row in group_summaries
            ],
        )
    )

    lines.extend(
        [
            "",
            "## exp005 泛化下降原因分析",
            "",
            (
                f"`split_source_aware.csv` 的 test 来源为 {', '.join(heldout_sources) or '未知'}，"
                f"held-out fusion_group 为 {compact_counter(heldout_group_counts)}。"
            ),
            (
                "exp005 的随机划分乐观问题主要来自 source leakage：训练和测试若混有同一数据来源，"
                "模型容易学习数据集风格、背景和采集域，而不是稳定的西瓜生理/病害语义。"
            ),
            (
                "在完全未见过的 `watermelon_disease_recognition` 上，错误明显集中在健康等级和异常预警；"
                "成熟度 0 错误不代表成熟度泛化好，因为 held-out 源全部是叶片图像，support 只覆盖 maturity=0。"
            ),
            "",
        ]
    )
    if error_facts:
        lines.extend(
            [
                (
                    f"error_summary 摘要: 样本级错误率 {error_facts.get('sample_error_rate', '未知')}；"
                    f"错误最多的任务为 {error_facts.get('most_wrong_task', '未知')}；"
                    f"异常报警误报 {error_facts.get('abnormal_false_positive', '未知')}，"
                    f"漏报 {error_facts.get('abnormal_false_negative', '未知')}。"
                ),
                "",
            ]
        )
    lines.extend(
        markdown_table(
            ["任务", "样本数", "错误数", "错误率", "Accuracy", "Macro F1", "support"],
            [
                [
                    row["task_name"],
                    row["sample_count"],
                    row["error_count"],
                    f"{row['error_rate']:.2%}",
                    f"{row['accuracy']:.4f}",
                    f"{row['macro_f1']:.4f}",
                    row["support"],
                ]
                for row in exp005_errors
            ],
        )
    )

    lines.extend(["", "## 任务可信度建议", ""])
    lines.extend(
        markdown_table(
            [
                "source_dataset",
                "样本数",
                "growth",
                "health",
                "maturity",
                "abnormal",
                "原因",
            ],
            [
                [
                    row["source_dataset"],
                    row["sample_count"],
                    f"{bool_text(row['recommended_growth_stage_valid'])}/{row['growth_stage_confidence']}",
                    f"{bool_text(row['recommended_health_level_valid'])}/{row['health_level_confidence']}",
                    f"{bool_text(row['recommended_maturity_level_valid'])}/{row['maturity_level_confidence']}",
                    f"{bool_text(row['recommended_abnormal_alert_valid'])}/{row['abnormal_alert_confidence']}",
                    row["reason"],
                ]
                for row in recommendation_rows
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 哪些任务标签可信",
            "",
            "- `health_level`: 叶片病害/健康来源最可信，包括 `watermelon_disease_recognition` 和 `kurdistan_watermelon_disease`；`fruq_db` 可作为果实质量异常补充，但不应与叶片病害完全等权。",
            "- `abnormal_alert`: disease/healthy/mild leaf 和 fruit quality 中的异常/腐烂信号可用；ripe/unripe 果实来源的全 0 异常标签更像负例假设。",
            "- `maturity_level`: `watermelon_ripe_semiripe_unripe` 和 `watermelon_ripe_unripe` 最适合；`fruq_db` 只适合作为 fresh/rotten/overripe 的补充弱监督。",
            "",
            "## 哪些任务标签风险较高",
            "",
            "- `growth_stage`: 当前主要来自融合规则、成熟度映射或环境时间先验，不是原始人工生长阶段标注；不建议作为强监督主任务。",
            "- leaf source 的 `maturity_level=0`: 只是非果实/未成熟占位，不能说明真实果实成熟度。",
            "- ripe/unripe fruit source 的 `health_level=0` 与 `abnormal_alert=0`: 多数是缺少病害标注时的健康假设，不能当作强健康监督。",
            "- `fruq_db`: 是通用水果质量域，不是西瓜大棚域；可增加异常多样性，但应降权并做域泛化验证。",
            "",
            "## 后续 masked loss 设计建议",
            "",
            "- 按 `source_dataset` 或 `fusion_group` 生成 per-sample task mask，不要让四个任务对每张图都等权反传。",
            "- `health_level`: leaf health/disease groups 权重 1.0；fruit quality groups 可设 0.3-0.5；ripe/unripe fruit groups 设 0 或仅作很低权重负例。",
            "- `abnormal_alert`: disease_leaf、mild_leaf_abnormal、healthy_leaf、mild_quality、rotten_quality、fresh_quality 可训练；ripe_fruit/unripe_fruit 建议 mask 或低权重，避免把“未标异常”学成“无异常”。",
            "- `maturity_level`: ripe_fruit/unripe_fruit 权重 1.0；fresh_quality/rotten_quality/mild_quality 可降权；leaf groups 应 mask。",
            "- `growth_stage`: 暂不作为强监督，建议 mask 或极低辅助权重；等真实大棚生育期标注后再提升权重。",
            "",
            "## 是否建议继续使用 dataset_fused",
            "",
            "建议继续使用，但定位应从“四任务强监督数据集”调整为“跨来源弱监督预训练数据集”。如果使用 task mask、source-aware 验证和任务降权，`dataset_fused` 仍然适合为 exp007 提供预训练和多任务表示学习基础。",
            "",
            "## 是否需要真实大棚数据微调",
            "",
            "需要。exp005 已经说明未见数据来源会显著拉低健康等级和异常报警；最终监测系统必须用真实大棚、目标相机、目标品种和人工审核标签做微调与验证，尤其是健康等级、异常报警和生长阶段。",
            "",
            "## 生成文件",
            "",
            "- `source_label_distribution.csv`: source_dataset 下 source_class、fusion_group 和四任务标签长表分布。",
            "- `fusion_group_label_distribution.csv`: fusion_group 下四任务标签长表分布。",
            "- `task_validity_recommendation.csv`: 每个 source_dataset 的任务有效性建议。",
            "- `label_task_validity_report.md`: 本报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_dir = (ROOT / args.dataset_dir).resolve()
    exp005_output_dir = (ROOT / args.exp005_output_dir).resolve()
    exp005_experiment_dir = (ROOT / args.exp005_experiment_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()

    manifest_path = dataset_dir / "fusion_manifest.csv"
    labels_path = dataset_dir / "labels.csv"
    split_path = dataset_dir / "split_source_aware.csv"
    quality_path = dataset_dir / "quality_report.json"
    evaluation_path = exp005_output_dir / "evaluation_report.json"
    error_summary_path = exp005_experiment_dir / "error_summary.md"

    manifest_rows = read_csv(manifest_path)
    label_rows = read_csv(labels_path)
    split_rows = read_csv(split_path)
    quality_report = read_json(quality_path)
    evaluation_report = read_json(evaluation_path)
    error_summary_text = error_summary_path.read_text(encoding="utf-8")

    label_alignment = validate_label_alignment(manifest_rows, label_rows)
    source_distribution_rows = build_source_distribution_rows(manifest_rows)
    fusion_group_distribution_rows = build_fusion_group_distribution_rows(manifest_rows)
    recommendation_rows = build_recommendation_rows(manifest_rows)

    write_csv(
        output_dir / "source_label_distribution.csv",
        source_distribution_rows,
        [
            "source_dataset",
            "sample_count",
            "distribution_field",
            "label_value",
            "label_name",
            "count",
            "percent_within_source",
        ],
    )
    write_csv(
        output_dir / "fusion_group_label_distribution.csv",
        fusion_group_distribution_rows,
        [
            "fusion_group",
            "sample_count",
            "task",
            "task_name",
            "label_value",
            "label_name",
            "count",
            "percent_within_fusion_group",
        ],
    )
    write_csv(
        output_dir / "task_validity_recommendation.csv",
        recommendation_rows,
        [
            "source_dataset",
            "sample_count",
            "recommended_growth_stage_valid",
            "recommended_health_level_valid",
            "recommended_maturity_level_valid",
            "recommended_abnormal_alert_valid",
            "growth_stage_confidence",
            "health_level_confidence",
            "maturity_level_confidence",
            "abnormal_alert_confidence",
            "dominant_source_classes",
            "fusion_groups",
            "reason",
        ],
    )

    report = build_report(
        manifest_rows,
        label_alignment,
        split_rows,
        quality_report,
        evaluation_report,
        error_summary_text,
        recommendation_rows,
    )
    (output_dir / "label_task_validity_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {output_dir / 'source_label_distribution.csv'}")
    print(f"Wrote {output_dir / 'fusion_group_label_distribution.csv'}")
    print(f"Wrote {output_dir / 'task_validity_recommendation.csv'}")
    print(f"Wrote {output_dir / 'label_task_validity_report.md'}")


if __name__ == "__main__":
    main()
