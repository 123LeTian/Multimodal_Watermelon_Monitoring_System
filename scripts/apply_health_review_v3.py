from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SOURCE_SPLITS = {
    "watermelon_disease_recognition": "train",
    "watermelon_ripe_semiripe_unripe": "train",
    "kurdistan_watermelon_disease": "val",
    "watermelon_ripe_unripe": "val",
    "fruq_db": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply reviewed health severity labels and create V3 source splits."
    )
    parser.add_argument(
        "--queue",
        default="experiments/health_severity_review_v3/health_review_queue.csv",
    )
    parser.add_argument(
        "--decisions",
        default="experiments/health_severity_review_v3/review_decisions.yaml",
    )
    parser.add_argument("--labels", default="dataset_fused/labels_masked.csv")
    parser.add_argument("--manifest", default="dataset_fused/fusion_manifest.csv")
    parser.add_argument(
        "--output-labels",
        default="dataset_fused/labels_reviewed_v3.csv",
    )
    parser.add_argument(
        "--output-split",
        default="dataset_fused/split_source_isolated_v3.csv",
    )
    parser.add_argument(
        "--output-report",
        default="dataset_fused/reviewed_v3_report.json",
    )
    parser.add_argument(
        "--output-reviewed-queue",
        default="experiments/health_severity_review_v3/health_review_completed.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    queue = read_csv(Path(args.queue))
    labels = read_csv(Path(args.labels))
    manifest_rows = read_csv(Path(args.manifest))
    decisions = yaml.safe_load(Path(args.decisions).read_text(encoding="utf-8"))

    queue_by_index = {int(row["review_index"]): row for row in queue}
    expected_indices = set(range(1, len(queue) + 1))
    if set(queue_by_index) != expected_indices:
        raise ValueError("review queue indices are not contiguous")

    groups = {
        "invalid": set(int(value) for value in decisions["invalid_indices"]),
        "mild": set(int(value) for value in decisions["mild_indices"]),
        "severe": set(int(value) for value in decisions["severe_indices"]),
    }
    for name, values in groups.items():
        unknown = values - expected_indices
        if unknown:
            raise ValueError(f"{name} decisions contain unknown indices: {sorted(unknown)}")
    if groups["invalid"] & groups["mild"]:
        raise ValueError("invalid and mild decisions overlap")
    if groups["invalid"] & groups["severe"]:
        raise ValueError("invalid and severe decisions overlap")
    if groups["mild"] & groups["severe"]:
        raise ValueError("mild and severe decisions overlap")

    default_level = int(decisions["default_health_level"])
    reviewed_by_image_id: dict[str, tuple[int, int, str]] = {}
    completed_queue: list[dict[str, object]] = []
    decision_counts: Counter[str] = Counter()
    for index in sorted(queue_by_index):
        row = dict(queue_by_index[index])
        if index in groups["invalid"]:
            level = default_level
            valid = 0
            reason = "severity_not_confirmable_from_partial_or_non_leaf_view"
            decision_counts["invalid"] += 1
        elif index in groups["mild"]:
            level = 1
            valid = 1
            reason = "localized_mild_symptoms"
            decision_counts["mild"] += 1
        elif index in groups["severe"]:
            level = 3
            valid = 1
            reason = "large_area_damage_deformation_wilting_or_necrosis"
            decision_counts["severe"] += 1
        else:
            level = default_level
            valid = 1
            reason = "multiple_regions_or_clear_lesions"
            decision_counts["moderate"] += 1
        row["review_health_level"] = level
        row["review_health_level_valid"] = valid
        row["review_reason"] = reason
        completed_queue.append(row)
        reviewed_by_image_id[row["image_id"]] = (level, valid, reason)

    changed_labels = 0
    invalidated_labels = 0
    for row in labels:
        decision = reviewed_by_image_id.get(row["image_id"])
        if decision is None:
            continue
        level, valid, _ = decision
        if int(row["health_level"]) != level:
            changed_labels += 1
        if int(row["health_level_valid"]) != valid and valid == 0:
            invalidated_labels += 1
        row["health_level"] = str(level)
        row["health_level_valid"] = str(valid)

    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    split_rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    label_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    labels_by_id = {row["image_id"]: row for row in labels}
    for image_id, manifest in manifest_by_id.items():
        source = manifest["source_dataset"]
        if source not in SOURCE_SPLITS:
            raise ValueError(f"source has no V3 split assignment: {source}")
        split = SOURCE_SPLITS[source]
        split_rows.append({"image_id": image_id, "split": split})
        split_counts[split] += 1
        label = labels_by_id[image_id]
        for task in ("growth_stage", "health_level", "maturity_level", "abnormal_alert"):
            if int(label[f"{task}_valid"]) > 0:
                label_counts[split][task][label[task]] += 1

    output_labels = Path(args.output_labels)
    output_split = Path(args.output_split)
    output_queue = Path(args.output_reviewed_queue)
    write_csv(output_labels, labels, list(labels[0]))
    write_csv(output_split, split_rows, ["image_id", "split"])
    write_csv(output_queue, completed_queue, list(completed_queue[0]))

    report = {
        "reviewed_samples": len(completed_queue),
        "decision_counts": dict(decision_counts),
        "changed_health_labels": changed_labels,
        "invalidated_health_labels": invalidated_labels,
        "source_splits": SOURCE_SPLITS,
        "split_counts": dict(split_counts),
        "valid_label_counts": {
            split: {
                task: dict(sorted(counts.items()))
                for task, counts in tasks.items()
            }
            for split, tasks in label_counts.items()
        },
        "outputs": {
            "labels": str(output_labels),
            "split": str(output_split),
            "reviewed_queue": str(output_queue),
        },
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
