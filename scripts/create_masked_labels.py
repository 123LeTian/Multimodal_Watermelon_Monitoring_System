from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("growth_stage", "health_level", "maturity_level", "abnormal_alert")
MASK_FIELDS = tuple(f"{task}_valid" for task in TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create labels_masked.csv with per-sample task validity masks."
    )
    parser.add_argument("--labels", default="dataset_fused/labels.csv")
    parser.add_argument("--manifest", default="dataset_fused/fusion_manifest.csv")
    parser.add_argument(
        "--recommendations",
        default="experiments/exp006_label_audit/task_validity_recommendation.csv",
    )
    parser.add_argument("--output", default="dataset_fused/labels_masked.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y", "是"})


def load_masks_by_source(path: Path) -> dict[str, dict[str, int]]:
    rows = read_csv(path)
    masks: dict[str, dict[str, int]] = {}
    for row in rows:
        source_dataset = row["source_dataset"]
        masks[source_dataset] = {
            "growth_stage_valid": truthy(row["recommended_growth_stage_valid"]),
            "health_level_valid": truthy(row["recommended_health_level_valid"]),
            "maturity_level_valid": truthy(row["recommended_maturity_level_valid"]),
            "abnormal_alert_valid": truthy(row["recommended_abnormal_alert_valid"]),
        }
    return masks


def main() -> None:
    args = parse_args()
    labels_path = (ROOT / args.labels).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    recommendations_path = (ROOT / args.recommendations).resolve()
    output_path = (ROOT / args.output).resolve()

    label_rows = read_csv(labels_path)
    manifest_rows = read_csv(manifest_path)
    masks_by_source = load_masks_by_source(recommendations_path)
    manifest_by_id = {row["image_id"]: row for row in manifest_rows}

    output_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    source_valid_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in label_rows:
        image_id = row["image_id"]
        manifest = manifest_by_id.get(image_id)
        if manifest is None:
            raise ValueError(f"missing manifest row for image_id={image_id}")

        source_dataset = manifest["source_dataset"]
        source_counts[source_dataset] += 1
        source_masks = masks_by_source.get(source_dataset)
        if source_masks is None:
            raise ValueError(f"missing task validity recommendation for {source_dataset}")

        output_row: dict[str, Any] = dict(row)
        for mask_field in MASK_FIELDS:
            value = int(source_masks[mask_field])
            output_row[mask_field] = value
            if value:
                valid_counts[mask_field] += 1
                source_valid_counts[source_dataset][mask_field] += 1
        output_rows.append(output_row)

    fieldnames = list(label_rows[0].keys()) + list(MASK_FIELDS)
    write_csv(output_path, output_rows, fieldnames)

    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"recommendations={recommendations_path}")
    print(f"output={output_path}")
    print(f"samples={len(output_rows)}")
    print("mask_counts:")
    for mask_field in MASK_FIELDS:
        print(f"  {mask_field}={valid_counts[mask_field]}")
    print("source_counts:")
    for source_dataset, count in sorted(source_counts.items()):
        masks = ", ".join(
            f"{field}={source_valid_counts[source_dataset][field]}" for field in MASK_FIELDS
        )
        print(f"  {source_dataset}: samples={count}, {masks}")


if __name__ == "__main__":
    main()
