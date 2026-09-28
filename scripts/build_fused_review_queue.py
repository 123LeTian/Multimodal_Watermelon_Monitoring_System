from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42

QUOTAS = {
    # The fused dataset cannot satisfy the full real-data stage x health table.
    # These quotas keep 5,000 samples with exact health targets and usable
    # maturity coverage while preserving provenance for manual review.
    (0, 0, 0): 250,
    (1, 0, 0): 250,
    (2, 0, 0): 250,
    (3, 0, 0): 250,
    (4, 0, 0): 600,
    (5, 0, 2): 900,
    (1, 1, 0): 70,
    (2, 1, 0): 70,
    (3, 1, 0): 60,
    (5, 1, 1): 800,
    (2, 2, 0): 450,
    (3, 2, 0): 450,
    (5, 3, 3): 600,
}

LABEL_FIELDS = (
    "image_id",
    "image_path",
    "capture_time",
    "plant_id",
    "batch_id",
    "device_id",
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
    "growth_stage_valid",
    "health_level_valid",
    "maturity_level_valid",
    "abnormal_alert_valid",
    "annotator_count",
    "label_status",
    "review_notes",
)

PROVENANCE_FIELDS = (
    "image_id",
    "source_image_id",
    "source_dataset",
    "source_class",
    "fusion_group",
    "source_path",
    "source_label_note",
    "source_capture_time",
    "review_warning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 5,000-image manual-review queue from dataset_fused."
    )
    parser.add_argument("--source-dir", default="dataset_fused")
    parser.add_argument("--output-dir", default="dataset_real_v001/review_queue_from_fused")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    rng = random.Random(SEED)
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["growth_stage"]),
            int(row["health_level"]),
            int(row["maturity_level"]),
        )
        grouped[key].append(row)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, str]] = []
    shortfalls = {}
    for key, target in QUOTAS.items():
        available = grouped.get(key, [])
        take = min(target, len(available))
        selected.extend(available[:take])
        if take < target:
            shortfalls[str(key)] = {"target": target, "available": len(available)}

    if len(selected) != 5000:
        raise ValueError(
            f"selection produced {len(selected)} rows, expected 5000; shortfalls={shortfalls}"
        )
    rng.shuffle(selected)

    summary = {
        "seed": SEED,
        "selected": len(selected),
        "quotas": {str(key): value for key, value in QUOTAS.items()},
        "shortfalls": shortfalls,
        "growth_stage_counts": dict(sorted(Counter(row["growth_stage"] for row in selected).items())),
        "health_level_counts": dict(sorted(Counter(row["health_level"] for row in selected).items())),
        "maturity_level_counts": dict(sorted(Counter(row["maturity_level"] for row in selected).items())),
        "abnormal_alert_counts": dict(sorted(Counter(row["abnormal_alert"] for row in selected).items())),
    }
    return selected, summary


def split_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["growth_stage"], row["health_level"], row["maturity_level"])].append(row)

    split_by_id: dict[str, str] = {}
    for group_rows in grouped.values():
        total = len(group_rows)
        train_end = round(total * 0.70)
        val_end = train_end + round(total * 0.15)
        for index, row in enumerate(group_rows):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            split_by_id[row["new_image_id"]] = split
    return split_by_id


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_by_id = {row["image_id"]: row for row in read_csv(source_dir / "labels_masked.csv")}
    manifest_by_id = {row["image_id"]: row for row in read_csv(source_dir / "fusion_manifest.csv")}
    selected, summary = choose_rows(list(labels_by_id.values()))

    label_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    for index, source_row in enumerate(selected, start=1):
        source_id = source_row["image_id"]
        manifest_row = manifest_by_id[source_id]
        source_image_path = source_dir / source_row["image_path"]
        extension = source_image_path.suffix.lower() or ".jpg"
        new_image_id = f"FUSED_REVIEW_{index:05d}"
        new_image_path = f"images/{new_image_id}{extension}"
        source_row["new_image_id"] = new_image_id

        if args.copy_images:
            shutil.copy2(source_image_path, output_dir / new_image_path)

        label_rows.append(
            {
                "image_id": new_image_id,
                "image_path": new_image_path,
                "capture_time": source_row["capture_time"],
                "plant_id": "",
                "batch_id": "",
                "device_id": "",
                "growth_stage": source_row["growth_stage"],
                "health_level": source_row["health_level"],
                "maturity_level": source_row["maturity_level"],
                "abnormal_alert": source_row["abnormal_alert"],
                "growth_stage_valid": source_row.get("growth_stage_valid", "0"),
                "health_level_valid": source_row.get("health_level_valid", "0"),
                "maturity_level_valid": source_row.get("maturity_level_valid", "0"),
                "abnormal_alert_valid": source_row.get("abnormal_alert_valid", "0"),
                "annotator_count": "0",
                "label_status": "weak_needs_human_review",
                "review_notes": "Imported from dataset_fused for manual review only; not real audited greenhouse ground truth.",
            }
        )
        provenance_rows.append(
            {
                "image_id": new_image_id,
                "source_image_id": source_id,
                "source_dataset": manifest_row["source_dataset"],
                "source_class": manifest_row["source_class"],
                "fusion_group": manifest_row["fusion_group"],
                "source_path": manifest_row["source_path"],
                "source_label_note": manifest_row["label_note"],
                "source_capture_time": source_row["capture_time"],
                "review_warning": "Weak-supervised label and synthetic capture alignment; must be manually verified before training as real data.",
            }
        )

    split_by_id = split_rows(selected)
    split_rows_out = [
        {"image_id": row["image_id"], "split": split_by_id[row["image_id"]]}
        for row in label_rows
    ]

    write_csv(output_dir / "labels.csv", LABEL_FIELDS, label_rows)
    write_csv(output_dir / "split.csv", ("image_id", "split"), split_rows_out)
    write_csv(output_dir / "provenance.csv", PROVENANCE_FIELDS, provenance_rows)
    shutil.copy2(source_dir / "environment.csv", output_dir / "environment.csv")

    summary["split_counts"] = dict(sorted(Counter(row["split"] for row in split_rows_out).items()))
    summary["source_dataset_counts"] = dict(
        sorted(Counter(row["source_dataset"] for row in provenance_rows).items())
    )
    summary["training_status"] = "not_ready_real_data"
    summary["blocking_reasons"] = [
        "labels are weak-supervised and annotator_count is 0",
        "plant_id and batch_id are unknown",
        "capture_time values are inherited from fused alignment, not verified camera metadata",
        "environment.csv is hourly fused weather data, not 10-minute same-greenhouse sensor data",
        "pH is fixed in dataset_fused and must not be treated as a real continuous sensor",
    ]
    (output_dir / "selection_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Fused Review Queue\n\n"
        "This directory contains 5,000 samples selected from `dataset_fused` for manual review.\n"
        "It is not a validated real greenhouse dataset and must not be used as final real-data ground truth.\n\n"
        "Required promotion steps:\n\n"
        "1. Verify each image against real capture records or replace it with a true greenhouse image.\n"
        "2. Fill `plant_id`, `batch_id`, and `device_id` from field records.\n"
        "3. Replace weak labels with human-reviewed labels and set `annotator_count`/`label_status`.\n"
        "4. Replace `environment.csv` with same-device 10-minute greenhouse sensor data.\n"
        "5. Run `python scripts/audit_real_dataset.py --dataset-dir dataset_real_v001/review_queue_from_fused --verify-images`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
