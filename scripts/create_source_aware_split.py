from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


LABEL_FIELDS = ("growth_stage", "health_level", "maturity_level", "abnormal_alert")
SUMMARY_FIELDS = (*LABEL_FIELDS, "source_dataset", "fusion_group")
DEFAULT_TEST_SOURCE = "watermelon_disease_recognition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-aware split for the fused watermelon dataset."
    )
    parser.add_argument("--dataset-dir", default="dataset_fused")
    parser.add_argument(
        "--output",
        default="split_source_aware.csv",
        help="Output split CSV path. Relative paths are written inside dataset-dir.",
    )
    parser.add_argument(
        "--report",
        default="split_source_aware_report.json",
        help="Output report JSON path. Relative paths are written inside dataset-dir.",
    )
    parser.add_argument(
        "--test-source",
        default=DEFAULT_TEST_SOURCE,
        help="Source dataset held out entirely as the test split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio sampled from non-test sources.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_output_path(dataset_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return dataset_dir / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(rows: list[dict[str, str]], test_source: str) -> None:
    if not rows:
        raise ValueError("fusion_manifest.csv is empty")

    required_fields = {
        "image_id",
        "source_dataset",
        "fusion_group",
        *LABEL_FIELDS,
    }
    missing = required_fields - set(rows[0])
    if missing:
        raise ValueError(f"fusion_manifest.csv missing fields: {sorted(missing)}")

    sources = {row["source_dataset"] for row in rows}
    if test_source not in sources:
        raise ValueError(
            f"test source {test_source!r} not found; available={sorted(sources)}"
        )


def validation_count(bucket_size: int, val_ratio: float) -> int:
    if bucket_size <= 1:
        return 0
    count = round(bucket_size * val_ratio)
    return min(max(count, 1), bucket_size - 1)


def create_split_rows(
    manifest_rows: list[dict[str, str]],
    test_source: str,
    val_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val-ratio must be greater than 0 and less than 0.5")

    random_generator = random.Random(seed)
    split_by_id: dict[str, str] = {}
    validation_buckets: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)

    for row in manifest_rows:
        image_id = row["image_id"]
        if row["source_dataset"] == test_source:
            split_by_id[image_id] = "test"
            continue

        # Stratify validation sampling so label/source/group proportions stay close
        # to the remaining training pool without mixing the held-out test source.
        bucket_key = (
            row["source_dataset"],
            row["fusion_group"],
            row["growth_stage"],
            row["health_level"],
            row["maturity_level"],
        )
        validation_buckets[bucket_key].append(image_id)

    for image_ids in validation_buckets.values():
        random_generator.shuffle(image_ids)
        val_count = validation_count(len(image_ids), val_ratio)
        validation_ids = set(image_ids[:val_count])
        for image_id in image_ids:
            split_by_id[image_id] = "val" if image_id in validation_ids else "train"

    return [
        {"image_id": row["image_id"], "split": split_by_id[row["image_id"]]}
        for row in manifest_rows
    ]


def nested_counter() -> defaultdict[str, Counter[str]]:
    return defaultdict(Counter)


def summarize(
    manifest_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    test_source: str,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    split_by_id = {row["image_id"]: row["split"] for row in split_rows}
    split_counts = Counter(split_by_id.values())
    summary_by_split: dict[str, dict[str, dict[str, int]]] = {}

    counters: dict[str, defaultdict[str, Counter[str]]] = {
        field: nested_counter() for field in SUMMARY_FIELDS
    }
    for row in manifest_rows:
        split = split_by_id[row["image_id"]]
        for field in SUMMARY_FIELDS:
            counters[field][split][row[field]] += 1

    for split in ("train", "val", "test"):
        summary_by_split[split] = {
            field: dict(sorted(counters[field][split].items()))
            for field in SUMMARY_FIELDS
        }

    return {
        "strategy": "hold out one source_dataset as test; stratified validation from remaining sources",
        "test_source": test_source,
        "val_ratio": val_ratio,
        "seed": seed,
        "sample_count": len(split_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "by_split": summary_by_split,
        "caveat": (
            "The test split is source-held-out, but it only contains labels available "
            "in the held-out source. Interpret unsupported-class macro metrics carefully."
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "fusion_manifest.csv"
    output_path = resolve_output_path(dataset_dir, args.output)
    report_path = resolve_output_path(dataset_dir, args.report)

    manifest_rows = read_rows(manifest_path)
    validate_manifest(manifest_rows, args.test_source)

    split_rows = create_split_rows(
        manifest_rows=manifest_rows,
        test_source=args.test_source,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    report = summarize(
        manifest_rows=manifest_rows,
        split_rows=split_rows,
        test_source=args.test_source,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_split(output_path, split_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_split={output_path}")
    print(f"saved_report={report_path}")
    print(json.dumps(report["split_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
