from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

LABEL_FIELDS = (
    "image_id",
    "image_path",
    "capture_time",
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)
ENVIRONMENT_FIELDS = ("timestamp", "temperature", "soil_humidity", "light", "ph")
SPLIT_FIELDS = ("image_id", "split")
MANIFEST_FIELDS = (
    "image_id",
    "image_path",
    "capture_time",
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
    "split",
    "source_dataset",
    "source_class",
    "fusion_group",
    "source_path",
    "label_note",
)

VALID_SPLITS = {"train", "val", "test"}
VALID_RANGES = {
    "growth_stage": range(6),
    "health_level": range(4),
    "maturity_level": range(4),
    "abnormal_alert": range(2),
}

EXPECTED_GROUP_RULES = {
    "healthy_leaf": {
        "health_level": {0},
        "maturity_level": {0},
        "abnormal_alert": {0},
        "growth_stage": {0, 1, 2, 3},
    },
    "mild_leaf_abnormal": {
        "health_level": {1},
        "maturity_level": {0},
        "abnormal_alert": {1},
        "growth_stage": {1, 2, 3},
    },
    "disease_leaf": {
        "health_level": {2},
        "maturity_level": {0},
        "abnormal_alert": {1},
        "growth_stage": {2, 3},
    },
    "unripe_fruit": {
        "health_level": {0},
        "maturity_level": {0},
        "abnormal_alert": {0},
        "growth_stage": {4},
    },
    "ripe_fruit": {
        "health_level": {0},
        "maturity_level": {2},
        "abnormal_alert": {0},
        "growth_stage": {5},
    },
    "fresh_quality": {
        "health_level": {0},
        "maturity_level": {2},
        "abnormal_alert": {0},
        "growth_stage": {5},
    },
    "mild_quality": {
        "health_level": {1},
        "maturity_level": {1},
        "abnormal_alert": {1},
        "growth_stage": {5},
    },
    "rotten_quality": {
        "health_level": {3},
        "maturity_level": {3},
        "abnormal_alert": {1},
        "growth_stage": {5},
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_fields(rows: list[dict[str, str]], expected: tuple[str, ...], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    actual = tuple(rows[0].keys())
    if actual != expected:
        raise ValueError(f"{name} fields mismatch: actual={actual}, expected={expected}")


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def check_required_files(dataset_dir: Path) -> dict[str, Path]:
    paths = {
        "images": dataset_dir / "images",
        "labels": dataset_dir / "labels.csv",
        "environment": dataset_dir / "environment.csv",
        "split": dataset_dir / "split.csv",
        "readme": dataset_dir / "README.md",
        "manifest": dataset_dir / "fusion_manifest.csv",
        "quality_report": dataset_dir / "quality_report.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required dataset items: {missing}")
    if not paths["images"].is_dir():
        raise NotADirectoryError(paths["images"])
    return paths


def check_environment(rows: list[dict[str, str]]) -> tuple[list[datetime], dict[str, Any]]:
    require_fields(rows, ENVIRONMENT_FIELDS, "environment.csv")
    timestamps = [parse_time(row["timestamp"]) for row in rows]
    if timestamps != sorted(timestamps):
        raise ValueError("environment.csv timestamps are not sorted")
    duplicate_count = len(timestamps) - len(set(timestamps))
    if duplicate_count:
        raise ValueError(f"environment.csv has duplicate timestamps: {duplicate_count}")

    range_checks = {
        "temperature": (0.0, 60.0),
        "soil_humidity": (0.0, 100.0),
        "light": (0.0, None),
        "ph": (0.0, 14.0),
    }
    violations = []
    for row in rows:
        for field, (lower, upper) in range_checks.items():
            value = float(row[field])
            if value < lower or (upper is not None and value > upper):
                violations.append({"timestamp": row["timestamp"], "field": field, "value": value})
    if violations:
        raise ValueError(f"environment range violations: {violations[:5]}")

    intervals = [
        int((right - left).total_seconds())
        for left, right in zip(timestamps, timestamps[1:])
    ]
    return timestamps, {
        "environment_rows": len(rows),
        "environment_start": timestamps[0].strftime(TIMESTAMP_FORMAT),
        "environment_end": timestamps[-1].strftime(TIMESTAMP_FORMAT),
        "environment_interval_seconds_min": min(intervals) if intervals else None,
        "environment_interval_seconds_max": max(intervals) if intervals else None,
    }


def check_labels_and_splits(
    dataset_dir: Path,
    labels: list[dict[str, str]],
    splits: list[dict[str, str]],
    env_times: list[datetime],
    verify_images: bool,
) -> dict[str, Any]:
    require_fields(labels, LABEL_FIELDS, "labels.csv")
    require_fields(splits, SPLIT_FIELDS, "split.csv")

    image_ids = [row["image_id"] for row in labels]
    if len(image_ids) != len(set(image_ids)):
        duplicates = [item for item, count in Counter(image_ids).items() if count > 1]
        raise ValueError(f"duplicate image_id values: {duplicates[:10]}")

    split_by_id = {row["image_id"]: row["split"] for row in splits}
    if set(split_by_id) != set(image_ids):
        raise ValueError("split.csv image_id set does not match labels.csv")
    invalid_splits = sorted(set(split_by_id.values()) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"invalid split values: {invalid_splits}")

    env_time_set = set(env_times)
    opened_images = 0
    missing_images = []
    bad_windows = []
    for row in labels:
        image_path = dataset_dir / row["image_path"]
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        if verify_images:
            with Image.open(image_path) as image:
                image.verify()
        opened_images += 1

        capture_time = parse_time(row["capture_time"])
        if capture_time not in env_time_set:
            bad_windows.append((row["image_id"], "capture_time_not_in_environment"))
        window_start = capture_time - timedelta(hours=24)
        window_count = sum(1 for time in env_times if window_start <= time <= capture_time)
        if window_count < 1:
            bad_windows.append((row["image_id"], "empty_24h_window"))

        for field, valid_range in VALID_RANGES.items():
            value = int(row[field])
            if value not in valid_range:
                raise ValueError(f"{row['image_id']} has invalid {field}: {value}")

    if missing_images:
        raise FileNotFoundError(f"missing image files: {missing_images[:5]}")
    if bad_windows:
        raise ValueError(f"bad environment windows: {bad_windows[:5]}")

    return {
        "labels": len(labels),
        "images_opened": opened_images,
        "split_counts": dict(sorted(Counter(split_by_id.values()).items())),
        "growth_stage_counts": dict(sorted(Counter(row["growth_stage"] for row in labels).items())),
        "health_level_counts": dict(sorted(Counter(row["health_level"] for row in labels).items())),
        "maturity_level_counts": dict(sorted(Counter(row["maturity_level"] for row in labels).items())),
        "abnormal_alert_counts": dict(sorted(Counter(row["abnormal_alert"] for row in labels).items())),
    }


def check_manifest(
    labels: list[dict[str, str]],
    splits: list[dict[str, str]],
    manifest: list[dict[str, str]],
) -> dict[str, Any]:
    require_fields(manifest, MANIFEST_FIELDS, "fusion_manifest.csv")
    labels_by_id = {row["image_id"]: row for row in labels}
    splits_by_id = {row["image_id"]: row["split"] for row in splits}
    manifest_by_id = {row["image_id"]: row for row in manifest}
    if set(manifest_by_id) != set(labels_by_id):
        raise ValueError("fusion_manifest.csv image IDs do not match labels.csv")

    source_missing = []
    rule_violations = []
    for image_id, row in manifest_by_id.items():
        label_row = labels_by_id[image_id]
        for field in LABEL_FIELDS:
            if row[field] != label_row[field]:
                raise ValueError(f"manifest mismatch for {image_id} field={field}")
        if row["split"] != splits_by_id[image_id]:
            raise ValueError(f"manifest split mismatch for {image_id}")

        source_path = ROOT / row["source_path"]
        if not source_path.exists():
            source_missing.append(str(source_path))

        group = row["fusion_group"]
        rules = EXPECTED_GROUP_RULES.get(group)
        if rules is None:
            rule_violations.append((image_id, group, "unknown_group"))
            continue
        for field, allowed in rules.items():
            value = int(row[field])
            if value not in allowed:
                rule_violations.append((image_id, group, field, value, sorted(allowed)))

    if source_missing:
        raise FileNotFoundError(f"missing source files listed in manifest: {source_missing[:5]}")
    if rule_violations:
        raise ValueError(f"label mapping rule violations: {rule_violations[:10]}")

    group_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        group_by_split[row["split"]][row["fusion_group"]] += 1

    return {
        "manifest_rows": len(manifest),
        "source_dataset_counts": dict(sorted(Counter(row["source_dataset"] for row in manifest).items())),
        "fusion_group_counts": dict(sorted(Counter(row["fusion_group"] for row in manifest).items())),
        "fusion_groups_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(group_by_split.items())
        },
    }


def check_quality_report(dataset_dir: Path, labels: list[dict[str, str]]) -> dict[str, Any]:
    report_path = dataset_dir / "quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("sample_count", -1)) != len(labels):
        raise ValueError("quality_report.json sample_count does not match labels.csv")
    return {"quality_report_sample_count": report["sample_count"]}


def write_report(dataset_dir: Path, summary: dict[str, Any]) -> None:
    output_path = dataset_dir / "validation_report.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(dataset_dir: Path, verify_images: bool) -> dict[str, Any]:
    paths = check_required_files(dataset_dir)
    labels = read_csv(paths["labels"])
    environment = read_csv(paths["environment"])
    splits = read_csv(paths["split"])
    manifest = read_csv(paths["manifest"])

    env_times, env_summary = check_environment(environment)
    label_summary = check_labels_and_splits(dataset_dir, labels, splits, env_times, verify_images)
    manifest_summary = check_manifest(labels, splits, manifest)
    quality_summary = check_quality_report(dataset_dir, labels)

    summary = {
        "validation": "passed",
        "dataset_dir": str(dataset_dir.resolve()),
        "schema": "matches project data handoff requirements plus fusion provenance files",
        **env_summary,
        **label_summary,
        **manifest_summary,
        **quality_summary,
    }
    write_report(dataset_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the fused watermelon training dataset.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset_fused"))
    parser.add_argument("--skip-image-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate(Path(args.dataset_dir), verify_images=not args.skip_image_verify)
    print("validation=passed")
    for key in (
        "dataset_dir",
        "labels",
        "images_opened",
        "environment_rows",
        "environment_start",
        "environment_end",
        "split_counts",
        "growth_stage_counts",
        "health_level_counts",
        "maturity_level_counts",
        "abnormal_alert_counts",
        "fusion_group_counts",
    ):
        print(f"{key}={summary[key]}")
    print(f"validation_report={Path(args.dataset_dir) / 'validation_report.json'}")


if __name__ == "__main__":
    main()
