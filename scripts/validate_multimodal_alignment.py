from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TASK_NAMES = (
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check image/environment time and device alignment before multimodal training."
        )
    )
    parser.add_argument("--dataset-dir", default="dataset_sample")
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--environment-file", default=None)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    parser.add_argument("--require-device-id", action="store_true")
    parser.add_argument("--require-task-masks", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    labels_path = Path(args.labels_file) if args.labels_file else dataset_dir / "labels.csv"
    environment_path = (
        Path(args.environment_file)
        if args.environment_file
        else dataset_dir / "environment.csv"
    )
    labels = read_rows(labels_path)
    environment = read_rows(environment_path)
    if not labels or not environment:
        raise ValueError("labels and environment files must not be empty")
    if not 0.0 <= args.minimum_coverage <= 1.0:
        raise ValueError("--minimum-coverage must be between 0 and 1")
    if args.window_hours <= 0:
        raise ValueError("--window-hours must be positive")

    label_columns = set(labels[0])
    environment_columns = set(environment[0])
    if "soil_humidity" not in environment_columns and "air_humidity" in environment_columns:
        raise ValueError(
            "environment data contains air_humidity, but this project requires "
            "soil_humidity; do not rename or mix the two measurements"
        )
    if args.require_device_id and (
        "device_id" not in label_columns or "device_id" not in environment_columns
    ):
        raise ValueError(
            "--require-device-id needs device_id in both labels and environment data"
        )

    mask_fields = {f"{task}_valid" for task in TASK_NAMES}
    present_masks = label_columns & mask_fields
    if present_masks and present_masks != mask_fields:
        raise ValueError(f"labels contain partial task masks: {sorted(present_masks)}")
    if args.require_task_masks and present_masks != mask_fields:
        raise ValueError(
            "all *_valid fields are required so weak labels cannot become ground truth"
        )

    env_by_device: dict[str, list[datetime]] = defaultdict(list)
    for row in environment:
        device_id = row.get("device_id", "").strip() or "default"
        env_by_device[device_id].append(datetime.strptime(row["timestamp"], TIME_FORMAT))
    for timestamps in env_by_device.values():
        timestamps.sort()

    window = timedelta(hours=args.window_hours)
    covered = 0
    uncovered_examples: list[dict[str, str]] = []
    for row in labels:
        capture_time = datetime.strptime(row["capture_time"], TIME_FORMAT)
        device_id = row.get("device_id", "").strip() or "default"
        timestamps = env_by_device.get(device_id, [])
        start = bisect_left(timestamps, capture_time - window)
        end = bisect_right(timestamps, capture_time)
        if start < end:
            covered += 1
        elif len(uncovered_examples) < 10:
            uncovered_examples.append(
                {
                    "image_id": row.get("image_id", ""),
                    "capture_time": row["capture_time"],
                    "device_id": device_id,
                }
            )

    coverage = covered / len(labels)
    all_env_times = [value for values in env_by_device.values() for value in values]
    label_times = [datetime.strptime(row["capture_time"], TIME_FORMAT) for row in labels]
    report = {
        "labels_file": str(labels_path),
        "environment_file": str(environment_path),
        "label_samples": len(labels),
        "environment_records": len(environment),
        "covered_samples": covered,
        "coverage": coverage,
        "required_coverage": args.minimum_coverage,
        "image_time_range": [
            min(label_times).strftime(TIME_FORMAT),
            max(label_times).strftime(TIME_FORMAT),
        ],
        "environment_time_range": [
            min(all_env_times).strftime(TIME_FORMAT),
            max(all_env_times).strftime(TIME_FORMAT),
        ],
        "device_ids_in_labels": sorted(
            {row.get("device_id", "").strip() or "default" for row in labels}
        ),
        "device_ids_in_environment": sorted(env_by_device),
        "task_masks_present": present_masks == mask_fields,
        "uncovered_examples": uncovered_examples,
        "status": "ok" if coverage >= args.minimum_coverage else "failed",
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if coverage < args.minimum_coverage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
