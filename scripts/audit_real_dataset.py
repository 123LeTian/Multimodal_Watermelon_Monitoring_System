from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

LABEL_FIELDS = (
    "image_id",
    "image_path",
    "capture_time",
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)
AUDIT_LABEL_FIELDS = ("plant_id", "batch_id", "device_id")
ENVIRONMENT_FIELDS = ("timestamp", "temperature", "soil_humidity", "light", "ph")
VALID_SPLITS = {"train", "val", "test"}

GROWTH_STAGE_NAMES = {
    "0": 0,
    "germination": 0,
    "发芽期": 0,
    "1": 1,
    "vine": 1,
    "伸蔓期": 1,
    "2": 2,
    "flowering": 2,
    "开花期": 2,
    "3": 3,
    "fruit_set": 3,
    "坐果期": 3,
    "4": 4,
    "expansion": 4,
    "膨大期": 4,
    "5": 5,
    "maturity": 5,
    "成熟期": 5,
}
HEALTH_NAMES = {
    "0": 0,
    "normal": 0,
    "正常": 0,
    "1": 1,
    "mild": 1,
    "轻微异常": 1,
    "2": 2,
    "moderate": 2,
    "中度异常": 2,
    "3": 3,
    "severe": 3,
    "严重异常": 3,
}
MATURITY_NAMES = {
    "0": 0,
    "immature": 0,
    "未成熟": 0,
    "1": 1,
    "near_ripe": 1,
    "接近成熟": 1,
    "2": 2,
    "ripe": 2,
    "成熟": 2,
    "3": 3,
    "overripe": 3,
    "过熟": 3,
}
ALERT_NAMES = {
    "0": 0,
    "none": 0,
    "无异常": 0,
    "false": 0,
    "1": 1,
    "alert": 1,
    "有异常": 1,
    "true": 1,
}
LABEL_MAPPINGS = {
    "growth_stage": GROWTH_STAGE_NAMES,
    "health_level": HEALTH_NAMES,
    "maturity_level": MATURITY_NAMES,
    "abnormal_alert": ALERT_NAMES,
}

STAGE_HEALTH_TARGETS = {
    0: {0: 250, 1: 70, 2: 50, 3: 30},
    1: {0: 350, 1: 100, 2: 90, 3: 60},
    2: {0: 350, 1: 140, 2: 130, 3: 80},
    3: {0: 400, 1: 160, 2: 140, 3: 100},
    4: {0: 600, 1: 240, 2: 220, 3: 140},
    5: {0: 550, 1: 290, 2: 270, 3: 190},
}
MATURITY_TARGETS = {0: 3000, 1: 800, 2: 900, 3: 300}
SPLIT_TARGETS = {"train": 3500, "val": 750, "test": 750}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a real greenhouse watermelon dataset before training."
    )
    parser.add_argument("--dataset-dir", default="dataset_real_v001")
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--environment-file", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--target-tolerance", type=float, default=0.10)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--expected-interval-minutes", type=int, default=10)
    parser.add_argument("--minimum-window-coverage", type=float, default=0.90)
    parser.add_argument("--max-consecutive-missing-minutes", type=int, default=30)
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), TIME_FORMAT)


def add_issue(issues: list[dict[str, str]], severity: str, message: str) -> None:
    issues.append({"severity": severity, "message": message})


def canonical_label(field: str, value: str) -> int:
    normalized = value.strip()
    mapping = LABEL_MAPPINGS[field]
    if normalized in mapping:
        return mapping[normalized]
    lowered = normalized.lower()
    if lowered in mapping:
        return mapping[lowered]
    raise ValueError(f"invalid {field}={value!r}")


def numeric_or_none(value: str) -> float | None:
    value = value.strip()
    if value == "":
        return None
    return float(value)


def ratio_status(actual: int, target: int, tolerance: float) -> str:
    lower = target * (1.0 - tolerance)
    upper = target * (1.0 + tolerance)
    if lower <= actual <= upper:
        return "ok"
    return "low" if actual < lower else "high"


def summarize_distribution(
    actual: Counter[Any],
    target: dict[Any, int],
    tolerance: float,
) -> dict[str, Any]:
    rows = {}
    for key, expected in target.items():
        count = actual.get(key, 0)
        rows[str(key)] = {
            "actual": count,
            "target": expected,
            "status": ratio_status(count, expected, tolerance),
        }
    return rows


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    dataset_dir = Path(args.dataset_dir)
    labels_path = Path(args.labels_file) if args.labels_file else dataset_dir / "labels.csv"
    environment_path = (
        Path(args.environment_file) if args.environment_file else dataset_dir / "environment.csv"
    )
    split_path = Path(args.split_file) if args.split_file else dataset_dir / "split.csv"
    labels = read_csv(labels_path)
    environment = read_csv(environment_path)
    splits = read_csv(split_path)
    return dataset_dir, labels_path, environment_path, split_path, labels, environment, splits


def audit_schema(
    labels: list[dict[str, str]],
    environment: list[dict[str, str]],
    splits: list[dict[str, str]],
    issues: list[dict[str, str]],
) -> None:
    if not labels:
        add_issue(issues, "error", "labels.csv is empty")
        return
    if not environment:
        add_issue(issues, "error", "environment.csv is empty")
    if not splits:
        add_issue(issues, "error", "split.csv is empty")

    label_columns = set(labels[0])
    missing_label_fields = set(LABEL_FIELDS) - label_columns
    if missing_label_fields:
        add_issue(issues, "error", f"labels.csv missing required fields: {sorted(missing_label_fields)}")
    missing_audit_fields = set(AUDIT_LABEL_FIELDS) - label_columns
    if missing_audit_fields:
        add_issue(
            issues,
            "error",
            "labels.csv should include plant_id, batch_id, and device_id to audit leakage "
            f"and same-device alignment; missing {sorted(missing_audit_fields)}",
        )

    if environment:
        env_columns = set(environment[0])
        missing_env_fields = set(ENVIRONMENT_FIELDS) - env_columns
        if missing_env_fields:
            add_issue(
                issues,
                "error",
                f"environment.csv missing required fields: {sorted(missing_env_fields)}",
            )
        if "device_id" in label_columns and "device_id" not in env_columns:
            add_issue(
                issues,
                "error",
                "labels.csv contains device_id but environment.csv does not; same-device matching cannot be verified",
            )

    if splits:
        split_columns = set(splits[0])
        if split_columns != {"image_id", "split"}:
            add_issue(
                issues,
                "error",
                f"split.csv fields should be exactly image_id,split; actual={sorted(split_columns)}",
            )


def audit_labels_and_images(
    dataset_dir: Path,
    labels: list[dict[str, str]],
    args: argparse.Namespace,
    issues: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    missing_images: list[str] = []
    unreadable_images: list[str] = []
    from PIL import Image

    for row in labels:
        image_id = row.get("image_id", "").strip()
        if image_id in seen_ids:
            duplicate_ids.append(image_id)
        seen_ids.add(image_id)

        parsed_row: dict[str, Any] = dict(row)
        try:
            parsed_row["capture_dt"] = parse_time(row["capture_time"])
            for field in LABEL_MAPPINGS:
                parsed_row[field] = canonical_label(field, row[field])
        except Exception as exc:
            add_issue(issues, "error", f"{image_id or '<missing image_id>'}: {exc}")
            continue

        image_path = dataset_dir / row.get("image_path", "")
        if not image_path.exists():
            missing_images.append(str(image_path))
        elif args.verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                unreadable_images.append(str(image_path))
        parsed.append(parsed_row)

    if duplicate_ids:
        add_issue(issues, "error", f"duplicate image_id values: {duplicate_ids[:10]}")
    if missing_images:
        add_issue(issues, "error", f"missing image files: {missing_images[:10]}")
    if unreadable_images:
        add_issue(issues, "error", f"unreadable image files: {unreadable_images[:10]}")

    total = len(parsed)
    if total:
        tolerance = args.target_tolerance
        if ratio_status(total, args.target_count, tolerance) != "ok":
            add_issue(
                issues,
                "warning",
                f"image count is {total}, target is {args.target_count} +/- {int(tolerance * 100)}%",
            )

    plant_ids = {row.get("plant_id", "").strip() for row in parsed if row.get("plant_id", "").strip()}
    batch_ids = {row.get("batch_id", "").strip() for row in parsed if row.get("batch_id", "").strip()}
    if plant_ids and len(plant_ids) < 50:
        add_issue(issues, "error", f"only {len(plant_ids)} plants found; target is 50-100 plants")
    if plant_ids and len(plant_ids) > 100:
        add_issue(issues, "warning", f"{len(plant_ids)} plants found; this is above the suggested 50-100 range")
    if batch_ids and len(batch_ids) < 2:
        add_issue(issues, "error", f"only {len(batch_ids)} planting batch found; at least 2 are required")

    annotator_counts = []
    for row in parsed:
        raw_count = row.get("annotator_count", "").strip()
        if raw_count:
            try:
                annotator_counts.append(int(raw_count))
            except ValueError:
                add_issue(issues, "warning", f"{row['image_id']}: annotator_count is not an integer")
    double_reviewed = sum(1 for count in annotator_counts if count >= 2)
    if parsed and annotator_counts:
        review_ratio = double_reviewed / len(parsed)
        if review_ratio < 0.20:
            add_issue(
                issues,
                "error",
                f"double-reviewed ratio is {review_ratio:.1%}; at least 20% is required",
            )
    elif parsed:
        add_issue(
            issues,
            "warning",
            "annotator_count is missing, so 20% double-review coverage cannot be verified",
        )

    approved_count = sum(
        1
        for row in parsed
        if row.get("label_status", "").strip().lower() in {"approved", "reviewed", "通过", "已审核"}
    )
    if parsed and "label_status" in parsed[0] and approved_count < len(parsed):
        add_issue(
            issues,
            "warning",
            f"{len(parsed) - approved_count} labels are not marked approved/reviewed",
        )

    summary = {
        "sample_count": total,
        "unique_plants": len(plant_ids),
        "unique_batches": len(batch_ids),
        "double_reviewed": double_reviewed,
        "double_reviewed_ratio": double_reviewed / total if total else 0.0,
    }
    return parsed, summary


def audit_distribution(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    growth_counts = Counter(row["growth_stage"] for row in rows)
    health_counts = Counter(row["health_level"] for row in rows)
    maturity_counts = Counter(row["maturity_level"] for row in rows)
    alert_counts = Counter(row["abnormal_alert"] for row in rows)
    stage_health = Counter((row["growth_stage"], row["health_level"]) for row in rows)

    stage_targets = {
        stage: sum(health_targets.values())
        for stage, health_targets in STAGE_HEALTH_TARGETS.items()
    }
    health_targets = Counter()
    for health_targets_for_stage in STAGE_HEALTH_TARGETS.values():
        health_targets.update(health_targets_for_stage)

    stage_health_report = {}
    for stage, health_targets_for_stage in STAGE_HEALTH_TARGETS.items():
        stage_health_report[str(stage)] = {}
        for health, target in health_targets_for_stage.items():
            actual = stage_health.get((stage, health), 0)
            status = ratio_status(actual, target, args.target_tolerance)
            stage_health_report[str(stage)][str(health)] = {
                "actual": actual,
                "target": target,
                "status": status,
            }

    severe_count = health_counts.get(3, 0)
    overripe_count = maturity_counts.get(3, 0)
    if severe_count < 400:
        add_issue(issues, "error", f"severe abnormal samples={severe_count}; minimum is 400")
    elif severe_count < 600:
        add_issue(issues, "warning", f"severe abnormal samples={severe_count}; target is about 600")
    if overripe_count < 300:
        add_issue(issues, "error", f"overripe samples={overripe_count}; minimum is 300")

    normal_alert = alert_counts.get(0, 0)
    abnormal_alert = alert_counts.get(1, 0)
    if normal_alert and not 2500 <= normal_alert <= 2800:
        add_issue(issues, "warning", f"normal-alert samples={normal_alert}; suggested range is 2500-2800")
    if abnormal_alert and not 2200 <= abnormal_alert <= 2500:
        add_issue(issues, "warning", f"abnormal-alert samples={abnormal_alert}; suggested range is 2200-2500")

    return {
        "growth_stage": summarize_distribution(growth_counts, stage_targets, args.target_tolerance),
        "health_level": summarize_distribution(dict(health_counts), dict(health_targets), args.target_tolerance),
        "maturity_level": summarize_distribution(maturity_counts, MATURITY_TARGETS, args.target_tolerance),
        "abnormal_alert": dict(sorted(alert_counts.items())),
        "stage_x_health": stage_health_report,
    }


def audit_environment(
    labels: list[dict[str, Any]],
    environment: list[dict[str, str]],
    args: argparse.Namespace,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    env_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_timestamps: list[str] = []
    seen_keys: set[tuple[str, datetime]] = set()
    missing_value_counts = Counter()
    value_ranges: dict[str, list[float]] = {field: [] for field in ENVIRONMENT_FIELDS if field != "timestamp"}

    for row in environment:
        try:
            timestamp = parse_time(row["timestamp"])
        except Exception as exc:
            add_issue(issues, "error", f"invalid environment timestamp {row.get('timestamp')!r}: {exc}")
            continue
        device_id = row.get("device_id", "").strip() or "default"
        key = (device_id, timestamp)
        if key in seen_keys:
            duplicate_timestamps.append(f"{device_id}:{timestamp.strftime(TIME_FORMAT)}")
        seen_keys.add(key)
        parsed = dict(row)
        parsed["timestamp_dt"] = timestamp
        for field in value_ranges:
            value = numeric_or_none(row.get(field, ""))
            if value is None:
                missing_value_counts[field] += 1
                continue
            value_ranges[field].append(value)
            if field == "temperature" and not -10.0 <= value <= 60.0:
                add_issue(issues, "warning", f"temperature out of range at {row['timestamp']}: {value}")
            if field == "soil_humidity" and not 0.0 <= value <= 100.0:
                add_issue(issues, "warning", f"soil_humidity out of range at {row['timestamp']}: {value}")
            if field == "light" and value < 0.0:
                add_issue(issues, "warning", f"light is negative at {row['timestamp']}: {value}")
            if field == "ph" and not 0.0 <= value <= 14.0:
                add_issue(issues, "warning", f"ph out of range at {row['timestamp']}: {value}")
        env_by_device[device_id].append(parsed)

    if duplicate_timestamps:
        add_issue(issues, "error", f"duplicate environment timestamps: {duplicate_timestamps[:10]}")

    for device_id, rows in env_by_device.items():
        rows.sort(key=lambda item: item["timestamp_dt"])
        times = [row["timestamp_dt"] for row in rows]
        if times != sorted(times):
            add_issue(issues, "error", f"environment timestamps are not sorted for device {device_id}")

    ph_values = value_ranges.get("ph", [])
    if ph_values and len(set(round(value, 4) for value in ph_values)) == 1 and len(ph_values) > 100:
        add_issue(
            issues,
            "warning",
            "ph is constant across more than 100 records; verify this is a real continuous sensor, not copied filler",
        )
    for field, count in missing_value_counts.items():
        if count:
            add_issue(issues, "warning", f"{field} has {count} missing environment values")

    expected_count = int(args.window_hours * 60 / args.expected_interval_minutes) + 1
    max_allowed_gap = timedelta(
        minutes=args.expected_interval_minutes + args.max_consecutive_missing_minutes
    )
    window = timedelta(hours=args.window_hours)
    bad_windows: list[dict[str, Any]] = []
    coverage_values: list[float] = []

    for row in labels:
        capture_time = row["capture_dt"]
        device_id = row.get("device_id", "").strip() or "default"
        device_rows = env_by_device.get(device_id, [])
        times = [item["timestamp_dt"] for item in device_rows]
        start_time = capture_time - window
        left = bisect_left(times, start_time)
        right = bisect_right(times, capture_time)
        window_times = times[left:right]
        coverage = len(window_times) / expected_count
        coverage_values.append(coverage)
        gap_failure = False
        if not window_times:
            gap_failure = True
        else:
            if window_times[0] - start_time > max_allowed_gap:
                gap_failure = True
            if capture_time - window_times[-1] > max_allowed_gap:
                gap_failure = True
            for previous, current in zip(window_times, window_times[1:]):
                if current - previous > max_allowed_gap:
                    gap_failure = True
                    break
        if coverage < args.minimum_window_coverage or gap_failure:
            bad_windows.append(
                {
                    "image_id": row["image_id"],
                    "device_id": device_id,
                    "capture_time": capture_time.strftime(TIME_FORMAT),
                    "records": len(window_times),
                    "expected_records": expected_count,
                    "coverage": round(coverage, 4),
                    "gap_failure": gap_failure,
                }
            )

    if bad_windows:
        add_issue(
            issues,
            "error",
            f"{len(bad_windows)} images fail 24h environment coverage/gap checks; examples={bad_windows[:5]}",
        )

    ranges = {}
    for field, values in value_ranges.items():
        ranges[field] = {
            "non_missing": len(values),
            "missing": missing_value_counts[field],
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    all_times = [row["timestamp_dt"] for rows in env_by_device.values() for row in rows]
    return {
        "environment_records": len(environment),
        "devices": sorted(env_by_device),
        "time_range": [
            min(all_times).strftime(TIME_FORMAT) if all_times else None,
            max(all_times).strftime(TIME_FORMAT) if all_times else None,
        ],
        "field_ranges": ranges,
        "expected_records_per_24h_window": expected_count,
        "min_window_coverage": min(coverage_values) if coverage_values else 0.0,
        "mean_window_coverage": sum(coverage_values) / len(coverage_values) if coverage_values else 0.0,
        "failed_windows": len(bad_windows),
        "failed_window_examples": bad_windows[:20],
    }


def audit_splits(
    labels: list[dict[str, Any]],
    splits: list[dict[str, str]],
    args: argparse.Namespace,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    split_by_id = {row["image_id"]: row["split"] for row in splits}
    split_values = set(split_by_id.values())
    invalid_splits = sorted(split_values - VALID_SPLITS)
    if invalid_splits:
        add_issue(issues, "error", f"invalid split values: {invalid_splits}")
    label_ids = {row["image_id"] for row in labels}
    if set(split_by_id) != label_ids:
        add_issue(issues, "error", "split.csv image_id set does not match labels.csv")

    split_counts = Counter(split_by_id.get(row["image_id"], "<missing>") for row in labels)
    for split, target in SPLIT_TARGETS.items():
        status = ratio_status(split_counts.get(split, 0), target, args.target_tolerance)
        if status != "ok":
            add_issue(
                issues,
                "warning",
                f"{split} split has {split_counts.get(split, 0)} samples; target is {target} +/- {int(args.target_tolerance * 100)}%",
            )

    labels_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        labels_by_split[split_by_id.get(row["image_id"], "<missing>")].append(row)
    coverage_by_split: dict[str, Any] = {}
    for split in VALID_SPLITS:
        rows = labels_by_split.get(split, [])
        stage_set = {row["growth_stage"] for row in rows}
        health_set = {row["health_level"] for row in rows}
        alert_set = {row["abnormal_alert"] for row in rows}
        coverage_by_split[split] = {
            "growth_stage": sorted(stage_set),
            "health_level": sorted(health_set),
            "abnormal_alert": sorted(alert_set),
        }
        if rows and len(stage_set) < 6:
            add_issue(issues, "error", f"{split} split does not cover all 6 growth stages")
        if rows and len(health_set) < 4:
            add_issue(issues, "error", f"{split} split does not cover all 4 health levels")
        if rows and len(alert_set) < 2:
            add_issue(issues, "error", f"{split} split does not cover both abnormal_alert classes")

    group_to_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in labels:
        plant_id = row.get("plant_id", "").strip()
        if not plant_id:
            continue
        capture_date = row["capture_dt"].date().isoformat()
        group_to_splits[(plant_id, capture_date)].add(split_by_id.get(row["image_id"], "<missing>"))
    leaks = [
        {"plant_id": plant_id, "date": date, "splits": sorted(values)}
        for (plant_id, date), values in group_to_splits.items()
        if len(values) > 1
    ]
    if leaks:
        add_issue(
            issues,
            "error",
            f"{len(leaks)} plant-day groups leak across splits; examples={leaks[:5]}",
        )

    return {
        "split_counts": dict(sorted(split_counts.items())),
        "coverage_by_split": coverage_by_split,
        "plant_day_leak_groups": len(leaks),
        "plant_day_leak_examples": leaks[:20],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    (
        dataset_dir,
        labels_path,
        environment_path,
        split_path,
        labels,
        environment,
        splits,
    ) = load_inputs(args)
    audit_schema(labels, environment, splits, issues)
    parsed_labels, label_summary = audit_labels_and_images(dataset_dir, labels, args, issues)
    distribution_summary = audit_distribution(parsed_labels, args, issues) if parsed_labels else {}
    environment_summary = (
        audit_environment(parsed_labels, environment, args, issues)
        if parsed_labels and environment
        else {}
    )
    split_summary = (
        audit_splits(parsed_labels, splits, args, issues) if parsed_labels and splits else {}
    )
    status = "failed" if any(issue["severity"] == "error" for issue in issues) else "ok"
    return {
        "status": status,
        "dataset_dir": str(dataset_dir.resolve()),
        "labels_file": str(labels_path.resolve()),
        "environment_file": str(environment_path.resolve()),
        "split_file": str(split_path.resolve()),
        "requirements": {
            "target_count": args.target_count,
            "target_tolerance": args.target_tolerance,
            "window_hours": args.window_hours,
            "expected_interval_minutes": args.expected_interval_minutes,
            "minimum_window_coverage": args.minimum_window_coverage,
            "max_consecutive_missing_minutes": args.max_consecutive_missing_minutes,
        },
        "labels": label_summary,
        "distribution": distribution_summary,
        "environment": environment_summary,
        "splits": split_summary,
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.report) if args.report else Path(args.dataset_dir) / "real_dataset_audit_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
