from __future__ import annotations

import csv
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
RAW_ENV_DIR = ROOT / "data_external" / "raw" / "environment_iot"
ENV_OUTPUT_DIR = ROOT / "data_external" / "environment_pretrain"
ENV_OUTPUT = ENV_OUTPUT_DIR / "environment.csv"
QUALITY_OUTPUT = ENV_OUTPUT_DIR / "quality_report.json"
README_OUTPUT = ENV_OUTPUT_DIR / "README.md"
SOURCES_OUTPUT = ENV_OUTPUT_DIR / "sources.json"
IMAGE_RAW_DIR = ROOT / "data_external" / "raw" / "digigreen_crop_disease" / "images"
IMAGE_OUTPUT_DIR = ROOT / "data_external" / "watermelon_disease_pretrain"
IMAGE_OUTPUT_IMAGES = IMAGE_OUTPUT_DIR / "images"
IMAGE_MANIFEST_OUTPUT = IMAGE_OUTPUT_DIR / "image_manifest.csv"
IMAGE_README_OUTPUT = IMAGE_OUTPUT_DIR / "README.md"
IMAGE_SOURCES_OUTPUT = IMAGE_OUTPUT_DIR / "sources.json"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SOURCE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
ENV_FIELDS = ("temperature", "soil_humidity", "light", "ph")

SOURCE_FILES = {
    "temperature": "Environment Temperature.xlsx",
    "soil_humidity": "Soil Moisture.xlsx",
    "light": "Environment Light Intensity.xlsx",
    "ph": "Soil pH.xlsx",
}

SOURCE_URLS = {
    "dataset_page": "https://data.mendeley.com/datasets/65jxyrxv7b/1",
    "api_page": "https://data.mendeley.com/public-api/datasets/65jxyrxv7b/files?folder_id=root&version=1",
}

WATERMELON_IMAGE_ROWS = (
    {
        "source_filename": "00538.jpg",
        "source_diagnosis": "Mosaic Virus",
        "source_country": "India",
        "source_state": "Odisha",
    },
    {
        "source_filename": "00060.jpg",
        "source_diagnosis": "Mosaic Virus; Nitrogen Deficiency",
        "source_country": "India",
        "source_state": "Odisha",
    },
    {
        "source_filename": "00520.jpg",
        "source_diagnosis": "Healthy",
        "source_country": "India",
        "source_state": "Bihar",
    },
)

IMAGE_SOURCE_PAGE = "https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images"


def read_sensor_workbook(path: Path) -> dict[int, dict[str, Any]]:
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        worksheet = workbook.active
        rows = list(worksheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    if len(rows) < 3:
        raise ValueError(f"{path.name} does not contain sensor records")

    records: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows[2:], start=3):
        if len(row) < 3 or row[0] in (None, ""):
            continue

        try:
            timestamp = datetime.strptime(str(row[0]).strip(), SOURCE_TIMESTAMP_FORMAT)
            entry_id = int(row[1])
            value = float(row[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid row {row_number} in {path.name}: {row!r}") from exc

        if entry_id in records:
            raise ValueError(f"duplicate Entry_id={entry_id} in {path.name}")
        records[entry_id] = {"timestamp": timestamp, "value": value}

    if not records:
        raise ValueError(f"{path.name} has no usable sensor records")
    return records


def build_environment_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sensor_records = {
        field: read_sensor_workbook(RAW_ENV_DIR / filename)
        for field, filename in SOURCE_FILES.items()
    }

    entry_sets = [set(records) for records in sensor_records.values()]
    common_ids = set.intersection(*entry_sets)
    all_ids = set.union(*entry_sets)
    missing_by_field = {
        field: sorted(all_ids - set(records))
        for field, records in sensor_records.items()
    }

    rows = []
    timestamp_mismatches = []
    for entry_id in sorted(common_ids):
        timestamps = {
            field: sensor_records[field][entry_id]["timestamp"]
            for field in ENV_FIELDS
        }
        if len(set(timestamps.values())) != 1:
            timestamp_mismatches.append(
                {
                    "entry_id": entry_id,
                    "timestamps": {
                        field: value.isoformat()
                        for field, value in timestamps.items()
                    },
                }
            )

        source_time = timestamps["temperature"]
        utc_time = source_time.astimezone(timezone.utc).replace(tzinfo=None)
        rows.append(
            {
                "timestamp": utc_time.strftime(TIMESTAMP_FORMAT),
                "temperature": sensor_records["temperature"][entry_id]["value"],
                "soil_humidity": sensor_records["soil_humidity"][entry_id]["value"],
                "light": sensor_records["light"][entry_id]["value"],
                "ph": sensor_records["ph"][entry_id]["value"],
            }
        )

    rows.sort(key=lambda row: row["timestamp"])
    timestamps = [
        datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
        for row in rows
    ]
    intervals = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
    ]

    ranges = {
        "temperature": (0.0, 60.0),
        "soil_humidity": (0.0, 100.0),
        "light": (0.0, None),
        "ph": (0.0, 14.0),
    }
    range_violations = []
    for row in rows:
        for field, (lower, upper) in ranges.items():
            value = float(row[field])
            if value < lower or (upper is not None and value > upper):
                range_violations.append(
                    {
                        "timestamp": row["timestamp"],
                        "field": field,
                        "value": value,
                        "expected": [lower, upper],
                    }
                )

    report = {
        "source_dataset": SOURCE_URLS["dataset_page"],
        "source_timezone": "UTC+05:30 as encoded in source timestamps",
        "output_timezone": "UTC",
        "source_files": SOURCE_FILES,
        "source_rows_by_field": {
            field: len(records)
            for field, records in sensor_records.items()
        },
        "source_entry_id_count": len(all_ids),
        "joined_record_count": len(rows),
        "missing_entry_ids_by_field": missing_by_field,
        "timestamp_mismatch_count": len(timestamp_mismatches),
        "timestamp_mismatch_examples": timestamp_mismatches[:10],
        "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
        "date_range_utc": (
            [timestamps[0].strftime(TIMESTAMP_FORMAT), timestamps[-1].strftime(TIMESTAMP_FORMAT)]
            if timestamps
            else []
        ),
        "interval_seconds": {
            "min": min(intervals) if intervals else None,
            "median": statistics.median(intervals) if intervals else None,
            "max": max(intervals) if intervals else None,
        },
        "range_violation_count": len(range_violations),
        "range_violation_examples": range_violations[:10],
        "maximum_contiguous_window_hours": (
            (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if timestamps
            else 0
        ),
        "meets_full_multimodal_schema": False,
        "schema_limitations": [
            "This package contains environment data only.",
            "There are no image timestamps or image labels to create paired samples.",
            "The available source interval is shorter than the required 24-hour window.",
        ],
    }
    return rows, report


def write_environment_csv(rows: list[dict[str, Any]]) -> None:
    ENV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ENV_OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("timestamp", *ENV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(report: dict[str, Any]) -> None:
    QUALITY_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SOURCES_OUTPUT.write_text(
        json.dumps(
            {
                "dataset_page": SOURCE_URLS["dataset_page"],
                "api_page": SOURCE_URLS["api_page"],
                "license": "CC BY 4.0",
                "retrieved_on": "2026-09-22",
                "source_files": SOURCE_FILES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    README_OUTPUT.write_text(
        """# Environment pretraining data

This directory contains a normalized environment-only package prepared from
the public Mendeley Data dataset listed in `sources.json`.

## Files

- `environment.csv`: canonical fields required by the project:
  `timestamp,temperature,soil_humidity,light,ph`
- `quality_report.json`: merge, timestamp, range, and coverage checks.
- `sources.json`: source URLs, license, and source file mapping.

The source workbooks share an `Entry_id` and timestamp. Their five sensor
tables were joined by `Entry_id`. Source timestamps include UTC+05:30 and
were normalized to UTC in `environment.csv`.

This is suitable for validating the environment preprocessing pipeline and
for a small environment-branch pretraining experiment. It is not a complete
multimodal training dataset: it has no watermelon images, image capture
timestamps, or four-task labels, and its total time span is shorter than the
required 24-hour history window. Do not use it as final model evaluation
data.
""",
        encoding="utf-8",
    )


def prepare_watermelon_image_manifest() -> None:
    IMAGE_OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "image_id",
        "image_path",
        "source_filename",
        "source_crop",
        "source_diagnosis",
        "source_country",
        "source_state",
    )
    rows = []
    for index, source_row in enumerate(WATERMELON_IMAGE_ROWS, start=1):
        source_path = IMAGE_RAW_DIR / source_row["source_filename"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        image_id = f"IMG_{index:06d}"
        output_name = f"{image_id}.jpg"
        shutil.copy2(source_path, IMAGE_OUTPUT_IMAGES / output_name)
        rows.append(
            {
                "image_id": image_id,
                "image_path": f"images/{output_name}",
                "source_filename": source_row["source_filename"],
                "source_crop": "Watermelon",
                "source_diagnosis": source_row["source_diagnosis"],
                "source_country": source_row["source_country"],
                "source_state": source_row["source_state"],
            }
        )

    with IMAGE_MANIFEST_OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    IMAGE_SOURCES_OUTPUT.write_text(
        json.dumps(
            {
                "dataset_page": IMAGE_SOURCE_PAGE,
                "license": "CC BY 4.0",
                "retrieved_on": "2026-09-22",
                "selection_rule": "crop == Watermelon",
                "selected_source_files": [
                    row["source_filename"] for row in WATERMELON_IMAGE_ROWS
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    IMAGE_README_OUTPUT.write_text(
        """# Watermelon image pretraining subset

This directory contains the three records in the source dataset whose crop
field is exactly `Watermelon`. The source diagnoses are preserved in
`image_manifest.csv`.

This is an image-only pretraining package. It intentionally does not contain
the project's `labels.csv`, because the source does not provide image capture
times, the four project labels, or paired environment windows. Do not assign
these records a fabricated growth stage, maturity level, or severity level.

The source dataset is CC BY 4.0. See `sources.json` for provenance.
""",
        encoding="utf-8",
    )


def main() -> None:
    rows, report = build_environment_rows()
    write_environment_csv(rows)
    write_metadata(report)
    prepare_watermelon_image_manifest()
    print(f"environment_rows={len(rows)}")
    print(f"output={ENV_OUTPUT}")
    print(f"quality_report={QUALITY_OUTPUT}")
    print(f"interval_median_seconds={report['interval_seconds']['median']}")
    print(f"coverage_hours={report['maximum_contiguous_window_hours']:.4f}")
    print(f"watermelon_images={len(WATERMELON_IMAGE_ROWS)}")
    print(f"image_manifest={IMAGE_MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
