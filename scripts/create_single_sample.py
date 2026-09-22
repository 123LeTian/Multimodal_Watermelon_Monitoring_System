from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from bisect import bisect_left, bisect_right
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.watermelon_dataset import (
    ENV_FIELDS,
    TIMESTAMP_FORMAT,
    parse_timestamp,
    read_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one standalone inference sample from a dataset."
    )
    parser.add_argument("--dataset-dir", default="dataset_sample")
    parser.add_argument("--image-id", default="IMG_000018")
    parser.add_argument("--output-dir", default="single_sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv_rows(dataset_dir / "labels.csv")
    selected = next(
        (row for row in labels if row["image_id"] == args.image_id),
        None,
    )
    if selected is None:
        raise ValueError(f"image_id not found: {args.image_id}")

    source_image = dataset_dir / selected["image_path"]
    if not source_image.exists():
        raise FileNotFoundError(source_image)

    environment_rows = read_csv_rows(dataset_dir / "environment.csv")
    environment_rows.sort(key=lambda row: parse_timestamp(row["timestamp"]))
    environment_times = [
        parse_timestamp(row["timestamp"]) for row in environment_rows
    ]
    capture_time = parse_timestamp(selected["capture_time"])
    window_start = capture_time - timedelta(hours=24)
    start = bisect_left(environment_times, window_start)
    end = bisect_right(environment_times, capture_time)
    selected_environment = environment_rows[start:end]

    if not selected_environment:
        raise ValueError(f"no environment rows for {args.image_id}")

    required_fields = {"timestamp", *ENV_FIELDS}
    if not required_fields <= set(selected_environment[0]):
        raise ValueError(
            f"environment.csv is missing fields: "
            f"{sorted(required_fields - set(selected_environment[0]))}"
        )

    output_image = output_dir / source_image.name
    shutil.copy2(source_image, output_image)

    output_environment = output_dir / "environment.csv"
    with output_environment.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", *ENV_FIELDS],
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in writer.fieldnames}
            for row in selected_environment
        )

    manifest = {
        "sample_id": args.image_id,
        "image": output_image.name,
        "environment": output_environment.name,
        "capture_time": selected["capture_time"],
        "window_start": window_start.strftime(TIMESTAMP_FORMAT),
        "window_end": selected["capture_time"],
        "environment_rows": len(selected_environment),
        "source": "dataset_sample simulated data",
        "warning": (
            "This is a simulated sample for inference pipeline testing. "
            "It must not be used as a real research sample."
        ),
    }
    (output_dir / "sample.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"created_sample={output_dir.resolve()}")
    print(f"image={output_image}")
    print(f"capture_time={selected['capture_time']}")
    print(f"environment_rows={len(selected_environment)}")
    print(f"manifest={output_dir / 'sample.json'}")


if __name__ == "__main__":
    main()
