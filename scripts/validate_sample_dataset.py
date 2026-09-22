from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_sample"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    labels = read_csv(DATASET_DIR / "labels.csv")
    environment = read_csv(DATASET_DIR / "environment.csv")
    splits = read_csv(DATASET_DIR / "split.csv")

    required_label_fields = {
        "image_id",
        "image_path",
        "capture_time",
        "growth_stage",
        "health_level",
        "maturity_level",
        "abnormal_alert",
    }
    required_env_fields = {"timestamp", "temperature", "soil_humidity", "light", "ph"}
    required_split_fields = {"image_id", "split"}

    if not labels:
        raise ValueError("labels.csv is empty")
    if not environment:
        raise ValueError("environment.csv is empty")
    if not splits:
        raise ValueError("split.csv is empty")

    if set(labels[0]) != required_label_fields:
        raise ValueError(f"labels.csv fields mismatch: {set(labels[0])}")
    if set(environment[0]) != required_env_fields:
        raise ValueError(f"environment.csv fields mismatch: {set(environment[0])}")
    if set(splits[0]) != required_split_fields:
        raise ValueError(f"split.csv fields mismatch: {set(splits[0])}")

    image_ids = [row["image_id"] for row in labels]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_id has duplicates")

    split_ids = {row["image_id"] for row in splits}
    if set(image_ids) != split_ids:
        raise ValueError("split.csv image_id set does not match labels.csv")

    split_values = {row["split"] for row in splits}
    if not split_values <= {"train", "val", "test"}:
        raise ValueError(f"invalid split values: {split_values}")

    env_times = [
        datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        for row in environment
    ]

    ranges = {
        "growth_stage": range(6),
        "health_level": range(4),
        "maturity_level": range(4),
        "abnormal_alert": range(2),
    }

    opened_images = 0
    matched_windows = 0
    for row in labels:
        image_path = DATASET_DIR / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as image:
            image.verify()
        opened_images += 1

        capture_time = datetime.strptime(row["capture_time"], "%Y-%m-%d %H:%M:%S")
        window_start = capture_time - timedelta(hours=24)
        window = [time for time in env_times if window_start <= time <= capture_time]
        if not window:
            raise ValueError(f"{row['image_id']} has no 24-hour environment window")
        matched_windows += 1

        for field, valid_range in ranges.items():
            value = int(row[field])
            if value not in valid_range:
                raise ValueError(f"{row['image_id']} has invalid {field}: {value}")

    split_counter = Counter(row["split"] for row in splits)

    print("validation=passed")
    print(f"labels={len(labels)}")
    print(f"environment_rows={len(environment)}")
    print(f"images_opened={opened_images}")
    print(f"windows_ok={matched_windows}")
    print(f"split_train={split_counter['train']}")
    print(f"split_val={split_counter['val']}")
    print(f"split_test={split_counter['test']}")


if __name__ == "__main__":
    main()
