from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ENV_FIELDS = ("temperature", "soil_humidity", "light", "ph")
LABEL_FIELDS = ("growth_stage", "health_level", "maturity_level", "abnormal_alert")
MASK_FIELDS = tuple(f"{field}_valid" for field in LABEL_FIELDS)
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ImageTransform:
    image_size: tuple[int, int] = (224, 224)
    normalize: bool = True

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)

        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


class WatermelonDataset(Dataset):
    """Dataset for multimodal watermelon growth monitoring samples.

    Each item contains one image, the environment sequence from the previous
    `window_hours`, and four labels for the multi-task prediction model.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str | None = None,
        split_file: str | Path | None = None,
        labels_file: str | Path | None = None,
        window_hours: int = 24,
        image_size: tuple[int, int] = (224, 224),
        normalize_image: bool = True,
        environment_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.split_file = self._resolve_split_file(split_file)
        self.labels_file = self._resolve_labels_file(labels_file)
        self.window = timedelta(hours=window_hours)
        self.environment_fields = environment_fields or ENV_FIELDS
        self.image_transform = ImageTransform(
            image_size=image_size,
            normalize=normalize_image,
        )

        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)} or None")

        self.labels = read_csv_rows(self.labels_file)
        self.environment = read_csv_rows(self.dataset_dir / "environment.csv")
        self.splits = read_csv_rows(self.split_file)
        self.has_task_masks = bool(self.labels and set(MASK_FIELDS).issubset(self.labels[0]))

        self._validate_required_files()
        self._filter_by_split()
        self._prepare_environment_cache()

    def _resolve_split_file(self, split_file: str | Path | None) -> Path:
        if split_file is None:
            return self.dataset_dir / "split.csv"

        path = Path(split_file)
        if path.is_absolute():
            return path
        candidate = self.dataset_dir / path
        if candidate.exists():
            return candidate
        return path

    def _resolve_labels_file(self, labels_file: str | Path | None) -> Path:
        if labels_file is None:
            return self.dataset_dir / "labels.csv"

        path = Path(labels_file)
        if path.is_absolute():
            return path
        candidate = self.dataset_dir / path
        if candidate.exists():
            return candidate
        return path

    def _validate_required_files(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(self.dataset_dir)
        if not self.labels_file.exists():
            raise FileNotFoundError(self.labels_file)
        if not self.labels:
            raise ValueError(f"{self.labels_file} is empty")
        if not self.environment:
            raise ValueError("environment.csv is empty")
        if not self.splits:
            raise ValueError(f"{self.split_file} is empty")

        required_label_fields = {"image_id", "image_path", "capture_time", *LABEL_FIELDS}
        required_env_fields = {"timestamp", *self.environment_fields}
        required_split_fields = {"image_id", "split"}

        label_fields = set(self.labels[0])
        if not required_label_fields.issubset(label_fields):
            raise ValueError(f"{self.labels_file} missing fields: {required_label_fields - label_fields}")
        present_mask_fields = label_fields & set(MASK_FIELDS)
        if present_mask_fields and present_mask_fields != set(MASK_FIELDS):
            raise ValueError(
                f"{self.labels_file} has partial task masks: {sorted(present_mask_fields)}"
            )
        if set(self.environment[0]) != required_env_fields:
            raise ValueError(f"environment.csv fields mismatch: {set(self.environment[0])}")
        if set(self.splits[0]) != required_split_fields:
            raise ValueError(f"{self.split_file} fields mismatch: {set(self.splits[0])}")

    def _filter_by_split(self) -> None:
        split_by_id = {row["image_id"]: row["split"] for row in self.splits}
        missing_ids = [row["image_id"] for row in self.labels if row["image_id"] not in split_by_id]
        if missing_ids:
            raise ValueError(f"{self.split_file} missing image_id values: {missing_ids[:5]}")

        if self.split is not None:
            self.labels = [
                row for row in self.labels if split_by_id[row["image_id"]] == self.split
            ]

        if not self.labels:
            raise ValueError(f"no samples found for split={self.split!r}")

    def _prepare_environment_cache(self) -> None:
        parsed_rows = []
        for row in self.environment:
            timestamp = parse_timestamp(row["timestamp"])
            values = [float(row[field]) for field in self.environment_fields]
            parsed_rows.append((timestamp, values))

        parsed_rows.sort(key=lambda item: item[0])
        self.env_times = [item[0] for item in parsed_rows]
        self.env_values = [item[1] for item in parsed_rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.labels[index]
        image_path = self.dataset_dir / row["image_path"]
        capture_time = parse_timestamp(row["capture_time"])
        environment, time_offsets = self._get_environment_window(capture_time)

        with Image.open(image_path) as image:
            image_tensor = self.image_transform(image)

        labels = {
            "growth_stage": int(row["growth_stage"]),
            "health_level": int(row["health_level"]),
            "maturity_level": int(row["maturity_level"]),
            "abnormal_alert": int(row["abnormal_alert"]),
        }
        task_masks = {
            field: float(row.get(f"{field}_valid", "1"))
            for field in LABEL_FIELDS
        }

        return {
            "image_id": row["image_id"],
            "image_path": str(image_path),
            "capture_time": row["capture_time"],
            "image": image_tensor,
            "environment": environment,
            "time_offsets": time_offsets,
            "labels": labels,
            "task_masks": task_masks,
        }

    def _get_environment_window(
        self,
        capture_time: datetime,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window_start = capture_time - self.window
        start = bisect_left(self.env_times, window_start)
        end = bisect_right(self.env_times, capture_time)

        if start == end:
            raise ValueError(f"no environment data found before {capture_time}")

        env_values = torch.tensor(self.env_values[start:end], dtype=torch.float32)
        offsets = [
            (time - capture_time).total_seconds() / self.window.total_seconds()
            for time in self.env_times[start:end]
        ]
        time_offsets = torch.tensor(offsets, dtype=torch.float32).unsqueeze(-1)

        return env_values, time_offsets


def watermelon_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    images = torch.stack([item["image"] for item in batch], dim=0)

    lengths = torch.tensor(
        [item["environment"].shape[0] for item in batch],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    env_dim = batch[0]["environment"].shape[1]

    environment = torch.zeros(batch_size, max_length, env_dim, dtype=torch.float32)
    time_offsets = torch.zeros(batch_size, max_length, 1, dtype=torch.float32)
    environment_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for row_index, item in enumerate(batch):
        length = item["environment"].shape[0]
        environment[row_index, :length] = item["environment"]
        time_offsets[row_index, :length] = item["time_offsets"]
        environment_mask[row_index, :length] = True

    labels = {
        field: torch.tensor([item["labels"][field] for item in batch], dtype=torch.long)
        for field in LABEL_FIELDS
    }
    task_masks = {
        field: torch.tensor([item["task_masks"][field] for item in batch], dtype=torch.float32)
        for field in LABEL_FIELDS
    }

    return {
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "capture_times": [item["capture_time"] for item in batch],
        "images": images,
        "environment": environment,
        "time_offsets": time_offsets,
        "environment_mask": environment_mask,
        "environment_lengths": lengths,
        "labels": labels,
        "task_masks": task_masks,
    }
