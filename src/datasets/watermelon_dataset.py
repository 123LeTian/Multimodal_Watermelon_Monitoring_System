from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
    augment: bool = False
    preserve_aspect_ratio: bool = False

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.augment:
            if torch.rand(()) < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(
                0.85 + 0.30 * float(torch.rand(()))
            )
            image = ImageEnhance.Contrast(image).enhance(
                0.85 + 0.30 * float(torch.rand(()))
            )
            image = ImageEnhance.Color(image).enhance(
                0.85 + 0.30 * float(torch.rand(()))
            )
            if torch.rand(()) < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(radius=0.8))

        if self.preserve_aspect_ratio:
            resized = ImageOps.contain(
                image,
                self.image_size,
                method=Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGB", self.image_size, color=(0, 0, 0))
            left = (self.image_size[0] - resized.width) // 2
            top = (self.image_size[1] - resized.height) // 2
            canvas.paste(resized, (left, top))
            image = canvas
        else:
            image = image.resize(self.image_size)

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
    """Image plus same-device environment readings from the preceding window."""

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
        environment_normalization: dict[str, tuple[float, float]] | None = None,
        optional_environment_fields: tuple[str, ...] = (),
        augment_image: bool = False,
        preserve_aspect_ratio: bool = False,
        strict_time_alignment: bool = True,
        require_device_id: bool = False,
        require_task_masks: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.split_file = self._resolve_split_file(split_file)
        self.labels_file = self._resolve_labels_file(labels_file)
        self.window = timedelta(hours=window_hours)
        self.environment_fields = environment_fields or ENV_FIELDS
        self.environment_normalization = environment_normalization or {}
        self.optional_environment_fields = set(optional_environment_fields)
        self.strict_time_alignment = strict_time_alignment
        self.require_device_id = require_device_id
        self.require_task_masks = require_task_masks
        self.image_transform = ImageTransform(
            image_size=image_size,
            normalize=normalize_image,
            augment=augment_image,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )

        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)} or None")

        self.labels = read_csv_rows(self.labels_file)
        self.environment = read_csv_rows(self.dataset_dir / "environment.csv")
        self.splits = read_csv_rows(self.split_file)
        self.has_task_masks = bool(
            self.labels and set(MASK_FIELDS).issubset(self.labels[0])
        )
        self._validate_required_files()
        self._filter_by_split()
        self._prepare_environment_cache()
        if self.strict_time_alignment:
            self._validate_environment_alignment()

    def _resolve_split_file(self, split_file: str | Path | None) -> Path:
        if split_file is None:
            return self.dataset_dir / "split.csv"
        path = Path(split_file)
        if path.is_absolute():
            return path
        candidate = self.dataset_dir / path
        return candidate if candidate.exists() else path

    def _resolve_labels_file(self, labels_file: str | Path | None) -> Path:
        if labels_file is None:
            return self.dataset_dir / "labels.csv"
        path = Path(labels_file)
        if path.is_absolute():
            return path
        candidate = self.dataset_dir / path
        return candidate if candidate.exists() else path

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
        required_split_fields = {"image_id", "split"}
        required_env_fields = {
            "timestamp",
            *(set(self.environment_fields) - self.optional_environment_fields),
        }
        label_fields = set(self.labels[0])
        environment_columns = set(self.environment[0])

        if not required_label_fields.issubset(label_fields):
            raise ValueError(
                f"{self.labels_file} missing fields: "
                f"{sorted(required_label_fields - label_fields)}"
            )
        present_mask_fields = label_fields & set(MASK_FIELDS)
        if present_mask_fields and present_mask_fields != set(MASK_FIELDS):
            raise ValueError(
                f"{self.labels_file} has partial task masks: {sorted(present_mask_fields)}"
            )
        if self.require_task_masks and not self.has_task_masks:
            raise ValueError(
                "labels file must contain all *_valid task masks; weak labels must not "
                "be trained as ground truth"
            )

        missing_environment_fields = required_env_fields - environment_columns
        if (
            "soil_humidity" in self.environment_fields
            and "soil_humidity" not in environment_columns
            and "air_humidity" in environment_columns
        ):
            raise ValueError(
                "environment.csv contains air_humidity but the project requires "
                "soil_humidity; these measurements are not interchangeable"
            )
        if missing_environment_fields:
            raise ValueError(
                "environment.csv missing required fields: "
                f"{sorted(missing_environment_fields)}"
            )
        if self.require_device_id and (
            "device_id" not in label_fields or "device_id" not in environment_columns
        ):
            raise ValueError(
                "device_id is required in both labels and environment data when "
                "data.require_device_id is enabled"
            )
        if set(self.splits[0]) != required_split_fields:
            raise ValueError(f"{self.split_file} fields mismatch: {set(self.splits[0])}")

        unknown_optional = self.optional_environment_fields - set(self.environment_fields)
        if unknown_optional:
            raise ValueError(
                "optional environment fields are not model inputs: "
                f"{sorted(unknown_optional)}"
            )
        for field, (_, scale) in self.environment_normalization.items():
            if field not in self.environment_fields:
                raise ValueError(f"normalization configured for unknown field: {field}")
            if float(scale) <= 0:
                raise ValueError(f"normalization scale must be positive for {field}")

    def _filter_by_split(self) -> None:
        split_by_id = {row["image_id"]: row["split"] for row in self.splits}
        missing_ids = [
            row["image_id"] for row in self.labels if row["image_id"] not in split_by_id
        ]
        if missing_ids:
            raise ValueError(f"{self.split_file} missing image_id values: {missing_ids[:5]}")
        if self.split is not None:
            self.labels = [
                row for row in self.labels if split_by_id[row["image_id"]] == self.split
            ]
        if not self.labels:
            raise ValueError(f"no samples found for split={self.split!r}")

    def _prepare_environment_cache(self) -> None:
        grouped: dict[str, list[tuple[datetime, list[float], list[float]]]] = defaultdict(list)
        for row in self.environment:
            timestamp = parse_timestamp(row["timestamp"])
            device_id = row.get("device_id", "").strip() or "default"
            values: list[float] = []
            masks: list[float] = []
            for field in self.environment_fields:
                raw_value = row.get(field, "").strip()
                if not raw_value:
                    if field not in self.optional_environment_fields:
                        raise ValueError(
                            f"missing required {field} at {row['timestamp']} "
                            f"for device {device_id}"
                        )
                    values.append(0.0)
                    masks.append(0.0)
                    continue
                value = float(raw_value)
                if field in self.environment_normalization:
                    center, scale = self.environment_normalization[field]
                    value = (value - float(center)) / float(scale)
                values.append(value)
                masks.append(1.0)
            grouped[device_id].append((timestamp, values, masks))

        self.env_times: dict[str, list[datetime]] = {}
        self.env_values: dict[str, list[list[float]]] = {}
        self.env_sensor_masks: dict[str, list[list[float]]] = {}
        for device_id, entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            self.env_times[device_id] = [item[0] for item in entries]
            self.env_values[device_id] = [item[1] for item in entries]
            self.env_sensor_masks[device_id] = [item[2] for item in entries]

    def _environment_bounds(
        self,
        row: dict[str, str],
    ) -> tuple[str, datetime, int, int]:
        device_id = row.get("device_id", "").strip() or "default"
        capture_time = parse_timestamp(row["capture_time"])
        times = self.env_times.get(device_id, [])
        start = bisect_left(times, capture_time - self.window)
        end = bisect_right(times, capture_time)
        return device_id, capture_time, start, end

    def _validate_environment_alignment(self) -> None:
        failures: list[tuple[str, str, str]] = []
        for row in self.labels:
            device_id, capture_time, start, end = self._environment_bounds(row)
            if start == end:
                failures.append(
                    (row["image_id"], device_id, capture_time.strftime(TIMESTAMP_FORMAT))
                )
                if len(failures) >= 5:
                    break
        if failures:
            hours = self.window.total_seconds() / 3600
            raise ValueError(
                "images have no same-device environment readings in the preceding "
                f"{hours:g} hours; examples={failures}. Run "
                "scripts/validate_multimodal_alignment.py before training."
            )

    def calculate_environment_normalization(self) -> dict[str, list[float]]:
        """Calculate mean/std from environment rows used by this dataset split."""
        used_rows: dict[str, set[int]] = defaultdict(set)
        for row in self.labels:
            device_id, _, start, end = self._environment_bounds(row)
            used_rows[device_id].update(range(start, end))

        result: dict[str, list[float]] = {}
        for field_index, field in enumerate(self.environment_fields):
            values = [
                self.env_values[device_id][row_index][field_index]
                for device_id, row_indices in used_rows.items()
                for row_index in row_indices
                if self.env_sensor_masks[device_id][row_index][field_index] > 0
            ]
            if not values:
                result[field] = [0.0, 1.0]
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            standard_deviation = float(tensor.std(unbiased=False))
            scale = standard_deviation if standard_deviation >= 1e-6 else 1.0
            result[field] = [float(tensor.mean()), scale]
        return result

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.labels[index]
        image_path = self.dataset_dir / row["image_path"]
        environment, sensor_mask, time_offsets = self._get_environment_window(row)
        with Image.open(image_path) as image:
            image_tensor = self.image_transform(image)

        labels = {field: int(row[field]) for field in LABEL_FIELDS}
        task_masks = {
            field: float(row.get(f"{field}_valid", "1")) for field in LABEL_FIELDS
        }
        return {
            "image_id": row["image_id"],
            "image_path": str(image_path),
            "capture_time": row["capture_time"],
            "image": image_tensor,
            "environment": environment,
            "environment_sensor_mask": sensor_mask,
            "time_offsets": time_offsets,
            "labels": labels,
            "task_masks": task_masks,
        }

    def _get_environment_window(
        self,
        row: dict[str, str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_id, capture_time, start, end = self._environment_bounds(row)
        if start == end:
            raise ValueError(
                f"no environment data found before {capture_time} for device {device_id}"
            )
        values = torch.tensor(
            self.env_values[device_id][start:end], dtype=torch.float32
        )
        sensor_mask = torch.tensor(
            self.env_sensor_masks[device_id][start:end], dtype=torch.float32
        )
        offsets = [
            (time - capture_time).total_seconds() / self.window.total_seconds()
            for time in self.env_times[device_id][start:end]
        ]
        time_offsets = torch.tensor(offsets, dtype=torch.float32).unsqueeze(-1)
        return values, sensor_mask, time_offsets


def watermelon_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    images = torch.stack([item["image"] for item in batch], dim=0)
    lengths = torch.tensor(
        [item["environment"].shape[0] for item in batch], dtype=torch.long
    )
    max_length = int(lengths.max().item())
    env_dim = batch[0]["environment"].shape[1]

    environment = torch.zeros(batch_size, max_length, env_dim, dtype=torch.float32)
    sensor_mask = torch.zeros(batch_size, max_length, env_dim, dtype=torch.float32)
    time_offsets = torch.zeros(batch_size, max_length, 1, dtype=torch.float32)
    environment_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = item["environment"].shape[0]
        environment[row_index, :length] = item["environment"]
        sensor_mask[row_index, :length] = item["environment_sensor_mask"]
        time_offsets[row_index, :length] = item["time_offsets"]
        environment_mask[row_index, :length] = True

    labels = {
        field: torch.tensor([item["labels"][field] for item in batch], dtype=torch.long)
        for field in LABEL_FIELDS
    }
    task_masks = {
        field: torch.tensor(
            [item["task_masks"][field] for item in batch], dtype=torch.float32
        )
        for field in LABEL_FIELDS
    }
    return {
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "capture_times": [item["capture_time"] for item in batch],
        "images": images,
        "environment": environment,
        "environment_sensor_mask": sensor_mask,
        "time_offsets": time_offsets,
        "environment_mask": environment_mask,
        "environment_lengths": lengths,
        "labels": labels,
        "task_masks": task_masks,
    }
