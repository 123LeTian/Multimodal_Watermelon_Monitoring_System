from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left, bisect_right
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.watermelon_dataset import (
    ENV_FIELDS,
    TIMESTAMP_FORMAT,
    ImageTransform,
    parse_timestamp,
)
from src.models import MultimodalWatermelonModel
from src.models import TaskOutputDims
from src.utils.config import deep_merge, load_config


CLASS_NAMES = {
    "growth_stage": ["发芽期", "伸蔓期", "开花期", "坐果期", "膨大期", "成熟期"],
    "health_level": ["正常", "轻微异常", "中度异常", "严重异常"],
    "maturity_level": ["未成熟", "接近成熟", "成熟", "过熟"],
    "abnormal_alert": ["无异常", "有异常"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for one watermelon image and environment window."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--capture-time", required=True)
    parser.add_argument(
        "--sensor-device-id",
        default=None,
        help="Sensor device_id to match when environment.csv contains multiple devices.",
    )
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--abnormal-alert-threshold",
        type=float,
        default=None,
        help="Optional positive-class threshold for abnormal alerts.",
    )
    parser.add_argument("--output", default="outputs/single_prediction.json")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def read_environment_window(
    environment_path: Path,
    capture_time_text: str,
    window_hours: int,
    environment_fields: tuple[str, ...] = ENV_FIELDS,
    environment_normalization: dict[str, tuple[float, float]] | None = None,
    optional_environment_fields: tuple[str, ...] = (),
    sensor_device_id: str | None = None,
    require_device_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if window_hours <= 0:
        raise ValueError("--window-hours must be positive")

    with environment_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("environment.csv is empty")

    optional_fields = set(optional_environment_fields)
    required_fields = {"timestamp", *(set(environment_fields) - optional_fields)}
    missing_fields = required_fields - set(rows[0])
    if (
        "soil_humidity" in environment_fields
        and "soil_humidity" not in rows[0]
        and "air_humidity" in rows[0]
    ):
        raise ValueError(
            "environment.csv contains air_humidity but the project requires "
            "soil_humidity; these measurements are not interchangeable"
        )
    if missing_fields:
        raise ValueError(f"environment.csv missing fields: {sorted(missing_fields)}")
    if require_device_id and "device_id" not in rows[0]:
        raise ValueError("environment.csv must contain device_id for this checkpoint")

    available_devices = sorted(
        {row.get("device_id", "").strip() or "default" for row in rows}
    )
    if sensor_device_id is None:
        if len(available_devices) > 1:
            raise ValueError(
                "environment.csv contains multiple devices; pass --sensor-device-id"
            )
        sensor_device_id = available_devices[0]
    rows = [
        row
        for row in rows
        if (row.get("device_id", "").strip() or "default") == sensor_device_id
    ]
    if not rows:
        raise ValueError(f"no environment rows found for device_id={sensor_device_id!r}")

    normalization = environment_normalization or {}
    parsed_rows = []
    for row in rows:
        values: list[float] = []
        masks: list[float] = []
        for field in environment_fields:
            raw_value = row.get(field, "").strip()
            if not raw_value:
                if field not in optional_fields:
                    raise ValueError(f"missing required {field} at {row['timestamp']}")
                values.append(0.0)
                masks.append(0.0)
                continue
            value = float(raw_value)
            if field in normalization:
                center, scale = normalization[field]
                value = (value - center) / scale
            values.append(value)
            masks.append(1.0)
        parsed_rows.append((parse_timestamp(row["timestamp"]), values, masks))
    parsed_rows.sort(key=lambda item: item[0])
    environment_times = [item[0] for item in parsed_rows]
    environment_values = [item[1] for item in parsed_rows]
    environment_sensor_masks = [item[2] for item in parsed_rows]

    capture_time = parse_timestamp(capture_time_text)
    window_start = capture_time - timedelta(hours=window_hours)
    start = bisect_left(environment_times, window_start)
    end = bisect_right(environment_times, capture_time)
    if start == end:
        raise ValueError(
            f"no environment data found in "
            f"[{window_start.strftime(TIMESTAMP_FORMAT)}, "
            f"{capture_time.strftime(TIMESTAMP_FORMAT)}]"
        )

    values = torch.tensor(environment_values[start:end], dtype=torch.float32)
    sensor_mask = torch.tensor(
        environment_sensor_masks[start:end], dtype=torch.float32
    )
    offsets = [
        (time - capture_time).total_seconds() / (window_hours * 3600)
        for time in environment_times[start:end]
    ]
    time_offsets = torch.tensor(offsets, dtype=torch.float32).view(-1, 1)
    metadata = {
        "window_start": window_start.strftime(TIMESTAMP_FORMAT),
        "window_end": capture_time.strftime(TIMESTAMP_FORMAT),
        "environment_rows": int(values.shape[0]),
        "sensor_device_id": sensor_device_id,
    }
    return values, sensor_mask, time_offsets, metadata


def build_model(
    config: dict[str, Any],
    image_pretrained: bool | None = None,
) -> MultimodalWatermelonModel:
    model_config = config["model"]
    task_config = model_config["output_tasks"]
    task_dims = TaskOutputDims(
        growth_stage=int(task_config["growth_stage"]),
        health_level=int(task_config["health_level"]),
        maturity_level=int(task_config["maturity_level"]),
        abnormal_alert=int(task_config["abnormal_alert"]),
    )
    return MultimodalWatermelonModel(
        sensor_dim=int(model_config["sensor_dim"]),
        time_dim=int(model_config["time_feature_dim"]),
        task_dims=task_dims,
        image_pretrained=(
            bool(model_config["image_pretrained"])
            if image_pretrained is None
            else image_pretrained
        ),
        input_mode=str(model_config.get("input_mode", "multimodal")),
        use_sensor_mask=bool(model_config.get("use_sensor_mask", False)),
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MultimodalWatermelonModel, Any, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing model_state_dict")

    config = load_config(None)
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, dict):
        config = deep_merge(config, saved_config)

    model = build_model(config, image_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config


def build_prediction(
    logits: torch.Tensor,
    task: str,
    positive_threshold: float | None = None,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    if positive_threshold is not None:
        if task != "abnormal_alert":
            raise ValueError("positive threshold is only supported for abnormal_alert")
        class_index = int(probabilities[0, 1].item() >= positive_threshold)
        confidence = probabilities[0, class_index]
    else:
        confidence, class_id = probabilities.max(dim=-1)
        class_index = int(class_id.item())

    prediction = {
        "class_id": class_index,
        "class_name": CLASS_NAMES[task][class_index],
        "confidence": float(confidence.item()),
        "probabilities": [
            float(value) for value in probabilities.squeeze(0).tolist()
        ],
    }
    if positive_threshold is not None:
        prediction["decision_threshold"] = positive_threshold
    return prediction


def main() -> None:
    args = parse_args()
    if (
        args.abnormal_alert_threshold is not None
        and not 0.0 <= args.abnormal_alert_threshold <= 1.0
    ):
        raise ValueError("abnormal alert threshold must be in [0, 1]")
    device = resolve_device(args.device)

    image_path = Path(args.image)
    environment_path = Path(args.environment)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not environment_path.exists():
        raise FileNotFoundError(environment_path)

    model, checkpoint, config = load_model(checkpoint_path, device)
    data_config = config["data"]
    environment_fields = tuple(data_config["environment_fields"])
    normalization_config = data_config.get("environment_normalization", {})
    if normalization_config == "auto":
        raise ValueError(
            "checkpoint contains unresolved automatic environment normalization"
        )
    environment_normalization = {
        field: (float(values[0]), float(values[1]))
        for field, values in normalization_config.items()
    }
    transform = ImageTransform(
        image_size=tuple(int(value) for value in data_config["image_size"]),
        preserve_aspect_ratio=bool(data_config.get("preserve_aspect_ratio", False)),
    )
    with Image.open(image_path) as image:
        image_tensor = transform(image).unsqueeze(0).to(device)

    environment, environment_sensor_mask, time_offsets, window_metadata = read_environment_window(
        environment_path=environment_path,
        capture_time_text=args.capture_time,
        window_hours=args.window_hours,
        environment_fields=environment_fields,
        environment_normalization=environment_normalization,
        optional_environment_fields=tuple(
            data_config.get("optional_environment_fields", ())
        ),
        sensor_device_id=args.sensor_device_id,
        require_device_id=bool(data_config.get("require_device_id", False)),
    )
    environment = environment.unsqueeze(0).to(device)
    environment_sensor_mask = environment_sensor_mask.unsqueeze(0).to(device)
    time_offsets = time_offsets.unsqueeze(0).to(device)
    environment_mask = torch.ones(
        1,
        environment.shape[1],
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            images=image_tensor,
            environment=environment,
            environment_sensor_mask=environment_sensor_mask,
            time_offsets=time_offsets,
            environment_mask=environment_mask,
        )

    predictions = {
        task: build_prediction(
            outputs[f"{task}_logits"],
            task,
            positive_threshold=(
                args.abnormal_alert_threshold
                if task == "abnormal_alert"
                else None
            ),
        )
        for task in CLASS_NAMES
    }
    result = {
        "warning": (
            "Predictions are produced by the current checkpoint. "
            "If the input is simulated data, this result has no research validity."
        ),
        "image": str(image_path),
        "capture_time": args.capture_time,
        "environment": str(environment_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_mode": config["model"].get("input_mode", "multimodal"),
        "device": str(device),
        "abnormal_alert_threshold": args.abnormal_alert_threshold,
        **window_metadata,
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"device={device}")
    print(f"image={image_path}")
    print(f"capture_time={args.capture_time}")
    print(f"environment_rows={window_metadata['environment_rows']}")
    for task, prediction in predictions.items():
        print(
            f"{task}: class_id={prediction['class_id']} "
            f"class_name={prediction['class_name']} "
            f"confidence={prediction['confidence']:.6f}"
        )
    print(f"saved_output={output_path}")


if __name__ == "__main__":
    main()
