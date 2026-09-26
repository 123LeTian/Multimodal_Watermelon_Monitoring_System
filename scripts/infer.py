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
    parser.add_argument("--window-hours", type=int, default=24)
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if window_hours <= 0:
        raise ValueError("--window-hours must be positive")

    with environment_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("environment.csv is empty")

    required_fields = {"timestamp", *environment_fields}
    missing_fields = required_fields - set(rows[0])
    if missing_fields:
        raise ValueError(f"environment.csv missing fields: {sorted(missing_fields)}")

    parsed_rows = sorted(
        (
            parse_timestamp(row["timestamp"]),
            [float(row[field]) for field in environment_fields],
        )
        for row in rows
    )
    environment_times = [item[0] for item in parsed_rows]
    environment_values = [item[1] for item in parsed_rows]

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
    offsets = [
        (time - capture_time).total_seconds() / (window_hours * 3600)
        for time in environment_times[start:end]
    ]
    time_offsets = torch.tensor(offsets, dtype=torch.float32).view(-1, 1)
    metadata = {
        "window_start": window_start.strftime(TIMESTAMP_FORMAT),
        "window_end": capture_time.strftime(TIMESTAMP_FORMAT),
        "environment_rows": int(values.shape[0]),
    }
    return values, time_offsets, metadata


def build_model(config: dict[str, Any]) -> MultimodalWatermelonModel:
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
        image_pretrained=bool(model_config["image_pretrained"]),
        input_mode=str(model_config.get("input_mode", "multimodal")),
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

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config


def build_prediction(
    logits: torch.Tensor,
    task: str,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    confidence, class_id = probabilities.max(dim=-1)
    class_index = int(class_id.item())
    return {
        "class_id": class_index,
        "class_name": CLASS_NAMES[task][class_index],
        "confidence": float(confidence.item()),
        "probabilities": [
            float(value) for value in probabilities.squeeze(0).tolist()
        ],
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    image_path = Path(args.image)
    environment_path = Path(args.environment)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not environment_path.exists():
        raise FileNotFoundError(environment_path)

    transform = ImageTransform()
    with Image.open(image_path) as image:
        image_tensor = transform(image).unsqueeze(0).to(device)

    model, checkpoint, config = load_model(checkpoint_path, device)
    data_config = config["data"]
    environment_fields = tuple(data_config["environment_fields"])

    environment, time_offsets, window_metadata = read_environment_window(
        environment_path=environment_path,
        capture_time_text=args.capture_time,
        window_hours=args.window_hours,
        environment_fields=environment_fields,
    )
    environment = environment.unsqueeze(0).to(device)
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
            time_offsets=time_offsets,
            environment_mask=environment_mask,
        )

    predictions = {
        task: build_prediction(outputs[f"{task}_logits"], task)
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
