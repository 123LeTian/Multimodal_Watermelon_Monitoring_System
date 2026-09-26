from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "watermelon_multimodal_monitoring",
        "description": "ResNet50 + Transformer multi-task classification model",
        "seed": 42,
    },
    "data": {
        "dataset_dir": "dataset_sample",
        "split_file": None,
        "image_size": [224, 224],
        "window_hours": 24,
        "environment_fields": ["temperature", "soil_humidity", "light", "ph"],
        "splits": {
            "train": "train",
            "validation": "val",
            "test": "test",
        },
    },
    "model": {
        "input_mode": "multimodal",
        "image_backbone": "resnet50",
        "image_pretrained": False,
        "environment_encoder": "transformer",
        "sensor_dim": 4,
        "time_feature_dim": 1,
        "output_tasks": {
            "growth_stage": 6,
            "health_level": 4,
            "maturity_level": 4,
            "abnormal_alert": 2,
        },
    },
    "training": {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "device": "auto",
    },
    "loss": {
        "type": "cross_entropy",
        "label_smoothing": 0.0,
        "task_weights": {
            "growth_stage": 1.0,
            "health_level": 1.0,
            "maturity_level": 1.0,
            "abnormal_alert": 1.0,
        },
    },
    "outputs": {
        "checkpoint_dir": "checkpoints",
        "best_checkpoint": "checkpoints/best_model.pth",
        "last_checkpoint": "checkpoints/last_model.pth",
        "evaluation_dir": "outputs",
        "experiment_dir": "experiments",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("the YAML configuration root must be a mapping")
    return deep_merge(config, loaded)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
