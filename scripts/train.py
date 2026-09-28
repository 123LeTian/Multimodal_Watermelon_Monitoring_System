from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import WatermelonDataset, watermelon_collate_fn
from src.models import MultimodalWatermelonModel, TaskOutputDims, VALID_INPUT_MODES
from src.training import MultiTaskLoss, classification_metrics
from src.utils import load_config, save_config


TASK_NAMES = (
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)

TASK_CLASS_COUNTS = {
    "growth_stage": TaskOutputDims.growth_stage,
    "health_level": TaskOutputDims.health_level,
    "maturity_level": TaskOutputDims.maturity_level,
    "abnormal_alert": TaskOutputDims.abnormal_alert,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the watermelon multimodal model.")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML configuration path. CLI arguments override YAML values.",
    )
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional split CSV path. Defaults to split.csv inside the dataset directory.",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional labels CSV path. Defaults to labels.csv inside the dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--window-hours", type=int, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
    )
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument(
        "--input-mode",
        default=None,
        choices=VALID_INPUT_MODES,
        help="Input modality mode for ablation experiments.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print the effective configuration without training.",
    )
    return parser.parse_args()


def set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    overrides = {
        "dataset_dir": ("data", "dataset_dir"),
        "split_file": ("data", "split_file"),
        "labels_file": ("data", "labels_file"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "lr": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "seed": ("project", "seed"),
        "num_workers": ("training", "num_workers"),
        "window_hours": ("data", "window_hours"),
        "label_smoothing": ("loss", "label_smoothing"),
        "input_mode": ("model", "input_mode"),
        "device": ("training", "device"),
    }
    for argument_name, config_path in overrides.items():
        value = getattr(args, argument_name)
        if value is not None:
            set_nested(config, config_path, value)

    if args.image_size is not None:
        set_nested(config, ("data", "image_size"), list(args.image_size))

    if args.checkpoint_dir is not None:
        set_nested(config, ("outputs", "checkpoint_dir"), args.checkpoint_dir)
        set_nested(
            config,
            ("outputs", "best_checkpoint"),
            str(Path(args.checkpoint_dir) / "best_model.pth"),
        )
        set_nested(
            config,
            ("outputs", "last_checkpoint"),
            str(Path(args.checkpoint_dir) / "last_model.pth"),
        )

    return config


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "images": batch["images"].to(device),
        "environment": batch["environment"].to(device),
        "environment_sensor_mask": batch["environment_sensor_mask"].to(device),
        "time_offsets": batch["time_offsets"].to(device),
        "environment_mask": batch["environment_mask"].to(device),
        "labels": {
            name: values.to(device)
            for name, values in batch["labels"].items()
        },
        "task_masks": {
            name: values.to(device)
            for name, values in batch.get("task_masks", {}).items()
        },
    }


def create_loader(
    dataset_dir: str | Path,
    split: str,
    split_file: str | Path | None,
    labels_file: str | Path | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    window_hours: int,
    image_size: tuple[int, int],
    environment_fields: tuple[str, ...],
    environment_normalization: dict[str, tuple[float, float]],
    optional_environment_fields: tuple[str, ...],
    augment_image: bool,
    preserve_aspect_ratio: bool,
    strict_time_alignment: bool,
    require_device_id: bool,
    require_task_masks: bool,
) -> DataLoader:
    dataset = WatermelonDataset(
        dataset_dir=dataset_dir,
        split=split,
        split_file=split_file,
        labels_file=labels_file,
        window_hours=window_hours,
        image_size=image_size,
        environment_fields=environment_fields,
        environment_normalization=environment_normalization,
        optional_environment_fields=optional_environment_fields,
        augment_image=augment_image,
        preserve_aspect_ratio=preserve_aspect_ratio,
        strict_time_alignment=strict_time_alignment,
        require_device_id=require_device_id,
        require_task_masks=require_task_masks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=watermelon_collate_fn,
    )


def run_epoch(
    model: MultimodalWatermelonModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    optimizer: AdamW | None = None,
    grad_clip_norm: float | None = None,
    collect_classification_metrics: bool = False,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    totals = defaultdict(float)
    sample_count = 0
    targets_by_task: dict[str, list[torch.Tensor]] = {
        task: [] for task in TASK_NAMES
    }
    predictions_by_task: dict[str, list[torch.Tensor]] = {
        task: [] for task in TASK_NAMES
    }

    for batch in loader:
        batch_on_device = move_batch_to_device(batch, device)
        current_batch_size = batch_on_device["images"].shape[0]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(
                images=batch_on_device["images"],
                environment=batch_on_device["environment"],
                environment_sensor_mask=batch_on_device["environment_sensor_mask"],
                time_offsets=batch_on_device["time_offsets"],
                environment_mask=batch_on_device["environment_mask"],
            )
            total_loss, details = criterion(
                outputs,
                batch_on_device["labels"],
                task_masks=batch_on_device.get("task_masks"),
                return_details=True,
            )

            if is_training:
                total_loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        sample_count += current_batch_size
        for key, value in details.items():
            if key.endswith("_valid_count"):
                totals[key] += float(value.detach().cpu())
            else:
                totals[key] += float(value.detach().cpu()) * current_batch_size

        if collect_classification_metrics:
            for task in TASK_NAMES:
                if task in batch_on_device.get("task_masks", {}):
                    valid_mask = batch_on_device["task_masks"][task].view(-1) > 0
                else:
                    valid_mask = torch.ones_like(
                        batch_on_device["labels"][task],
                        dtype=torch.bool,
                        device=device,
                    )
                predictions = outputs[f"{task}_logits"].argmax(dim=1)
                targets_by_task[task].append(
                    batch_on_device["labels"][task][valid_mask].detach().cpu()
                )
                predictions_by_task[task].append(
                    predictions[valid_mask].detach().cpu()
                )

    metrics: dict[str, float] = {}
    for key, value in totals.items():
        if key.endswith("_valid_count"):
            metrics[key] = value
        else:
            metrics[key] = value / max(sample_count, 1)

    if collect_classification_metrics:
        for task in TASK_NAMES:
            targets = torch.cat(targets_by_task[task], dim=0)
            predictions = torch.cat(predictions_by_task[task], dim=0)
            result = classification_metrics(
                targets=targets,
                predictions=predictions,
                num_classes=TASK_CLASS_COUNTS[task],
            )
            metrics[f"{task}_accuracy"] = result.accuracy
            metrics[f"{task}_macro_f1"] = result.macro_f1
            if TASK_CLASS_COUNTS[task] == 2:
                metrics[f"{task}_positive_recall"] = result.recall_by_class[1]
                metrics[f"{task}_positive_f1"] = result.f1_by_class[1]
    return metrics


def selection_value(
    validation_metrics: dict[str, float],
    training_config: dict[str, Any],
) -> tuple[float, bool]:
    metric_name = str(training_config.get("selection_metric", "val_loss"))
    if metric_name == "val_loss":
        return validation_metrics["total_loss"], False
    if metric_name != "weighted_macro_f1":
        raise ValueError(
            "training.selection_metric must be 'val_loss' or 'weighted_macro_f1'"
        )

    weights = training_config.get("selection_metric_weights", {})
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            "training.selection_metric_weights must be a non-empty mapping"
        )
    unknown_tasks = set(weights) - set(TASK_NAMES)
    if unknown_tasks:
        raise ValueError(f"unknown selection metric tasks: {sorted(unknown_tasks)}")

    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0:
        raise ValueError("selection metric weights must sum to a positive value")
    weighted_score = sum(
        float(weight) * validation_metrics[f"{task}_macro_f1"]
        for task, weight in weights.items()
    )
    return weighted_score / total_weight, True


def load_initial_checkpoint(
    model: MultimodalWatermelonModel,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("initial checkpoint is missing model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def save_checkpoint(
    path: Path,
    model: MultimodalWatermelonModel,
    optimizer: AdamW,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [f"{prefix}_{key}={value:.6f}" for key, value in sorted(metrics.items())]
    return " ".join(parts)


def build_model(config: dict[str, Any]) -> MultimodalWatermelonModel:
    model_config = config["model"]
    task_config = model_config["output_tasks"]
    missing_tasks = set(TASK_NAMES) - set(task_config)
    extra_tasks = set(task_config) - set(TASK_NAMES)
    if missing_tasks or extra_tasks:
        raise ValueError(
            f"model.output_tasks mismatch; missing={sorted(missing_tasks)}, "
            f"extra={sorted(extra_tasks)}"
        )

    environment_fields = config["data"]["environment_fields"]
    sensor_dim = int(model_config["sensor_dim"])
    if len(environment_fields) != sensor_dim:
        raise ValueError(
            "data.environment_fields length must equal model.sensor_dim: "
            f"{len(environment_fields)} != {sensor_dim}"
        )

    task_dims = TaskOutputDims(
        **{task: int(task_config[task]) for task in TASK_NAMES}
    )
    return MultimodalWatermelonModel(
        sensor_dim=sensor_dim,
        time_dim=int(model_config["time_feature_dim"]),
        task_dims=task_dims,
        image_pretrained=bool(model_config["image_pretrained"]),
        input_mode=str(model_config.get("input_mode", "multimodal")),
        use_sensor_mask=bool(model_config.get("use_sensor_mask", False)),
    )


def resolve_environment_normalization(
    config: dict[str, Any],
    split: str,
) -> dict[str, tuple[float, float]]:
    data_config = config["data"]
    configured = data_config.get("environment_normalization", {})
    if configured != "auto":
        if not isinstance(configured, dict):
            raise ValueError("data.environment_normalization must be a mapping or 'auto'")
        return {
            field: (float(values[0]), float(values[1]))
            for field, values in configured.items()
        }

    raw_dataset = WatermelonDataset(
        dataset_dir=data_config["dataset_dir"],
        split=split,
        split_file=data_config.get("split_file"),
        labels_file=data_config.get("labels_file"),
        window_hours=int(data_config["window_hours"]),
        image_size=tuple(int(value) for value in data_config["image_size"]),
        environment_fields=tuple(data_config["environment_fields"]),
        optional_environment_fields=tuple(
            data_config.get("optional_environment_fields", ())
        ),
        strict_time_alignment=bool(data_config.get("strict_time_alignment", True)),
        require_device_id=bool(data_config.get("require_device_id", False)),
        require_task_masks=bool(data_config.get("require_task_masks", False)),
    )
    calculated = raw_dataset.calculate_environment_normalization()
    data_config["environment_normalization"] = calculated
    return {
        field: (float(values[0]), float(values[1]))
        for field, values in calculated.items()
    }


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    data_config = config["data"]
    training_config = config["training"]
    loss_config = config["loss"]
    output_config = config["outputs"]

    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    window_hours = int(data_config["window_hours"])
    image_size = tuple(int(value) for value in data_config["image_size"])
    environment_fields = tuple(data_config["environment_fields"])
    optional_environment_fields = tuple(
        data_config.get("optional_environment_fields", ())
    )

    if len(image_size) != 2:
        raise ValueError("data.image_size must contain exactly two values")
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("training.batch_size must be at least 1")
    if window_hours < 1:
        raise ValueError("data.window_hours must be at least 1")

    set_seed(int(config["project"]["seed"]))
    device = resolve_device(str(training_config["device"]))
    pin_memory = device.type == "cuda"
    split_config = data_config["splits"]
    environment_normalization = resolve_environment_normalization(
        config,
        split=str(split_config["train"]),
    )
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / "effective_config.yaml")

    train_loader = create_loader(
        dataset_dir=data_config["dataset_dir"],
        split=split_config["train"],
        split_file=data_config.get("split_file"),
        labels_file=data_config.get("labels_file"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        window_hours=window_hours,
        image_size=image_size,
        environment_fields=environment_fields,
        environment_normalization=environment_normalization,
        optional_environment_fields=optional_environment_fields,
        augment_image=bool(data_config.get("train_augmentation", False)),
        preserve_aspect_ratio=bool(data_config.get("preserve_aspect_ratio", False)),
        strict_time_alignment=bool(data_config.get("strict_time_alignment", True)),
        require_device_id=bool(data_config.get("require_device_id", False)),
        require_task_masks=bool(data_config.get("require_task_masks", False)),
    )
    val_loader = create_loader(
        dataset_dir=data_config["dataset_dir"],
        split=split_config["validation"],
        split_file=data_config.get("split_file"),
        labels_file=data_config.get("labels_file"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        window_hours=window_hours,
        image_size=image_size,
        environment_fields=environment_fields,
        environment_normalization=environment_normalization,
        optional_environment_fields=optional_environment_fields,
        augment_image=False,
        preserve_aspect_ratio=bool(data_config.get("preserve_aspect_ratio", False)),
        strict_time_alignment=bool(data_config.get("strict_time_alignment", True)),
        require_device_id=bool(data_config.get("require_device_id", False)),
        require_task_masks=bool(data_config.get("require_task_masks", False)),
    )

    model = build_model(config).to(device)
    initial_checkpoint_path = training_config.get("initial_checkpoint")
    initial_checkpoint = load_initial_checkpoint(
        model=model,
        checkpoint_path=initial_checkpoint_path,
        device=device,
    )
    criterion = MultiTaskLoss(
        task_weights=loss_config["task_weights"],
        class_weights=loss_config.get("class_weights"),
        label_smoothing=float(loss_config["label_smoothing"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_name = str(training_config.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(training_config.get("min_learning_rate", 1e-6)),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError("training.scheduler must be 'none' or 'cosine'")
    grad_clip_value = training_config.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_value is None else float(grad_clip_value)
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("training.grad_clip_norm must be positive or null")
    patience_value = training_config.get("early_stopping_patience")
    early_stopping_patience = (
        None if patience_value is None else int(patience_value)
    )
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be positive or null")

    print(f"config={args.config or 'built-in defaults'}")
    print(f"device={device}")
    print(f"dataset_dir={data_config['dataset_dir']}")
    print(f"labels_file={data_config.get('labels_file') or 'labels.csv'}")
    print(f"split_file={data_config.get('split_file') or 'split.csv'}")
    print(f"task_masks_enabled={getattr(train_loader.dataset, 'has_task_masks', False)}")
    print(f"train_samples={len(train_loader.dataset)}")
    print(f"val_samples={len(val_loader.dataset)}")
    print(f"epochs={epochs}")
    print(f"batch_size={batch_size}")
    print(f"pin_memory={pin_memory}")
    print(f"persistent_workers={num_workers > 0}")
    print(f"input_mode={config['model'].get('input_mode', 'multimodal')}")
    print(f"image_size={image_size}")
    print(f"window_hours={window_hours}")
    print(f"environment_normalization={environment_normalization}")
    print(f"use_sensor_mask={config['model'].get('use_sensor_mask', False)}")
    print(f"scheduler={scheduler_name}")
    print(f"initial_checkpoint={initial_checkpoint_path or 'none'}")
    if initial_checkpoint is not None:
        print(f"initial_checkpoint_epoch={initial_checkpoint.get('epoch')}")
    print(
        f"selection_metric={training_config.get('selection_metric', 'val_loss')}"
    )
    print(f"class_weights={loss_config.get('class_weights', {})}")

    selection_metric_name = str(training_config.get("selection_metric", "val_loss"))
    maximize_selection = selection_metric_name == "weighted_macro_f1"
    best_selection_value = float("-inf") if maximize_selection else float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = Path(output_config["best_checkpoint"])
    last_path = Path(output_config["last_checkpoint"])
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            collect_classification_metrics=True,
        )

        current_selection_value, maximize_selection = selection_value(
            val_metrics,
            training_config,
        )
        val_metrics["selection_value"] = current_selection_value

        print(f"epoch={epoch}")
        print(format_metrics("train", train_metrics))
        print(format_metrics("val", val_metrics))

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train": train_metrics, "val": val_metrics},
            args=args,
            config=config,
        )

        current_val_loss = val_metrics["total_loss"]
        is_improved = (
            current_selection_value > best_selection_value
            if maximize_selection
            else current_selection_value < best_selection_value
        )
        if is_improved:
            best_selection_value = current_selection_value
            best_val_loss = current_val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                args=args,
                config=config,
            )
            print(f"saved_best={best_path}")
        else:
            stale_epochs += 1

        print(f"saved_last={last_path}")
        if scheduler is not None:
            scheduler.step()
        if (
            early_stopping_patience is not None
            and stale_epochs >= early_stopping_patience
        ):
            print(f"early_stopping_epoch={epoch}")
            break

    summary = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric_name,
        "best_selection_value": best_selection_value,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "effective_config": str(checkpoint_dir / "effective_config.yaml"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
