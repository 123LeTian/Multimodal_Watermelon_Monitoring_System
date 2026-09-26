from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import WatermelonDataset, watermelon_collate_fn
from src.models import MultimodalWatermelonModel, TaskOutputDims
from src.training import MultiTaskLoss, TASK_NAMES, classification_metrics
from src.utils.config import deep_merge, load_config


TASK_CLASS_COUNTS = {
    "growth_stage": TaskOutputDims.growth_stage,
    "health_level": TaskOutputDims.health_level,
    "maturity_level": TaskOutputDims.maturity_level,
    "abnormal_alert": TaskOutputDims.abnormal_alert,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained watermelon model.")
    parser.add_argument("--dataset-dir", default="dataset_sample")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional split CSV path. Defaults to split.csv inside the dataset directory or checkpoint config.",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional labels CSV path. Defaults to checkpoint config or labels.csv inside the dataset directory.",
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
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


def move_labels_to_device(
    labels: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: values.to(device) for name, values in labels.items()}


def move_task_masks_to_device(
    task_masks: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if task_masks is None:
        return None
    return {name: values.to(device) for name, values in task_masks.items()}


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


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = load_config(None)
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, dict):
        config = deep_merge(config, saved_config)
    return config


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MultimodalWatermelonModel, dict[str, Any], dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing model_state_dict")

    config = checkpoint_config(checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config


def evaluate(
    model: MultimodalWatermelonModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    loss_totals = {f"{task}_loss": 0.0 for task in TASK_NAMES}
    loss_totals["total_loss"] = 0.0
    sample_count = 0

    targets_by_task = {task: [] for task in TASK_NAMES}
    predictions_by_task = {task: [] for task in TASK_NAMES}
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            environment = batch["environment"].to(device)
            time_offsets = batch["time_offsets"].to(device)
            environment_mask = batch["environment_mask"].to(device)
            labels = move_labels_to_device(batch["labels"], device)
            task_masks = move_task_masks_to_device(batch.get("task_masks"), device)

            outputs = model(
                images=images,
                environment=environment,
                time_offsets=time_offsets,
                environment_mask=environment_mask,
            )
            total_loss, details = criterion(
                outputs,
                labels,
                task_masks=task_masks,
                return_details=True,
            )

            batch_size = images.shape[0]
            sample_count += batch_size
            for key, value in details.items():
                if key.endswith("_valid_count"):
                    loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())
                else:
                    loss_totals[key] += float(value.detach().cpu()) * batch_size

            batch_predictions = {}
            for task in TASK_NAMES:
                predictions = outputs[f"{task}_logits"].argmax(dim=1)
                batch_predictions[task] = predictions
                if task_masks is None or task not in task_masks:
                    valid_mask = torch.ones_like(labels[task], dtype=torch.bool, device=device)
                else:
                    valid_mask = task_masks[task].view(-1) > 0
                targets_by_task[task].append(labels[task][valid_mask].detach().cpu())
                predictions_by_task[task].append(predictions[valid_mask].detach().cpu())

            for row_index, image_id in enumerate(batch["image_ids"]):
                row: dict[str, Any] = {
                    "image_id": image_id,
                    "capture_time": batch["capture_times"][row_index],
                }
                for task in TASK_NAMES:
                    row[f"{task}_true"] = int(labels[task][row_index].detach().cpu())
                    row[f"{task}_pred"] = int(batch_predictions[task][row_index].detach().cpu())
                    if task_masks is None or task not in task_masks:
                        row[f"{task}_valid"] = 1
                    else:
                        row[f"{task}_valid"] = int(task_masks[task][row_index].detach().cpu() > 0)
                prediction_rows.append(row)

    losses = {}
    for key, value in loss_totals.items():
        if key.endswith("_valid_count"):
            losses[key] = value
        else:
            losses[key] = value / max(sample_count, 1)

    metrics: dict[str, dict[str, Any]] = {}
    for task in TASK_NAMES:
        targets = torch.cat(targets_by_task[task], dim=0)
        predictions = torch.cat(predictions_by_task[task], dim=0)
        result = classification_metrics(
            targets=targets,
            predictions=predictions,
            num_classes=TASK_CLASS_COUNTS[task],
        )
        metrics[task] = {
            "evaluated_samples": int(targets.numel()),
            "accuracy": result.accuracy,
            "macro_precision": result.macro_precision,
            "macro_recall": result.macro_recall,
            "macro_f1": result.macro_f1,
            "precision_by_class": result.precision_by_class,
            "recall_by_class": result.recall_by_class,
            "f1_by_class": result.f1_by_class,
            "support": result.support,
            "confusion_matrix": result.confusion_matrix,
        }
        if TASK_CLASS_COUNTS[task] == 2:
            metrics[task]["positive_precision"] = result.precision_by_class[1]
            metrics[task]["positive_recall"] = result.recall_by_class[1]
            metrics[task]["positive_f1"] = result.f1_by_class[1]

    return losses, metrics, prediction_rows


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrices(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "true_class", "predicted_class", "count"])
        for task, task_metrics in metrics.items():
            matrix = task_metrics["confusion_matrix"]
            for true_class, row in enumerate(matrix):
                for predicted_class, count in enumerate(row):
                    writer.writerow([task, true_class, predicted_class, count])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, config = load_model(Path(args.checkpoint), device)
    data_config = config["data"]
    loss_config = config["loss"]
    image_size = tuple(int(value) for value in data_config["image_size"])
    environment_fields = tuple(data_config["environment_fields"])

    split_file = args.split_file or data_config.get("split_file")
    labels_file = args.labels_file or data_config.get("labels_file")

    dataset = WatermelonDataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        split_file=split_file,
        labels_file=labels_file,
        window_hours=int(data_config["window_hours"]),
        image_size=image_size,
        environment_fields=environment_fields,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=watermelon_collate_fn,
    )
    criterion = MultiTaskLoss(
        task_weights=loss_config["task_weights"],
        label_smoothing=float(loss_config["label_smoothing"]),
    )

    losses, metrics, predictions = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
    )

    report = {
        "note": (
            "This report may be generated from simulated sample data. "
            "Do not use simulated-data metrics as final research conclusions."
        ),
        "dataset_dir": str(Path(args.dataset_dir)),
        "labels_file": str(Path(labels_file)) if labels_file else None,
        "split_file": str(Path(split_file)) if split_file else None,
        "split": args.split,
        "samples": len(dataset),
        "task_masks_enabled": bool(getattr(dataset, "has_task_masks", False)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_mode": config["model"].get("input_mode", "multimodal"),
        "device": str(device),
        "losses": losses,
        "metrics": metrics,
    }

    report_path = output_dir / "evaluation_report.json"
    predictions_path = output_dir / "predictions.csv"
    confusion_path = output_dir / "confusion_matrix.csv"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions(predictions_path, predictions)
    write_confusion_matrices(confusion_path, metrics)

    print(f"device={device}")
    print(f"split={args.split}")
    print(f"labels_file={labels_file or 'labels.csv'}")
    print(f"split_file={split_file or 'split.csv'}")
    print(f"task_masks_enabled={getattr(dataset, 'has_task_masks', False)}")
    print(f"samples={len(dataset)}")
    print(f"total_loss={losses['total_loss']:.6f}")
    for task in TASK_NAMES:
        task_metrics = metrics[task]
        print(
            f"{task}: "
            f"evaluated_samples={task_metrics['evaluated_samples']} "
            f"accuracy={task_metrics['accuracy']:.6f} "
            f"macro_f1={task_metrics['macro_f1']:.6f}"
        )
        if TASK_CLASS_COUNTS[task] == 2:
            print(
                f"{task}_positive: "
                f"precision={task_metrics['positive_precision']:.6f} "
                f"recall={task_metrics['positive_recall']:.6f} "
                f"f1={task_metrics['positive_f1']:.6f}"
            )
    print(f"saved_report={report_path}")
    print(f"saved_predictions={predictions_path}")
    print(f"saved_confusion_matrix={confusion_path}")


if __name__ == "__main__":
    main()
