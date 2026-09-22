from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import WatermelonDataset, watermelon_collate_fn
from src.models import MultimodalWatermelonModel
from src.training import MultiTaskLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check multi-task loss calculation and backward propagation."
    )
    parser.add_argument("--dataset-dir", default="dataset_sample")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = WatermelonDataset(dataset_dir=args.dataset_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=watermelon_collate_fn,
    )
    batch = next(iter(loader))

    model = MultimodalWatermelonModel().to(device)
    criterion = MultiTaskLoss()
    model.train()

    outputs = model(
        images=batch["images"].to(device),
        environment=batch["environment"].to(device),
        time_offsets=batch["time_offsets"].to(device),
        environment_mask=batch["environment_mask"].to(device),
    )
    labels = {
        name: values.to(device) for name, values in batch["labels"].items()
    }
    total_loss, details = criterion(outputs, labels, return_details=True)
    total_loss.backward()

    gradient_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_count = sum(
        parameter.requires_grad for parameter in model.parameters()
    )

    print(f"device={device}")
    print(f"batch_size={batch['images'].shape[0]}")
    print(f"total_loss={total_loss.item():.6f}")
    for task in ("growth_stage", "health_level", "maturity_level", "abnormal_alert"):
        print(f"{task}_loss={details[f'{task}_loss'].item():.6f}")
    print(f"parameters_with_gradients={gradient_count}/{trainable_count}")


if __name__ == "__main__":
    main()
