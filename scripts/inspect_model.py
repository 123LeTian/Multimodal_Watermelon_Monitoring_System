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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect model forward output shapes.")
    parser.add_argument("--dataset-dir", default="dataset_sample")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for the forward pass.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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
    model.eval()

    with torch.no_grad():
        outputs = model(
            images=batch["images"].to(device),
            environment=batch["environment"].to(device),
            time_offsets=batch["time_offsets"].to(device),
            environment_mask=batch["environment_mask"].to(device),
        )

    print(f"device={device}")
    print(f"samples={len(dataset)}")
    print(f"image_ids={batch['image_ids']}")
    print(f"parameters={count_parameters(model)}")
    print(f"images_shape={tuple(batch['images'].shape)}")
    print(f"environment_shape={tuple(batch['environment'].shape)}")

    for name, tensor in outputs.items():
        print(f"{name}={tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
