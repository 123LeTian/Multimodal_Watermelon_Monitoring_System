from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import WatermelonDataset, watermelon_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one watermelon dataset batch.")
    parser.add_argument(
        "--dataset-dir",
        default="dataset_sample",
        help="Path to dataset folder containing images, labels.csv, environment.csv, and split.csv.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset = WatermelonDataset(dataset_dir=dataset_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=watermelon_collate_fn,
    )

    batch = next(iter(loader))
    print(f"dataset_dir={dataset_dir.resolve()}")
    print(f"split={args.split}")
    print(f"samples={len(dataset)}")
    print(f"image_ids={batch['image_ids']}")
    print(f"images_shape={tuple(batch['images'].shape)}")
    print(f"environment_shape={tuple(batch['environment'].shape)}")
    print(f"time_offsets_shape={tuple(batch['time_offsets'].shape)}")
    print(f"environment_mask_shape={tuple(batch['environment_mask'].shape)}")
    print(f"environment_lengths={batch['environment_lengths'].tolist()}")

    for name, values in batch["labels"].items():
        print(f"{name}={values.tolist()}")


if __name__ == "__main__":
    main()
