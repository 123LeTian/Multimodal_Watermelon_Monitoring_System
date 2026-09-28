from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.datasets import WatermelonDataset, watermelon_collate_fn
from src.models import EnvironmentTransformerEncoder
from src.training import MultiTaskLoss, find_binary_threshold


LABEL_HEADER = [
    "image_id",
    "image_path",
    "capture_time",
    "device_id",
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
    "growth_stage_valid",
    "health_level_valid",
    "maturity_level_valid",
    "abnormal_alert_valid",
]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


class EnhancedPipelineTests(unittest.TestCase):
    def make_dataset(
        self,
        environment_header: list[str] | None = None,
        environment_rows: list[list[object]] | None = None,
    ) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "images").mkdir()
        Image.new("RGB", (32, 16), color=(30, 120, 40)).save(root / "images" / "one.png")
        write_csv(
            root / "labels.csv",
            LABEL_HEADER,
            [[
                "IMG_1",
                "images/one.png",
                "2026-09-21 12:00:00",
                "sensor-a",
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
            ]],
        )
        write_csv(root / "split.csv", ["image_id", "split"], [["IMG_1", "train"]])
        write_csv(
            root / "environment.csv",
            environment_header
            or ["timestamp", "device_id", "temperature", "soil_humidity", "light", "ph"],
            environment_rows
            or [[
                "2026-09-21 11:00:00",
                "sensor-a",
                25.0,
                65.0,
                "",
                "",
            ]],
        )
        return temporary, root

    def test_optional_sensors_and_weak_label_masks_are_preserved(self) -> None:
        temporary, root = self.make_dataset()
        with temporary:
            dataset = WatermelonDataset(
                root,
                split="train",
                optional_environment_fields=("light", "ph"),
                require_device_id=True,
                require_task_masks=True,
                preserve_aspect_ratio=True,
                environment_normalization={
                    "temperature": (20.0, 5.0),
                    "soil_humidity": (50.0, 5.0),
                },
            )
            item = dataset[0]
            self.assertEqual(item["environment"].tolist(), [[1.0, 3.0, 0.0, 0.0]])
            self.assertEqual(
                item["environment_sensor_mask"].tolist(),
                [[1.0, 1.0, 0.0, 0.0]],
            )
            self.assertEqual(item["task_masks"]["growth_stage"], 0.0)
            batch = watermelon_collate_fn([item])
            self.assertEqual(batch["images"].shape, (1, 3, 224, 224))
            self.assertEqual(batch["environment_sensor_mask"].shape, (1, 1, 4))

    def test_air_humidity_is_not_accepted_as_soil_humidity(self) -> None:
        temporary, root = self.make_dataset(
            environment_header=[
                "timestamp",
                "device_id",
                "temperature",
                "air_humidity",
                "light",
                "ph",
            ],
            environment_rows=[[
                "2026-09-21 11:00:00",
                "sensor-a",
                25.0,
                80.0,
                1000.0,
                6.5,
            ]],
        )
        with temporary:
            with self.assertRaisesRegex(ValueError, "not interchangeable"):
                WatermelonDataset(root, split="train", require_device_id=True)

    def test_same_device_alignment_is_enforced(self) -> None:
        temporary, root = self.make_dataset(
            environment_rows=[[
                "2026-09-21 11:00:00",
                "sensor-b",
                25.0,
                65.0,
                1000.0,
                6.5,
            ]],
        )
        with temporary:
            with self.assertRaisesRegex(ValueError, "same-device"):
                WatermelonDataset(root, split="train", require_device_id=True)

    def test_environment_encoder_accepts_sensor_masks(self) -> None:
        encoder = EnvironmentTransformerEncoder(
            sensor_dim=4,
            model_dim=8,
            output_dim=6,
            num_heads=2,
            num_layers=1,
            use_sensor_mask=True,
        )
        output = encoder(
            environment=torch.zeros(2, 3, 4),
            environment_sensor_mask=torch.ones(2, 3, 4),
            time_offsets=torch.zeros(2, 3, 1),
            environment_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertEqual(output.shape, (2, 6))

    def test_multitask_loss_supports_per_class_weights(self) -> None:
        outputs = {
            "growth_stage_logits": torch.zeros(2, 6),
            "health_level_logits": torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 2),
            "maturity_level_logits": torch.zeros(2, 4),
            "abnormal_alert_logits": torch.zeros(2, 2),
        }
        labels = {
            "growth_stage": torch.zeros(2, dtype=torch.long),
            "health_level": torch.tensor([0, 1]),
            "maturity_level": torch.zeros(2, dtype=torch.long),
            "abnormal_alert": torch.zeros(2, dtype=torch.long),
        }
        masks = {
            "growth_stage": torch.zeros(2),
            "health_level": torch.ones(2),
            "maturity_level": torch.zeros(2),
            "abnormal_alert": torch.zeros(2),
        }
        unweighted = MultiTaskLoss()(outputs, labels, masks)
        weighted = MultiTaskLoss(
            class_weights={"health_level": [1.0, 3.0, 1.0, 1.0]}
        )(outputs, labels, masks)
        self.assertGreater(float(weighted), float(unweighted))

    def test_threshold_calibration_honors_minimum_recall(self) -> None:
        result = find_binary_threshold(
            targets=torch.tensor([0, 0, 1, 1]),
            positive_probabilities=torch.tensor([0.1, 0.4, 0.45, 0.9]),
            minimum_recall=1.0,
            thresholds=[0.4, 0.5],
        )
        self.assertEqual(result.threshold, 0.4)
        self.assertEqual(result.positive_recall, 1.0)
        self.assertTrue(result.recall_requirement_met)


if __name__ == "__main__":
    unittest.main()
