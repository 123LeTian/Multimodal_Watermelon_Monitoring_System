from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
HEALTH_NAMES = {
    0: "正常",
    1: "轻微异常",
    2: "中度异常",
    3: "严重异常",
}
QUOTAS = (
    (0, "val", "Healthy", 5),
    (1, "val", "Iron_Deficiency_Chlorosis", 5),
    (2, "val", "Anthracnose", 2),
    (2, "val", "Down_Mildew", 1),
    (2, "val", "Spider_mite_infestation", 2),
    (3, "train", "Mosaic_Virus", 3),
    (3, "train", "Downy_Mildew", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 20 balanced V3 inference examples.")
    parser.add_argument(
        "--output-dir",
        default="experiments/inference_examples_v3",
    )
    parser.add_argument("--alert-threshold", type=float, default=0.84)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_quality(path: Path) -> dict[str, float | int]:
    with Image.open(path) as source:
        width, height = source.size
        image = source.convert("RGB")
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    gray = array.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float(dx.var() + dy.var())
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    score = (
        math.log1p(sharpness)
        + 0.35 * math.log1p(contrast)
        + 0.15 * math.log1p(min(width, height))
        + 0.75 * exposure
    )
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "sharpness": round(sharpness, 3),
        "quality_score": round(score, 6),
    }


def prediction_rows() -> dict[str, dict[str, str]]:
    paths = (
        ROOT / "outputs/reviewed_v3/train_calibrated/predictions.csv",
        ROOT / "outputs/reviewed_v3/val_argmax/predictions.csv",
        ROOT / "outputs/reviewed_v3/test_calibrated/predictions.csv",
    )
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_rows(path):
            result[row["image_id"]] = row
    return result


def build_candidates(alert_threshold: float) -> list[dict[str, object]]:
    labels = {row["image_id"]: row for row in read_rows(ROOT / "dataset_fused/labels_reviewed_v3.csv")}
    splits = {row["image_id"]: row["split"] for row in read_rows(ROOT / "dataset_fused/split_source_isolated_v3.csv")}
    predictions = prediction_rows()
    candidates: list[dict[str, object]] = []

    for manifest in read_rows(ROOT / "dataset_fused/fusion_manifest.csv"):
        image_id = manifest["image_id"]
        split = splits.get(image_id)
        label = labels.get(image_id)
        prediction = predictions.get(image_id)
        if split not in {"train", "val", "test"} or label is None or prediction is None:
            continue
        if label["health_level_valid"] != "1" or label["abnormal_alert_valid"] != "1":
            continue
        health_true = int(label["health_level"])
        health_pred = int(prediction["health_level_pred"])
        alert_true = int(label["abnormal_alert"])
        alert_probability = float(prediction["abnormal_alert_probability"])
        alert_pred = int(alert_probability >= alert_threshold)
        if health_pred != health_true or alert_pred != alert_true:
            continue
        maturity_valid = int(label["maturity_level_valid"])
        maturity_true = int(label["maturity_level"])
        maturity_pred = int(prediction["maturity_level_pred"])
        if maturity_valid and maturity_pred != maturity_true:
            continue

        image_path = ROOT / "dataset_fused" / label["image_path"]
        quality = image_quality(image_path)
        if min(int(quality["width"]), int(quality["height"])) < 224:
            continue
        candidates.append(
            {
                "image_id": image_id,
                "split": split,
                "source_dataset": manifest["source_dataset"],
                "source_class": manifest["source_class"],
                "capture_time": label["capture_time"],
                "health_level": health_true,
                "health_name": HEALTH_NAMES[health_true],
                "health_pred": health_pred,
                "maturity_level": maturity_true,
                "maturity_valid": maturity_valid,
                "maturity_pred": maturity_pred,
                "abnormal_alert": alert_true,
                "abnormal_alert_pred": alert_pred,
                "abnormal_alert_probability": round(alert_probability, 8),
                "source_image_path": str(image_path.relative_to(ROOT)),
                **quality,
            }
        )
    return candidates


def select_examples(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        key = (int(row["health_level"]), str(row["split"]), str(row["source_class"]))
        grouped[key].append(row)

    selected: list[dict[str, object]] = []
    for health, split, source_class, count in QUOTAS:
        key = (health, split, source_class)
        ranked = sorted(
            grouped[key],
            key=lambda row: (
                float(row["quality_score"]),
                abs(float(row["abnormal_alert_probability"]) - 0.84),
            ),
            reverse=True,
        )
        if len(ranked) < count:
            raise ValueError(f"not enough candidates for {key}: {len(ranked)} < {count}")
        selected.extend(ranked[:count])

    selected.sort(key=lambda row: (int(row["health_level"]), str(row["source_class"]), str(row["image_id"])))
    for index, row in enumerate(selected, start=1):
        row["example_no"] = index
    return selected


def copy_inputs(selected: list[dict[str, object]], output_dir: Path) -> None:
    images_dir = output_dir / "images"
    environment_dir = output_dir / "environment"
    images_dir.mkdir(parents=True, exist_ok=True)
    environment_dir.mkdir(parents=True, exist_ok=True)
    environment_rows = read_rows(ROOT / "dataset_fused/environment.csv")
    parsed_environment = [
        (datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT), row)
        for row in environment_rows
    ]

    for row in selected:
        image_id = str(row["image_id"])
        source_path = ROOT / str(row["source_image_path"])
        image_name = f"{image_id}{source_path.suffix.lower()}"
        shutil.copy2(source_path, images_dir / image_name)
        capture_time = datetime.strptime(str(row["capture_time"]), TIMESTAMP_FORMAT)
        window_start = capture_time - timedelta(hours=24)
        window = [
            environment_row
            for timestamp, environment_row in parsed_environment
            if window_start <= timestamp <= capture_time
        ]
        if len(window) != 25:
            raise ValueError(f"{image_id} expected 25 environment rows, found {len(window)}")
        write_rows(environment_dir / f"{image_id}.csv", window)
        row["image_path"] = f"images/{image_name}"
        row["environment_path"] = f"environment/{image_id}.csv"
        row["output_path"] = f"outputs/{image_id}.json"


def make_contact_sheet(selected: list[dict[str, object]], output_path: Path) -> None:
    columns = 5
    rows = 4
    cell_width = 300
    cell_height = 260
    image_height = 210
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image_path = output_path.parent / str(row["image_path"])
        with Image.open(image_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (cell_width - 12, image_height - 12))
        left = x + (cell_width - image.width) // 2
        top = y + 6 + (image_height - 12 - image.height) // 2
        sheet.paste(image, (left, top))
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="gray")
        text = (
            f"#{int(row['example_no']):02d} {row['image_id']} H{row['health_level']} "
            f"{row['source_class']}\n{row['split']} {row['width']}x{row['height']} "
            f"Q={float(row['quality_score']):.2f}"
        )
        draw.multiline_text((x + 6, y + image_height + 2), text, fill="black", font=font, spacing=2)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args.alert_threshold)
    ranking = sorted(
        candidates,
        key=lambda row: (int(row["health_level"]), -float(row["quality_score"])),
    )
    write_rows(output_dir / "candidate_ranking.csv", ranking)
    selected = select_examples(candidates)
    copy_inputs(selected, output_dir)
    write_rows(output_dir / "selection_manifest.csv", selected)
    make_contact_sheet(selected, output_dir / "contact_sheet.jpg")
    print(f"eligible_candidates={len(candidates)}")
    print(f"selected_examples={len(selected)}")
    for health in range(4):
        print(f"health_{health}={sum(int(row['health_level']) == health for row in selected)}")
    print(f"saved_manifest={output_dir / 'selection_manifest.csv'}")
    print(f"saved_contact_sheet={output_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
