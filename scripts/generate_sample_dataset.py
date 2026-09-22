from __future__ import annotations

import csv
import math
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_sample"
IMAGES_DIR = DATASET_DIR / "images"
RANDOM_SEED = 20260921


def reset_output_dir() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def draw_sample_image(path: Path, index: int, growth_stage: int, health_level: int) -> None:
    width, height = 256, 256
    rng = random.Random(RANDOM_SEED + index)

    background = (232, 244, 221)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    # Soil band.
    draw.rectangle((0, 205, width, height), fill=(126, 86, 50))

    # Vines.
    vine_color = (45, 126, 52) if health_level <= 1 else (121, 124, 44)
    for offset in range(0, 5):
        y = 150 + offset * 6
        points = []
        for x in range(15, 245, 8):
            points.append((x, y + int(math.sin((x + index * 17) / 22) * 12)))
        draw.line(points, fill=vine_color, width=3)

    # Leaves.
    leaf_count = 5 + growth_stage * 2
    for _ in range(leaf_count):
        cx = rng.randint(25, 230)
        cy = rng.randint(45, 175)
        radius = rng.randint(10, 22)
        color = (38, rng.randint(110, 175), 52)
        if health_level >= 2 and rng.random() < 0.45:
            color = (170, 154, 52)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        draw.line((cx, cy, cx + rng.randint(-18, 18), cy + rng.randint(12, 25)), fill=(30, 90, 35), width=2)

    # Flowers for flowering and fruit-setting stages.
    if growth_stage in {2, 3}:
        for _ in range(4):
            cx = rng.randint(45, 210)
            cy = rng.randint(65, 155)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(247, 217, 68))

    # Fruit for fruit-setting, enlargement, and maturity stages.
    if growth_stage >= 3:
        fruit_radius = 18 + max(0, growth_stage - 3) * 12
        cx, cy = 132, 165
        fruit_color = (58, 154, 72) if health_level < 3 else (102, 122, 55)
        draw.ellipse(
            (cx - fruit_radius, cy - fruit_radius, cx + fruit_radius, cy + fruit_radius),
            fill=fruit_color,
            outline=(24, 95, 35),
            width=3,
        )
        for stripe in range(-fruit_radius + 6, fruit_radius, 10):
            draw.arc(
                (cx + stripe - 14, cy - fruit_radius, cx + stripe + 14, cy + fruit_radius),
                90,
                270,
                fill=(28, 111, 44),
                width=2,
            )

    # Visible abnormal marks for unhealthy samples.
    if health_level > 0:
        for _ in range(health_level * 3):
            cx = rng.randint(40, 220)
            cy = rng.randint(55, 180)
            r = rng.randint(3, 7)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(132, 75, 38))

    image.save(path, format="PNG")


def write_environment_csv() -> None:
    start = datetime(2026, 9, 20, 8, 0, 0)
    end = datetime(2026, 9, 21, 18, 30, 0)
    rows = []
    current = start
    step = timedelta(minutes=10)
    rng = random.Random(RANDOM_SEED)

    while current <= end:
        minutes_from_start = int((current - start).total_seconds() // 60)
        day_phase = (current.hour * 60 + current.minute) / 1440
        temperature = 26 + 5 * math.sin(2 * math.pi * (day_phase - 0.2)) + rng.uniform(-0.4, 0.4)
        soil_humidity = 68 - 9 * math.sin(2 * math.pi * (day_phase - 0.2)) + rng.uniform(-1.0, 1.0)
        daylight = max(0, math.sin(math.pi * (current.hour + current.minute / 60 - 6) / 12))
        light = daylight * 36000 + rng.uniform(0, 900)
        ph = 6.7 + 0.08 * math.sin(minutes_from_start / 360) + rng.uniform(-0.03, 0.03)

        rows.append(
            {
                "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": f"{temperature:.2f}",
                "soil_humidity": f"{soil_humidity:.2f}",
                "light": f"{max(light, 0):.2f}",
                "ph": f"{ph:.2f}",
            }
        )
        current += step

    with (DATASET_DIR / "environment.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "temperature", "soil_humidity", "light", "ph"])
        writer.writeheader()
        writer.writerows(rows)


def write_labels_and_images() -> list[dict[str, str]]:
    rows = []
    base_time = datetime(2026, 9, 21, 8, 30, 0)

    for index in range(1, 21):
        image_id = f"IMG_{index:06d}"
        growth_stage = (index - 1) % 6
        health_level = [0, 0, 1, 0, 2, 1, 0, 3, 0, 1][(index - 1) % 10]
        maturity_level = 0 if growth_stage < 4 else min(3, growth_stage - 3)
        abnormal_alert = 1 if health_level > 0 else 0
        capture_time = base_time + timedelta(minutes=30 * (index - 1))

        image_path = f"images/{image_id}.png"
        draw_sample_image(IMAGES_DIR / f"{image_id}.png", index, growth_stage, health_level)

        rows.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "capture_time": capture_time.strftime("%Y-%m-%d %H:%M:%S"),
                "growth_stage": str(growth_stage),
                "health_level": str(health_level),
                "maturity_level": str(maturity_level),
                "abnormal_alert": str(abnormal_alert),
            }
        )

    with (DATASET_DIR / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "image_id",
            "image_path",
            "capture_time",
            "growth_stage",
            "health_level",
            "maturity_level",
            "abnormal_alert",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_split_csv(label_rows: list[dict[str, str]]) -> None:
    split_rows = []
    for index, row in enumerate(label_rows, start=1):
        if index <= 14:
            split = "train"
        elif index <= 17:
            split = "val"
        else:
            split = "test"
        split_rows.append({"image_id": row["image_id"], "split": split})

    with (DATASET_DIR / "split.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows(split_rows)


def write_readme(label_rows: list[dict[str, str]]) -> None:
    readme = """# 模拟数据集说明

本文件夹是按照 `数据交接规范表.md` 生成的模拟数据集，仅用于测试数据读取、模型前向传播和最小训练流程。

注意：本数据不是大棚真实采集数据，不能用于论文结论、模型效果汇报或项目最终实验结果。

1. 数据采集地点：模拟生成
2. 数据采集时间范围：2026-09-20 08:00:00 至 2026-09-21 18:30:00
3. 图片采集设备：模拟生成图片
4. 环境传感器型号：模拟空气温度、土壤湿度、光照、pH 数据
5. 环境数据采样频率：每 10 分钟 1 条
6. 数据标注人员：模拟标签
7. 标签审核人员：暂缺
8. 数据集总图片数：20
9. 环境数据总记录数：208
10. 训练集、验证集、测试集数量：训练集 14，验证集 3，测试集 3
11. 已知问题：图片和标签均为模拟生成，只能用于流程测试
"""
    (DATASET_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    reset_output_dir()
    write_environment_csv()
    label_rows = write_labels_and_images()
    write_split_csv(label_rows)
    write_readme(label_rows)
    print(f"Created sample dataset at: {DATASET_DIR}")
    print("Images: 20")
    print("Environment rows: 208")
    print("Split: train=14, val=3, test=3")


if __name__ == "__main__":
    main()
