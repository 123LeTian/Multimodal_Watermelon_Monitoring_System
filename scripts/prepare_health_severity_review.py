from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare numbered contact sheets for health-severity review."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", default="dataset_fused/fusion_manifest.csv")
    parser.add_argument("--dataset-dir", default="dataset_fused")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    predictions = read_csv(Path(args.predictions))
    manifest_rows = read_csv(Path(args.manifest))
    manifest = {row["image_id"]: row for row in manifest_rows}

    candidates = []
    for prediction in predictions:
        if not (
            int(prediction["health_level_valid"]) == 1
            and int(prediction["health_level_true"]) == 2
            and int(prediction["health_level_pred"]) == 1
        ):
            continue
        image_id = prediction["image_id"]
        source = manifest.get(image_id)
        if source is None:
            raise KeyError(f"missing manifest row for {image_id}")
        candidates.append(
            {
                "image_id": image_id,
                "image_path": source["image_path"],
                "source_dataset": source["source_dataset"],
                "source_class": source["source_class"],
                "source_path": source["source_path"],
                "old_health_level": prediction["health_level_true"],
                "model_health_level": prediction["health_level_pred"],
                "review_health_level": "",
                "review_health_level_valid": "",
                "review_reason": "",
            }
        )

    candidates.sort(key=lambda row: (row["source_class"], row["image_id"]))
    for index, row in enumerate(candidates, start=1):
        row["review_index"] = index

    output_dir = Path(args.output_dir)
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "health_review_queue.csv"
    fieldnames = [
        "review_index",
        "image_id",
        "image_path",
        "source_dataset",
        "source_class",
        "source_path",
        "old_health_level",
        "model_health_level",
        "review_health_level",
        "review_health_level_valid",
        "review_reason",
    ]
    with queue_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    tile_width = 320
    tile_height = 280
    image_box = (300, 225)
    items_per_sheet = args.rows * args.columns
    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    dataset_dir = Path(args.dataset_dir)

    for sheet_start in range(0, len(candidates), items_per_sheet):
        sheet_rows = candidates[sheet_start : sheet_start + items_per_sheet]
        sheet = Image.new(
            "RGB",
            (args.columns * tile_width, args.rows * tile_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for local_index, row in enumerate(sheet_rows):
            column = local_index % args.columns
            grid_row = local_index // args.columns
            left = column * tile_width
            top = grid_row * tile_height
            image_path = dataset_dir / row["image_path"]
            with Image.open(image_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.thumbnail(image_box, Image.Resampling.LANCZOS)
                image_left = left + (tile_width - image.width) // 2
                image_top = top + 4
                sheet.paste(image, (image_left, image_top))
            draw.rectangle(
                (left, top, left + tile_width - 1, top + tile_height - 1),
                outline=(80, 80, 80),
                width=1,
            )
            text_top = top + 232
            draw.text(
                (left + 8, text_top),
                f"#{row['review_index']}  {row['image_id']}",
                fill="black",
                font=font,
            )
            draw.text(
                (left + 8, text_top + 23),
                row["source_class"],
                fill=(70, 70, 70),
                font=small_font,
            )

        first_index = sheet_rows[0]["review_index"]
        last_index = sheet_rows[-1]["review_index"]
        sheet.save(
            sheets_dir / f"review_{first_index:04d}_{last_index:04d}.jpg",
            quality=92,
        )

    print(f"candidates={len(candidates)}")
    print(f"queue={queue_path}")
    print(f"sheets={len(list(sheets_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
