from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_external" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "dataset_fused"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_FIELDS = (
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)

BASE_GROUP_QUOTAS = {
    "healthy_leaf": 1200,
    "mild_leaf_abnormal": 800,
    "disease_leaf": 2000,
    "unripe_fruit": 1800,
    "ripe_fruit": 1800,
    "fresh_quality": 800,
    "mild_quality": 800,
    "rotten_quality": 800,
}

SOURCES = {
    "watermelon_disease_recognition": {
        "path": "data_external/raw/watermelon_disease_recognition",
        "role": "Watermelon health and disease image labels.",
    },
    "kurdistan_watermelon_disease": {
        "path": "data_external/raw/kurdistan_watermelon_disease/augmented_extracted",
        "role": "Watermelon health, disease, pest, and nutrient deficiency image labels.",
    },
    "watermelon_ripe_semiripe_unripe": {
        "path": "data_external/raw/watermelon_ripe_semiripe_unripe",
        "role": "Watermelon ripeness image labels.",
    },
    "watermelon_ripe_unripe": {
        "path": "data_external/raw/watermelon_ripe_unripe",
        "role": "Watermelon ripe/unripe image labels.",
    },
    "fruq_db": {
        "path": "data_external/raw/fruq_db",
        "role": "Fruit quality labels, used in limited quantity for overripe/rotten examples.",
    },
    "watermelon_environment": {
        "path": "data_external/raw/watermelon_environment/watermelon_2023_south_Italy_ancillary_weather+LAI+GM+SWC.xlsx",
        "role": "Real watermelon field weather and soil water data.",
    },
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    source_dataset: str
    source_class: str
    fusion_group: str
    health_level: int
    maturity_level: int
    abnormal_alert: int
    growth_stage_choices: tuple[int, ...]
    label_note: str


@dataclass(frozen=True)
class EnvironmentRow:
    timestamp: datetime
    temperature: float
    soil_humidity: float
    light: float
    ph: float


def iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def add_candidates(
    candidates: dict[str, list[Candidate]],
    paths: list[Path],
    *,
    source_dataset: str,
    source_class: str,
    fusion_group: str,
    health_level: int,
    maturity_level: int,
    abnormal_alert: int,
    growth_stage_choices: tuple[int, ...],
    label_note: str,
) -> None:
    for path in paths:
        candidates[fusion_group].append(
            Candidate(
                path=path,
                source_dataset=source_dataset,
                source_class=source_class,
                fusion_group=fusion_group,
                health_level=health_level,
                maturity_level=maturity_level,
                abnormal_alert=abnormal_alert,
                growth_stage_choices=growth_stage_choices,
                label_note=label_note,
            )
        )


def collect_disease_recognition(candidates: dict[str, list[Candidate]]) -> None:
    root = RAW_DIR / "watermelon_disease_recognition" / "Watermelon Disease Recognition Dataset"
    class_roots = [
        root / "original_image" / "Watermelon",
        root / "augmented_image" / "Augmented_Image",
    ]
    class_map = {
        "Healthy": ("healthy_leaf", 0, 0, 0, (0, 1, 2, 3), "Healthy watermelon leaf image."),
        "Anthracnose": ("disease_leaf", 2, 0, 1, (2, 3), "Watermelon leaf disease mapped to moderate health abnormality."),
        "Downy_Mildew": ("disease_leaf", 2, 0, 1, (2, 3), "Watermelon leaf disease mapped to moderate health abnormality."),
        "Mosaic_Virus": ("disease_leaf", 2, 0, 1, (2, 3), "Watermelon leaf disease mapped to moderate health abnormality."),
    }
    for class_root in class_roots:
        for class_name, mapping in class_map.items():
            paths = iter_images(class_root / class_name)
            fusion_group, health, maturity, alert, growth_choices, note = mapping
            add_candidates(
                candidates,
                paths,
                source_dataset="watermelon_disease_recognition",
                source_class=class_name,
                fusion_group=fusion_group,
                health_level=health,
                maturity_level=maturity,
                abnormal_alert=alert,
                growth_stage_choices=growth_choices,
                label_note=note,
            )


def collect_kurdistan_disease(candidates: dict[str, list[Candidate]]) -> None:
    root = RAW_DIR / "kurdistan_watermelon_disease" / "augmented_extracted"
    class_map = {
        "Healthy": ("healthy_leaf", 0, 0, 0, (0, 1, 2, 3), "Healthy watermelon leaf image."),
        "Iron_Deficiency_Chlorosis": ("mild_leaf_abnormal", 1, 0, 1, (1, 2, 3), "Nutrient deficiency mapped to mild health abnormality."),
        "Anthracnose": ("disease_leaf", 2, 0, 1, (2, 3), "Watermelon disease mapped to moderate health abnormality."),
        "Down_Mildew": ("disease_leaf", 2, 0, 1, (2, 3), "Watermelon disease mapped to moderate health abnormality."),
        "Spider_mite_infestation": ("disease_leaf", 2, 0, 1, (2, 3), "Pest infestation mapped to moderate health abnormality."),
    }
    for class_name, mapping in class_map.items():
        paths = iter_images(root / class_name)
        fusion_group, health, maturity, alert, growth_choices, note = mapping
        add_candidates(
            candidates,
            paths,
            source_dataset="kurdistan_watermelon_disease",
            source_class=class_name,
            fusion_group=fusion_group,
            health_level=health,
            maturity_level=maturity,
            abnormal_alert=alert,
            growth_stage_choices=growth_choices,
            label_note=note,
        )


def collect_roboflow_ripeness(candidates: dict[str, list[Candidate]]) -> None:
    datasets = [
        (RAW_DIR / "watermelon_ripe_semiripe_unripe", "watermelon_ripe_semiripe_unripe"),
        (RAW_DIR / "watermelon_ripe_unripe", "watermelon_ripe_unripe"),
    ]
    for root, source_dataset in datasets:
        for split in ("train", "valid", "test"):
            split_dir = root / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                class_key = class_dir.name.lower().strip()
                has_unripe = "unripe" in class_key
                has_ripe = "ripe" in class_key.replace("unripe", "")
                if class_key == "empty" or (has_ripe and has_unripe):
                    continue
                if has_unripe:
                    add_candidates(
                        candidates,
                        iter_images(class_dir),
                        source_dataset=source_dataset,
                        source_class=class_dir.name,
                        fusion_group="unripe_fruit",
                        health_level=0,
                        maturity_level=0,
                        abnormal_alert=0,
                        growth_stage_choices=(4,),
                        label_note="Unripe watermelon fruit image.",
                    )
                elif has_ripe:
                    add_candidates(
                        candidates,
                        iter_images(class_dir),
                        source_dataset=source_dataset,
                        source_class=class_dir.name,
                        fusion_group="ripe_fruit",
                        health_level=0,
                        maturity_level=2,
                        abnormal_alert=0,
                        growth_stage_choices=(5,),
                        label_note="Ripe watermelon fruit image.",
                    )


def collect_fruq(candidates: dict[str, list[Candidate]]) -> None:
    root = RAW_DIR / "fruq_db" / "FruQ-DB"
    class_map = {
        "Fresh": ("fresh_quality", 0, 2, 0, (5,), "Fresh fruit quality image used as mature healthy quality supplement."),
        "Mild": ("mild_quality", 1, 1, 1, (5,), "Mild fruit quality degradation mapped to near-mature mild abnormality."),
        "Rotten": ("rotten_quality", 3, 3, 1, (5,), "Rotten fruit quality image mapped to severe abnormal overripe sample."),
    }
    for class_name, mapping in class_map.items():
        fusion_group, health, maturity, alert, growth_choices, note = mapping
        add_candidates(
            candidates,
            iter_images(root / class_name),
            source_dataset="fruq_db",
            source_class=class_name,
            fusion_group=fusion_group,
            health_level=health,
            maturity_level=maturity,
            abnormal_alert=alert,
            growth_stage_choices=growth_choices,
            label_note=note,
        )


def collect_candidates() -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    collect_disease_recognition(candidates)
    collect_kurdistan_disease(candidates)
    collect_roboflow_ripeness(candidates)
    collect_fruq(candidates)
    return candidates


def scale_quotas(target_size: int) -> dict[str, int]:
    base_total = sum(BASE_GROUP_QUOTAS.values())
    scaled: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for group, quota in BASE_GROUP_QUOTAS.items():
        raw = quota * target_size / base_total
        scaled[group] = int(raw)
        fractions.append((raw - int(raw), group))
    remainder = target_size - sum(scaled.values())
    for _, group in sorted(fractions, reverse=True)[:remainder]:
        scaled[group] += 1
    return scaled


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_candidates(
    candidates: dict[str, list[Candidate]],
    target_size: int,
    seed: int,
) -> tuple[list[Candidate], dict[str, int]]:
    rng = random.Random(seed)
    quotas = scale_quotas(target_size)
    selected: list[Candidate] = []
    seen_hashes: set[str] = set()
    selected_by_group: dict[str, int] = {}

    for group, quota in quotas.items():
        group_candidates = list(candidates.get(group, []))
        rng.shuffle(group_candidates)
        group_selected: list[Candidate] = []
        for candidate in group_candidates:
            digest = file_sha1(candidate.path)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            group_selected.append(candidate)
            if len(group_selected) >= quota:
                break
        if len(group_selected) < quota:
            raise ValueError(
                f"not enough unique images for {group}: "
                f"needed={quota}, found={len(group_selected)}"
            )
        selected.extend(group_selected)
        selected_by_group[group] = len(group_selected)

    rng.shuffle(selected)
    return selected, selected_by_group


def read_xlsx_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", ns):
        texts = [node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
        strings.append("".join(texts))
    return strings


def sheet_targets_by_name(zip_file: ZipFile) -> dict[str, str]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    targets_by_id = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    result = {}
    for sheet in workbook.find("a:sheets", ns):
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = targets_by_id[rel_id]
        result[sheet.attrib["name"]] = "xl/" + target.lstrip("/")
    return result


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str | float | None:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    value = cell.findtext("a:v", default="", namespaces=ns)
    if value == "":
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    try:
        return float(value)
    except ValueError:
        return value


def read_sheet_rows(zip_file: ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str | float | None]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zip_file.read(sheet_path))
    rows = []
    for row in root.find("a:sheetData", ns).findall("a:row", ns):
        values = []
        for cell in row.findall("a:c", ns):
            values.append(cell_value(cell, shared_strings))
        rows.append(values)
    return rows


def excel_serial_to_datetime(value: float) -> datetime:
    return datetime(1899, 12, 30) + timedelta(days=float(value))


def build_environment_rows(xlsx_path: Path) -> list[EnvironmentRow]:
    with ZipFile(xlsx_path) as zip_file:
        shared_strings = read_xlsx_shared_strings(zip_file)
        sheets = sheet_targets_by_name(zip_file)
        weather_rows = read_sheet_rows(zip_file, sheets["weather (hourly)"], shared_strings)
        swc_rows = read_sheet_rows(zip_file, sheets["SWC+FC+WP+Irrigations"], shared_strings)

    swc_by_date: list[tuple[datetime, float]] = []
    for row in swc_rows[2:]:
        if len(row) < 2 or row[0] is None:
            continue
        swc_value = row[1] if row[1] is not None else (row[2] if len(row) > 2 else None)
        if swc_value is None:
            continue
        swc_by_date.append((excel_serial_to_datetime(float(row[0])), float(swc_value) * 100.0))
    swc_by_date.sort(key=lambda item: item[0])
    if not swc_by_date:
        raise ValueError("SWC sheet has no usable soil water rows")

    environment: list[EnvironmentRow] = []
    swc_index = 0
    for row in weather_rows[2:]:
        if len(row) < 5 or row[0] is None:
            continue
        timestamp = excel_serial_to_datetime(float(row[0])).replace(microsecond=0)
        while swc_index + 1 < len(swc_by_date) and swc_by_date[swc_index + 1][0] <= timestamp:
            swc_index += 1
        par = max(float(row[1] or 0.0), 0.0)
        temperature = float(row[2])
        soil_humidity = min(max(swc_by_date[swc_index][1], 0.0), 100.0)
        environment.append(
            EnvironmentRow(
                timestamp=timestamp,
                temperature=temperature,
                soil_humidity=soil_humidity,
                light=par,
                ph=6.8,
            )
        )

    environment.sort(key=lambda item: item.timestamp)
    if len(environment) < 25:
        raise ValueError("environment source must contain at least 25 hourly rows")
    return environment


def stage_time_buckets(environment: list[EnvironmentRow]) -> dict[int, list[datetime]]:
    eligible = [row.timestamp for row in environment[24:]]
    if len(eligible) < 6:
        raise ValueError("not enough environment rows after the first 24 hours")
    bucket_ranges = {
        0: (0.00, 0.12),
        1: (0.12, 0.28),
        2: (0.28, 0.44),
        3: (0.44, 0.60),
        4: (0.60, 0.78),
        5: (0.78, 1.00),
    }
    buckets: dict[int, list[datetime]] = {}
    total = len(eligible)
    for stage, (start_ratio, end_ratio) in bucket_ranges.items():
        start = min(int(total * start_ratio), total - 1)
        end = min(max(int(total * end_ratio), start + 1), total)
        buckets[stage] = eligible[start:end]
    return buckets


def split_name(index: int, count: int) -> str:
    ratio = index / count
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def prepare_output_dir(output_dir: Path, clear_output: bool) -> Path:
    if output_dir.exists() and clear_output:
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def copy_selected_images(
    selected: list[Candidate],
    output_dir: Path,
    environment: list[EnvironmentRow],
    seed: int,
    verify_images: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed + 1000)
    images_dir = output_dir / "images"
    buckets = stage_time_buckets(environment)
    bucket_offsets: Counter[int] = Counter()

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.fusion_group].append(candidate)

    split_by_source: dict[Path, str] = {}
    for group_items in grouped.values():
        rng.shuffle(group_items)
        for index, candidate in enumerate(group_items):
            split_by_source[candidate.path] = split_name(index, len(group_items))

    labels: list[dict[str, str]] = []
    splits: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []

    for index, candidate in enumerate(selected, start=1):
        growth_stage = rng.choice(candidate.growth_stage_choices)
        available_times = buckets[growth_stage]
        time_index = bucket_offsets[growth_stage] % len(available_times)
        bucket_offsets[growth_stage] += 1
        capture_time = available_times[time_index]

        image_id = f"IMG_{index:06d}"
        suffix = candidate.path.suffix.lower()
        output_name = f"{image_id}{suffix}"
        output_path = images_dir / output_name
        shutil.copy2(candidate.path, output_path)

        if verify_images:
            with Image.open(output_path) as image:
                image.verify()

        split = split_by_source[candidate.path]
        label_row = {
            "image_id": image_id,
            "image_path": f"images/{output_name}",
            "capture_time": capture_time.strftime(TIMESTAMP_FORMAT),
            "growth_stage": str(growth_stage),
            "health_level": str(candidate.health_level),
            "maturity_level": str(candidate.maturity_level),
            "abnormal_alert": str(candidate.abnormal_alert),
        }
        labels.append(label_row)
        splits.append({"image_id": image_id, "split": split})
        manifest.append(
            {
                **label_row,
                "split": split,
                "source_dataset": candidate.source_dataset,
                "source_class": candidate.source_class,
                "fusion_group": candidate.fusion_group,
                "source_path": str(candidate.path.relative_to(ROOT)),
                "label_note": candidate.label_note,
            }
        )

    return labels, splits, manifest


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_environment_csv(output_dir: Path, environment: list[EnvironmentRow]) -> None:
    rows = [
        {
            "timestamp": row.timestamp.strftime(TIMESTAMP_FORMAT),
            "temperature": f"{row.temperature:.3f}",
            "soil_humidity": f"{row.soil_humidity:.3f}",
            "light": f"{row.light:.3f}",
            "ph": f"{row.ph:.2f}",
        }
        for row in environment
    ]
    write_csv(output_dir / "environment.csv", ("timestamp", "temperature", "soil_humidity", "light", "ph"), rows)


def write_readme(output_dir: Path, sample_count: int) -> None:
    readme = f"""# Fused Watermelon Multimodal Dataset

This dataset was generated by `scripts/build_fused_watermelon_dataset.py`.

It keeps the project training schema unchanged:

- `images/`: sampled watermelon-related images copied from public source datasets.
- `labels.csv`: one complete four-task label row per image.
- `environment.csv`: hourly watermelon-field environment rows.
- `split.csv`: train/val/test split.
- `fusion_manifest.csv`: provenance and label-mapping details for every image.
- `quality_report.json`: counts and validation summary.

The dataset contains {sample_count} sampled images. It is intentionally sampled
and balanced instead of using every downloaded image. Labels are weak-supervised
fusion labels derived from original source classes plus agricultural priors. The
environment time series is real watermelon-field weather/SWC data from the
downloaded South Italy watermelon workbook. Soil pH is set to 6.8 because the
source workbook does not provide pH.

Use this dataset for project training and demonstration. Do not describe it as a
single-site manually annotated greenhouse dataset.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def counter_from_rows(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def write_quality_report(
    output_dir: Path,
    labels: list[dict[str, str]],
    splits: list[dict[str, str]],
    manifest: list[dict[str, str]],
    selected_by_group: dict[str, int],
    available_by_group: dict[str, int],
    environment: list[EnvironmentRow],
    seed: int,
) -> None:
    report = {
        "seed": seed,
        "sample_count": len(labels),
        "environment_rows": len(environment),
        "environment_date_range": [
            environment[0].timestamp.strftime(TIMESTAMP_FORMAT),
            environment[-1].timestamp.strftime(TIMESTAMP_FORMAT),
        ],
        "available_by_group": dict(sorted(available_by_group.items())),
        "selected_by_group": dict(sorted(selected_by_group.items())),
        "split_counts": counter_from_rows(splits, "split"),
        "source_dataset_counts": counter_from_rows(manifest, "source_dataset"),
        "source_class_counts": counter_from_rows(manifest, "source_class"),
        "growth_stage_counts": counter_from_rows(labels, "growth_stage"),
        "health_level_counts": counter_from_rows(labels, "health_level"),
        "maturity_level_counts": counter_from_rows(labels, "maturity_level"),
        "abnormal_alert_counts": counter_from_rows(labels, "abnormal_alert"),
        "sources": SOURCES,
        "notes": [
            "Roboflow `empty` classes are skipped.",
            "Ambiguous classes containing both ripe and unripe are skipped.",
            "Kurdistan HEIC original captures are not used by default; the extracted augmented JPG/PNG images are used.",
            "FruQ-DB is used in limited quantity for fruit quality and rotten/overripe supervision.",
            "Every label row has a matching 24-hour environment window.",
        ],
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_generated_dataset(output_dir: Path, labels: list[dict[str, str]], splits: list[dict[str, str]], environment: list[EnvironmentRow]) -> None:
    image_ids = [row["image_id"] for row in labels]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image_id values in labels")
    if {row["image_id"] for row in labels} != {row["image_id"] for row in splits}:
        raise ValueError("split.csv image IDs do not match labels.csv")

    env_times = [row.timestamp for row in environment]
    env_time_set = set(env_times)
    for row in labels:
        image_path = output_dir / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        capture_time = datetime.strptime(row["capture_time"], TIMESTAMP_FORMAT)
        if capture_time not in env_time_set:
            raise ValueError(f"capture_time is not in environment.csv: {capture_time}")
        window_start = capture_time - timedelta(hours=24)
        if not any(window_start <= time <= capture_time for time in env_times):
            raise ValueError(f"no environment window for {row['image_id']}")
        if not 0 <= int(row["growth_stage"]) <= 5:
            raise ValueError(f"invalid growth_stage for {row['image_id']}")
        if not 0 <= int(row["health_level"]) <= 3:
            raise ValueError(f"invalid health_level for {row['image_id']}")
        if not 0 <= int(row["maturity_level"]) <= 3:
            raise ValueError(f"invalid maturity_level for {row['image_id']}")
        if not 0 <= int(row["abnormal_alert"]) <= 1:
            raise ValueError(f"invalid abnormal_alert for {row['image_id']}")


def build_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, clear_output=args.clear_output)

    candidates = collect_candidates()
    available_by_group = {group: len(items) for group, items in candidates.items()}
    selected, selected_by_group = sample_candidates(
        candidates=candidates,
        target_size=args.target_size,
        seed=args.seed,
    )

    environment_path = RAW_DIR / "watermelon_environment" / "watermelon_2023_south_Italy_ancillary_weather+LAI+GM+SWC.xlsx"
    environment = build_environment_rows(environment_path)
    write_environment_csv(output_dir, environment)

    labels, splits, manifest = copy_selected_images(
        selected=selected,
        output_dir=output_dir,
        environment=environment,
        seed=args.seed,
        verify_images=not args.skip_verify_images,
    )

    write_csv(
        output_dir / "labels.csv",
        ("image_id", "image_path", "capture_time", *LABEL_FIELDS),
        labels,
    )
    write_csv(output_dir / "split.csv", ("image_id", "split"), splits)
    write_csv(
        output_dir / "fusion_manifest.csv",
        (
            "image_id",
            "image_path",
            "capture_time",
            *LABEL_FIELDS,
            "split",
            "source_dataset",
            "source_class",
            "fusion_group",
            "source_path",
            "label_note",
        ),
        manifest,
    )
    validate_generated_dataset(output_dir, labels, splits, environment)
    write_quality_report(
        output_dir=output_dir,
        labels=labels,
        splits=splits,
        manifest=manifest,
        selected_by_group=selected_by_group,
        available_by_group=available_by_group,
        environment=environment,
        seed=args.seed,
    )
    write_readme(output_dir, sample_count=len(labels))

    print(f"output_dir={output_dir}")
    print(f"images={len(labels)}")
    print(f"environment_rows={len(environment)}")
    print(f"split_counts={counter_from_rows(splits, 'split')}")
    print(f"growth_stage_counts={counter_from_rows(labels, 'growth_stage')}")
    print(f"health_level_counts={counter_from_rows(labels, 'health_level')}")
    print(f"maturity_level_counts={counter_from_rows(labels, 'maturity_level')}")
    print(f"abnormal_alert_counts={counter_from_rows(labels, 'abnormal_alert')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced fused watermelon multimodal training dataset."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--skip-verify-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
