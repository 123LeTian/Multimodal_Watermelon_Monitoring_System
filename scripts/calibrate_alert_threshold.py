from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import find_binary_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an abnormal-alert threshold from validation predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    targets: list[int] = []
    probabilities: list[float] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "abnormal_alert_true",
            "abnormal_alert_valid",
            "abnormal_alert_probability",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["abnormal_alert_valid"]) <= 0:
                continue
            targets.append(int(row["abnormal_alert_true"]))
            probabilities.append(float(row["abnormal_alert_probability"]))

    result = find_binary_threshold(
        targets=torch.tensor(targets),
        positive_probabilities=torch.tensor(probabilities),
        minimum_recall=args.minimum_recall,
    )
    report = {
        "predictions": str(predictions_path),
        "evaluated_samples": len(targets),
        **result.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
