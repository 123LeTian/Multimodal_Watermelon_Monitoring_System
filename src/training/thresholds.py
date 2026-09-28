from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from src.training.metrics import classification_metrics


@dataclass(frozen=True)
class BinaryThresholdResult:
    threshold: float
    accuracy: float
    macro_f1: float
    positive_precision: float
    positive_recall: float
    positive_f1: float
    minimum_recall: float
    recall_requirement_met: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def find_binary_threshold(
    targets: torch.Tensor,
    positive_probabilities: torch.Tensor,
    minimum_recall: float = 0.85,
    thresholds: Iterable[float] | None = None,
) -> BinaryThresholdResult:
    targets = targets.long().view(-1).cpu()
    probabilities = positive_probabilities.float().view(-1).cpu()
    if targets.numel() == 0:
        raise ValueError("threshold calibration requires at least one sample")
    if targets.numel() != probabilities.numel():
        raise ValueError("target and probability lengths differ")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be in [0, 1]")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("positive probabilities must be in [0, 1]")

    candidates = list(thresholds or (index / 100 for index in range(5, 96)))
    if not candidates:
        raise ValueError("at least one threshold candidate is required")

    results: list[BinaryThresholdResult] = []
    for threshold in candidates:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold candidates must be in [0, 1]")
        predictions = (probabilities >= threshold).long()
        metrics = classification_metrics(targets, predictions, num_classes=2)
        recall = metrics.recall_by_class[1]
        results.append(
            BinaryThresholdResult(
                threshold=threshold,
                accuracy=metrics.accuracy,
                macro_f1=metrics.macro_f1,
                positive_precision=metrics.precision_by_class[1],
                positive_recall=recall,
                positive_f1=metrics.f1_by_class[1],
                minimum_recall=minimum_recall,
                recall_requirement_met=recall >= minimum_recall,
            )
        )

    eligible = [result for result in results if result.recall_requirement_met]
    if eligible:
        return max(
            eligible,
            key=lambda result: (
                result.positive_f1,
                result.macro_f1,
                result.threshold,
            ),
        )
    return max(
        results,
        key=lambda result: (
            result.positive_recall,
            result.positive_f1,
            result.macro_f1,
        ),
    )
