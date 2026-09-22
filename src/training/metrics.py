from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    precision_by_class: list[float]
    recall_by_class: list[float]
    f1_by_class: list[float]
    confusion_matrix: list[list[int]]
    support: list[int]


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if targets.ndim != 1:
        targets = targets.view(-1)
    if predictions.ndim != 1:
        predictions = predictions.view(-1)
    if targets.numel() != predictions.numel():
        raise ValueError(
            f"target and prediction lengths differ: "
            f"{targets.numel()} vs {predictions.numel()}"
        )

    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, prediction in zip(targets.long(), predictions.long(), strict=True):
        if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(target), int(prediction)] += 1
        else:
            raise ValueError(
                f"class index out of range: target={int(target)}, "
                f"prediction={int(prediction)}, num_classes={num_classes}"
            )
    return matrix


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    matrix = confusion_matrix(targets, predictions, num_classes)
    total = int(matrix.sum().item())
    correct = int(torch.diag(matrix).sum().item())
    accuracy = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1_scores = []
    support = matrix.sum(dim=1)

    for class_index in range(num_classes):
        true_positive = float(matrix[class_index, class_index].item())
        predicted_positive = float(matrix[:, class_index].sum().item())
        actual_positive = float(matrix[class_index, :].sum().item())

        precision = (
            true_positive / predicted_positive
            if predicted_positive > 0
            else 0.0
        )
        recall = (
            true_positive / actual_positive
            if actual_positive > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=sum(precisions) / num_classes,
        macro_recall=sum(recalls) / num_classes,
        macro_f1=sum(f1_scores) / num_classes,
        precision_by_class=precisions,
        recall_by_class=recalls,
        f1_by_class=f1_scores,
        confusion_matrix=matrix.tolist(),
        support=[int(value) for value in support.tolist()],
    )
