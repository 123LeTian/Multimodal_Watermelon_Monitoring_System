from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


TASK_NAMES = (
    "growth_stage",
    "health_level",
    "maturity_level",
    "abnormal_alert",
)

DEFAULT_TASK_WEIGHTS = {
    "growth_stage": 1.0,
    "health_level": 1.0,
    "maturity_level": 1.0,
    "abnormal_alert": 1.0,
}


class MultiTaskLoss(nn.Module):
    """Weighted cross-entropy loss for the four classification tasks."""

    def __init__(
        self,
        task_weights: Mapping[str, float] | None = None,
        class_weights: Mapping[str, list[float]] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        weights = dict(DEFAULT_TASK_WEIGHTS)
        if task_weights is not None:
            unknown_tasks = set(task_weights) - set(TASK_NAMES)
            if unknown_tasks:
                raise ValueError(f"unknown task weights: {sorted(unknown_tasks)}")
            weights.update(task_weights)

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("task weights must be non-negative")
        if sum(weights.values()) == 0:
            raise ValueError("at least one task weight must be positive")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

        configured_class_weights = dict(class_weights or {})
        unknown_class_weight_tasks = set(configured_class_weights) - set(TASK_NAMES)
        if unknown_class_weight_tasks:
            raise ValueError(
                f"unknown class weight tasks: {sorted(unknown_class_weight_tasks)}"
            )
        for task, values in configured_class_weights.items():
            if not values:
                raise ValueError(f"class weights for {task} must not be empty")
            if any(float(value) <= 0 for value in values):
                raise ValueError(f"class weights for {task} must be positive")

        self.task_weights = weights
        self.criteria = nn.ModuleDict(
            {
                task: nn.CrossEntropyLoss(
                    weight=(
                        torch.tensor(configured_class_weights[task], dtype=torch.float32)
                        if task in configured_class_weights
                        else None
                    ),
                    label_smoothing=label_smoothing,
                )
                for task in TASK_NAMES
            }
        )

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        task_masks: Mapping[str, torch.Tensor] | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate total loss and, optionally, each task's loss.

        `outputs` must contain keys such as `growth_stage_logits`.
        `labels` must contain one integer class-index tensor per task.
        If `task_masks` is provided, each task loss is computed only for
        samples with mask value > 0. A fully masked task contributes zero.
        """
        task_losses: dict[str, torch.Tensor] = {}
        task_valid_counts: dict[str, torch.Tensor] = {}

        for task in TASK_NAMES:
            logits_key = f"{task}_logits"
            if logits_key not in outputs:
                raise KeyError(f"missing model output: {logits_key}")
            if task not in labels:
                raise KeyError(f"missing target labels: {task}")

            logits = outputs[logits_key]
            targets = labels[task].long().view(-1)

            if logits.ndim != 2:
                raise ValueError(
                    f"{logits_key} must have shape (batch, classes), "
                    f"got {tuple(logits.shape)}"
                )
            if logits.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"batch size mismatch for {task}: "
                    f"logits={logits.shape[0]}, targets={targets.shape[0]}"
                )

            if task_masks is None or task not in task_masks:
                mask = torch.ones_like(targets, dtype=torch.bool, device=logits.device)
            else:
                mask = task_masks[task].to(device=logits.device).view(-1) > 0
                if mask.shape[0] != targets.shape[0]:
                    raise ValueError(
                        f"mask size mismatch for {task}: "
                        f"mask={mask.shape[0]}, targets={targets.shape[0]}"
                    )

            valid_count = mask.sum()
            task_valid_counts[task] = valid_count
            if int(valid_count.item()) == 0:
                task_losses[task] = logits.sum() * 0.0
            else:
                task_losses[task] = self.criteria[task](logits[mask], targets[mask])

        total_loss = sum(
            self.task_weights[task] * task_losses[task]
            for task in TASK_NAMES
        )

        if return_details:
            details: dict[str, torch.Tensor] = {
                f"{task}_loss": task_losses[task] for task in TASK_NAMES
            }
            details.update(
                {f"{task}_valid_count": task_valid_counts[task] for task in TASK_NAMES}
            )
            details["total_loss"] = total_loss
            return total_loss, details

        return total_loss


def calculate_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    task_masks: Mapping[str, torch.Tensor] | None = None,
    task_weights: Mapping[str, float] | None = None,
    class_weights: Mapping[str, list[float]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience function returning total and per-task losses."""
    criterion = MultiTaskLoss(
        task_weights=task_weights,
        class_weights=class_weights,
    )
    total_loss, details = criterion(outputs, labels, task_masks=task_masks, return_details=True)
    return total_loss, details
