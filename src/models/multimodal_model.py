from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class TaskOutputDims:
    growth_stage: int = 6
    health_level: int = 4
    maturity_level: int = 4
    abnormal_alert: int = 2


class ResNetImageEncoder(nn.Module):
    """ResNet50 image branch for watermelon growth images."""

    def __init__(
        self,
        output_dim: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projection(features)


class EnvironmentTransformerEncoder(nn.Module):
    """Transformer branch for 24-hour environment time series."""

    def __init__(
        self,
        sensor_dim: int = 4,
        time_dim: int = 1,
        model_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(sensor_dim + time_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        environment: torch.Tensor,
        time_offsets: torch.Tensor,
        environment_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([environment, time_offsets], dim=-1)
        inputs = self.input_projection(inputs)

        padding_mask = ~environment_mask
        encoded = self.encoder(inputs, src_key_padding_mask=padding_mask)

        mask = environment_mask.unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / lengths

        return self.output_projection(pooled)


class MultimodalWatermelonModel(nn.Module):
    """Minimal ResNet + Transformer model with four task heads."""

    def __init__(
        self,
        image_feature_dim: int = 256,
        environment_feature_dim: int = 256,
        fusion_dim: int = 256,
        task_dims: TaskOutputDims | None = None,
        sensor_dim: int = 4,
        time_dim: int = 1,
        image_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.task_dims = task_dims or TaskOutputDims()
        self.image_encoder = ResNetImageEncoder(
            output_dim=image_feature_dim,
            pretrained=image_pretrained,
        )
        self.environment_encoder = EnvironmentTransformerEncoder(
            sensor_dim=sensor_dim,
            time_dim=time_dim,
            output_dim=environment_feature_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + environment_feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.heads = nn.ModuleDict(
            {
                "growth_stage": nn.Linear(fusion_dim, self.task_dims.growth_stage),
                "health_level": nn.Linear(fusion_dim, self.task_dims.health_level),
                "maturity_level": nn.Linear(fusion_dim, self.task_dims.maturity_level),
                "abnormal_alert": nn.Linear(fusion_dim, self.task_dims.abnormal_alert),
            }
        )

    def forward(
        self,
        images: torch.Tensor,
        environment: torch.Tensor,
        time_offsets: torch.Tensor,
        environment_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(images)
        environment_features = self.environment_encoder(
            environment=environment,
            time_offsets=time_offsets,
            environment_mask=environment_mask,
        )
        fused = self.fusion(torch.cat([image_features, environment_features], dim=-1))

        return {
            f"{name}_logits": head(fused)
            for name, head in self.heads.items()
        }
