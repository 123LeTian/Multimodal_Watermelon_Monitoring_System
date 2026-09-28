from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


VALID_INPUT_MODES = ("multimodal", "image_only", "env_only")


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
        use_sensor_mask: bool = False,
    ) -> None:
        super().__init__()
        self.use_sensor_mask = use_sensor_mask
        sensor_input_dim = sensor_dim * (2 if use_sensor_mask else 1)
        self.input_projection = nn.Linear(sensor_input_dim + time_dim, model_dim)
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
        environment_sensor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [environment]
        if self.use_sensor_mask:
            if environment_sensor_mask is None:
                environment_sensor_mask = torch.ones_like(environment)
            if environment_sensor_mask.shape != environment.shape:
                raise ValueError(
                    "environment_sensor_mask must have the same shape as environment"
                )
            features.append(environment_sensor_mask.to(environment.dtype))
        features.append(time_offsets)
        inputs = torch.cat(features, dim=-1)
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
        input_mode: str = "multimodal",
        use_sensor_mask: bool = False,
    ) -> None:
        super().__init__()
        if input_mode not in VALID_INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {VALID_INPUT_MODES}, got {input_mode!r}"
            )

        self.input_mode = input_mode
        self.task_dims = task_dims or TaskOutputDims()
        self.image_encoder = None
        self.environment_encoder = None

        fusion_input_dim = 0
        if self.uses_image:
            self.image_encoder = ResNetImageEncoder(
                output_dim=image_feature_dim,
                pretrained=image_pretrained,
            )
            fusion_input_dim += image_feature_dim
        if self.uses_environment:
            self.environment_encoder = EnvironmentTransformerEncoder(
                sensor_dim=sensor_dim,
                time_dim=time_dim,
                output_dim=environment_feature_dim,
                use_sensor_mask=use_sensor_mask,
            )
            fusion_input_dim += environment_feature_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
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

    @property
    def uses_image(self) -> bool:
        return self.input_mode in {"multimodal", "image_only"}

    @property
    def uses_environment(self) -> bool:
        return self.input_mode in {"multimodal", "env_only"}

    def forward(
        self,
        images: torch.Tensor,
        environment: torch.Tensor,
        time_offsets: torch.Tensor,
        environment_mask: torch.Tensor,
        environment_sensor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = []
        if self.uses_image:
            if self.image_encoder is None:
                raise RuntimeError("image encoder is not initialized")
            features.append(self.image_encoder(images))
        if self.uses_environment:
            if self.environment_encoder is None:
                raise RuntimeError("environment encoder is not initialized")
            features.append(
                self.environment_encoder(
                    environment=environment,
                    time_offsets=time_offsets,
                    environment_mask=environment_mask,
                    environment_sensor_mask=environment_sensor_mask,
                )
            )

        fused = self.fusion(torch.cat(features, dim=-1))

        return {
            f"{name}_logits": head(fused)
            for name, head in self.heads.items()
        }
