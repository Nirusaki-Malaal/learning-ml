from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float]


class VisualizationRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)
    x_feature: str = "radius_mean"
    y_feature: str = "texture_mean"
    grid_size: int = 28
