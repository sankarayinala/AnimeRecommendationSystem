"""Pydantic response models for the anime recommendation API."""

from typing import List
from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    """Response schema returned by the recommendation endpoint."""
    user_id: int
    recommendations: List[str]
    model: str = "Hybrid Recommendation Model v1.0"


class HealthResponse(BaseModel):
    """Response schema returned by the health endpoint."""
    status: str
    message: str
    uptime_seconds: float