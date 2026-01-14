"""API schemas for Vietnamese NER service."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Entity model."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    confidence: Optional[float] = Field(None, description="Confidence score")


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    text: str = Field(..., description="Input text for NER")
    return_confidence: bool = Field(
        False,
        description="Whether to return confidence scores"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    text: str = Field(..., description="Original input text")
    entities: List[Entity] = Field(..., description="Detected entities")
    processing_time: float = Field(..., description="Processing time in seconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of texts for NER")
    return_confidence: bool = Field(False)


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_processing_time: float = Field(
        ...,
        description="Total processing time"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")