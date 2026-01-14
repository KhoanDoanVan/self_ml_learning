"""FastAPI application for Vietnamese NER service."""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    Entity,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from inference.predictor import NERPredictor
from utils.logger import setup_logger

logger = setup_logger(__name__, "logs/api.log")

# Global predictor instance
predictor: Optional[NERPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    global predictor
    
    # Startup: Load model
    logger.info("Loading NER model...")
    try:
        # Initialize predictor
        # predictor = NERPredictor.from_pretrained(...)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Vietnamese NER API",
    description="Named Entity Recognition API for Vietnamese text",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Vietnamese NER API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single text prediction",
            "/predict/batch": "Batch prediction"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict entities in text."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Predict entities
        entities_list = predictor.predict_text(request.text)
        
        # Convert to response format
        entities = [
            Entity(
                text=text,
                label=label,
                start=start,
                end=end
            )
            for text, label, start, end in entities_list
        ]
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            text=request.text,
            entities=entities,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict entities in multiple texts."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Predict entities for all texts
        all_entities = predictor.predict_batch(
            request.texts,
            batch_size=16
        )
        
        # Convert to response format
        predictions = []
        for text, entities_list in zip(request.texts, all_entities):
            entities = [
                Entity(
                    text=ent_text,
                    label=label,
                    start=start,
                    end=end
                )
                for ent_text, label, start, end in entities_list
            ]
            
            predictions.append(
                PredictionResponse(
                    text=text,
                    entities=entities,
                    processing_time=0.0  # Individual time not tracked
                )
            )
        
        total_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time=total_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )