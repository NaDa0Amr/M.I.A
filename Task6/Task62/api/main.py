"""
FastAPI-based REST API for Image Caption Generation.

Endpoints:
    POST /predict       - Upload an image and receive a generated caption
    GET  /health        - Health check endpoint
    GET  /              - API documentation redirect
"""

import io
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
import torch

from src.inference.pipeline import CaptionPipeline

app = FastAPI(
    title="Neural Image Captioning API",
    description="Upload an image and receive an AI-generated caption using a CNN-LSTM model trained on Flickr8k.",
    version="1.0.0",
)

# Global pipeline instance (loaded once at startup)
pipeline = None


@app.on_event("startup")
async def load_model():
    """Load the captioning model on server startup."""
    global pipeline
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = CaptionPipeline.from_checkpoint("checkpoints", device=device)
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")
        print("  The API will return 503 until a valid checkpoint is available.")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipeline is not None else "model_not_loaded",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    strategy: str = Query("beam", enum=["greedy", "beam"], description="Decoding strategy"),
    beam_size: int = Query(3, ge=1, le=10, description="Beam size (only used with beam strategy)"),
    max_length: int = Query(20, ge=5, le=50, description="Maximum caption length"),
    temperature: float = Query(1.0, ge=0.1, le=2.0, description="Sampling temperature"),
):
    """
    Generate a caption for an uploaded image.

    - **file**: Image file (JPEG, PNG, or WEBP format)
    - **strategy**: `greedy` for fast decoding or `beam` for higher quality
    - **beam_size**: Number of beams for beam search (default: 3)
    - **max_length**: Maximum number of words in the generated caption
    - **temperature**: Controls randomness (1.0 = normal, <1 = more focused, >1 = more diverse)

    Returns a JSON object with the generated caption, tokens, and confidence score.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure checkpoints are available in the 'checkpoints' directory.",
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WEBP.",
        )

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        # Generate caption
        result = pipeline.generate(
            image=image,
            strategy=strategy,
            beam_size=beam_size,
            max_length=max_length,
            temperature=temperature,
        )

        return JSONResponse(
            content={
                "caption": result["caption"],
                "tokens": result["tokens"],
                "token_ids": result["token_ids"],
                "score": round(result["score"], 4),
                "strategy": strategy,
                "beam_size": beam_size if strategy == "beam" else None,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
