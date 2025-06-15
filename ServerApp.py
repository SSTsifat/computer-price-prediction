import fastapi
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from typing import Dict, Any, Tuple
import logging
from dataclasses import dataclass
import util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with custom configuration
app = FastAPI(
    title="Laptop Price Predictor API",
    description="Advanced API for predicting laptop prices using machine learning",
    version="2.0.0"
)

# Custom CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class LaptopSpecs:
    inches: float
    cpu: float
    ram: float
    weight: float
    resolution: str
    ssd: float
    hdd: float
    graphics: str
    os: str
    is_gaming: bool

    @classmethod
    def from_form_data(cls, form_data: Dict[str, Any]) -> 'LaptopSpecs':
        resolution = form_data["resolution"].split("x")
        return cls(
            inches=float(form_data["inches"]),
            cpu=float(form_data["cpu"]),
            ram=float(form_data["ram"]),
            weight=float(form_data["weight"]),
            resolution=form_data["resolution"],
            ssd=float(form_data["ssd"]),
            hdd=float(form_data["hdd"]),
            graphics=str(form_data["graphics"]),
            os=str(form_data["os"]),
            is_gaming=bool(form_data.get("tg") == "on")
        )

def process_resolution(resolution: str) -> Tuple[float, float]:
    """Process resolution string into horizontal and vertical components."""
    h_res, v_res = map(float, resolution.split("x"))
    return h_res, v_res

@app.post("/api/predict_price")
async def predict_price(request: Request) -> JSONResponse:
    """
    Predict laptop price based on specifications.
    Returns both best and average predicted prices.
    """
    try:
        form_data = await request.form()
        logger.info(f"Received prediction request with data: {dict(form_data)}")
        
        # Create LaptopSpecs object from form data
        specs = LaptopSpecs.from_form_data(form_data)
        h_res, v_res = process_resolution(specs.resolution)
        
        # Log processed inputs
        logger.info("Processed specifications:", {
            "Inches": specs.inches,
            "CPU": specs.cpu,
            "RAM": specs.ram,
            "Weight": specs.weight,
            "Resolution": f"{h_res}x{v_res}",
            "Storage": f"SSD: {specs.ssd}GB, HDD: {specs.hdd}GB",
            "Graphics": specs.graphics,
            "OS": specs.os,
            "Gaming": specs.is_gaming
        })

        # Get predictions
        best_prediction, avg_prediction = util.Predict(
            specs.inches, specs.cpu, specs.ram, specs.weight,
            h_res, v_res, specs.ssd, specs.hdd,
            specs.graphics, specs.os, int(specs.is_gaming)
        )
        
        # Round predictions to one decimal place
        predictions = {
            "predictedPrice": round(float(best_prediction), 1),
            "bestPredictedPrice": round(float(best_prediction), 1),
            "avgPredictedPrice": round(float(avg_prediction), 1)
        }
        
        logger.info(f"Predictions generated: {predictions}")
        return JSONResponse(status_code=200, content=predictions)
        
    except ValueError as ve:
        logger.error(f"Invalid input data: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Prediction error: {str(e)}\n{error_details}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "details": error_details
            }
        )

def initialize_app():
    """Initialize the application and load required models."""
    try:
        util.load()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5002,
        log_level="info"
    )
