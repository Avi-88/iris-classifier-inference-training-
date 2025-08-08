import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# --- Application Setup ---
app = FastAPI(
    title="ML Inference API",
    description="An API to serve the Iris classification model.",
    version="1.0.0"
)

# Add Prometheus metrics instrumentation to the API
Instrumentator().instrument(app).expose(app)


# --- Model Loading ---
# Load the model path from an environment variable for configurability.
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model_not_found.joblib")
model = None

@app.on_event("startup")
def load_model():
    """Load the ML model from disk when the API starts up."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"--- Model loaded successfully from {MODEL_PATH} ---")
    except FileNotFoundError:
        print(f"--- ERROR: Model file not found at {MODEL_PATH} ---")
        # The model remains None, and endpoints will return an error.

# --- API Data Models ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOut(BaseModel):
    predicted_species: str
    prediction_index: int

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint providing API status."""
    return {
        "status": "online",
        "model_path": MODEL_PATH,
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOut)
def predict(payload: IrisInput):
    """Endpoint to make predictions on new data."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service is unavailable.")

    # Convert Pydantic model to a numpy array for the model
    input_data = np.array([[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width
    ]])
    
    # Make prediction
    try:
        prediction_index = model.predict(input_data)[0]
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(prediction_index, "unknown")
        
        return {
            "predicted_species": predicted_species,
            "prediction_index": int(prediction_index)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")