from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load scaler
scaler = joblib.load('data/scaler.pkl')
# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Stock Price Prediction API",
    description="ML model API for predicting stock prices",
    version="1.0.0"
)

# Enable CORS (allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL AND METRICS
# ============================================================================

print("Loading trained model...")
model = joblib.load('data/model1.pkl')
print("✓ Model loaded successfully!")

# Model metrics from evaluation
MODEL_METRICS = {
    "model_name": "Linear Regression",
    "mae": 3.42,
    "rmse": 4.59,
    "r2": 0.9413,
    "mape": 1.25,
    "directional_accuracy": 49.09,
    "num_features": 15
}

# ============================================================================
# DEFINE REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Input features for prediction"""
    features: list  # 15 features from the preprocessed data

class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_price: float
    confidence_range: float
    upper_bound: float
    lower_bound: float
    status: str = "success"

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    num_features: int

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Stock Price Prediction API is running!",
        "endpoints": {
            "info": "/info - Get model information",
            "predict": "/predict - Make a prediction",
            "docs": "/docs - API documentation"
        }
    }

# ============================================================================
# INFO ENDPOINT
# ============================================================================

@app.get("/info", response_model=ModelInfo)
def get_model_info():
    """
    Get model metrics and information
    
    Returns:
        ModelInfo: Model name, MAE, RMSE, R², MAPE, directional accuracy
    """
    return ModelInfo(**MODEL_METRICS)

# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a prediction given features
    
    Args:
        request: PredictionRequest with 15 features
        
    Returns:
        PredictionResponse: Predicted price with confidence bounds
        
    Example:
        POST /predict
        {
            "features": [180.5, 182.0, 185.0, 179.0, 50000000, ...]
        }
    """
    try:
        # Validate input
        if len(request.features) != MODEL_METRICS["num_features"]:
            return {
                "predicted_price": 0,
                "confidence_range": 0,
                "upper_bound": 0,
                "lower_bound": 0,
                "status": f"error: expected {MODEL_METRICS['num_features']} features, got {len(request.features)}"
            }
        
        # Convert to numpy array
        features_array = scaler.transform(np.array([request.features]))
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Calculate confidence bounds (using MAE as uncertainty estimate)
        confidence_range = MODEL_METRICS["mae"]
        upper_bound = prediction + confidence_range
        lower_bound = prediction - confidence_range
        
        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            confidence_range=round(confidence_range, 2),
            upper_bound=round(upper_bound, 2),
            lower_bound=round(lower_bound, 2),
            status="success"
        )
        
    except Exception as e:
        return {
            "predicted_price": 0,
            "confidence_range": 0,
            "upper_bound": 0,
            "lower_bound": 0,
            "status": f"error: {str(e)}"
        }

# ============================================================================
# TEST DATA ENDPOINT (for frontend testing)
# ============================================================================

@app.get("/test-predict")
def test_predict():
    """
    Test endpoint that returns a sample prediction
    Useful for frontend testing without sending real data
    """
    # Sample feature values (you can modify these)
    sample_features = [
        187.5,      # close
        189.2,      # high
        186.1,      # low
        185.5,      # open
        65000000,   # volume
        186.5,      # sma_5
        0.5,        # daily_return
        187.0,      # close_lag1
        55.2,       # rsi_14
        0.08,       # macd
        0.06,       # macd_signal
        0.02,       # macd_histogram
        195.0,      # bb_upper
        188.0,      # bb_middle
        181.0       # bb_lower
    ]
    
    request = PredictionRequest(features=sample_features)
    return predict(request)

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.get("/error-test")
def error_test():
    """Test error handling"""
    return {
        "error": "This is a test error response",
        "status": "error"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)