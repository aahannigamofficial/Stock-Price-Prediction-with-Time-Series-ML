from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description="ML model API for predicting stock prices with LSTM",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

print("Loading LSTM model...")
model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse')
scaler = joblib.load('data/scaler_lstm.pkl')
metrics = joblib.load('models/lstm_metrics.pkl')

print("✓ LSTM model loaded successfully!")

# Model configuration
MODEL_METRICS = {
    "model_name": "LSTM (Multi-Stock)",
    "mae": float(metrics['mae']),
    "rmse": float(metrics['rmse']),
    "r2": float(metrics['r2']),
    "mape": float(metrics['mape']),
    "directional_accuracy": float(metrics['directional_accuracy']),
    "num_features": 13,
    "lookback": 30
}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    features: list  # 13 features (30-day sequence will be created)

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_range: float
    upper_bound: float
    lower_bound: float
    status: str = "success"

class ModelInfo(BaseModel):
    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    num_features: int
    lookback: int

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "LSTM Stock Price Prediction API v2.0",
        "endpoints": {
            "info": "/info - Get model information",
            "predict": "/predict - Make a prediction",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/info", response_model=ModelInfo)
def get_model_info():
    """Get model metrics and information"""
    return ModelInfo(**MODEL_METRICS)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a prediction given features
    For LSTM, we need a sequence of features
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
        
        # For simplicity, create a sequence by repeating the features 30 times
        # In production, you'd have a proper 30-day sequence
        features_array = np.array([request.features])
        features_scaled = scaler.transform(features_array)
        
        # Create sequence (30 days, but we're using the same scaled features)
        sequence = np.repeat(features_scaled, MODEL_METRICS["lookback"], axis=0).reshape(1, MODEL_METRICS["lookback"], MODEL_METRICS["num_features"])
        
        # Make prediction
        prediction = model.predict(sequence, verbose=0)[0][0]
        
        # Calculate confidence bounds
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

@app.get("/test-predict")
def test_predict():
    """Test prediction endpoint"""
    sample_features = [187.5, 189.2, 186.1, 185.5, 65000000, 186.5, 0.5, 187.0, 55.2, 0.08, 0.06, 0.02, 195.0]
    request = PredictionRequest(features=sample_features)
    return predict(request)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Note: using port 8001 to avoid conflict with v1