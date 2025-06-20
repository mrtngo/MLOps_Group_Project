from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import os
import pickle

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    ETHUSDT_price: float
    BNBUSDT_price: float
    XRPUSDT_price: float
    ADAUSDT_price: float
    SOLUSDT_price: float
    BTCUSDT_funding_rate: float
    ETHUSDT_funding_rate: float
    BNBUSDT_funding_rate: float
    XRPUSDT_funding_rate: float
    ADAUSDT_funding_rate: float
    SOLUSDT_funding_rate: float

# Define the feature sets for the models
ALL_FEATURES = [
    "ETHUSDT_price", "BNBUSDT_price", "XRPUSDT_price", "ADAUSDT_price", "SOLUSDT_price",
    "BTCUSDT_funding_rate", "ETHUSDT_funding_rate", "BNBUSDT_funding_rate",
    "XRPUSDT_funding_rate", "ADAUSDT_funding_rate", "SOLUSDT_funding_rate"
]
REG_FEATURES = [
    'ETHUSDT_price', 'BNBUSDT_price', 'XRPUSDT_price', 'ADAUSDT_price', 'SOLUSDT_price', 
    'ETHUSDT_funding_rate', 'XRPUSDT_funding_rate', 'ADAUSDT_funding_rate', 
    'SOLUSDT_funding_rate', 'BTCUSDT_funding_rate'
]
CLASS_FEATURES = [
    'BNBUSDT_price', 'XRPUSDT_price', 'ADAUSDT_price', 
    'ETHUSDT_funding_rate', 'ADAUSDT_funding_rate'
]

# Load the trained models and preprocessing pipeline
try:
    reg_model = joblib.load("models/linear_regression.pkl")
    class_model = joblib.load("models/logistic_regression.pkl")
    with open("models/preprocessing_pipeline.pkl", "rb") as f:
        preprocessor_pipeline = pickle.load(f)
except FileNotFoundError:
    reg_model = None
    class_model = None
    preprocessor_pipeline = None

app = FastAPI(
    title="Crypto Price Prediction API",
    description="An API to predict cryptocurrency price direction and value.",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    """Ensure models and preprocessor are loaded at startup."""
    global reg_model, class_model, preprocessor_pipeline
    if not all([reg_model, class_model, preprocessor_pipeline]):
        try:
            reg_model = joblib.load("models/linear_regression.pkl")
            class_model = joblib.load("models/logistic_regression.pkl")
            with open("models/preprocessing_pipeline.pkl", "rb") as f:
                preprocessor_pipeline = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError("Models or preprocessor not found. Please train the models first.")

@app.get("/")
def read_root():
    """Welcome message for the API root."""
    return {"message": "Welcome to the Crypto Price Prediction API"}

@app.get("/health")
def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predicts the BTC price and direction based on input features.
    """
    if not all([reg_model, class_model, preprocessor_pipeline]):
        raise HTTPException(status_code=503, detail="Models or preprocessor are not loaded")

    try:
        # Extract the scaler and feature lists from the loaded pipeline
        scaler = preprocessor_pipeline['scaler']
        all_features = preprocessor_pipeline['all_feature_cols']
        reg_features = preprocessor_pipeline['selected_features_reg']
        class_features = preprocessor_pipeline['selected_features_class']

        # Convert input data to a DataFrame and ensure correct column order
        input_df = pd.DataFrame([input_data.dict()], columns=all_features)
        
        # Preprocess the data (e.g., scale it)
        scaled_data = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_data, columns=all_features)
        
        # Select features for each model
        reg_input_df = scaled_df[reg_features]
        class_input_df = scaled_df[class_features]

        # Predict using the regression and classification models
        price_prediction = reg_model.predict(reg_input_df)[0]
        direction_prediction = class_model.predict(class_input_df)[0]
        direction_probability = class_model.predict_proba(class_input_df)[0][1]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "predicted_btc_price": float(price_prediction),
        "predicted_price_direction": int(direction_prediction),
        "prediction_probability": float(direction_probability),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 