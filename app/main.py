from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import os
import pickle
import io

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

def perform_prediction(input_df: pd.DataFrame, preprocessor_dict: dict, reg_model, class_model):
    """Helper function to perform preprocessing and prediction."""
    scaler = preprocessor_dict['scaler']
    selected_features_reg = preprocessor_dict['selected_features_reg']
    selected_features_class = preprocessor_dict['selected_features_class']

    # Ensure columns are in the correct order
    input_df = input_df[preprocessor_dict['all_feature_cols']]

    # Scale the features
    scaled_features = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_features, columns=input_df.columns, index=input_df.index)

    # Select features for each model
    X_reg = scaled_df[selected_features_reg]
    X_class = scaled_df[selected_features_class]

    # Make predictions
    price_prediction = reg_model.predict(X_reg)
    direction_prediction = class_model.predict(X_class)
    direction_probability = class_model.predict_proba(X_class)[:, 1]

    return {
        "price_prediction": price_prediction.tolist(),
        "direction_prediction": direction_prediction.tolist(),
        "direction_probability": direction_probability.tolist()
    }

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
    """Endpoint for single predictions."""
    if not all([reg_model, class_model, preprocessor_pipeline]):
        raise HTTPException(status_code=503, detail="Models or preprocessor not loaded")

    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Use the helper function for prediction
        predictions = perform_prediction(input_df, preprocessor_pipeline, reg_model, class_model)

        # Since this is a single prediction, return the first element of the lists
        return {
            "price_prediction": predictions["price_prediction"][0],
            "direction_prediction": predictions["direction_prediction"][0],
            "direction_probability": predictions["direction_probability"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Endpoint for batch predictions from a CSV file."""
    if not all([reg_model, class_model, preprocessor_pipeline]):
        raise HTTPException(status_code=503, detail="Models or preprocessor not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        # Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Use the helper function for prediction
        predictions = perform_prediction(input_df, preprocessor_pipeline, reg_model, class_model)
        
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 