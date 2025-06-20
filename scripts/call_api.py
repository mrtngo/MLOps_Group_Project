import requests
import json
import pandas as pd
import argparse

def call_predict_api(input_data):
    """Sends a request to the /predict endpoint."""
    url = "http://127.0.0.1:8000/predict"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, data=json.dumps(input_data), headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: API returned status code {response.status_code}")
            try:
                # Try to print the detailed error from the API response
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Non-JSON response from API", "content": response.text}
                
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Sample input data (you can replace this with your own)
    sample_data = {
        "ETHUSDT_price": 1800.0,
        "BNBUSDT_price": 300.0,
        "XRPUSDT_price": 0.5,
        "ADAUSDT_price": 0.3,
        "SOLUSDT_price": 25.0,
        "BTCUSDT_funding_rate": 0.0001,
        "ETHUSDT_funding_rate": 0.0001,
        "BNBUSDT_funding_rate": 0.0001,
        "XRPUSDT_funding_rate": 0.0001,
        "ADAUSDT_funding_rate": 0.0001,
        "SOLUSDT_funding_rate": 0.0001
    }
    
    prediction = call_predict_api(sample_data)
    print("API Response:")
    print(json.dumps(prediction, indent=2)) 