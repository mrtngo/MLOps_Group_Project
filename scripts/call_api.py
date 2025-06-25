import argparse
import json
import os

import requests


def call_predict_api(url: str, input_data):
    """Sends a request to the /predict endpoint."""
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(input_data), headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if e.response:
            print(f"Response: {e.response.text}")
        return None


def call_predict_batch_api(url: str, file_path: str):
    """Sends a request to the /predict_batch endpoint with a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at {file_path}")
        return None

    files = {"file": (os.path.basename(file_path), open(file_path, "rb"), "text/csv")}

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if e.response:
            print(f"Response: {e.response.text}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the prediction API.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="URL of the prediction endpoint.",
    )
    parser.add_argument(
        "--input", type=str, help="Path to input CSV file for batch prediction."
    )

    args = parser.parse_args()

    if args.input:
        # Batch prediction mode
        if "predict_batch" not in args.url:
            print(
                "Warning: URL does not seem to point to a _batch endpoint. Adjusting..."
            )
            args.url = args.url.replace("/predict", "/predict_batch")

        print(f"Requesting batch prediction from {args.url} with file {args.input}")
        prediction = call_predict_batch_api(args.url, args.input)
    else:
        # Single prediction mode
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
            "SOLUSDT_funding_rate": 0.0001,
        }
        print(f"Requesting single prediction from {args.url}")
        prediction = call_predict_api(args.url, sample_data)

    if prediction:
        print("\nAPI Response:")
        print(json.dumps(prediction, indent=2))
