import requests
import pandas as pd
import time
import yaml
import os
from typing import Dict
from datetime import datetime, timedelta


def load_config(config_path: str) -> Dict:
    """
    Load configuration schema from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary containing schema and settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config("config.yaml")

BASE_URL = "https://api.binance.com"  # Spot API for prices
FUTURES_URL = "https://fapi.binance.com"  # Futures API for funding rates

# SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT"]
SYMBOLS = config.get("symbols", [])


def fetch_binance_klines(symbol, interval="8h", days=365):
    # url = f"{BASE_URL}/api/v3/klines"
    url = config.get('data_source', {}).get("raw_path_spot")
    end_time = int(time.time() * 1000)  # Current time in ms
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # 1 year ago in ms

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000  # Max limit per request
    }
    
    all_data = []
    while start_time < end_time:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            start_time = int(data[-1][0]) + 1  # Next batch
            params["startTime"] = start_time
            time.sleep(0.5)  # Avoid rate limits
        else:
            print(f"Error fetching {symbol} klines: {response.status_code}, {response.text}")
            break
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                         "close_time", "quote_volume", "trades", "taker_base", 
                                         "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].rename(columns={"close": f"{symbol}_price"})


def fetch_binance_funding_rate(symbol, days=365):
    """Fetch historical funding rates from Binance Futures."""
    # url = f"{FUTURES_URL}/fapi/v1/fundingRate"
    url = config.get('data_source', {}).get("raw_path_futures")
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    params = {
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    
    all_data = []
    while start_time < end_time:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            start_time = int(data[-1]["fundingTime"]) + 1
            params["startTime"] = start_time
            time.sleep(0.5)
        else:
            print(f"Error fetching {symbol} funding: {response.status_code}, {response.text}")
            break
    
    # Convert to DataFrame (timestamp, funding rate)
    df = pd.DataFrame(all_data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["fundingTime", "fundingRate"]].rename(columns={"fundingTime": "timestamp", "fundingRate": f"{symbol}_funding_rate"})


def fetch_data():
    # Fetch and combine data
    price_dfs = []
    funding_dfs = []
    for symbol in SYMBOLS:
        print(f"Fetching data for {symbol}...")
        price_df = fetch_binance_klines(symbol)
        funding_df = fetch_binance_funding_rate(symbol)
        price_dfs.append(price_df)
        funding_dfs.append(funding_df)
    # Merge price data
    combined_price = price_dfs[0]
    for df in price_dfs[1:]:
        combined_price = combined_price.merge(df, on="timestamp", how="outer")
    # Merge funding rate data
    combined_funding = funding_dfs[0]
    for df in funding_dfs[1:]:
        combined_funding = combined_funding.merge(df, on="timestamp", how="outer")

    # Merge prices and funding rates (align timestamps)
    data = combined_price.merge(combined_funding, on="timestamp", how="inner")
    data = data.dropna()  # Drop rows with missing values
    print(data)
    data.to_csv(config.get('data_source', {}).get("processed_path"), index=False)
    return data


fetch_data()