# import requests
# import pandas as pd
# import time
# import yaml
# import os
# from typing import Dict
# from datetime import datetime, timedelta


# def load_config(config_path: str) -> Dict:
#     """
#     Load configuration schema from a YAML file.

#     Args:
#         config_path (str): Path to the YAML configuration file.

#     Returns:
#         dict: Parsed configuration dictionary containing schema and settings.
#     """
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# config = load_config("config.yaml")

# SYMBOLS = config.get("symbols", [])


# def fetch_binance_klines(symbol, start_time, end_time, interval="8h", days=365):
#     url = config.get('data_source', {}).get("raw_path_spot")
#     end_time = int(time.time() * 1000)  # Current time in ms
#     start_time = end_time - (days * 24 * 60 * 60 * 1000)  # 1 year ago in ms

#     params = {
#         "symbol": symbol,
#         "interval": interval,
#         "startTime": start_time,
#         "endTime": end_time,
#         "limit": 1000  # Max limit per request
#     }
    
#     all_data = []
#     while start_time < end_time:
#         response = requests.get(url, params=params)
#         if response.status_code == 200:
#             data = response.json()
#             if not data:
#                 break
#             all_data.extend(data)
#             start_time = int(data[-1][0]) + 1  # Next batch
#             params["startTime"] = start_time
#             time.sleep(0.5)  # Avoid rate limits
#         else:
#             print(f"Error fetching {symbol} klines: {response.status_code}, {response.text}")
#             break
    
#     df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", 
#                                          "close_time", "quote_volume", "trades", "taker_base", 
#                                          "taker_quote", "ignore"])
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df["close"] = df["close"].astype(float)
#     return df[["timestamp", "close"]].rename(columns={"close": f"{symbol}_price"})


# def fetch_binance_funding_rate(symbol, days=365):
#     """Fetch historical funding rates from Binance Futures."""
#     # url = f"{FUTURES_URL}/fapi/v1/fundingRate"
#     url = config.get('data_source', {}).get("raw_path_futures")
#     end_time = int(time.time() * 1000)
#     start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
#     params = {
#         "symbol": symbol,
#         "startTime": start_time,
#         "endTime": end_time,
#         "limit": 1000
#     }
    
#     all_data = []
#     while start_time < end_time:
#         response = requests.get(url, params=params)
#         if response.status_code == 200:
#             data = response.json()
#             if not data:
#                 break
#             all_data.extend(data)
#             start_time = int(data[-1]["fundingTime"]) + 1
#             params["startTime"] = start_time
#             time.sleep(0.5)
#         else:
#             print(f"Error fetching {symbol} funding: {response.status_code}, {response.text}")
#             break
    
#     # Convert to DataFrame (timestamp, funding rate)
#     df = pd.DataFrame(all_data)
#     df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
#     df["fundingRate"] = df["fundingRate"].astype(float)
#     return df[["fundingTime", "fundingRate"]].rename(columns={"fundingTime": "timestamp", "fundingRate": f"{symbol}_funding_rate"})


# def fetch_data():
#     # Fetch and combine data
#     price_dfs = []
#     funding_dfs = []
#     for symbol in SYMBOLS:
#         print(f"Fetching data for {symbol}...")
#         price_df = fetch_binance_klines(symbol)
#         funding_df = fetch_binance_funding_rate(symbol)
#         price_dfs.append(price_df)
#         funding_dfs.append(funding_df)
#     # Merge price data
#     combined_price = price_dfs[0]
#     for df in price_dfs[1:]:
#         combined_price = combined_price.merge(df, on="timestamp", how="outer")
#     # Merge funding rate data
#     combined_funding = funding_dfs[0]
#     for df in funding_dfs[1:]:
#         combined_funding = combined_funding.merge(df, on="timestamp", how="outer")

#     # Merge prices and funding rates (align timestamps)
#     data = combined_price.merge(combined_funding, on="timestamp", how="inner")
#     data = data.dropna()  # Drop rows with missing values
#     print(data)
#     # data.to_csv(config.get('data_source', {}).get("processed_path"), index=False)
#     return data


import requests
import time
import yaml 
import pandas as pd
from typing import Dict, Optional
# from datetime import datetime, timedelta

# ───────────────────────────── helpers ──────────────────────────────


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def date_to_ms(date_str: str) -> int:
    """
    Convert a YYYY-MM-DD string to Unix epoch milliseconds (UTC).
    """
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def default_window(days: int = 365) -> tuple[int, int]:
    """
    Last `days` expressed as (start_ms, end_ms).
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    return start_ms, end_ms


cfg = load_config("config.yaml")
SYMBOLS = cfg.get("symbols", [])


def fetch_binance_klines(
    symbol: str,
    start_date: Optional[str] = None,   # ← YYYY-MM-DD  or None
    end_date:   Optional[str] = None,   # ← YYYY-MM-DD  or None
    interval: str = "8h",
    days: int = 365
):
    url = cfg["data_source"]["raw_path_spot"]

    if start_date and end_date:
        start_ms = date_to_ms(start_date)
        # include the full end-day by adding 1 day − 1 ms
        end_ms = date_to_ms(end_date) + 86_400_000 - 1
    else:
        start_ms, end_ms = default_window(days)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit": 1000
    }

    rows = []
    while params["startTime"] < end_ms:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"[{symbol}] HTTP {r.status_code}: {r.text}")
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        params["startTime"] = batch[-1][0] + 1   # next candle
        time.sleep(0.4)                          # anti-429

    df = pd.DataFrame(
        rows,
        columns=["timestamp","open","high","low","close","volume",
                 "close_time","quote_volume","trades",
                 "taker_base","taker_quote","ignore"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].rename(columns={"close": f"{symbol}_price"})


def fetch_binance_funding_rate(
    symbol: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    days: int = 365
):
    url = cfg["data_source"]["raw_path_futures"]

    if start_date and end_date:
        start_ms = date_to_ms(start_date)
        end_ms = date_to_ms(end_date) + 86_400_000 - 1
    else:
        start_ms, end_ms = default_window(days)

    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit": 1000
    }

    rows = []
    while params["startTime"] < params["endTime"]:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"[{symbol}] HTTP {r.status_code}: {r.text}")
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        params["startTime"] = batch[-1]["fundingTime"] + 1
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "fundingRate"]].rename(
        columns={"fundingRate": f"{symbol}_funding_rate"}
    )


def fetch_data(start_date: Optional[str] = None,
               end_date:   Optional[str] = None):
    price_dfs, funding_dfs = [], []

    for symbol in SYMBOLS:
        print(f"Fetching {symbol} …", end=" ", flush=True)
        price_dfs.append(fetch_binance_klines(symbol, start_date, end_date))
        funding_dfs.append(fetch_binance_funding_rate(symbol, start_date, end_date))
        print("✓")

    # merge helpers
    def merge_frames(frames):
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="timestamp", how="outer")
        return out
    
    combined_price = merge_frames(price_dfs)
    combined_funding = merge_frames(funding_dfs)
    data = combined_price.merge(combined_funding, on="timestamp", how="inner")
    return data

df = fetch_data("2023-01-01", "2023-12-31")
print(df.head(1000))