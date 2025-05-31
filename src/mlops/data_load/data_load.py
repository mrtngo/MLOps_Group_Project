import requests
import time
import yaml 
import pandas as pd
from typing import Dict, Optional

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
        columns=cfg.get("data_load", {}).get("column_names", [])
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