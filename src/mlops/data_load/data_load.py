import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import requests
import yaml

import wandb

# ───────────────────────────── setup logging ──────────────────────────────

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ───────────────────────────── helpers ──────────────────────────────


def load_config(path: str) -> Dict:
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
            logger.info(f"Config loaded from {path}")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def date_to_ms(date_str: str) -> int:
    """
    Convert a YYYY-MM-DD string to Unix epoch milliseconds (UTC).
    """
    try:
        return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)
    except Exception as e:
        logger.error(f"Error converting date '{date_str}' to ms: {e}")
        raise


def default_window(days: int = 365) -> tuple[int, int]:
    """
    Last `days` expressed as (start_ms, end_ms).
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    logger.debug(f"Default window: {days} days ({start_ms} to {end_ms})")
    return start_ms, end_ms


def load_symbols(config: Dict) -> Tuple[List[str], Dict]:
    """Load symbols from config with error handling."""
    try:
        symbols = config.get("symbols", [])
        if not symbols:
            logger.warning("No symbols found in config")
        else:
            logger.info(f"Loaded {len(symbols)} symbols: {symbols}")
        return symbols, config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return [], {}


def fetch_binance_klines(
    symbol: str,
    config: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "8h",
    days: int = 365,
):
    try:
        url = config["data_source"]["raw_path_spot"]
        logger.info(f"Fetching klines for {symbol} (interval: {interval})")

        if start_date and end_date:
            start_ms = date_to_ms(start_date)
            # include the full end-day by adding 1 day − 1 ms
            end_ms = date_to_ms(end_date) + 86_400_000 - 1
            logger.info(f"Date range: {start_date} to {end_date}")
        else:
            start_ms, end_ms = default_window(days)
            logger.info(f"Using default window: {days} days")

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        rows = []
        request_count = 0

        while params["startTime"] < end_ms:
            try:
                request_count += 1
                r = requests.get(url, params=params, timeout=10)

                if r.status_code != 200:
                    logger.error(f"[{symbol}] HTTP {r.status_code}: {r.text}")
                    break

                batch = r.json()
                if not batch:
                    logger.info(f"[{symbol}] No more data available")
                    break

                rows.extend(batch)
                params["startTime"] = batch[-1][0] + 1  # next candle

                if request_count % 10 == 0:
                    msg = f"[{symbol}] Fetched {len(rows)} records so far"
                    logger.debug(msg)

                time.sleep(0.4)  # anti-429

            except requests.RequestException as e:
                logger.error(f"[{symbol}] Request failed: {e}")
                break
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected error during fetch: {e}")
                break

        if not rows:
            logger.warning(f"[{symbol}] No data fetched")
            return pd.DataFrame(columns=["timestamp", f"{symbol}_price"])

        logger.info(f"[{symbol}] Successfully fetched {len(rows)} klines")

        df = pd.DataFrame(
            rows, columns=config.get("data_load", {}).get("column_names", [])
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        return df[["timestamp", "close"]].rename(columns={"close": f"{symbol}_price"})

    except KeyError as e:
        logger.error(f"Missing config key: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        raise


def fetch_binance_funding_rate(
    symbol: str,
    config: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 365,
):
    try:
        url = config["data_source"]["raw_path_futures"]
        logger.info(f"Fetching funding rates for {symbol}")

        if start_date and end_date:
            start_ms = date_to_ms(start_date)
            end_ms = date_to_ms(end_date) + 86_400_000 - 1
        else:
            start_ms, end_ms = default_window(days)

        params = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        rows = []
        request_count = 0

        while params["startTime"] < params["endTime"]:
            try:
                request_count += 1
                r = requests.get(url, params=params, timeout=10)

                if r.status_code != 200:
                    logger.error(f"[{symbol}] HTTP {r.status_code}: {r.text}")
                    break

                batch = r.json()
                if not batch:
                    logger.info(f"[{symbol}] No more funding data available")
                    break

                rows.extend(batch)
                params["startTime"] = batch[-1]["fundingTime"] + 1

                if request_count % 10 == 0:
                    msg = f"[{symbol}] Fetched {len(rows)} funding records"
                    logger.debug(f"{msg} so far")

                time.sleep(0.4)

            except requests.RequestException as e:
                logger.error(f"[{symbol}] Request failed: {e}")
                break
            except Exception as e:
                msg = f"[{symbol}] Unexpected error during funding fetch: {e}"
                logger.error(msg)
                break

        if not rows:
            logger.warning(f"[{symbol}] No funding data fetched")
            columns = ["timestamp", f"{symbol}_funding_rate"]
            return pd.DataFrame(columns=columns)

        msg = f"[{symbol}] Successfully fetched {len(rows)} funding rates"
        logger.info(msg)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df[["timestamp", "fundingRate"]].rename(
            columns={"fundingRate": f"{symbol}_funding_rate"}
        )

    except KeyError as e:
        logger.error(f"Missing config key: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching funding rates for {symbol}: {e}")
        raise


def fetch_data(
    config: Dict, start_date: Optional[str] = None, end_date: Optional[str] = None
):
    try:
        print(f"start date {start_date}")
        logger.info("Starting data fetch process")

        # Load symbols from config
        symbols, _ = load_symbols(config)

        price_dfs, funding_dfs = [], []
        failed_symbols = []

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                price_df = fetch_binance_klines(symbol, config, start_date, end_date)
                funding_df = fetch_binance_funding_rate(
                    symbol, config, start_date, end_date
                )

                if not price_df.empty:
                    price_dfs.append(price_df)
                if not funding_df.empty:
                    funding_dfs.append(funding_df)

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")

        if not price_dfs and not funding_dfs:
            logger.error("No data fetched for any symbol")
            return pd.DataFrame()

        def merge_frames(frames):
            if not frames:
                return pd.DataFrame()
            result = frames[0]
            for frame in frames[1:]:
                result = result.merge(frame, on="timestamp", how="outer")
            return result.sort_values("timestamp").reset_index(drop=True)

        price_data = merge_frames(price_dfs)
        funding_data = merge_frames(funding_dfs)

        if price_data.empty and funding_data.empty:
            logger.error("No data available after merging")
            return pd.DataFrame()

        if not price_data.empty and not funding_data.empty:
            final_data = price_data.merge(funding_data, on="timestamp", how="outer")
        elif not price_data.empty:
            final_data = price_data
        else:
            final_data = funding_data

        logger.info(f"Final dataset shape: {final_data.shape}")
        return final_data

    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        raise
