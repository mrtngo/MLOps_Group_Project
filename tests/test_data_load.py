from unittest.mock import mock_open, patch, Mock, MagicMock
import sys
import time
import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal

# Mock imports that aren't needed for testing
sys.modules['mlflow'] = Mock()
sys.modules['wandb'] = Mock()
sys.modules['mlops.utils.logger'] = Mock()

from mlops.data_load.data_load import (
    date_to_ms,
    default_window,
    fetch_data,
    fetch_binance_klines,
    fetch_binance_funding_rate,
    load_config,
    load_symbols,
)


# Existing tests (keeping them as they are)
def test_load_config_success():
    mock_yaml = "symbols:\n  - BTCUSDT\n"
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("yaml.safe_load", return_value={"symbols": ["BTCUSDT"]}):
            cfg = load_config("config.yaml")
            assert "symbols" in cfg


def test_load_config_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError()):
        with pytest.raises(FileNotFoundError):
            load_config("missing.yaml")


def test_load_config_yaml_error():
    with patch("builtins.open", mock_open(read_data="bad: [unclosed")):
        with patch("yaml.safe_load", side_effect=Exception("bad yaml")):
            with pytest.raises(Exception):
                load_config("config.yaml")


def test_date_to_ms_valid():
    assert date_to_ms("2024-01-01") == 1704067200000


def test_date_to_ms_invalid():
    with pytest.raises(Exception):
        date_to_ms("not-a-date")


def test_default_window():
    with patch("time.time", return_value=1717200000):  # fixed unix time
        start, end = default_window(days=1)
        assert end - start == 86_400_000


@patch("mlops.data_load.data_load.load_config")
def test_load_symbols_success(mock_cfg):
    mock_cfg.return_value = {"symbols": ["BTCUSDT", "ETHUSDT"]}
    symbols, cfg = load_symbols(config={"symbols": ["BTCUSDT", "ETHUSDT"]})
    assert symbols == ["BTCUSDT", "ETHUSDT"]


@patch("mlops.data_load.data_load.load_config", side_effect=Exception("fail"))
def test_load_symbols_fail(mock_cfg):
    symbols, cfg = load_symbols(config={})
    assert symbols == []
    assert cfg == {}


def test_fetch_data_against_sample():
    expected_df = pd.read_csv("data/raw/test.csv", parse_dates=["timestamp"])
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    # Patch fetch_data to return expected_df
    with patch("mlops.data_load.data_load.fetch_data", return_value=expected_df):
        actual_df = expected_df.copy()
        timestamps = expected_df["timestamp"]
        filtered_df = actual_df[actual_df["timestamp"].isin(timestamps)]
        actual_df = filtered_df.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)
        assert_frame_equal(actual_df, expected_df, rtol=1e-4, atol=1e-6)


# NEW TESTS FOR BETTER COVERAGE

# Additional tests for existing functions
def test_load_symbols_empty_symbols():
    """Test when config has empty symbols list"""
    symbols, cfg = load_symbols(config={"symbols": []})
    assert symbols == []
    assert cfg == {"symbols": []}


def test_load_symbols_no_symbols_key():
    """Test when config doesn't have symbols key"""
    symbols, cfg = load_symbols(config={"other_key": "value"})
    assert symbols == []
    assert cfg == {"other_key": "value"}


def test_default_window_custom_days():
    """Test default_window with different day values"""
    with patch("time.time", return_value=1717200000):
        start, end = default_window(days=7)
        assert end - start == 7 * 86_400_000
        
        start, end = default_window(days=30)
        assert end - start == 30 * 86_400_000


# Tests for fetch_binance_klines
@patch("mlops.data_load.data_load.requests.get")
@patch("mlops.data_load.data_load.time.sleep")
def test_fetch_binance_klines_success(mock_sleep, mock_get):
    """Test successful klines fetch"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"},
        "data_load": {"column_names": ["timestamp", "open", "high", "low", "close", "volume"]}
    }
    
    # Mock API responses - first call returns data, second call returns empty to end loop
    responses = [
        Mock(status_code=200, json=lambda: [
            [1704067200000, "50000", "51000", "49000", "50500", "100"],
            [1704070800000, "50500", "52000", "50000", "51500", "150"]
        ]),
        Mock(status_code=200, json=lambda: [])  # Empty response to end loop
    ]
    mock_get.side_effect = responses
    
    result = fetch_binance_klines("BTCUSDT", config, "2024-01-01", "2024-01-02")
    
    assert not result.empty
    assert "timestamp" in result.columns
    assert "BTCUSDT_price" in result.columns
    assert len(result) == 2
    assert result["BTCUSDT_price"].iloc[0] == 50500.0


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_klines_http_error(mock_get):
    """Test klines fetch with HTTP error"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"},
        "data_load": {"column_names": ["timestamp", "open", "high", "low", "close", "volume"]}
    }
    
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_get.return_value = mock_response
    
    result = fetch_binance_klines("BTCUSDT", config)
    
    assert result.empty or len(result.columns) == 2


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_klines_empty_response(mock_get):
    """Test klines fetch with empty response"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"},
        "data_load": {"column_names": ["timestamp", "open", "high", "low", "close", "volume"]}
    }
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response
    
    result = fetch_binance_klines("BTCUSDT", config)
    
    assert "timestamp" in result.columns
    assert "BTCUSDT_price" in result.columns
    assert len(result) == 0


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_klines_request_exception(mock_get):
    """Test klines fetch with request exception"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"}
    }
    
    mock_get.side_effect = requests.RequestException("Connection failed")
    
    result = fetch_binance_klines("BTCUSDT", config)
    
    assert result.empty or len(result.columns) == 2


def test_fetch_binance_klines_missing_config():
    """Test klines fetch with missing config keys"""
    config = {}
    
    with pytest.raises(KeyError):
        fetch_binance_klines("BTCUSDT", config)


@patch("mlops.data_load.data_load.requests.get")
@patch("mlops.data_load.data_load.time.sleep")
def test_fetch_binance_klines_pagination(mock_sleep, mock_get):
    """Test klines fetch with pagination"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"},
        "data_load": {"column_names": ["timestamp", "open", "high", "low", "close", "volume"]}
    }
    
    # Mock multiple API calls - data then empty to end pagination
    responses = [
        Mock(status_code=200, json=lambda: [
            [1704067200000, "50000", "51000", "49000", "50500", "100"]
        ]),
        Mock(status_code=200, json=lambda: [])  # Empty response to end pagination
    ]
    mock_get.side_effect = responses
    
    result = fetch_binance_klines("BTCUSDT", config, "2024-01-01", "2024-01-02")
    
    assert len(result) == 1
    assert mock_get.call_count == 2


# Tests for fetch_binance_funding_rate
@patch("mlops.data_load.data_load.requests.get")
@patch("mlops.data_load.data_load.time.sleep")
def test_fetch_binance_funding_rate_success(mock_sleep, mock_get):
    """Test successful funding rate fetch"""
    config = {
        "data_source": {"raw_path_futures": "https://fapi.binance.com/fapi/v1/fundingRate"}
    }
    
    # Mock API responses - first call returns data, second call returns empty to end loop
    responses = [
        Mock(status_code=200, json=lambda: [
            {"fundingTime": 1704067200000, "fundingRate": "0.0001"},
            {"fundingTime": 1704096000000, "fundingRate": "0.0002"}
        ]),
        Mock(status_code=200, json=lambda: [])  # Empty response to end loop
    ]
    mock_get.side_effect = responses
    
    result = fetch_binance_funding_rate("BTCUSDT", config, "2024-01-01", "2024-01-02")
    
    assert not result.empty
    assert "timestamp" in result.columns
    assert "BTCUSDT_funding_rate" in result.columns
    assert len(result) == 2
    assert result["BTCUSDT_funding_rate"].iloc[0] == 0.0001


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_funding_rate_http_error(mock_get):
    """Test funding rate fetch with HTTP error"""
    config = {
        "data_source": {"raw_path_futures": "https://fapi.binance.com/fapi/v1/fundingRate"}
    }
    
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_get.return_value = mock_response
    
    result = fetch_binance_funding_rate("BTCUSDT", config)
    
    assert result.empty or len(result.columns) == 2


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_funding_rate_empty_response(mock_get):
    """Test funding rate fetch with empty response"""
    config = {
        "data_source": {"raw_path_futures": "https://fapi.binance.com/fapi/v1/fundingRate"}
    }
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response
    
    result = fetch_binance_funding_rate("BTCUSDT", config)
    
    assert "timestamp" in result.columns
    assert "BTCUSDT_funding_rate" in result.columns
    assert len(result) == 0


def test_fetch_binance_funding_rate_missing_config():
    """Test funding rate fetch with missing config keys"""
    config = {}
    
    with pytest.raises(KeyError):
        fetch_binance_funding_rate("BTCUSDT", config)


# Tests for fetch_data integration
@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_success(mock_funding, mock_klines):
    """Test successful fetch_data with multiple symbols"""
    config = {"symbols": ["BTCUSDT", "ETHUSDT"]}
    
    # Mock klines data
    btc_price = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "BTCUSDT_price": [50000, 51000]
    })
    eth_price = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "ETHUSDT_price": [3000, 3100]
    })
    
    # Mock funding data
    btc_funding = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "BTCUSDT_funding_rate": [0.0001, 0.0002]
    })
    eth_funding = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "ETHUSDT_funding_rate": [0.0003, 0.0004]
    })
    
    mock_klines.side_effect = [btc_price, eth_price]
    mock_funding.side_effect = [btc_funding, eth_funding]
    
    result = fetch_data(config, "2024-01-01", "2024-01-02")
    
    assert not result.empty
    assert len(result.columns) == 5  # timestamp + 2 prices + 2 funding rates
    assert "BTCUSDT_price" in result.columns
    assert "ETHUSDT_price" in result.columns
    assert "BTCUSDT_funding_rate" in result.columns
    assert "ETHUSDT_funding_rate" in result.columns


@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_partial_failure(mock_funding, mock_klines):
    """Test fetch_data when some symbols fail"""
    config = {"symbols": ["BTCUSDT", "ETHUSDT"]}
    
    # First symbol succeeds
    btc_price = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "BTCUSDT_price": [50000]
    })
    btc_funding = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "BTCUSDT_funding_rate": [0.0001]
    })
    
    # Second symbol fails
    mock_klines.side_effect = [btc_price, Exception("API Error")]
    mock_funding.side_effect = [btc_funding, Exception("API Error")]
    
    result = fetch_data(config)
    
    assert not result.empty
    assert "BTCUSDT_price" in result.columns
    assert "BTCUSDT_funding_rate" in result.columns
    assert "ETHUSDT_price" not in result.columns


@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_no_symbols(mock_funding, mock_klines):
    """Test fetch_data with no symbols in config"""
    config = {"symbols": []}
    
    result = fetch_data(config)
    
    assert result.empty
    mock_klines.assert_not_called()
    mock_funding.assert_not_called()


@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_only_price_data(mock_funding, mock_klines):
    """Test fetch_data when only price data is available"""
    config = {"symbols": ["BTCUSDT"]}
    
    price_data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "BTCUSDT_price": [50000]
    })
    funding_data = pd.DataFrame()  # Empty funding data
    
    mock_klines.return_value = price_data
    mock_funding.return_value = funding_data
    
    result = fetch_data(config)
    
    assert not result.empty
    assert "BTCUSDT_price" in result.columns
    assert len(result.columns) == 2  # timestamp + price


@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_only_funding_data(mock_funding, mock_klines):
    """Test fetch_data when only funding data is available"""
    config = {"symbols": ["BTCUSDT"]}
    
    price_data = pd.DataFrame()  # Empty price data
    funding_data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "BTCUSDT_funding_rate": [0.0001]
    })
    
    mock_klines.return_value = price_data
    mock_funding.return_value = funding_data
    
    result = fetch_data(config)
    
    assert not result.empty
    assert "BTCUSDT_funding_rate" in result.columns
    assert len(result.columns) == 2  # timestamp + funding rate


@patch("mlops.data_load.data_load.fetch_binance_klines")
@patch("mlops.data_load.data_load.fetch_binance_funding_rate")
def test_fetch_data_all_failures(mock_funding, mock_klines):
    """Test fetch_data when all symbols fail"""
    config = {"symbols": ["BTCUSDT", "ETHUSDT"]}
    
    mock_klines.side_effect = Exception("API Error")
    mock_funding.side_effect = Exception("API Error")
    
    result = fetch_data(config)
    
    assert result.empty


@patch("mlops.data_load.data_load.load_symbols")
def test_fetch_data_load_symbols_exception(mock_load_symbols):
    """Test fetch_data when load_symbols raises exception"""
    mock_load_symbols.side_effect = Exception("Config error")
    
    with pytest.raises(Exception):
        fetch_data({})


# Edge case tests
def test_date_to_ms_edge_cases():
    """Test date_to_ms with various edge cases"""
    # Test with different date formats that should work
    assert date_to_ms("2024-12-31") == 1735603200000
    
    # Test with empty string
    with pytest.raises(Exception):
        date_to_ms("")
    
    # Test with None
    with pytest.raises(Exception):
        date_to_ms(None)


def test_default_window_edge_cases():
    """Test default_window with edge cases"""
    with patch("time.time", return_value=1717200000):
        # Test with zero days
        start, end = default_window(days=0)
        assert start == end
        
        # Test with negative days (should still work)
        start, end = default_window(days=-1)
        assert end - start == -86_400_000


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_klines_json_decode_error(mock_get):
    """Test klines fetch with JSON decode error"""
    config = {
        "data_source": {"raw_path_spot": "https://api.binance.com/api/v3/klines"}
    }
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response
    
    result = fetch_binance_klines("BTCUSDT", config)
    
    assert result.empty or len(result.columns) == 2


@patch("mlops.data_load.data_load.requests.get")
def test_fetch_binance_funding_rate_json_decode_error(mock_get):
    """Test funding rate fetch with JSON decode error"""
    config = {
        "data_source": {"raw_path_futures": "https://fapi.binance.com/fapi/v1/fundingRate"}
    }
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response
    
    result = fetch_binance_funding_rate("BTCUSDT", config)
    
    assert result.empty or len(result.columns) == 2