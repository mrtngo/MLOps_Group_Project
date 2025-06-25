from unittest.mock import mock_open, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlops.data_load.data_load import (
    date_to_ms,
    default_window,
    fetch_data,
    load_config,
    load_symbols,
)


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
