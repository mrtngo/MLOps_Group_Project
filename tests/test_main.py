import pytest
import pandas as pd
import sys
from mlops import main


@pytest.fixture
def dummy_df():
    """Create dummy input dataframe for testing"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=5),
        "BTCUSDT_price": [100, 101, 102, 103, 104],
        "ETHUSDT_price": [50, 51, 52, 53, 54],
        "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.018],
        "price_direction": [1, 0, 1, 1, 0]
    })


def test_main_all_stage(monkeypatch, dummy_df):
    """
    Test the full pipeline execution using --stage=all.
    All functions are mocked to isolate main logic.
    """
    monkeypatch.setattr(
        "mlops.main.fetch_data", lambda start_date, end_date: dummy_df
    )
    monkeypatch.setattr("mlops.main.load_config", lambda _: {
        "data_validation": {"schema": {"columns": []}},
        "data_source": {"processed_path": "tests/tmp/test_output.csv"}
    })
    monkeypatch.setattr(
        "mlops.main.validate_data", lambda df, schema, logger, ms, oe: df
    )
    feature_return = (
        ["BTCUSDT_price", "ETHUSDT_price", "BTCUSDT_funding_rate"],
        "BTCUSDT_price"
    )
    monkeypatch.setattr(
        "mlops.main.define_features_and_label", lambda: feature_return
    )
    monkeypatch.setattr(
        "mlops.main.create_price_direction_label", lambda df, label: df
    )
    monkeypatch.setattr(
        "mlops.main.train_model", lambda df: (
            "regression_model", "classification_model")
    )
    metrics_return = (
        {"RMSE": 1.23}, {"Accuracy": 0.9, "ROC AUC": 0.88}
    )
    monkeypatch.setattr(
        "mlops.main.evaluate_models", lambda df: metrics_return)
    monkeypatch.setattr("mlops.main.setup_logger", lambda: None)

    test_args = ["main", "--stage", "all"]
    monkeypatch.setattr(sys, "argv", test_args)

    main.main()  # Should complete without error


def test_main_infer_stage(monkeypatch, dummy_df):
    """
    Test the inference mode using --stage=infer and ensuring inference runs.
    """
    monkeypatch.setattr(
        "mlops.main.fetch_data", lambda start_date, end_date: dummy_df
    )
    monkeypatch.setattr("mlops.main.load_config", lambda _: {
        "data_validation": {"schema": {"columns": []}}
    })
    monkeypatch.setattr(
        "mlops.main.run_inference", lambda df, config, path: None
    )
    monkeypatch.setattr("mlops.main.setup_logger", lambda: None)

    test_args = [
        "main", "--stage", "infer", "--output-csv", "tests/tmp/predictions.csv"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main.main()  # Should complete without error


def test_main_infer_stage_missing_output(monkeypatch):
    """
    Test if the script exits with error when --output-csv is not provided
    for inference.
    """
    monkeypatch.setattr("mlops.main.setup_logger", lambda: None)
    monkeypatch.setattr("mlops.main.load_config", lambda _: {})
    monkeypatch.setattr(
        "mlops.main.fetch_data", lambda start_date, end_date: pd.DataFrame()
    )

    test_args = ["main", "--stage", "infer", "--output-csv", ""]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
