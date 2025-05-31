# src/main.py

import logging
from data_load.data_load import fetch_data
from data_validation.data_validation import load_config, validate_data
from features.features import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features,
    select_features
)
from preproccess.preproccessing import scale_features, smote_oversample

def setup_logger():
    """
    Configure logging using parameters from config.yaml
    """
    config = load_config("config.yaml")
    log_cfg = config.get("logging", {})

    log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_format = log_cfg.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    date_format = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = log_cfg.get("log_file", None)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        filename=log_file,
        filemode="a"
    )

    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger("").addHandler(console)

def preprocess_data(df, feature_cols, y_class):
    """
    Performs scaling and optional oversampling on selected features.
    """
    config = load_config("config.yaml")
    sampling_cfg = config.get("preprocessing", {}).get("sampling", {})
    threshold = sampling_cfg.get("threshold_ratio", 1.5)

    # Scale selected features
    X_scaled, scaler = scale_features(df, feature_cols)

    # Optionally apply SMOTE
    X_balanced, y_balanced = smote_oversample(X_scaled, y_class)

    return X_balanced, y_balanced

def run_until_feature_engineering():
    setup_logger()
    logger = logging.getLogger("Pipeline")

    logger.info("🚀 Starting pipeline up to feature engineering")

    # 1. Load raw data
    logger.info("📥 Loading data...")
    df = fetch_data()

    # 2. Load schema from config and convert list → dict
    config = load_config("config.yaml")
    schema_list = config.get("data_validation", {}).get("schema", {}).get("columns", [])
    schema = {col["name"]: col for col in schema_list}

    # 3. Validate data
    logger.info("✅ Validating data...")
    df = validate_data(df, schema, logger, missing_strategy="drop", on_error="warn")

    # 4. Feature engineering
    logger.info("🧠 Creating features and labels...")
    feature_cols, label_col = define_features_and_label()
    df = create_price_direction_label(df, label_col)
    X, y_reg, y_class = prepare_features(df, feature_cols, label_col)

    # 5. Feature selection
    logger.info("🎯 Selecting top features...")
    selected_cols = select_features(df, feature_cols)

    # 6. Preprocessing: scaling + SMOTE
    logger.info("🧪 Preprocessing selected features...")
    X_processed, y_processed = preprocess_data(df, selected_cols, y_class)

    logger.info("✅ Feature engineering and preprocessing complete.")
    return X_processed, y_reg, y_processed

if __name__ == "__main__":
    X_processed, y_reg, y_class = run_until_feature_engineering()

