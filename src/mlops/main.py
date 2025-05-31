# src/mlops/main.py

import logging
import pandas as pd

from data_load.data_load import fetch_data
from data_validation.data_validation import load_config, validate_data
from features.features import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features,
    select_features
)
from preproccess.preproccessing import scale_features, smote_oversample
from models.models import ModelTrainer

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
    # Scale selected features
    X_scaled, _ = scale_features(df, feature_cols)
    # Apply SMOTE
    X_balanced, y_balanced = smote_oversample(X_scaled, y_class)
    return X_balanced, y_balanced

def run_until_feature_engineering():
    setup_logger()
    logger = logging.getLogger("Pipeline")
    logger.info("🚀 Starting pipeline up to feature engineering")

    # 1. Load raw data
    logger.info("📥 Loading data...")
    df = fetch_data()

    # 2. Load schema from config and validate
    config = load_config("config.yaml")
    schema_list = config.get("data_validation", {}).get("schema", {}).get("columns", [])
    schema = {col["name"]: col for col in schema_list}

    logger.info("✅ Validating data...")
    df = validate_data(df, schema, logger, missing_strategy="drop", on_error="warn")

    # 3. Feature engineering
    logger.info("🧠 Creating features and labels...")
    feature_cols, label_col = define_features_and_label()
    df = create_price_direction_label(df, label_col)
    X, y_reg, y_class = prepare_features(df, feature_cols, label_col)

    # 4. Preprocessing
    logger.info("🧪 Preprocessing features...")
    X_preprocessed, y_class_balanced = preprocess_data(df, feature_cols, y_class)

    # 5. Feature selection
    logger.info("🎯 Selecting top features...")
    X_df = pd.DataFrame(X_preprocessed, columns=feature_cols)
    X_df_with_target = X_df.copy()
    X_df_with_target[config.get("target")] = y_reg.values
    selected_cols = select_features(X_df_with_target, feature_cols)

    X_selected = X_df[selected_cols]

    logger.info("✅ Feature engineering and preprocessing complete.")
    return X_selected, y_reg, y_class_balanced

if __name__ == "__main__":
    X_processed, y_reg, y_class = run_until_feature_engineering()
    #trainer = ModelTrainer()
    #price_model, direction_model = trainer.train_from_arrays(X_processed, y_reg, y_class)
