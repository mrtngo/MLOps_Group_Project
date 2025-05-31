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
from preproccess.preproccessing import scale_features, smote_oversample, split_data
#from models.models import ModelTrainer

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
    X_train_scaled, X_test_scaled, _ = scale_features(df, feature_cols)
    # Apply SMOTE
    X_balanced, y_balanced = smote_oversample(X_train_scaled, y_class)
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
    #run twice to get both regression and classification targets
    logger.info("🧪 Preprocessing features...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X, y_reg)
    X_train_class, X_test_class, y_train_class, y_test_class = split_data(X, y_class)
    # X_preprocessed_reg, y_reg_balanced = preprocess_data(df, feature_cols, y_train_reg)
    X_preprocessed_class, y_class_balanced = preprocess_data(df, feature_cols, y_train_class)

    # 5. Feature selection
    logger.info("🎯 Selecting top features...")
    X_df_reg = pd.DataFrame(X_preprocessed_class, columns=feature_cols)
    X_df_class = pd.DataFrame(X_preprocessed_class, columns=feature_cols)
    X_df_reg_with_target = X_df_reg.copy()
    X_df_class_with_target = X_df_class.copy()
    X_df_class_with_target[config.get("target")] = y_reg.values
    X_df_reg_with_target[config.get("target")] = y_reg.values
    X_df_class_with_target['price_direction'] = y_class.values
    selected_cols_reg = select_features(X_df_reg_with_target, feature_cols)
    selected_cols_class = select_features(X_df_class_with_target, feature_cols)

    X_selected_reg = X_df_reg[selected_cols_reg]
    X_selected_class = X_df_class[selected_cols_class]

    logger.info("✅ Feature engineering and preprocessing complete.")
    return X_selected_reg, X_selected_class, y_class_balanced, y_reg

if __name__ == "__main__":
    X_processed_reg, X_processed_class, y_reg, y_class = run_until_feature_engineering()
    #trainer = ModelTrainer()
    #price_model, direction_model = trainer.train_from_arrays(X_processed, y_reg, y_class)
