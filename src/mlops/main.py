# src/mlops/main.py

import logging
import argparse
import sys
import os

from mlops.data_load.data_load import fetch_data
from mlops.data_validation.data_validation import load_config, validate_data
from mlops.features.features import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features,
)
from mlops.preproccess.preproccessing import (
    scale_features,
    smote_oversample,
)

from mlops.models.models import train_model
from mlops.evaluation.evaluation import evaluate_models
from mlops.inference.inference import run_inference


def setup_logger():
    """
    Configure logging using parameters from config.yaml
    """
    print("hello")
    config = load_config("conf/config.yaml")
    log_cfg = config.get("logging", {})

    log_level = getattr(
        logging, log_cfg.get("level", "INFO").upper(), logging.INFO
    )
    log_format = log_cfg.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = log_cfg.get("log_file", None)

    # Create log directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

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


def run_full_pipeline(start_date, end_date):
    """
    Runs the complete MLOps pipeline: data loading, validation, preprocessing,
    feature engineering, model training, and evaluation.
    """
    logger = logging.getLogger("Pipeline")
    logger.info("Starting complete MLOps pipeline")

    try:
        # 1. Load raw data
        logger.info("Step 1: Loading data...")
        df = fetch_data(start_date=start_date, end_date=end_date)
        logger.info(f"Raw data loaded | shape={df.shape}")

        # 2. Load schema from config and validate
        logger.info("Step 2: Validating data...")
        config = load_config("conf/config.yaml")
        schema_list = config.get("data_validation", {}).get(
            "schema", {}
        ).get("columns", [])
        schema = {col["name"]: col for col in schema_list}
        missing_strategy = config.get("data_validation", {}).get(
            "missing_values_strategy", "drop"
        )

        df_validated = validate_data(
            df, schema, logger, missing_strategy, "warn"
        )
        logger.info(f"Data validation completed | shape={df_validated.shape}")

        # Save processed data
        processed_path = config.get("data_source", {}).get(
            "processed_path", "./data/processed/processed_data.csv"
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_validated.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")

        # 3. Feature engineering and preprocessing
        logger.info("Step 3: Feature engineering and preprocessing...")
        feature_cols, label_col = define_features_and_label()
        df_with_direction = create_price_direction_label(
            df_validated, label_col
        )
        logger.info("Price direction labels created")

        # 4. Model training
        logger.info("Step 4: Training models...")
        price_model, direction_model = train_model(df_with_direction)
        logger.info("Model training completed successfully")

        # 5. Model evaluation
        logger.info("Step 5: Evaluating models...")
        regression_metrics, classification_metrics = evaluate_models(
            df_with_direction
        )
        logger.info("Model evaluation completed successfully")

        # Print summary metrics
        logger.info("=" * 50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 50)
        rmse_value = regression_metrics.get('RMSE', 'N/A')
        if rmse_value != 'N/A':
            logger.info(f"Linear Regression RMSE: {rmse_value:.4f}")
        else:
            logger.info("Linear Regression RMSE: N/A")

        accuracy_value = classification_metrics.get('Accuracy', 'N/A')
        if accuracy_value != 'N/A':
            logger.info(f"Logistic Regression Accuracy: {accuracy_value:.4f}")
        else:
            logger.info("Logistic Regression Accuracy: N/A")

        roc_auc_value = classification_metrics.get('ROC AUC', 'N/A')
        if roc_auc_value != 'N/A':
            logger.info(f"Logistic Regression ROC AUC: {roc_auc_value:.4f}")
        else:
            logger.info("Logistic Regression ROC AUC: N/A")

        logger.info("=" * 50)

        logger.info("Complete MLOps pipeline finished successfully!")
        return df_with_direction, price_model, direction_model

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    """
    Main entry point with command line argument support.
    """

    parser = argparse.ArgumentParser(description="MLOps pipeline orchestrator")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "infer"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/output.csv",
        help="Output CSV file for inference stage",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date for fetching the date",
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="End date for fetching the date",
    )

    args = parser.parse_args()
    setup_logger()
    logger = logging.getLogger("Main")
    logger.info(f"Pipeline started | stage={args.stage}")

    try:
        config = load_config(args.config)
        if args.stage == "all":
            run_full_pipeline(args.start_date, args.end_date)

        elif args.stage == "infer":
            # Inference
            if not args.output_csv:
                logger.error("Inference stage requires --output-csv")
                sys.exit(1)

            logger.info("=== Model Inference ===")

            # Fetch and validate input data
            input_df = fetch_data(
                start_date=args.start_date, end_date=args.end_date
            )
            logger.info(f"Loaded input data | shape={input_df.shape}")

            # Optional: validate inference input against schema
            schema_list = config.get("data_validation", {}).get(
                "schema", {}
            ).get("columns", [])
            schema = {col["name"]: col for col in schema_list}

            # Run inference
            run_inference(input_df, args.config, args.output_csv)
            output_msg = f"Inference completed | output saved to {args.output_csv}"
            logger.info(output_msg)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


def preprocess_data(df, feature_cols, y_class):
    """
    Performs scaling and optional oversampling on selected features.
    This function is kept for backward compatibility but is now handled
    within the ModelTrainer class in models.py
    """
    config = load_config("conf/config.yaml")

    logger = logging.getLogger("Preprocessing")
    deprecation_msg = (
        "preprocess_data() is deprecated. "
        "Preprocessing is now handled in ModelTrainer."
    )
    logger.warning(deprecation_msg)

    # Scale selected features
    X_train_scaled, X_test_scaled, _ = scale_features(df, feature_cols)
    # Apply SMOTE
    X_balanced, y_balanced = smote_oversample(X_train_scaled, y_class)
    return X_balanced, y_balanced


def run_until_feature_engineering():
    """
    Legacy function - runs pipeline up to feature engineering.
    Kept for backward compatibility.
    """
    logger = logging.getLogger("Pipeline")
    deprecation_msg = (
        "run_until_feature_engineering() is deprecated. "
        "Use run_full_pipeline() instead."
    )
    logger.warning(deprecation_msg)

    logger.info("Starting pipeline up to feature engineering")

    # 1. Load raw data
    logger.info("Loading data...")
    df = fetch_data()

    # 2. Load schema from config and validate
    config = load_config("conf/config.yaml")
    schema_list = config.get("data_validation", {}).get(
        "schema", {}
    ).get("columns", [])
    schema = {col["name"]: col for col in schema_list}

    logger.info("Validating data...")
    df = validate_data(
        df, schema, logger, missing_strategy="drop", on_error="warn"
    )

    # 3. Feature engineering
    logger.info("Creating features and labels...")
    feature_cols, label_col = define_features_and_label()
    df = create_price_direction_label(df, label_col)
    X, y_reg, y_class = prepare_features(df, feature_cols, label_col)

    logger.info("Feature engineering completed.")
    return X, y_reg, y_class, df


if __name__ == "__main__":
    main()
