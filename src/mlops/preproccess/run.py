import argparse
import logging
import os
import sys
import pickle
import pandas as pd
import mlflow
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.mlops.data_validation.data_validation import load_config
from src.mlops.features.features import define_features_and_label, prepare_features, select_features
from src.mlops.preproccess.preproccessing import split_data, scale_features, smote_oversample

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing(input_artifact: str):
    """
    Executes the feature engineering and preprocessing step.
    - Loads validated data.
    - Creates features and labels.
    - Splits, scales, and oversamples the data.
    - Logs all processed data and the preprocessing pipeline as artifacts.
    """
    logger.info("--- Starting Standalone Preprocessing Step (Scaling, Splitting, SMOTE) ---")

    config = load_config("conf/config.yaml")

    # Set MLflow experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get("experiment_name", "MLOps-Group-Project-Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'")

    # Initialize a new W&B run
    wandb_config = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        entity=wandb_config.get("entity"),
        name="preprocessing-standalone",
        job_type="preprocessing"
    )

    try:
        with mlflow.start_run(run_name="preprocessing") as mlrun:
            # --- 1. Load Data ---
            logger.info(f"Loading feature-engineered data from: {input_artifact}")
            df = pd.read_csv(input_artifact)
            # MLflow already logs the input artifact parameter, no need to do it manually
            # mlflow.log_param("input_artifact", input_artifact)
            # wandb.config.update({"input_artifact": input_artifact})

            # --- 2. Feature Engineering ---
            feature_cols, label_col = define_features_and_label(config)
            X, y_reg, y_class = prepare_features(df, feature_cols, label_col)
            logger.info("Feature engineering complete.")

            # --- 3. Data Splitting ---
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X, y_reg, config)
            X_train_class, X_test_class, y_train_class, y_test_class = split_data(X, y_class, config)
            logger.info("Data splitting complete.")

            # --- 4. Scaling and Feature Selection ---
            X_train_reg_scaled, _, scaler = scale_features(X_train_reg, feature_cols)
            X_test_reg_scaled = scaler.transform(X_test_reg[feature_cols])
            
            X_train_class_scaled, _, _ = scale_features(X_train_class, feature_cols)
            X_test_class_scaled = scaler.transform(X_test_class[feature_cols])

            # For feature selection, create temp DFs and add the correct target column
            df_reg_train = pd.DataFrame(X_train_reg_scaled, columns=feature_cols, index=X_train_reg.index)
            df_reg_train[config.get("target")] = y_train_reg.values
            selected_features_reg = select_features(
                df_reg_train, feature_cols, target_col=config.get("target"), config=config
            )
            
            df_class_train = pd.DataFrame(X_train_class_scaled, columns=feature_cols, index=X_train_class.index)
            df_class_train["price_direction"] = y_train_class.values
            selected_features_class = select_features(
                df_class_train, feature_cols, target_col="price_direction", config=config
            )
            
            logger.info(f"Selected regression features: {selected_features_reg}")
            logger.info(f"Selected classification features: {selected_features_class}")

            # Filter datasets based on selected features
            X_train_reg_selected = pd.DataFrame(X_train_reg_scaled, columns=feature_cols)[selected_features_reg]
            X_test_reg_selected = pd.DataFrame(X_test_reg_scaled, columns=feature_cols)[selected_features_reg]
            
            X_train_class_selected = pd.DataFrame(X_train_class_scaled, columns=feature_cols)[selected_features_class]
            X_test_class_scaled = pd.DataFrame(X_test_class_scaled, columns=feature_cols)[selected_features_class]

            # --- 5. Oversampling (SMOTE) ---
            smote_params = config.get("preprocessing", {}).get("sampling", {})
            if smote_params.get("method") == "smote":
                X_train_class_balanced, y_train_class_balanced = smote_oversample(X_train_class_selected, y_train_class, config)
                logger.info("SMOTE applied to classification training data.")
            else:
                X_train_class_balanced, y_train_class_balanced = X_train_class_selected, y_train_class
            
            # --- 6. Generate and Log Visualizations ---
            logger.info("Generating and logging visualizations to W&B...")

            # a) Class Distribution Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            y_train_class.value_counts().plot(kind='bar', ax=ax1, title='Class Distribution Before SMOTE')
            ax1.set_xlabel("Price Direction")
            ax1.set_ylabel("Count")

            pd.Series(y_train_class_balanced).value_counts().plot(kind='bar', ax=ax2, title='Class Distribution After SMOTE')
            ax2.set_xlabel("Price Direction")
            ax2.set_ylabel("Count")

            plt.tight_layout()
            wandb.log({"class_distribution_comparison": wandb.Image(plt)})
            plt.close(fig) # Close the plot to free up memory

            # b) Correlation Matrix Heatmap
            plt.figure(figsize=(12, 10))
            correlation_matrix = X_train_reg_selected.corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
            plt.title('Correlation Matrix of Selected Regression Features')
            wandb.log({"correlation_heatmap": wandb.Image(plt)})
            plt.close()
            
            # c) Feature Distribution Plots
            num_features = len(selected_features_reg)
            num_cols = 4  # Adjust layout as needed
            num_rows = (num_features + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
            axes = axes.flatten()

            for i, feature in enumerate(selected_features_reg):
                sns.histplot(X_train_reg_selected[feature], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
            
            # Hide unused subplots
            for i in range(num_features, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            wandb.log({"feature_distributions": wandb.Image(plt)})
            plt.close(fig)

            logger.info("Visualizations logged successfully.")

            # --- 7. Save and Log Artifacts ---
            logger.info("Saving and logging all processed artifacts...")
            
            # Create a directory for the artifacts
            output_dir = config.get("artifacts", {}).get("processed_data_path", "data/processed/training_data")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save dataframes
            artifacts_to_save = {
                "X_train_reg": X_train_reg_selected, "y_train_reg": y_train_reg,
                "X_test_reg": X_test_reg_selected, "y_test_reg": y_test_reg,
                "X_train_class": X_train_class_balanced, "y_train_class": y_train_class_balanced,
                "X_test_class": X_test_class_scaled, "y_test_class": y_test_class
            }
            for name, df_to_save in artifacts_to_save.items():
                path = os.path.join(output_dir, f"{name}.csv")
                pd.DataFrame(df_to_save).to_csv(path, index=False)
            
            # Save preprocessing pipeline
            pipeline_path = config.get("artifacts", {}).get("preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
            os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
            pipeline = {
                'scaler': scaler, 
                'selected_features_reg': selected_features_reg, 
                'selected_features_class': selected_features_class,
                'all_feature_cols': feature_cols
            }
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)

            # Log to MLflow and W&B
            mlflow.log_artifacts(output_dir, "processed-data")
            mlflow.log_artifact(pipeline_path, "preprocessing-pipeline")
            
            artifact = wandb.Artifact("processed-data", type="dataset", description="Processed data splits for training and testing.")
            artifact.add_dir(output_dir)
            artifact.add_file(pipeline_path)
            wandb.log_artifact(artifact)
            
            logger.info("All artifacts logged successfully.")

        logger.info("--- Preprocessing Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Preprocessing step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline step.")
    config = load_config("conf/config.yaml")
    default_input = config.get("artifacts", {}).get("feature_engineered_path", "data/processed/feature_engineered_data.csv")
    
    parser.add_argument(
        "--input-artifact", 
        default=default_input,
        help="Path to the feature-engineered data CSV file."
    )
    args = parser.parse_args()

    run_preprocessing(args.input_artifact) 