import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TEST_FILE = "data/processed/test.csv"
MODEL_FILE = "models/student_marks_prediction.joblib"
METRICS_FILE = "reports/metrics.json"
RUN_ID_FILE = "reports/mlflow_run_id.txt"
LOG_DIR = "logs"


# Create folders
os.makedirs("reports", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "model_evaluation.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --------------------------------------------------
# MLflow setup
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():
    try:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_FILE}"
            )

        model = joblib.load(MODEL_FILE)

        logger.debug(f"Model loaded from {MODEL_FILE}")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# --------------------------------------------------
# Load test data
# --------------------------------------------------

def load_data():
    try:
        if not os.path.exists(TEST_FILE):
            raise FileNotFoundError(
                f"Test file not found: {TEST_FILE}"
            )

        data = pd.read_csv(TEST_FILE)

        logger.debug(f"Test data loaded from {TEST_FILE}")
        logger.debug(f"Test dataset shape: {data.shape}")

        return data

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, data):
    try:
        target_column = "exam_score"

        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]

        logger.debug(f"Test samples: {len(X_test)}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
        }

        logger.info(f"R2 Score: {metrics['r2_score']}")
        logger.info(f"MAE: {metrics['mae']}")
        logger.info(f"MSE: {metrics['mse']}")
        logger.info(f"RMSE: {metrics['rmse']}")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


# --------------------------------------------------
# Save metrics locally
# --------------------------------------------------

def save_metrics(metrics):
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {METRICS_FILE}")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


# --------------------------------------------------
# Log metrics to MLflow
# --------------------------------------------------

def log_to_mlflow(metrics, data):
    try:
        if not os.path.exists(RUN_ID_FILE):
            raise FileNotFoundError(
                f"MLflow Run ID file not found: {RUN_ID_FILE}"
            )

        with open(RUN_ID_FILE, "r") as f:
            run_id = f.read().strip()

        if not run_id:
            raise ValueError("MLflow Run ID is empty.")

        # Get the run's original experiment before resuming it
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id

        # Ensure the active experiment matches the run being resumed
        mlflow.set_experiment(experiment_id=experiment_id)

        # Resume the existing MLflow run
        with mlflow.start_run(run_id=run_id):
            dataset = mlflow.data.from_pandas(
                data,
                source=TEST_FILE,
                name="student_test_data",
                targets="exam_score",
            )

            mlflow.log_input(dataset, context="testing")
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(METRICS_FILE)

        logger.info(
            f"Test dataset, metrics and artifact logged to MLflow run: {run_id}"
        )

    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        raise


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    logger.info("Starting model evaluation")

    try:
        model = load_model()
        data = load_data()
        metrics = evaluate_model(model, data)
        save_metrics(metrics)
        log_to_mlflow(metrics, data)

        logger.info("Model evaluation stage completed successfully")

    except Exception as e:
        logger.error(f"Model evaluation stage failed: {e}")
        raise


if __name__ == "__main__":
    main()
