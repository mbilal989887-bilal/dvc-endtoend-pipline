
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
        logger.info(f"Model loaded from {MODEL_FILE}")

        return model

    except Exception:
        logger.exception("Error loading model")
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

        if "exam_score" not in data.columns:
            raise ValueError(
                "Target column 'exam_score' is missing from test data"
            )

        logger.info(f"Test data loaded from {TEST_FILE}")
        logger.info(f"Test dataset shape: {data.shape}")

        return data

    except Exception:
        logger.exception("Error loading test data")
        raise


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, data):
    try:
        target_column = "exam_score"

        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]

        logger.info(f"Test samples: {len(X_test)}")

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "mse": round(mse, 4),
            "rmse": round(float(np.sqrt(mse)), 4),
        }

        for name, value in metrics.items():
            logger.info("%s: %s", name, value)

        return metrics

    except Exception:
        logger.exception("Error evaluating model")
        raise


# --------------------------------------------------
# Save metrics locally
# --------------------------------------------------

def save_metrics(metrics):
    try:
        with open(METRICS_FILE, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved to %s", METRICS_FILE)

    except Exception:
        logger.exception("Error saving metrics")
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

        with open(RUN_ID_FILE, "r") as file:
            run_id = file.read().strip()

        if not run_id:
            raise ValueError("MLflow Run ID is empty")

        # Confirm that this run exists in the current MLflow database.
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id

        logger.info(
            "Resuming MLflow run %s in experiment %s",
            run_id,
            experiment_id,
        )

        # Resume the training run; do not select a different experiment.
        with mlflow.start_run(run_id=run_id):
            dataset = mlflow.data.from_pandas(
                data,
                source=TEST_FILE,
                name="student_test_data",
                targets="exam_score",
            )

            mlflow.log_input(
                dataset,
                context="testing",
            )

            mlflow.log_metrics(metrics)

            if os.path.exists(METRICS_FILE):
                mlflow.log_artifact(METRICS_FILE)
            else:
                raise FileNotFoundError(
                    f"Metrics file not found: {METRICS_FILE}"
                )

        logger.info(
            "Test dataset, metrics, and artifact logged to MLflow run %s",
            run_id,
        )

    except Exception:
        logger.exception("Failed to log evaluation results to MLflow")
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

    except Exception:
        logger.exception("Model evaluation stage failed")
        raise


if __name__ == "__main__":
    main()

