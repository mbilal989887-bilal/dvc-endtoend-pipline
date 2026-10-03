import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


TEST_FILE = "data/processed/test.csv"

MODEL_FILE = (
    "models/student_marks_prediction.joblib"
)

REPORTS_DIR = "reports"

METRICS_FILE = os.path.join(
    REPORTS_DIR,
    "metrics.json"
)


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    os.path.join(
        LOG_DIR,
        "model_evaluation.log"
    )
)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    """Load trained model."""

    try:

        model = joblib.load(file_path)

        logger.debug(
            "Model loaded from %s",
            file_path
        )

        return model

    except FileNotFoundError as e:

        logger.error(
            "Model file not found: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading model: %s",
            e
        )

        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load test data."""

    try:

        df = pd.read_csv(file_path)

        logger.debug(
            "Test data loaded from %s",
            file_path
        )

        logger.debug(
            "Test dataset shape: %s",
            df.shape
        )

        return df

    except FileNotFoundError as e:

        logger.error(
            "Test data not found: %s",
            e
        )

        raise

    except pd.errors.ParserError as e:

        logger.error(
            "CSV parsing error: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Unexpected error loading test data: %s",
            e
        )

        raise


def evaluate_model(
    model,
    X_test,
    y_test
):
    """Calculate regression metrics."""

    try:

        logger.info(
            "Starting model evaluation"
        )

        y_pred = model.predict(X_test)

        r2 = r2_score(
            y_test,
            y_pred
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        metrics = {
            "r2_score": round(
                float(r2),
                4
            ),
            "mae": round(
                float(mae),
                4
            ),
            "mse": round(
                float(mse),
                4
            ),
            "rmse": round(
                float(rmse),
                4
            )
        }

        logger.info(
            "R2 Score: %.4f",
            r2
        )

        logger.info(
            "MAE: %.4f",
            mae
        )

        logger.info(
            "MSE: %.4f",
            mse
        )

        logger.info(
            "RMSE: %.4f",
            rmse
        )

        return metrics

    except Exception as e:

        logger.error(
            "Error during model evaluation: %s",
            e
        )

        raise


def save_metrics(
    metrics: dict,
    file_path: str
):
    """Save evaluation metrics."""

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(
            file_path,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            "Metrics saved to %s",
            file_path
        )

    except Exception as e:

        logger.error(
            "Error saving metrics: %s",
            e
        )

        raise


def main():

    try:

        logger.info(
            "Starting model evaluation"
        )

        model = load_model(
            MODEL_FILE
        )

        df = load_data(
            TEST_FILE
        )

        X_test = df.drop(
            "exam_score",
            axis=1
        )

        y_test = df["exam_score"]

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        save_metrics(
            metrics,
            METRICS_FILE
        )

        logger.info(
            "Model evaluation completed successfully"
        )

    except Exception as e:

        logger.error(
            "Failed to complete model evaluation: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()