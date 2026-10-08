import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression


TRAIN_FILE = "data/processed/train.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "student_marks_prediction.joblib")
LOG_DIR = "logs"
REPORTS_DIR = "reports"


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# Logging setup
logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "model_training.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student Marks Prediction")


def load_data():
    try:
        if not os.path.exists(TRAIN_FILE):
            raise FileNotFoundError(
                f"Training file not found: {TRAIN_FILE}"
            )

        data = pd.read_csv(TRAIN_FILE)

        logger.debug(f"Training data loaded from {TRAIN_FILE}")
        logger.debug(f"Training dataset shape: {data.shape}")

        return data

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def train_model(data):
    try:
        target_column = "exam_score"

        X = data.drop(columns=[target_column])
        y = data[target_column]

        logger.debug(f"Training samples: {len(X)}")
        logger.debug(f"Features used: {list(X.columns)}")

        logger.info("Starting Linear Regression training")

        model = LinearRegression()
        model.fit(X, y)

        logger.info("Model training completed")

        return model, X, y

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model):
    try:
        joblib.dump(model, MODEL_FILE)
        logger.info(f"Model saved to {MODEL_FILE}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    logger.info("Starting model training")

    try:
        # Load data
        data = load_data()

        # Train model
        model, X, y = train_model(data)

        # Save model locally
        save_model(model)

        # Start MLflow run
        with mlflow.start_run() as run:

            # --------------------------------
            # Dataset tracking
            # --------------------------------
            dataset = mlflow.data.from_pandas(
                data,
                source=TRAIN_FILE,
                name="student_training_data",
                targets="exam_score"
            )

            mlflow.log_input(
                dataset,
                context="training"
            )

            logger.info("Training dataset logged to MLflow")

            # --------------------------------
            # Log training parameters
            # --------------------------------
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("target_column", "exam_score")
            mlflow.log_param("training_samples", len(X))
            mlflow.log_param("number_of_features", len(X.columns))

            # --------------------------------
            # Log trained model
            # --------------------------------
            mlflow.sklearn.log_model(
                sk_model=model,
                name="student_marks_model"
            )

            # --------------------------------
            # Save MLflow Run ID
            # --------------------------------
            with open(
                os.path.join(REPORTS_DIR, "mlflow_run_id.txt"),
                "w"
            ) as f:
                f.write(run.info.run_id)

            logger.info(
                f"MLflow run created: {run.info.run_id}"
            )

            logger.info(
                "Dataset, model and parameters logged to MLflow successfully"
            )

        logger.info("Model training stage completed successfully")

    except Exception as e:
        logger.error(f"Model training stage failed: {e}")
        raise


if __name__ == "__main__":
    main()