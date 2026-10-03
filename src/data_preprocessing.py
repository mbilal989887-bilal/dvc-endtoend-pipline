import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


RAW_FILE = "data/raw/student_exam_scores.csv"

PROCESSED_DIR = "data/processed"

TRAIN_FILE = os.path.join(
    PROCESSED_DIR,
    "train.csv"
)

TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "test.csv"
)


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    os.path.join(
        LOG_DIR,
        "data_preprocessing.log"
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


def load_params(params_path: str) -> dict:
    """Load preprocessing parameters."""

    try:

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug(
            "Parameters loaded from %s",
            params_path
        )

        return params

    except FileNotFoundError as e:

        logger.error(
            "Parameter file not found: %s",
            e
        )

        raise

    except yaml.YAMLError as e:

        logger.error(
            "YAML error: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Unexpected error loading parameters: %s",
            e
        )

        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""

    try:

        df = pd.read_csv(file_path)

        logger.debug(
            "Data loaded from %s",
            file_path
        )

        logger.debug(
            "Dataset shape: %s",
            df.shape
        )

        return df

    except FileNotFoundError as e:

        logger.error(
            "Data file not found: %s",
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
            "Unexpected error loading data: %s",
            e
        )

        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the student dataset."""

    try:

        logger.debug(
            "Starting data preprocessing"
        )

        # Clean column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
        )

        logger.debug(
            "Column names cleaned"
        )

        # Remove columns not used by the model
        columns_to_drop = [
            "student_id",
            "sleep_hours"
        ]

        existing_columns = [
            col
            for col in columns_to_drop
            if col in df.columns
        ]

        df = df.drop(
            columns=existing_columns
        )

        logger.debug(
            "Unused columns removed"
        )

        # Remove missing values
        before = len(df)

        df = df.dropna()

        after = len(df)

        logger.debug(
            "Removed %d rows containing missing values",
            before - after
        )

        logger.debug(
            "Final dataset shape: %s",
            df.shape
        )

        return df

    except KeyError as e:

        logger.error(
            "Column error during preprocessing: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Unexpected preprocessing error: %s",
            e
        )

        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
):
    """Save processed train and test datasets."""

    try:

        os.makedirs(
            PROCESSED_DIR,
            exist_ok=True
        )

        train_data.to_csv(
            TRAIN_FILE,
            index=False
        )

        test_data.to_csv(
            TEST_FILE,
            index=False
        )

        logger.debug(
            "Training data saved to %s",
            TRAIN_FILE
        )

        logger.debug(
            "Testing data saved to %s",
            TEST_FILE
        )

    except Exception as e:

        logger.error(
            "Error saving processed data: %s",
            e
        )

        raise


def main():

    try:

        logger.info(
            "Starting data preprocessing"
        )

        params = load_params("params.yaml")

        test_size = params[
            "data_preprocessing"
        ][
            "test_size"
        ]

        random_state = params[
            "data_preprocessing"
        ][
            "random_state"
        ]

        df = load_data(RAW_FILE)

        df = preprocess_data(df)

        X = df.drop(
            "exam_score",
            axis=1
        )

        y = df["exam_score"]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state
            )
        )

        train_data = X_train.copy()
        train_data["exam_score"] = y_train.values

        test_data = X_test.copy()
        test_data["exam_score"] = y_test.values

        save_data(
            train_data,
            test_data
        )

        logger.info(
            "Training data shape: %s",
            train_data.shape
        )

        logger.info(
            "Testing data shape: %s",
            test_data.shape
        )

        logger.info(
            "Data preprocessing completed successfully"
        )

    except Exception as e:

        logger.error(
            "Failed to complete preprocessing: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()