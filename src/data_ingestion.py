import logging
import os
import shutil


SOURCE_FILE = "student_exam_scores.csv"
RAW_DIR = "data/raw"
RAW_FILE = os.path.join(RAW_DIR, "student_exam_scores.csv")


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "data_ingestion.log")
)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(source_file: str):
    """Check that the source dataset exists."""

    try:
        if not os.path.exists(source_file):
            raise FileNotFoundError(
                f"Dataset not found: {source_file}"
            )

        logger.debug(
            "Source dataset found: %s",
            source_file
        )

        return source_file

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error while checking data: %s",
            e
        )
        raise


def save_data(source_file: str, destination_file: str):
    """Copy raw data into the DVC raw-data directory."""

    try:
        os.makedirs(
            os.path.dirname(destination_file),
            exist_ok=True
        )

        shutil.copy2(
            source_file,
            destination_file
        )

        logger.debug(
            "Raw data saved to: %s",
            destination_file
        )

    except Exception as e:
        logger.error(
            "Error while saving raw data: %s",
            e
        )
        raise


def main():

    try:

        logger.info("Starting data ingestion")

        source_file = load_data(SOURCE_FILE)

        save_data(
            source_file,
            RAW_FILE
        )

        logger.info(
            "Data ingestion completed successfully"
        )

        logger.info(
            "Raw data saved to: %s",
            RAW_FILE
        )

    except Exception as e:

        logger.error(
            "Failed to complete data ingestion: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()