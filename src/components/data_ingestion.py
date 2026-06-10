"""
data_ingestion.py

This module handles the ingestion of raw heart disease data,
including loading, validation (basic), train-test splitting,
and saving artifacts for downstream ML pipelines.

"""

from dataclasses import dataclass
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# LOGGING CONFIGURATION
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# CONFIGURATION CLASS
# =========================
@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion paths.
    """
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.2
    random_state: int = 42


# =========================
# DATA INGESTION CLASS
# =========================
class DataIngestion:
    """
    Handles data ingestion for the heart disease dataset.

    Responsibilities:
    - Load dataset from CSV
    - Perform basic validation
    - Split into train/test sets
    - Save artifacts for downstream pipeline
    """

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str):
        """
        Main method to ingest data and create train/test datasets.

        Parameters
        ----------
        file_path : str
            Path to the raw dataset CSV file.

        Returns
        -------
        tuple
            (train_data_path, test_data_path)
        """

        logging.info("Starting data ingestion process...")

        try:
            # =========================
            # LOAD DATA
            # =========================
            df = pd.read_csv(file_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            # =========================
            # BASIC VALIDATION
            # =========================
            if df.empty:
                raise ValueError("Dataset is empty")

            if "target" not in df.columns:
                raise ValueError("Target column not found in dataset")

            # =========================
            # SAVE RAW DATA
            # =========================
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.config.raw_data_path}")

            # =========================
            # TRAIN-TEST SPLIT
            # =========================
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df["target"]
            )

            logging.info("Train-test split completed")

            # =========================
            # SAVE TRAIN DATA
            # =========================
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            train_set.to_csv(self.config.train_data_path, index=False)
            logging.info(f"Train data saved at {self.config.train_data_path}")

            # =========================
            # SAVE TEST DATA
            # =========================
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Test data saved at {self.config.test_data_path}")

            logging.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise e