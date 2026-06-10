"""
data_transformation.py

Responsible for transforming raw heart disease data into
model-ready format using preprocessing pipelines.

Includes:
- Missing value handling
- Feature scaling
- Train/test transformation
- Preprocessor persistence

"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# CONFIG CLASS
# =========================
@dataclass
class DataTransformationConfig:
    """
    Stores paths for transformation artifacts.
    """
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


# =========================
# DATA TRANSFORMATION CLASS
# =========================
class DataTransformation:
    """
    Handles preprocessing of heart disease dataset.

    Steps:
    1. Separate features and target
    2. Build preprocessing pipeline
    3. Transform train/test data
    4. Save preprocessing object
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    # =========================
    # PREPROCESSOR BUILDER
    # =========================
    def get_data_transformer_object(self):
        """
        Creates preprocessing pipeline.

        Returns
        -------
        ColumnTransformer
            Preprocessing pipeline object
        """

        logging.info("Creating preprocessing object...")

        try:
            # All features are treated as numeric in this baseline setup
            numeric_features = [
                "age", "sex", "cp", "trestbps", "chol",
                "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal"
            ]

            # =========================
            # NUMERIC PIPELINE
            # =========================
            numeric_pipeline = Pipeline(steps=[
                # Handle missing values
                ("imputer", SimpleImputer(strategy="median")),

                # Scale features
                ("scaler", StandardScaler())
            ])

            # =========================
            # COMBINE TRANSFORMERS
            # =========================
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numeric_features)
                ]
            )

            logging.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            logging.error(f"Error creating preprocessor: {str(e)}")
            raise e

    # =========================
    # MAIN TRANSFORMATION PIPELINE
    # =========================
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing to train and test datasets.

        Parameters
        ----------
        train_path : str
            Path to training dataset
        test_path : str
            Path to test dataset

        Returns
        -------
        tuple
            (train_array, test_array, preprocessor_path)
        """

        try:
            logging.info("Starting data transformation...")

            # =========================
            # LOAD DATA
            # =========================
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # =========================
            # GET PREPROCESSOR
            # =========================
            preprocessor = self.get_data_transformer_object()

            target_column = "target"

            # =========================
            # SPLIT FEATURES & TARGET
            # =========================
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # =========================
            # FIT ON TRAIN
            # =========================
            input_feature_train_arr = preprocessor.fit_transform(
                input_feature_train_df
            )

            # =========================
            # TRANSFORM TEST
            # =========================
            input_feature_test_arr = preprocessor.transform(
                input_feature_test_df
            )

            # =========================
            # COMBINE FEATURES + TARGET
            # =========================
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # =========================
            # SAVE PREPROCESSOR
            # =========================
            os.makedirs(
                os.path.dirname(self.config.preprocessor_obj_file_path),
                exist_ok=True
            )

            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)

            logging.info(
                f"Preprocessor saved at {self.config.preprocessor_obj_file_path}"
            )

            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise e