"""
prediction_pipeline.py

This module handles inference for the Heart Disease ML system.

It:
- Loads trained model
- Loads preprocessing object
- Transforms input data
- Returns prediction + probability

Author: MLOps Pipeline
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# PREDICTION PIPELINE
# =========================
class PredictionPipeline:
    """
    Handles inference for heart disease prediction system.
    """

    def __init__(self):
        # Path to saved artifacts
        self.model_path = os.path.join(
            "artifacts", "model_pusher", "production_model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts", "preprocessor.pkl"
        )

        # Load artifacts once (efficient inference)
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        logging.info("Model and preprocessor loaded successfully")

    # =========================
    # PREDICT METHOD
    # =========================
    def predict(self, data: dict):
        """
        Makes prediction for a single patient.

        Parameters
        ----------
        data : dict
            Patient feature dictionary

        Returns
        -------
        dict
            Prediction result + probability
        """

        try:
            logging.info("Starting prediction process...")

            # =========================
            # CONVERT INPUT TO DATAFRAME
            # =========================
            df = pd.DataFrame([data])

            # =========================
            # PREPROCESS INPUT
            # =========================
            processed_data = self.preprocessor.transform(df)

            # =========================
            # PREDICTION
            # =========================
            prediction = self.model.predict(processed_data)

            # =========================
            # PROBABILITY (if supported)
            # =========================
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(processed_data)[0][1]
            else:
                probability = None

            # =========================
            # RESPONSE FORMAT
            # =========================
            result = {
                "prediction": int(prediction[0]),
                "prediction_label": (
                    "Heart Disease Detected" if prediction[0] == 1
                    else "No Heart Disease"
                ),
                "probability": float(probability) if probability is not None else None
            }

            logging.info(f"Prediction completed: {result}")

            return result

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise e