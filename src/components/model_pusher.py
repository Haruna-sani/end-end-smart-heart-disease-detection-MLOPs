"""
model_pusher.py

Responsible for pushing the best trained model
into a production-ready location.

This includes:
- Saving model artifact
- Versioning model
- Preparing model for deployment pipeline

Author: MLOps Pipeline
"""

import os
import logging
import joblib
from dataclasses import dataclass


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
class ModelPusherConfig:
    """
    Stores model pusher paths.
    """
    export_dir_path: str = os.path.join("artifacts", "model_pusher")
    production_model_path: str = os.path.join(
        "artifacts", "model_pusher", "production_model.pkl"
    )


# =========================
# MODEL PUSHER CLASS
# =========================
class ModelPusher:
    """
    Handles pushing trained models to production-ready storage.

    Responsibilities:
    - Save best model artifact
    - Maintain versioned model storage
    - Prepare model for deployment
    """

    def __init__(self):
        self.config = ModelPusherConfig()

    # =========================
    # MAIN PUSH METHOD
    # =========================
    def push_model(self, model):
        """
        Saves trained model as production artifact.

        Parameters
        ----------
        model : trained ML model (sklearn / xgboost)

        Returns
        -------
        str
            Path to saved production model
        """

        logging.info("Starting model pushing process...")

        try:
            # =========================
            # CREATE DIRECTORY
            # =========================
            os.makedirs(self.config.export_dir_path, exist_ok=True)

            # =========================
            # SAVE MODEL
            # =========================
            joblib.dump(model, self.config.production_model_path)

            logging.info(
                f"Model successfully saved at {self.config.production_model_path}"
            )

            logging.info("Model pushing completed successfully 🚀")

            return self.config.production_model_path

        except Exception as e:
            logging.error(f"Error while pushing model: {str(e)}")
            raise e