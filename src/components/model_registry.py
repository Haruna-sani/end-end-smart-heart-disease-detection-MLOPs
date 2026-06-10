"""
model_registry.py

MLflow Model Registry handler for Heart Disease Classification System.

Features:
- Register trained model to MLflow registry
- Version control
- Stage transition (Staging → Production)
- Load production model
- CI/CD compatible design
"""

from __future__ import annotations

import sys
import mlflow
import mlflow.sklearn

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException


# =====================================================
# CONFIG
# =====================================================

@dataclass
class ModelRegistryConfig:
    """
    Configuration for MLflow Model Registry.
    """

    registered_model_name: str = "HeartDiseaseClassifier"


# =====================================================
# MODEL REGISTRY
# =====================================================

class ModelRegistry:
    """
    Handles MLflow Model Registry operations.

    Responsibilities:
    - Register models
    - Promote models across stages
    - Load production models
    """

    def __init__(self) -> None:
        self.config = ModelRegistryConfig()
        self.client = mlflow.tracking.MlflowClient()

    # =================================================
    # REGISTER MODEL
    # =================================================

    def register_model(
        self,
        model,
        run_id: str,
        artifact_path: str = "model"
    ) -> int:
        """
        Register a trained model in MLflow Model Registry.

        Parameters
        ----------
        model : sklearn model
        run_id : str
            MLflow run ID
        artifact_path : str
            Path where model is logged in MLflow

        Returns
        -------
        int
            Registered model version
        """

        try:
            logger.info("Registering model to MLflow Model Registry...")

            if not run_id:
                raise ValueError("run_id is required for model registration")

            model_uri = f"runs:/{run_id}/{artifact_path}"

            result = mlflow.register_model(
                model_uri=model_uri,
                name=self.config.registered_model_name
            )

            version = result.version

            logger.info(
                f"Model registered successfully: "
                f"{self.config.registered_model_name} v{version}"
            )

            return int(version)

        except Exception as e:
            logger.error("Model registration failed.")
            raise CustomException(e, sys)

    # =================================================
    # PROMOTE MODEL STAGE
    # =================================================

    def transition_stage(
        self,
        version: int,
        stage: str = "Staging"
    ) -> None:
        """
        Transition model version to a deployment stage.

        Parameters
        ----------
        version : int
            Model version
        stage : str
            Target stage: "Staging" or "Production"
        """

        try:
            logger.info(
                f"Transitioning model v{version} → {stage}"
            )

            self.client.transition_model_version_stage(
                name=self.config.registered_model_name,
                version=version,
                stage=stage
            )

            logger.info(
                f"Model v{version} successfully moved to {stage}"
            )

        except Exception as e:
            logger.error("Model stage transition failed.")
            raise CustomException(e, sys)

    # =================================================
    # LOAD PRODUCTION MODEL
    # =================================================

    def load_production_model(self):
        """
        Load the latest Production model from MLflow registry.

        Returns
        -------
        model
        """

        try:
            logger.info("Loading Production model from MLflow registry...")

            model_uri = (
                f"models:/"
                f"{self.config.registered_model_name}/Production"
            )

            model = mlflow.sklearn.load_model(model_uri)

            logger.info("Production model loaded successfully.")

            return model

        except Exception as e:
            logger.error("Failed to load production model.")
            raise CustomException(e, sys)