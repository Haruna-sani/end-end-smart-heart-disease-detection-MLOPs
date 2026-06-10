"""
mlflow_config.py

Central configuration for MLflow tracking, experiments, and runs.

Used across:
- Model Trainer
- Training Pipeline
- CI/CD workflows
"""

import mlflow


# =====================================================
# CORE CONFIGURATION
# =====================================================

MLFLOW_TRACKING_URI = "file:./mlruns"

EXPERIMENT_NAME = "heart_disease_classification"

RUN_NAME = "heart_disease_training"


# =====================================================
# MLflow INITIALIZER (IMPORTANT FOR PRODUCTION)
# =====================================================

def setup_mlflow() -> None:
    """
    Initializes MLflow tracking environment.

    Ensures:
    - Tracking URI is set
    - Experiment exists or is created
    - Safe reuse in CI/CD pipelines
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)