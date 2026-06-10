"""
model_trainer.py

Core training module for Heart Disease Classification System.

Features:
- Optuna Hyperparameter Optimization
- Random Forest + XGBoost
- Cross Validation
- Best Model Selection
- MLflow Experiment Tracking
- Model Registry Compatible (run_id exposed)
- SHAP-ready outputs
"""

from __future__ import annotations

import sys
import numpy as np
import optuna
import mlflow
import mlflow.sklearn

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException
from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    RUN_NAME
)


# =====================================================
# CONFIG
# =====================================================

@dataclass
class ModelTrainerConfig:
    random_state: int = 42
    n_trials: int = 20
    cv_folds: int = 5


# =====================================================
# TRAINER
# =====================================================

class ModelTrainer:

    def __init__(self) -> None:

        self.config = ModelTrainerConfig()

        self.best_model = None
        self.best_score = -np.inf
        self.best_params = None
        self.run_id = None   # 🔥 IMPORTANT for Model Registry

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    # =================================================
    # SPLIT DATA
    # =================================================

    @staticmethod
    def split_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        X = data[:, :-1]
        y = data[:, -1]

        return X, y

    # =================================================
    # RF OPTUNA
    # =================================================

    def rf_objective(self, trial, X, y):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": self.config.random_state
        }

        model = RandomForestClassifier(**params)

        score = cross_val_score(
            model,
            X,
            y,
            cv=self.config.cv_folds,
            scoring="accuracy",
            n_jobs=-1
        ).mean()

        return score

    # =================================================
    # XGB OPTUNA
    # =================================================

    def xgb_objective(self, trial, X, y):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": self.config.random_state,
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)

        score = cross_val_score(
            model,
            X,
            y,
            cv=self.config.cv_folds,
            scoring="accuracy",
            n_jobs=-1
        ).mean()

        return score

    # =================================================
    # TRAIN PIPELINE
    # =================================================

    def initiate_model_trainer(self, train_array, test_array) -> Dict[str, Any]:

        try:
            logger.info("Starting training pipeline...")

            X_train, y_train = self.split_data(train_array)
            X_test, y_test = self.split_data(test_array)

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # =================================================
            # MLflow Run START (CRITICAL)
            # =================================================

            with mlflow.start_run(run_name=RUN_NAME):

                # 🔥 capture run_id for Model Registry
                self.run_id = mlflow.active_run().info.run_id

                # =================================================
                # RANDOM FOREST
                # =================================================

                rf_study = optuna.create_study(direction="maximize")

                rf_study.optimize(
                    lambda t: self.rf_objective(t, X_train, y_train),
                    n_trials=self.config.n_trials
                )

                rf_model = RandomForestClassifier(
                    **rf_study.best_params,
                    random_state=self.config.random_state
                )

                rf_model.fit(X_train, y_train)
                rf_score = rf_model.score(X_test, y_test)

                # =================================================
                # XGBOOST
                # =================================================

                xgb_study = optuna.create_study(direction="maximize")

                xgb_study.optimize(
                    lambda t: self.xgb_objective(t, X_train, y_train),
                    n_trials=self.config.n_trials
                )

                xgb_model = XGBClassifier(
                    **xgb_study.best_params,
                    random_state=self.config.random_state,
                    eval_metric="logloss"
                )

                xgb_model.fit(X_train, y_train)
                xgb_score = xgb_model.score(X_test, y_test)

                # =================================================
                # MODEL SELECTION
                # =================================================

                if rf_score > xgb_score:

                    self.best_model = rf_model
                    self.best_score = rf_score
                    self.best_params = rf_study.best_params
                    model_name = "RandomForest"

                else:

                    self.best_model = xgb_model
                    self.best_score = xgb_score
                    self.best_params = xgb_study.best_params
                    model_name = "XGBoost"

                # =================================================
                # MLFLOW LOGGING
                # =================================================

                mlflow.log_param("model", model_name)
                mlflow.log_param("cv_folds", self.config.cv_folds)
                mlflow.log_param("optuna_trials", self.config.n_trials)

                for k, v in self.best_params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("rf_accuracy", rf_score)
                mlflow.log_metric("xgb_accuracy", xgb_score)
                mlflow.log_metric("best_accuracy", self.best_score)

                mlflow.sklearn.log_model(
                    self.best_model,
                    artifact_path="model"
                )

                logger.info(f"Best model: {model_name}")
                logger.info(f"Best score: {self.best_score}")

                # =================================================
                # RETURN (IMPORTANT FOR PIPELINE + REGISTRY)
                # =================================================

                return {
                    "best_model": self.best_model,
                    "best_score": self.best_score,
                    "best_params": self.best_params,
                    "model_name": model_name,
                    "rf_score": rf_score,
                    "xgb_score": xgb_score,
                    "run_id": self.run_id   # 🔥 REQUIRED FOR REGISTRY
                }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise CustomException(e, sys)