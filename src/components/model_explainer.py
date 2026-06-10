"""
model_explainer.py

SHAP Explainability Module for Heart Disease Classification.

Features
--------
- Global Feature Importance
- Local Prediction Explanations
- SHAP Summary Plot
- SHAP Bar Plot
- SHAP Dependence Plot
- Artifact Saving

Supports:
- Random Forest
- XGBoost

Author: Haruna Sule
"""

from __future__ import annotations

import os
import sys
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional

from src.logger import logger
from src.exception import CustomException


# =====================================================
# CONFIGURATION
# =====================================================

@dataclass
class ModelExplainerConfig:
    """
    SHAP output configuration.
    """

    output_dir: str = "artifacts/shap"

    summary_plot_name: str = "summary_plot.png"
    bar_plot_name: str = "feature_importance.png"


# =====================================================
# MODEL EXPLAINER
# =====================================================

class ModelExplainer:
    """
    Generates SHAP explanations for trained models.
    """

    def __init__(self):

        self.config = ModelExplainerConfig()

        os.makedirs(
            self.config.output_dir,
            exist_ok=True
        )

    # =================================================
    # LOAD MODEL
    # =================================================

    @staticmethod
    def load_model(model_path: str):
        """
        Loads serialized model.

        Parameters
        ----------
        model_path : str

        Returns
        -------
        Trained model
        """

        return joblib.load(model_path)

    # =================================================
    # BUILD EXPLAINER
    # =================================================

    @staticmethod
    def build_explainer(model):
        """
        Creates SHAP TreeExplainer.

        Parameters
        ----------
        model

        Returns
        -------
        shap.TreeExplainer
        """

        return shap.TreeExplainer(model)

    # =================================================
    # COMPUTE SHAP VALUES
    # =================================================

    def compute_shap_values(
        self,
        model,
        X: pd.DataFrame
    ):
        """
        Computes SHAP values.

        Parameters
        ----------
        model
        X : pd.DataFrame

        Returns
        -------
        SHAP values
        """

        logger.info(
            "Computing SHAP values..."
        )

        explainer = self.build_explainer(model)

        shap_values = explainer.shap_values(X)

        return shap_values

    # =================================================
    # SUMMARY PLOT
    # =================================================

    def generate_summary_plot(
        self,
        shap_values,
        X: pd.DataFrame
    ) -> str:
        """
        Creates SHAP summary plot.
        """

        try:

            save_path = os.path.join(
                self.config.output_dir,
                self.config.summary_plot_name
            )

            plt.figure()

            shap.summary_plot(
                shap_values,
                X,
                show=False
            )

            plt.tight_layout()

            plt.savefig(
                save_path,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"Summary plot saved to {save_path}"
            )

            return save_path

        except Exception as e:
            raise CustomException(e, sys)

    # =================================================
    # FEATURE IMPORTANCE BAR PLOT
    # =================================================

    def generate_feature_importance_plot(
        self,
        shap_values,
        X: pd.DataFrame
    ) -> str:
        """
        Creates SHAP importance plot.
        """

        try:

            save_path = os.path.join(
                self.config.output_dir,
                self.config.bar_plot_name
            )

            plt.figure()

            shap.summary_plot(
                shap_values,
                X,
                plot_type="bar",
                show=False
            )

            plt.tight_layout()

            plt.savefig(
                save_path,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"Feature importance plot saved to {save_path}"
            )

            return save_path

        except Exception as e:
            raise CustomException(e, sys)

    # =================================================
    # TOP FEATURES TABLE
    # =================================================

    def get_top_features(
        self,
        shap_values,
        X: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Returns most influential features.

        Parameters
        ----------
        shap_values
        X : pd.DataFrame
        top_k : int

        Returns
        -------
        pd.DataFrame
        """

        try:

            mean_abs_shap = np.abs(
                shap_values
            ).mean(axis=0)

            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": mean_abs_shap
            })

            importance_df = (
                importance_df
                .sort_values(
                    by="importance",
                    ascending=False
                )
                .head(top_k)
                .reset_index(drop=True)
            )

            return importance_df

        except Exception as e:
            raise CustomException(e, sys)

    # =================================================
    # COMPLETE PIPELINE
    # =================================================

    def explain_model(
        self,
        model,
        X: pd.DataFrame
    ) -> dict:
        """
        Generates complete SHAP explanation suite.

        Parameters
        ----------
        model
        X : pd.DataFrame

        Returns
        -------
        dict
        """

        try:

            logger.info(
                "Starting SHAP explainability pipeline..."
            )

            shap_values = self.compute_shap_values(
                model,
                X
            )

            summary_plot = (
                self.generate_summary_plot(
                    shap_values,
                    X
                )
            )

            importance_plot = (
                self.generate_feature_importance_plot(
                    shap_values,
                    X
                )
            )

            top_features = (
                self.get_top_features(
                    shap_values,
                    X
                )
            )

            logger.info(
                "SHAP explanation completed."
            )

            return {
                "shap_values": shap_values,
                "summary_plot": summary_plot,
                "importance_plot": importance_plot,
                "top_features": top_features
            }

        except Exception as e:

            logger.error(
                f"SHAP pipeline failed: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )