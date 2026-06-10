"""
model_evaluation.py

Responsible for evaluating trained ML models
for heart disease classification.

Outputs:
- Metrics report (JSON)
- Confusion matrix
- ROC-AUC score
- MLflow-ready metrics dictionary

Author: MLOps Pipeline
"""

import os
import json
import logging
import numpy as np

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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
class ModelEvaluationConfig:
    """
    Stores evaluation artifact paths.
    """
    report_file_path: str = os.path.join(
        "artifacts", "model_evaluation_report.json"
    )


# =========================
# MODEL EVALUATION CLASS
# =========================
class ModelEvaluation:
    """
    Evaluates trained classification models
    for heart disease prediction.
    """

    def __init__(self):
        self.config = ModelEvaluationConfig()

    # =========================
    # MAIN EVALUATION METHOD
    # =========================
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates trained model performance.

        Parameters
        ----------
        model : trained ML model
        X_test : np.array or pd.DataFrame
        y_test : np.array or pd.Series

        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """

        logging.info("Starting model evaluation...")

        try:
            # =========================
            # PREDICTIONS
            # =========================
            y_pred = model.predict(X_test)

            # For ROC-AUC (probability required)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None

            # =========================
            # METRICS
            # =========================
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
            }

            # ROC-AUC (only if probabilities exist)
            if y_prob is not None:
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            else:
                metrics["roc_auc"] = None

            # =========================
            # CONFUSION MATRIX
            # =========================
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # =========================
            # SAVE REPORT
            # =========================
            os.makedirs(
                os.path.dirname(self.config.report_file_path),
                exist_ok=True
            )

            with open(self.config.report_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(
                f"Evaluation report saved at {self.config.report_file_path}"
            )

            logging.info(f"Model metrics: {metrics}")

            return metrics

        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise e