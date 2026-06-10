"""
data_validation.py

Production-grade validation module for Heart Disease ML pipeline.

Fixes included:
- JSON serialization issues (numpy types)
- Over-strict validation logic
- Stable reporting for CI/CD pipelines

Author: MLOps System
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass


# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# CONFIG
# =========================
@dataclass
class DataValidationConfig:
    report_file_path: str = os.path.join(
        "artifacts", "data_validation_report.json"
    )


# =========================
# DATA VALIDATION CLASS
# =========================
class DataValidation:

    def __init__(self):
        self.config = DataValidationConfig()

        self.expected_columns = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "target"
        ]

        self.valid_targets = {0, 1}

    # =================================================
    # SAFE JSON CONVERTER
    # =================================================
    def _to_python_type(self, obj):
        """
        Converts numpy / pandas types to JSON serializable Python types.
        """

        if isinstance(obj, dict):
            return {k: self._to_python_type(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._to_python_type(i) for i in obj]

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        return obj

    # =================================================
    # MAIN VALIDATION
    # =================================================
    def validate_data(self, dataframe: pd.DataFrame) -> bool:

        logging.info("Starting data validation...")

        report = {}
        critical_errors = []
        warnings = []

        # -------------------------
        # COLUMN CHECK
        # -------------------------
        missing_cols = [
            col for col in self.expected_columns
            if col not in dataframe.columns
        ]

        report["missing_columns"] = missing_cols

        if missing_cols:
            critical_errors.append(f"Missing columns: {missing_cols}")

        # -------------------------
        # EMPTY DATA CHECK
        # -------------------------
        if dataframe.empty:
            critical_errors.append("Dataset is empty")

        # -------------------------
        # MISSING VALUES
        # -------------------------
        missing_values = dataframe.isnull().sum().to_dict()
        report["missing_values"] = missing_values

        if sum(missing_values.values()) > 0:
            warnings.append("Missing values detected")

        # -------------------------
        # DUPLICATES (WARNING ONLY)
        # -------------------------
        duplicates = int(dataframe.duplicated().sum())
        report["duplicate_rows"] = duplicates

        if duplicates > 0:
            warnings.append(f"Duplicate rows found: {duplicates}")

        # -------------------------
        # TARGET VALIDATION
        # -------------------------
        if "target" in dataframe.columns:
            unique_targets = set(dataframe["target"].dropna().unique())
        else:
            unique_targets = set()

        report["target_distribution"] = list(unique_targets)

        if not unique_targets:
            critical_errors.append("Target column missing or empty")

        elif not unique_targets.issubset(self.valid_targets):
            critical_errors.append(
                f"Invalid target values: {unique_targets}"
            )

        # -------------------------
        # RANGE VALIDATION
        # -------------------------
        range_issues = {}

        if "age" in dataframe.columns:
            range_issues["age_out_of_range"] = int(
                ((dataframe["age"] < 0) | (dataframe["age"] > 120)).sum()
            )

        if "trestbps" in dataframe.columns:
            range_issues["bp_out_of_range"] = int(
                ((dataframe["trestbps"] < 50) | (dataframe["trestbps"] > 250)).sum()
            )

        if "chol" in dataframe.columns:
            range_issues["chol_out_of_range"] = int(
                ((dataframe["chol"] < 50) | (dataframe["chol"] > 600)).sum()
            )

        report["range_validation"] = range_issues

        if any(v > 0 for v in range_issues.values()):
            warnings.append("Medical range anomalies detected")

        # -------------------------
        # FINAL DECISION
        # -------------------------
        validation_passed = len(critical_errors) == 0

        report["critical_errors"] = critical_errors
        report["warnings"] = warnings
        report["validation_status"] = "PASS" if validation_passed else "FAIL"

        # -------------------------
        # SAVE REPORT (SAFE)
        # -------------------------
        os.makedirs(
            os.path.dirname(self.config.report_file_path),
            exist_ok=True
        )

        clean_report = self._to_python_type(report)

        with open(self.config.report_file_path, "w") as f:
            json.dump(clean_report, f, indent=4)

        # -------------------------
        # LOGGING
        # -------------------------
        if validation_passed:
            logging.info("Data validation PASSED")
        else:
            logging.error(f"Data validation FAILED: {critical_errors}")

        for w in warnings:
            logging.warning(w)

        logging.info(
            f"Validation report saved at {self.config.report_file_path}"
        )

        return validation_passed