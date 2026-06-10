"""
training_pipeline.py

End-to-end orchestration pipeline for Heart Disease ML system.

This pipeline:
- Loads raw data
- Validates data quality
- Transforms features
- Trains models (Optuna + RF + XGBoost)
- Evaluates best model
- Registers model to MLflow Model Registry
- Promotes model to Production
"""

import os
import logging
import mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_registry import ModelRegistry


# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# TRAINING PIPELINE
# =========================
class TrainingPipeline:
    """
    Orchestrates full ML training workflow.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

        # Components
        self.ingestion = DataIngestion()
        self.validation = DataValidation()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluation()
        self.registry = ModelRegistry()

    # =========================
    # MAIN EXECUTION METHOD
    # =========================
    def run_pipeline(self):
        """
        Executes full training pipeline step-by-step.
        """

        try:
            logging.info("🚀 Starting Training Pipeline")

            # =========================
            # STEP 1: DATA INGESTION
            # =========================
            train_path, test_path = self.ingestion.initiate_data_ingestion(
                self.data_path
            )

            # =========================
            # STEP 2: DATA VALIDATION
            # =========================
            import pandas as pd
            raw_df = pd.read_csv(self.data_path)

            validation_status = self.validation.validate_data(raw_df)

            if not validation_status:
                raise Exception("Data validation failed. Pipeline stopped.")

            # =========================
            # STEP 3: DATA TRANSFORMATION
            # =========================
            train_arr, test_arr, _ = (
                self.transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            # =========================
            # STEP 4: MODEL TRAINING (MLflow run happens inside trainer)
            # =========================
            training_output = self.trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            best_model = training_output["best_model"]
            run_id = training_output["run_id"]   # 🔥 IMPORTANT FIX

            # =========================
            # STEP 5: MODEL EVALUATION
            # =========================
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            metrics = self.evaluator.evaluate_model(
                best_model,
                X_test,
                y_test
            )

            # =========================
            # STEP 6: MODEL REGISTRY (MLflow Model Registry)
            # =========================

            logging.info("📦 Registering model in MLflow Registry...")

            version = self.registry.register_model(
                model=best_model,
                run_id=run_id
            )

            self.registry.transition_stage(
                version=version,
                stage="Production"
            )

            # =========================
            # COMPLETION
            # =========================

            logging.info("🎯 Pipeline completed successfully")
            logging.info(f"Model Version: {version}")
            logging.info(f"Final Metrics: {metrics}")

            return {
                "model_version": version,
                "metrics": metrics,
                "model": best_model,
                "run_id": run_id
            }

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise e


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    DATA_PATH = os.path.join("data", "heart.csv")

    pipeline = TrainingPipeline(DATA_PATH)
    pipeline.run_pipeline()