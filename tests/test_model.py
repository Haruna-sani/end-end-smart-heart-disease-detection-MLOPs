"""
test_model_real_data.py

Integration test for ModelTrainer using real dataset.
"""

import os
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# =========================
# TEST MODEL WITH REAL DATA
# =========================
def test_model_training_with_real_data():

    # =========================
    # PATH
    # =========================
    data_path = "data/heart.csv"
    assert os.path.exists(data_path), "Dataset not found!"

    # =========================
    # STEP 1: INGESTION
    # =========================
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(data_path)

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # =========================
    # STEP 2: TRANSFORMATION (IMPORTANT FIX)
    # =========================
    transformer = DataTransformation()

    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path,
        test_path
    )

    assert train_arr.shape[1] == test_arr.shape[1]

    # =========================
    # STEP 3: MODEL TRAINING
    # =========================
    trainer = ModelTrainer()

    output = trainer.initiate_model_trainer(train_arr, test_arr)

    # =========================
    # ASSERTIONS
    # =========================
    assert output["best_model"] is not None
    assert output["best_score"] > 0
    assert "model_name" in output