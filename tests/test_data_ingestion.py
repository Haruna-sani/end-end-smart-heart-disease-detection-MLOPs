"""
test_data_ingestion_unit.py
"""

import os
from src.components.data_ingestion import DataIngestion


def test_data_ingestion_only(tmp_path):

    # fake dataset path (or small real sample)
    data_path = "data/heart.csv"

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion(data_path)

    # ONLY ingestion assertions
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # DO NOT test model here