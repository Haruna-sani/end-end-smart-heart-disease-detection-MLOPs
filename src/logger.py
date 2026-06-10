"""
logger.py

Centralized logging configuration for the MLOps pipeline.

Provides:
- Console logging
- File logging
- Standard format across project


"""

import os
import logging
from datetime import datetime


# =========================
# LOG DIRECTORY SETUP
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"ml_pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
)


# =========================
# LOGGING CONFIGURATION
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


# =========================
# LOGGER OBJECT
# =========================
logger = logging.getLogger("heart_disease_ml")