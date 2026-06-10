#  Heart Disease End-to-End MLOps Pipeline

A complete end-to-end Machine Learning Operations (MLOps) system for heart disease prediction.

This project implements a production-grade ML pipeline including:
- Data engineering
- Model training
- Experiment tracking
- Model registry
- CI/CD automation
- Deployment readiness

---

##  Project Objective

The goal of this project is to build an automated ML system that:

- Predicts heart disease from clinical data
- Automates ML lifecycle (ingestion → training → deployment)
- Tracks experiments using MLflow
- Performs hyperparameter tuning using Optuna
- Supports CI/CD using GitHub Actions
- Enables model versioning and registry

---

##  System Architecture

Each stage is modular and implemented inside `src/components`.

---

##  Machine Learning Approach

### Models Used
- Random Forest Classifier
- XGBoost Classifier

### Hyperparameter Optimization
- Optuna for automatic tuning
- Cross-validation (K-Fold) for robustness

### Model Selection
- Best model selected using highest test accuracy

---

##  Data Pipeline

### 1. Data Ingestion
- Loads raw dataset
- Splits into train/test sets
- Saves artifacts for reproducibility

### 2. Data Validation
- Schema validation
- Missing value detection
- Duplicate detection
- Medical range checks (age, BP, cholesterol)

### 3. Data Transformation
- Feature scaling
- Encoding categorical variables
- Preprocessing pipeline creation

---

##  Experiment Tracking (MLflow)

MLflow is used for tracking experiments.

```bash



### Logged Metrics
- Accuracy
- Hyperparameters
- Model artifacts

---

## 🏷️ Model Registry

- Models are registered using MLflow Model Registry
- Supports:
  - Versioning
  - Stage transitions (Staging → Production)
  - Model loading for inference

---

## 🔁 CI/CD Pipeline (GitHub Actions)

Automated pipeline runs on every push to `main`:

### Steps:
1. Checkout repository
2. Setup Python environment
3. Install dependencies
4. Run training pipeline

```bash
python -m src.pipelines.training_pipeline

```
# ⚙️ Project Structure
```
heart-disease-mlops/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── model_registry.py
│   │   ├── model_pusher.py
│   │   ├── model_explainer.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   ├── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── config/
│       └── mlflow_config.py
│
├── data/
├── artifacts/
├── mlruns/
├── requirements.txt
├── setup.py
└── README.md
```
# Model Explainability (SHAP)
- SHAP (SHapley Additive exPlanations) is used
- Explains feature importance per prediction
- Improves interpretability for medical use cases


# Tech Stack
- Python 3.10+
- Scikit-learn
- XGBoost
- Optuna
- MLflow
- GitHub Actions
- Pandas
- NumPy

#  How to Run the Project
- Clone Repository
```bash
git clone https://github.com/Haruna-sani/end-end-smart-heart-disease-detection-MLOPs.git
cd heart-disease-mlops

```
