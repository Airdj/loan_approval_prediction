
# Project Overview
## Quick Summary

- Problem: Predicting probabilities for obtaining a loan
- Model: LGBM Classification
- Metric: Area under the ROC curve(also count of False Positives)
- Best Validation ROC: 0.97277
- Dataset: Kaggle Loan Approval

## Features
- Advanced feature engineering
- Hyperparameter tuning with Optuna
- Stacking ensemble
- FastAPI + Gradio deployment
- Docker support
- MLflow experiment tracking
- Automated inference pipeline

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- LightGBM
- FastAPI
- Gradio
- Docker
- MLflow

# Project structure

The project follows a modular structure:
- `configs/` – configuration files
- `data/` – raw and processed datasets
- `features/` – feature engineering
- `notebooks/` – EDA
- `app/` – gradio + FastAPI
- `models/` – model definitions, training, tuning, evaluating

```bash
.
├── configs
│   ├── base_models_params.yaml
│   └── tuned_models_params.yaml
├── data
│   ├── processed
│   │   ├── processed_final_eval_df.csv
│   │   ├── processed_final_inference_df.csv
│   │   ├── processed_final_test_df.csv
│   │   └── processed_final_train_df.csv
│   └── raw
│       ├── credit_risk_dataset.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── notebooks
│   └── EDA_loan_approval_predictions.ipynb
├── src
│   ├── app
│   │   ├── app.py
│   │   └── main.py
│   ├── data
│   │   ├── load_data.py
│   │   └── preprocess_data.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   ├── evaluate.py
│   │   ├── model_factory.py
│   │   ├── train.py
│   │   └── tune.py
│   ├── pipelines
│   │   ├── inference_pipeline.py
│   │   └── train_pipeline.py
│   ├── serving
│   │   ├── base_model.pkl
│   │   ├── inference.py
│   │   └── tuned_model.pkl
│   ├── utils
│   │   ├── build_features_pipeline_artifacts.pkl
│   │   ├── columns.csv
│   │   ├── helpers.py
│   │   ├── preprocess_pipeline_artifacts.pkl
│   │   └── validate_data.py
│   └── __init__.py
├── tests
│   ├── test_fastapi.py
│   └── test_inference.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── docker_requirements.txt
├── mlflow.db
└── requirements.txt
```

## Define the problem and analyze it from a broader perspective

### 1. Define the objective in business terms

The goal of the project is to develop a machine learning model to predict whether a loan applicant should be approved, supporting banks and financial institutions in making more informed lending decisions. 
It is important to note that banks often adopt a conservative risk strategy, focusing on minimizing potential loan defaults, which may result in the rejection of some otherwise creditworthy applicants.


### 2. Describe how your solution will be used
My solution will be used as a real estate price prediction system and can be implemented in several forms, such as:

* a web application
* an API
* an analytical dashboard
* a reporting module
* integration with mobile applications

### 3. Identify existing solutions or workarounds (if any)
#### Traditional valuation methods:

* Rule based system
* Classical statistical scoring
* Scorecards

---

#### Existing technological solutions:

* Hybrid decision engines
* Alternative data scoring
* Real-time decisioning

### 4. In which categories should the problem be defined (unsupervised/supervised, incremental/static, etc.)?

* **Learning:** supervised
* **Prediction type:** classification (binary: approve / reject), optionally regression (probability of default, PD)
* **Learning mode:** initially static, later incremental / periodically retrained
* **Type of data:** tabular (core), optionally temporal (transaction history, behavioral trends) and geospatial (location stability, regional risk)

### 5. How will the model's performance be measured?

The results are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.


### 6. Is the performance measurement linked to the business objective?

The business objective is to develop a model that helps any financial institutions or banks with their decision-making in approving loans.
Model performance is evaluated using the ROC curve and the Area Under the ROC Curve (AUC), which directly reflects the model’s ability to distinguish between creditworthy and non-creditworthy applicants across all decision thresholds.
ROC-based evaluation directly supports risk-based lending decisions and portfolio profitability.

### 7. What is the minimum performance required to meet the business objective?

The minimum required performance is an ROC AUC ≥ 0.65, ensuring sufficient class separation to support risk-based credit decisions across operating points on the ROC curve.

### 8. Are there any comparable problems? Can you leverage existing experience or tools?

The problem is comparable to standard credit scoring and PD modeling tasks, where ROC/AUC-based evaluation, established modeling techniques, and mature tooling are already widely used and can be directly leveraged.

#### Tools:
* **Machine learning:** well-established statistical and machine learning models


### 9. How can the problem be solved manually?
The problem can be addressed manually through expert-driven credit assessment using predefined rules and analyst judgment, but this approach is not scalable and lacks consistent risk ranking compared to ROC-based model evaluation.


### 10. Make a list of assumptions established by you (or others)
#### Assumptions regarding the data:

* **Availability of historical data:** It is assumed that there is a sufficient amount of historical data about previous loans of a person.
* **High data quality:** It is assumed that the data does not contain significant gaps, errors, or anomalies. In case of missing values, it is assumed they can be filled using interpolation, medians, or other imputation techniques.
* **Consistency of units:** All data is measured in consistent units (e.g., person income in USD, person employment lenght in months).

---

#### **Assumptions regarding the model:**

* **Classification model:** It is assumed that the model will be a classification model, as the goal is to predict a binary outcome (approve/reject) or the probability of default.
* **Model simplicity at the beginning:** Initially, simpler and interpretable models will be used (e.g., logistic regression, decision trees) as a baseline. If these models do not achieve sufficient performance, more advanced methods such as gradient boosting (XGBoost, LightGBM) or neural networks may be applied.
* **Model stability:** The model is expected to maintain stable and consistent discrimination (as measured by ROC/AUC) on new, unseen data. Proper validation techniques such as cross-validation, hold-out sets, and back-testing will be applied.

---

#### **Assumptions regarding model performance:**

* **Expected discrimination level:** It is assumed that the model will achieve a ROC AUC sufficient for business needs, e.g., AUC ≥ 0.65, ensuring meaningful separation between creditworthy and non-creditworthy applicants..
* **Expected speed of operation:** The model should be fast enough for real-time decisioning, allowing instant credit approval decisions (e.g., for online loan applications).
* **Regular model updates:** The model will require periodic retraining or recalibration on new data to adapt to changing borrower behavior, economic conditions, and portfolio risk.
* **Stability across segments:** The model is expected to maintain consistent performance across customer segments and products, avoiding bias or degradation in specific subgroups.
* **Compliance and monitoring:** The model’s performance metrics (AUC, ROC) will be monitored continuously, ensuring alignment with business objectives and regulatory requirements.

# Data

## Data acquisition

### 1. Specify the type and amount of data needed
The provided data is in CSV format , separated to test and train set. Train set contains 58645 rows and 13 columns and the Test set contains 39098 rows and  12 columns. There is also an extension to train set with additional 32581 rows.

### 2. Identify the source from which you can obtain the data and document it
https://www.kaggle.com/competitions/playground-series-s4e10/data

https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction

### 3. Check how much storage space will be needed to store the data
The data requires less than 10 MB of disk space.


# Installation and Setup

In this section, detailed instructions are provided on how to set up the project on a local machine. Follow the steps below to ensure a smooth and reproducible environment.

## Codes and Resources Used

This section provides essential information about the software requirements used in this project.

- **Editor Used:** PyCharm  
- **Python Version:** Python 3.12  

It is recommended to use the same versions to avoid compatibility issues.

---

## Python Packages Used

Below is the list of dependencies required to run the project. It is recommended to install them inside a virtual environment.

### General Purpose
- joblib
- PyYAML 
- GreatExpectations
- FastAPI
- gradio

### Data Manipulation
- numpy 
- pandas 

### Data Visualization
- matplotlib
- seaborn

### Machine Learning
- scikit_learn 
- catboost 
- lightgbm 
- mlflow
- optuna
- xgboost

---

## Installation Steps

### 1. Clone the repository
```bash
cd <your-project-folder>
git clone https://github.com/Airdj/loan_approval_prediction.git
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermode -aG docker $USER
```

### 5. Run the container
```bash
docker compose up --build
```
### 6. Check FastAPI and Gradio
```bash
fastapi: 0.0.0.0/8000
gradio: 0.0.0.0/7860
```

## Notebook

The notebook contains:
- EDA
- Feature analysis
- Model comparison
# Modeling

## Models tested
- CatBoost Classifier
- KNeighbors Regression
- Decision Tree
- Random Forest
- AdaBoost Classifier
- SVC Classifier
- SGD Classifier
- Ridge Classification
- Logistic Regression
- Linear SVC
- Gradient Boosting Classification
- XGBoost Classification
- LightGBM Classification

## Final model
LGBM Classification with hyperparameter tuning

## Feature engineering
- Handling missing values
- Combine new features
- Encoding categorical variables
- Handling outliers
- Feature scaling
- Multicolinearity check

# Results and evaluation
Scores are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

Best performing model on Kaggle public test set: 0.97277

Best performing model on Kaggle private test set: 0.96794

### Base Model - Local Evaluation

**Metrics**

| Metric | Value |
|---------|---------:|
| Accuracy | 0.9264 |
| Precision | 0.9305 |
| Recall | 0.9264 |
| F1 Score | 0.9199 |

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 | 0.92 | 1.00 | 0.96 | 9082 |
| 1 | 0.98 | 0.63 | 0.77 | 2144 |
| **Accuracy** | - | - | **0.93** | **11226** |
| **Macro Avg** | 0.95 | 0.81 | 0.86 | 11226 |
| **Weighted Avg** | 0.93 | 0.93 | 0.92 | 11226 |

**Confusion Matrix**

| Actual \\ Predicted | 0 | 1 |
|--------------------|----:|----:|
| **0** | 9053 | 29 |
| **1** | 797 | 1347 |


### Tuned Model - Local Evaluation

**Metrics**

| Metric | Value |
|---------|---------:|
| Accuracy | 0.9253 |
| Precision | 0.9308 |
| Recall | 0.9253 |
| F1 Score | 0.9181 |

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 | 0.92 | 1.00 | 0.96 | 9082 |
| 1 | 0.99 | 0.61 | 0.76 | 2144 |
| **Accuracy** | - | - | **0.93** | **11226** |
| **Macro Avg** | 0.95 | 0.81 | 0.86 | 11226 |
| **Weighted Avg** | 0.93 | 0.93 | 0.92 | 11226 |

**Confusion Matrix**

| Actual \\ Predicted | 0 | 1 |
|--------------------|----:|----:|
| **0** | 9072 | 10 |
| **1** | 829 | 1315 |


## Validation
- Train/test split
- Cross-validation (e.g., kfold)
- Preventing data leakage
# Future work
- Deploying the model as an API (FastAPI)
- Creating a web application interface
- Monitoring model performance over time
- SHAP
- Feature importance analysis
