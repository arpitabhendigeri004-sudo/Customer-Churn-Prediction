# Customer Churn Prediction Model 🚀

## 📌 Overview

An end-to-end Machine Learning project to predict customer churn using telecom data.
Includes data preprocessing, model training, evaluation, and deployment via FastAPI.

---

## 🧠 Problem Statement

Predict whether a customer will churn in the next billing cycle to help businesses take proactive retention actions.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Random Forest)
* Matplotlib, Seaborn
* FastAPI (Deployment)
* Uvicorn

---

## 📊 Features

* Data cleaning & preprocessing
* Label encoding
* Feature scaling
* Class imbalance handling
* Threshold tuning
* Model evaluation (Confusion Matrix, Classification Report)
* Feature importance visualization
* REST API for real-time predictions

---

## 🚀 How to Run

### 1. Clone repo

```bash
git clone <your-repo-link>
cd Customer-Churn-Prediction
```

### 2. Create environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train model

```bash
python main.py
```

### 5. Run API

```bash
uvicorn src.app:app --reload
```

### 6. Open Swagger

http://127.0.0.1:8000/docs

---

## 📈 Sample Output

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.26
}
```

---

## 📊 Visual Outputs

* Confusion Matrix
* Feature Importance
* Churn Distribution
* Correlation Heatmap

---

## 🧠 Key Learnings

* Handling imbalanced datasets
* Improving recall using threshold tuning
* Maintaining training-serving consistency
* Deploying ML model using FastAPI

---

## 🎯 Industry Relevance

Used in telecom, SaaS, fintech, OTT platforms for:

* Customer retention
* Revenue optimization
* Targeted marketing

---

## 👩‍💻 Author

Arpita Bhendigeri
