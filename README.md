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
## screenshots
<img width="960" height="540" alt="ss1" src="https://github.com/user-attachments/assets/9a1bb764-3bf6-4daa-9e29-4f0acf827e2e" />
<img width="960" height="540" alt="ss2" src="https://github.com/user-attachments/assets/2815070f-4e21-49e5-81e1-615dc5a11051" />
<img width="960" height="540" alt="ss3" src="https://github.com/user-attachments/assets/456da9d0-ae36-46db-b246-54c3463fe625" />
<img width="960" height="540" alt="ss4" src="https://github.com/user-attachments/assets/da15a44f-2241-4daa-b8cf-7d8c6376210c" />
<img width="960" height="540" alt="ss5" src="https://github.com/user-attachments/assets/95cea224-94ba-4c08-aa52-d954ea1318f3" />
<img width="960" height="540" alt="ss6" src="https://github.com/user-attachments/assets/286e9f03-f373-436f-b307-1d996f761697" />
<img width="960" height="540" alt="ss8" src="https://github.com/user-attachments/assets/c764334d-24ca-42d4-b6a9-70fe7574278a" />


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
