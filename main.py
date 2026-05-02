import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/churn.csv")

# Drop ID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Handle missing values
df.fillna(0, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model (with class balancing)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions with threshold tuning
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_prob > threshold).astype(int)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
joblib.dump({
    "model": model,
    "scaler": scaler,
    "threshold": threshold
}, "models/churn_model.pkl")

print("\n✅ Model & Scaler saved successfully!")