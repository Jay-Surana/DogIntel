# scripts/train_ecg.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ✅ Load dataset
df = pd.read_csv('datasets/ecg/dataset.csv')

# ✅ Clean and convert target
df['bad_ecg'] = df['bad_ecg'].apply(lambda x: 0 if x == '[]' else 1)

# ✅ Features to use
features = ['duration', 'weight', 'age']
X = df[features].fillna(0)
y = df['bad_ecg']

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)}")
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
os.makedirs("models/ecg_model", exist_ok=True)
joblib.dump(model, "models/ecg_model/ecg_rf_model.pkl")
