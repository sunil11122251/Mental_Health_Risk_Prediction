import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ LOAD DATA ------------------

data = pd.read_csv("/content/Student Mental health.csv")

# Forward fill missing values
data = data.ffill()

# Drop unnecessary columns
data = data.drop(['Timestamp', 'What is your course?'], axis=1)

# ------------------ FIX CGPA COLUMN ------------------

cgpa_mapping = {
    '0 - 1.99': 1.5,
    '2.00 - 2.49': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 4.00': 3.75
}

data['What is your CGPA?'] = data['What is your CGPA?'].map(cgpa_mapping)

# Remove rows with unmapped CGPA values
data.dropna(inplace=True)

# ------------------ CREATE RISK LEVEL TARGET ------------------
# Based on Depression, Anxiety and Panic

def assign_risk(row):
    if row['Do you have Depression?'] == 'Yes' and row['Do you have Anxiety?'] == 'Yes':
        return 'High'
    elif row['Do you have Depression?'] == 'Yes' or row['Do you have Panic attack?'] == 'Yes':
        return 'Medium'
    else:
        return 'Low'

data['Risk_Level'] = data.apply(assign_risk, axis=1)

# ------------------ ENCODE CATEGORICAL COLUMNS ------------------

le = LabelEncoder()

categorical_cols = [
    'Choose your gender',
    'Your current year of Study',
    'Marital status',
    'Do you have Anxiety?',
    'Do you have Panic attack?',
    'Did you seek any specialist for a treatment?',
    'Risk_Level'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# ------------------ FEATURES & TARGET ------------------

X = data.drop(['Do you have Depression?', 'Risk_Level'], axis=1)
y = data['Risk_Level']   # Target: Low / Medium / High

# Save feature names for later use
feature_names = X.columns.tolist()
print("Features used:", feature_names)

# ------------------ SCALING ------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------ TRAIN TEST SPLIT ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL TRAINING ------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------ MODEL EVALUATION ------------------

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ------------------ SAMPLE PREDICTION ------------------
# Format:
# [Gender, Age, Year, CGPA, Marital, Anxiety, Panic, Treatment]

sample_df = pd.DataFrame(
    [[1, 21, 2, 3.25, 1, 1, 0, 0]],
    columns=feature_names
)

# Scale sample
sample_scaled = scaler.transform(sample_df)

# Predict
pred_class = model.predict(sample_scaled)[0]
pred_prob = model.predict_proba(sample_scaled).max()

# Convert numeric back to label
risk_label = le.inverse_transform([pred_class])[0]

# Recommendation logic
if risk_label == 'High':
    recommendation = "Immediate counseling and professional support required"
elif risk_label == 'Medium':
    recommendation = "Regular monitoring and stress management advised"
else:
    recommendation = "No immediate risk, maintain healthy lifestyle"

print("\n--- FINAL OUTPUT ---")
print("Predicted Risk Level:", risk_label)
print("Risk Probability Score:", round(pred_prob, 2))
print("Recommendation:", recommendation)
