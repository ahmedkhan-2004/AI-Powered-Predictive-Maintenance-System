import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Loading and preparing data...")
# Use processed data file
if os.path.exists("processed_sensor_data.csv"):
    df = pd.read_csv("processed_sensor_data.csv")
else:
    print("Processed data file not found. Run data_processing.py first.")
    exit()

# Feature selection
feature_columns = ["Temperature", "Humidity", "Vibration", 
                  "AccelMagnitude", "GyroMagnitude", 
                  "Temp_Change", "Accel_Change"]
X = df[feature_columns]
y = df["Failure"]

# Check class balance
print(f"Class distribution - Normal: {sum(y==0)}, Failure: {sum(y==1)}")

# If highly imbalanced, add synthetic failure data
if sum(y==1) < 5:
    print("Adding synthetic failure data...")
    # Find thresholds for abnormal conditions
    temp_threshold = df["Temperature"].quantile(0.85)
    accel_threshold = df["AccelMagnitude"].quantile(0.85)
    
    # Create synthetic failures
    n_synthetic = min(20, len(df) // 10)
    normal_samples = df[y==0].sample(n_synthetic)
    
    # Modify values to simulate failures
    synthetic_failures = normal_samples.copy()
    synthetic_failures["Temperature"] = temp_threshold + np.random.uniform(1, 5, n_synthetic)
    synthetic_failures["AccelMagnitude"] = accel_threshold + np.random.uniform(1, 5, n_synthetic)
    synthetic_failures["Vibration"] = 1
    synthetic_failures["Failure"] = 1
    
    # Add to dataset
    df = pd.concat([df, synthetic_failures])
    X = df[feature_columns]
    y = df["Failure"]
    print(f"Updated class distribution - Normal: {sum(y==0)}, Failure: {sum(y==1)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# Save the model and scaler
joblib.dump(model, "predictive_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\nModel, scaler, and feature columns saved to disk")
print("Model training complete!")

# Print thresholds for use in the dashboard
print("\nSuggested thresholds for dashboard:")
print(f"Temperature warning: > {df['Temperature'].quantile(0.75):.1f}°C")
print(f"Temperature danger: > {df['Temperature'].quantile(0.90):.1f}°C")
print(f"Acceleration warning: > {df['AccelMagnitude'].quantile(0.75):.2f}")
print(f"Acceleration danger: > {df['AccelMagnitude'].quantile(0.90):.2f}")
import sys
sys.exit()
