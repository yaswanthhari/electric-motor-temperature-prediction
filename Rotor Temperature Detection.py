#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Cell 2: Load and explore data
df = pd.read_csv('pmsm_temperature_data.csv')
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

# Cell 3: Exploratory Data Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['torque', 'current', 'rpm', 'ambient_temp', 'coolant_temp', 'rotor_temp']
for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    axes[row, col].hist(df[feature], bins=50, edgecolor='black')
    axes[row, col].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Cell 4: Correlation Analysis
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Cell 5: Prepare features for training
feature_columns = ['torque', 'current', 'rpm', 'ambient_temp', 'coolant_temp']
X = df[feature_columns]
y = df['rotor_temp']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Cell 6: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open('transform.save', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved scaler as transform.save")

# Cell 7: Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("✅ Model training completed")

# Cell 8: Model evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} °C")
print(f"R² Score: {r2:.4f}")

# Cell 9: Feature importance
importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(importance['feature'], importance['importance'])
plt.title('Feature Importance for Rotor Temperature Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

# Cell 10: Save the model
with open('model.save', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved model as model.save")

# Cell 11: Test predictions
sample_data = X_test[:5]
sample_predictions = model.predict(sample_data)
actual_values = y_test[:5].values

print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"Predicted: {sample_predictions[i]:.2f}°C | Actual: {actual_values[i]:.2f}°C | Error: {abs(sample_predictions[i] - actual_values[i]):.2f}°C")

