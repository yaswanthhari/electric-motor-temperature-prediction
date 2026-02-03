"""
Create and save the scaler file that's missing
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("⚡ Creating missing scaler file...")

# Load the training data to fit the scaler
train_df = pd.read_csv('data/processed/train_data.csv')

# Remove target column
feature_columns = [col for col in train_df.columns if col != 'motor_temperature']
X_train = train_df[feature_columns].values

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

print(f"✅ Scaler created and saved to: models/scaler.pkl")
print(f"   Features scaled: {len(feature_columns)}")
print(f"   Feature names: {feature_columns}")

# Test that it works
print("\n🔧 Testing scaler...")
test_data = np.array([[0, 25, 75, 1000, 45, 17100, 2850, 0.85, 14, 1]])
scaled_data = scaler.transform(test_data)
print(f"   Original: {test_data[0]}")
print(f"   Scaled: {scaled_data[0]}")
print(f"   Mean after scaling: {scaled_data.mean():.6f}")
print(f"   Std after scaling: {scaled_data.std():.6f}")