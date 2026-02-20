# fix_and_run.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("ELECTRIC MOTOR TEMPERATURE PREDICTION - FIX AND RUN")
print("=" * 60)

# Step 1: Check/Create dataset
dataset_file = 'pmsm_temperature_data.csv'

if os.path.exists(dataset_file) and os.path.getsize(dataset_file) > 0:
    print(f"✅ Dataset exists with size: {os.path.getsize(dataset_file):,} bytes")
    df = pd.read_csv(dataset_file)
else:
    print("📊 Creating new dataset...")
    np.random.seed(42)
    n_samples = 10000
    
    # Generate data
    torque = np.random.uniform(0, 200, n_samples)
    current = np.random.uniform(0, 500, n_samples)
    rpm = np.random.uniform(0, 6000, n_samples)
    ambient_temp = np.random.uniform(20, 35, n_samples)
    coolant_temp = ambient_temp + np.random.uniform(-2, 5, n_samples)
    rotor_temp = (ambient_temp + 20 + 0.1*current + 0.05*torque + 0.001*rpm + np.random.normal(0, 5, n_samples))
    
    df = pd.DataFrame({
        'torque': torque,
        'current': current,
        'rpm': rpm,
        'ambient_temp': ambient_temp,
        'coolant_temp': coolant_temp,
        'rotor_temp': rotor_temp
    })
    
    df.to_csv(dataset_file, index=False)
    print(f"✅ Created dataset with {n_samples} samples")
    print(f"   File size: {os.path.getsize(dataset_file):,} bytes")

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# Step 2: Train model
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

feature_columns = ['torque', 'current', 'rpm', 'ambient_temp', 'coolant_temp']
X = df[feature_columns]
y = df['rotor_temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('transform.save', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved transform.save")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("✅ Model trained")

# Save model
with open('model.save', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved model.save")

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance:")
print(f"   Mean Absolute Error: {mae:.2f}°C")
print(f"   R² Score: {r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔍 Feature Importance:")
for _, row in importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Test prediction
print("\n🎯 Test Predictions:")
test_samples = X_test.head(3)
actuals = y_test.head(3)
for i, (idx, sample) in enumerate(test_samples.iterrows()):
    sample_scaled = scaler.transform([sample.values])
    pred = model.predict(sample_scaled)[0]
    actual = actuals.iloc[i]
    print(f"   Sample {i+1}: Predicted={pred:.1f}°C, Actual={actual:.1f}°C, Error={abs(pred-actual):.1f}°C")

print("\n" + "=" * 60)
print("✅ FIX COMPLETE! Ready to run Flask app.")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python app.py")
print("2. Open browser: http://localhost:5000")