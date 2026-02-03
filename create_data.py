import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("⚡ Creating Electric Motor Temperature Dataset")
print("="*60)

# Create directory structure
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Generate realistic motor data
np.random.seed(42)
n_samples = 10000

# Timestamps - every 10 seconds
timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i*10) for i in range(n_samples)]

print(f"Generating {n_samples} samples...")

data = {
    'timestamp': timestamps,
    'profile_id': np.random.choice(['Motor_001', 'Motor_002', 'Motor_003', 'Motor_004'], n_samples),
    
    # Operational parameters
    'ambient': np.random.normal(25, 3, n_samples),  # Ambient temperature
    'current': np.random.uniform(10, 100, n_samples),  # Current in Amps
    'voltage': np.random.uniform(220, 480, n_samples),  # Voltage
    'rpm': np.random.uniform(1000, 3600, n_samples),  # Rotations per minute
    'torque': np.random.uniform(5, 50, n_samples),  # Torque in Nm
    'load_percentage': np.random.uniform(50, 100, n_samples),  # Load %
    'cooling_efficiency': np.random.uniform(0.7, 0.95, n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate realistic motor temperatures (main target)
# Formula: Temperature ≈ Ambient + (I²R losses) + (friction) - (cooling)
df['motor_temperature'] = (
    df['ambient'] + 
    0.8 * df['current']**2 * 0.001 +  # I²R heating
    0.3 * df['rpm'] * 0.001 +         # Friction heating
    0.4 * df['load_percentage'] * 0.5 - # Load effect
    df['cooling_efficiency'] * 10 +   # Cooling effect
    np.random.normal(0, 2, n_samples) # Random noise
)

# Add more features
df['electrical_power'] = df['current'] * df['voltage']  # Power in Watts
df['power_factor'] = df['electrical_power'] / (df['voltage'] * df['current'])
df['thermal_load'] = df['current']**2 * 0.8  # Simplified thermal load
df['temperature_rise'] = df['motor_temperature'] - df['ambient']

# Time-based features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_operating_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)

# Save to CSV
raw_path = 'data/raw/motor_temperature_dataset.csv'
df.to_csv(raw_path, index=False)

print(f"\n✅ Dataset created successfully!")
print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"💾 Saved to: {raw_path}")

print(f"\n📋 Columns ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n📈 Sample data (first 3 rows):")
print(df.head(3).to_string())

print(f"\n📊 Basic statistics:")
print(f"  Motor Temperature: {df['motor_temperature'].mean():.1f}°C (avg), "
      f"{df['motor_temperature'].min():.1f}°C to {df['motor_temperature'].max():.1f}°C")
print(f"  Current: {df['current'].mean():.1f}A (avg)")
print(f"  RPM: {df['rpm'].mean():.0f} (avg)")
print(f"  Load: {df['load_percentage'].mean():.0f}% (avg)")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Launch Jupyter: jupyter notebook")
print("2. Create new notebook in notebooks/ folder")
print("3. Start with: import pandas as pd")
print("4. Load data: df = pd.read_csv('data/raw/motor_temperature_dataset.csv')")
print("="*60)