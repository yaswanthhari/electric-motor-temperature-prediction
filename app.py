"""
Electric Motor Temperature Prediction API - Final Version
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

print("⚡ Loading ML model and scaler...")

# Load model and scaler
try:
    model = joblib.load('models/best_model_linear_regression.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
    
    # Load feature names from training data
    train_df = pd.read_csv('data/processed/train_data.csv')
    FEATURE_NAMES = [col for col in train_df.columns if col != 'motor_temperature']
    
    print(f"   Model: Linear Regression (R² = 1.0000)")
    print(f"   Features: {len(FEATURE_NAMES)}")
    print(f"   Feature names: {FEATURE_NAMES}")
    
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("⚠️ Using fallback mode")
    model = None
    scaler = None
    FEATURE_NAMES = []

print("="*60)
print("⚡ Electric Motor Temperature Prediction API")
print("="*60)
print(f"API running on: http://localhost:5000")
print("="*60)

def predict_with_model(features_array):
    """Make prediction using the loaded model"""
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded")
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction

@app.route('/')
def home():
    """Home page with API documentation"""
    status_color = "green" if model else "red"
    status_text = "Operational" if model else "Limited Mode"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Electric Motor Temperature Prediction API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 20px 0; }}
            .operational {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .limited {{ background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
            .endpoint {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            code {{ background: #eee; padding: 2px 5px; border-radius: 3px; }}
            .example {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }}
        </style>
    </head>
    <body>
        <h1>⚡ Electric Motor Temperature Prediction API</h1>
        
        <div class="status {'operational' if model else 'limited'}">
            <h3>Status: <span style="color: {status_color}">● {status_text}</span></h3>
            <p>Model: {'Linear Regression (R² = 1.0000)' if model else 'Simplified Predictor'}</p>
            <p>Features: {len(FEATURE_NAMES) if FEATURE_NAMES else '10'}</p>
        </div>
        
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Make a single temperature prediction</p>
            <div class="example">
                <p><strong>Example Request:</strong></p>
                <code>
POST /predict
Content-Type: application/json

{{
    "ambient": 25.0,
    "current": 45.5,
    "voltage": 380.0,
    "rpm": 2850.0,
    "load_percentage": 78.9,
    "cooling_efficiency": 0.85,
    "hour": 14
}}
                </code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>POST /batch_predict</h3>
            <p>Make multiple predictions at once</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check API health and model status</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /features</h3>
            <p>Get information about expected features</p>
        </div>
        
        <h2>Testing the API</h2>
        <p>You can test the API using curl:</p>
        <code>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"ambient": 25.0, "current": 45.5, "voltage": 380.0, "rpm": 2850.0, "load_percentage": 78.9, "cooling_efficiency": 0.85, "hour": 14}}'
        </code>
        
        <h2>Temperature Thresholds</h2>
        <ul>
            <li>✅ <strong>NORMAL:</strong> &lt; 80°C (Green)</li>
            <li>⚠️ <strong>WARNING:</strong> 80°C - 100°C (Orange)</li>
            <li>🚨 <strong>CRITICAL:</strong> &gt;= 100°C (Red)</li>
        </ul>
    </body>
    </html>
    """

@app.route('/health', methods=['GET'])
def health_check():
    """Check API health and model status"""
    if model is None or scaler is None:
        return jsonify({
            'status': 'limited',
            'message': 'API running in limited mode (no model files)',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    # Test prediction with sample data
    try:
        sample_features = np.array([[0, 25, 75, 1000, 45, 17100, 2850, 0.85, 14, 1]])
        prediction = predict_with_model(sample_features)
        
        return jsonify({
            'status': 'healthy',
            'model': 'Linear Regression',
            'accuracy': 'R² = 1.0000',
            'test_prediction': float(prediction),
            'message': 'API is fully operational',
            'features_expected': len(FEATURE_NAMES),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Model test failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get information about expected features"""
    return jsonify({
        'feature_count': len(FEATURE_NAMES),
        'features': FEATURE_NAMES,
        'description': 'Features expected by the model in exact order',
        'note': 'temperature_rise = motor_temperature - ambient (calculated internally)'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a single temperature prediction"""
    try:
        # Get JSON data
        data = request.json
        
        # Validate required fields
        required_fields = ['ambient', 'current', 'voltage', 'rpm', 'load_percentage']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract and calculate features
        ambient = float(data['ambient'])
        current = float(data['current'])
        voltage = float(data['voltage'])
        rpm = float(data['rpm'])
        load_percentage = float(data['load_percentage'])
        cooling_efficiency = float(data.get('cooling_efficiency', 0.85))
        hour = data.get('hour', datetime.now().hour)
        
        # Calculate derived features (matching training)
        electrical_power = current * voltage
        thermal_load = current ** 2 * 0.8
        
        # For is_operating_hour (8 AM to 6 PM)
        is_operating_hour = 1 if 8 <= hour <= 18 else 0
        
        # Create feature array in correct order
        features = np.array([[
            0,  # temperature_rise (placeholder - will be updated)
            ambient,
            load_percentage,
            thermal_load,
            current,
            electrical_power,
            rpm,
            cooling_efficiency,
            hour,
            is_operating_hour
        ]])
        
        # Predict temperature (iterative to get temperature_rise right)
        if model and scaler:
            # First prediction
            predicted_temp = predict_with_model(features)
            
            # Calculate actual temperature_rise
            temperature_rise = predicted_temp - ambient
            
            # Update features with correct temperature_rise
            features[0, 0] = temperature_rise
            
            # Final prediction
            predicted_temp = predict_with_model(features)
        else:
            # Fallback calculation
            temperature_rise = 0.8 * current + 0.001 * rpm + 0.4 * load_percentage - 10 * cooling_efficiency
            predicted_temp = ambient + temperature_rise
        
        # Determine status
        if predicted_temp < 80:
            status = "NORMAL"
            recommendation = "✅ Motor operating within safe limits"
            color = "green"
            emoji = "✅"
        elif predicted_temp < 100:
            status = "WARNING"
            recommendation = "⚠️ Temperature approaching critical level. Consider reducing load by 10-15%."
            color = "orange"
            emoji = "⚠️"
        else:
            status = "CRITICAL"
            recommendation = "🚨 IMMEDIATE SHUTDOWN REQUIRED! Temperature exceeds safe limits."
            color = "red"
            emoji = "🚨"
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'motor_temperature': round(float(predicted_temp), 2),
                'temperature_rise': round(float(temperature_rise), 2),
                'status': status,
                'color': color,
                'emoji': emoji,
                'timestamp': datetime.now().isoformat()
            },
            'recommendation': recommendation,
            'input_parameters': {
                'ambient': ambient,
                'current': current,
                'voltage': voltage,
                'rpm': rpm,
                'load_percentage': load_percentage,
                'cooling_efficiency': cooling_efficiency,
                'hour': hour
            },
            'calculated_features': {
                'electrical_power': round(electrical_power, 2),
                'thermal_load': round(thermal_load, 2),
                'is_operating_hour': is_operating_hour
            },
            'model_info': {
                'type': 'Linear Regression' if model else 'Simplified Predictor',
                'accuracy': 'R² = 1.0000' if model else 'Approximate',
                'features_used': len(FEATURE_NAMES) if FEATURE_NAMES else 10
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple data points"""
    try:
        data = request.json
        
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                'success': False,
                'error': 'Expected {"data": list_of_objects}'
            }), 400
        
        predictions = []
        
        for item in data['data']:
            try:
                # Extract parameters
                ambient = float(item.get('ambient', 25))
                current = float(item.get('current', 45))
                voltage = float(item.get('voltage', 380))
                rpm = float(item.get('rpm', 2850))
                load_percentage = float(item.get('load_percentage', 75))
                cooling_efficiency = float(item.get('cooling_efficiency', 0.85))
                hour = item.get('hour', datetime.now().hour)
                
                # Calculate features
                electrical_power = current * voltage
                thermal_load = current ** 2 * 0.8
                is_operating_hour = 1 if 8 <= hour <= 18 else 0
                
                # Create feature array
                features = np.array([[
                    0,  # temperature_rise (placeholder)
                    ambient,
                    load_percentage,
                    thermal_load,
                    current,
                    electrical_power,
                    rpm,
                    cooling_efficiency,
                    hour,
                    is_operating_hour
                ]])
                
                # Predict
                if model and scaler:
                    # First prediction
                    predicted_temp = predict_with_model(features)
                    
                    # Calculate temperature_rise
                    temperature_rise = predicted_temp - ambient
                    features[0, 0] = temperature_rise
                    
                    # Final prediction
                    predicted_temp = predict_with_model(features)
                else:
                    # Fallback
                    temperature_rise = 0.8 * current + 0.001 * rpm + 0.4 * load_percentage - 10 * cooling_efficiency
                    predicted_temp = ambient + temperature_rise
                
                # Determine status
                if predicted_temp < 80:
                    status = "NORMAL"
                elif predicted_temp < 100:
                    status = "WARNING"
                else:
                    status = "CRITICAL"
                
                predictions.append({
                    'motor_temperature': round(float(predicted_temp), 2),
                    'temperature_rise': round(float(temperature_rise), 2),
                    'status': status,
                    'input_parameters': item
                })
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'input_parameters': item
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)