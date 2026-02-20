from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
try:
    model = pickle.load(open('model.save', 'rb'))
    scaler = pickle.load(open('transform.save', 'rb'))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            torque = float(request.form['torque'])
            current = float(request.form['current'])
            rpm = float(request.form['rpm'])
            ambient_temp = float(request.form['ambient_temp'])
            coolant_temp = float(request.form['coolant_temp'])
            
            # Prepare features
            features = np.array([[torque, current, rpm, ambient_temp, coolant_temp]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            return render_template('index.html', 
                                 prediction_text=f'Predicted Rotor Temperature: {prediction:.2f}°C',
                                 torque=torque,
                                 current=current,
                                 rpm=rpm,
                                 ambient=ambient_temp,
                                 coolant=coolant_temp)
        except Exception as e:
            return render_template('index.html', 
                                 prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        features = np.array([[data['torque'], data['current'], data['rpm'], 
                            data['ambient_temp'], data['coolant_temp']]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return jsonify({'predicted_temperature': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)