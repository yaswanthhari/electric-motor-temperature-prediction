# -*- coding: utf-8 -*- 
from flask import Flask, request, jsonify, render_template 
import pickle 
import numpy as np 
import os 
 
app = Flask(__name__) 
 
# Load models 
model = None 
scaler = None 
 
try: 
    model_path = 'model.save' 
    transform_path = 'transform.save' 
    print(f"Loading model from {model_path}") 
    with open(model_path, 'rb') as f: 
        model = pickle.load(f) 
    print("? Model loaded") 
    with open(transform_path, 'rb') as f: 
        scaler = pickle.load(f) 
    print("? Scaler loaded") 
    models_loaded = True 
except Exception as e: 
    print(f"? Error: {e}") 
    models_loaded = False 
 
@app.route('/') 
def home(): 
    """Main web interface""" 
    return render_template('index.html', models_loaded=models_loaded) 
 
@app.route('/health') 
def health(): 
    """Health check endpoint""" 
    return jsonify({ 
        "status": "healthy" if models_loaded else "degraded", 
        "model_loaded": model is not None, 
        "scaler_loaded": scaler is not None, 
        "service": "IBM Motor Temperature Scoring Endpoint" 
    }) 
 
@app.route('/info') 
def info(): 
    """Model information endpoint""" 
    if not models_loaded: 
        return jsonify({"error": "Models not loaded"}), 503 
    return jsonify({ 
        "model_type": str(type(model).__name__), 
        "scaler_type": str(type(scaler).__name__), 
        "features": ["torque", "current", "rpm", "ambient_temp", "coolant_temp"], 
        "models_loaded": True 
    }) 
 
@app.route('/score', methods=['POST']) 
def score(): 
    """Make a temperature prediction""" 
    if not models_loaded: 
        return jsonify({"error": "Models not loaded"}), 503 
    try: 
        data = request.get_json() 
        if not data: 
            return jsonify({"error": "No JSON data received"}), 400 
 
        # Validate required fields 
        required = ['torque', 'current', 'rpm', 'ambient_temp', 'coolant_temp'] 
        for field in required: 
            if field not in data: 
                return jsonify({"error": f"Missing field: {field}"}), 400 
 
        # Prepare features 
        features = np.array([[ 
            float(data['torque']), 
            float(data['current']), 
            float(data['rpm']), 
            float(data['ambient_temp']), 
            float(data['coolant_temp']) 
        ]]) 
 
        # Scale and predict 
        features_scaled = scaler.transform(features) 
        prediction = model.predict(features_scaled)[0] 
 
        return jsonify({ 
            "prediction": float(prediction), 
            "status": "success" 
        }) 
    except ValueError as e: 
        return jsonify({"error": f"Invalid value: {e}"}), 400 
    except Exception as e: 
        return jsonify({"error": str(e)}), 500 
 
@app.route('/api') 
def api(): 
    """API information endpoint""" 
    return jsonify({ 
        "service": "IBM Motor Temperature Scoring Endpoint", 
        "version": "1.0", 
        "endpoints": { 
            "GET /": "Web interface", 
            "GET /health": "Health check", 
            "GET /info": "Model information", 
            "GET /api": "This information", 
            "POST /score": "Make prediction (send JSON)" 
        }, 
        "example_request": { 
            "torque": 150, 
            "current": 250, 
            "rpm": 3000, 
            "ambient_temp": 25, 
            "coolant_temp": 30 
        } 
    }) 
 
@app.errorhandler(404) 
def not_found(e): 
    """Handle 404 errors""" 
    return jsonify({ 
        "error": "Endpoint not found", 
        "available_endpoints": [ 
            "GET /", "GET /health", "GET /info", "GET /api", "POST /score" 
        ] 
    }), 404 
 
if __name__ == '__main__': 
    print("\n" + "="*60) 
    print("IBM MOTOR TEMPERATURE SCORING ENDPOINT") 
    print("="*60) 
    print(f"Models loaded: {models_loaded}") 
    print(f"\nAvailable endpoints:") 
    print(f"  ?? Web Interface: http://localhost:5001/") 
    print(f"  ?? Health check:  http://localhost:5001/health") 
    print(f"  ??  Model info:    http://localhost:5001/info") 
    print(f"  ?? API info:      http://localhost:5001/api") 
    print(f"  ?? Prediction:    POST to http://localhost:5001/score") 
    print("="*60 + "\n") 
    app.run(debug=True, port=5001) 
