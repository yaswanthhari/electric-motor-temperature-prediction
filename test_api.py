"""
Test script for the Electric Motor Temperature Prediction API
"""

import requests
import json
import time

def test_api():
    """Test the prediction API"""
    
    base_url = "http://localhost:5000"
    
    print("⚡ Testing Electric Motor Temperature Prediction API")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Single prediction
    print("\n2. Testing single prediction...")
    test_data = {
        "ambient": 25.5,
        "current": 45.5,
        "voltage": 380.0,
        "rpm": 2850.0,
        "load_percentage": 78.9,
        "cooling_efficiency": 0.85,
        "hour": 14
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        if result.get('success'):
            pred = result['prediction']
            print(f"   ✅ Prediction successful!")
            print(f"   Motor Temperature: {pred['motor_temperature']}°C")
            print(f"   Status: {pred['status']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Batch prediction
    print("\n3. Testing batch prediction...")
    batch_data = {
        "data": [
            {
                "ambient": 25.0,
                "current": 40.0,
                "voltage": 380.0,
                "rpm": 2500.0,
                "load_percentage": 70.0,
                "cooling_efficiency": 0.9
            },
            {
                "ambient": 30.0,
                "current": 60.0,
                "voltage": 400.0,
                "rpm": 3000.0,
                "load_percentage": 90.0,
                "cooling_efficiency": 0.7
            },
            {
                "ambient": 20.0,
                "current": 30.0,
                "voltage": 350.0,
                "rpm": 2000.0,
                "load_percentage": 60.0,
                "cooling_efficiency": 0.95
            }
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/batch_predict", json=batch_data)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        if result.get('success'):
            print(f"   ✅ Batch prediction successful!")
            print(f"   Number of predictions: {result['count']}")
            
            for i, pred in enumerate(result['predictions'][:2]):  # Show first 2
                if 'error' not in pred:
                    print(f"\n   Prediction {i+1}:")
                    print(f"     Temperature: {pred['motor_temperature']}°C")
                    print(f"     Status: {pred['status']}")
                else:
                    print(f"   ❌ Prediction {i+1} failed: {pred['error']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Get features
    print("\n4. Getting feature information...")
    try:
        response = requests.get(f"{base_url}/features")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Features expected: {data['feature_count']}")
        print(f"   Features: {', '.join(data['features'][:5])}...")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ API Testing Complete!")
    print("\nTo run the API:")
    print("   python app.py")
    print("\nTo run the dashboard:")
    print("   streamlit run dashboard.py")
    print("="*60)

if __name__ == "__main__":
    test_api()