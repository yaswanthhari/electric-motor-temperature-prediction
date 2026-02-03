"""
Command-line predictor for electric motor temperature
"""

import joblib
import numpy as np
import argparse
import sys

def predict_temperature(args):
    """Predict motor temperature from command line arguments"""
    
    try:
        # Load model and scaler
        model = joblib.load('models/best_model_linear_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Calculate derived features
        electrical_power = args.current * args.voltage
        thermal_load = args.current ** 2 * 0.8
        is_operating_hour = 1 if 8 <= args.hour <= 18 else 0
        
        # Create feature array
        features = np.array([[
            0,  # temperature_rise (placeholder)
            args.ambient,
            args.load,
            thermal_load,
            args.current,
            electrical_power,
            args.rpm,
            args.cooling,
            args.hour,
            is_operating_hour
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        predicted_temp = model.predict(features_scaled)[0]
        
        # Calculate temperature rise and predict again
        temperature_rise = predicted_temp - args.ambient
        features[0, 0] = temperature_rise
        features_scaled = scaler.transform(features)
        predicted_temp = model.predict(features_scaled)[0]
        
        # Determine status
        if predicted_temp < 80:
            status = "NORMAL"
            emoji = "✅"
        elif predicted_temp < 100:
            status = "WARNING"
            emoji = "⚠️"
        else:
            status = "CRITICAL"
            emoji = "🚨"
        
        # Print results
        print("\n" + "="*60)
        print("⚡ ELECTRIC MOTOR TEMPERATURE PREDICTION")
        print("="*60)
        print(f"\n📊 Input Parameters:")
        print(f"   Ambient Temperature: {args.ambient}°C")
        print(f"   Current: {args.current}A")
        print(f"   Voltage: {args.voltage}V")
        print(f"   RPM: {args.rpm}")
        print(f"   Load: {args.load}%")
        print(f"   Cooling Efficiency: {args.cooling}")
        print(f"   Hour of Day: {args.hour}:00")
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Motor Temperature: {predicted_temp:.2f}°C")
        print(f"   Temperature Rise: {temperature_rise:.2f}°C")
        print(f"   Status: {emoji} {status}")
        
        print(f"\n💡 Recommendation:")
        if status == "NORMAL":
            print("   Motor operating within safe limits. Continue normal operation.")
        elif status == "WARNING":
            print("   Temperature approaching critical level. Consider reducing load by 10-15%.")
        else:
            print("   CRITICAL! Immediate shutdown required. Inspect motor and cooling system.")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model (run 03_modeling.ipynb)")
        print("2. Model files exist in models/ folder")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Predict electric motor temperature',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --ambient 25 --current 45 --voltage 380 --rpm 2850 --load 75
  python predict.py -a 30 -c 60 -v 400 -r 3000 -l 90 --cooling 0.7 --hour 15
        """
    )
    
    parser.add_argument('-a', '--ambient', type=float, required=True,
                       help='Ambient temperature in °C')
    parser.add_argument('-c', '--current', type=float, required=True,
                       help='Motor current in Amps')
    parser.add_argument('-v', '--voltage', type=float, required=True,
                       help='Motor voltage in Volts')
    parser.add_argument('-r', '--rpm', type=float, required=True,
                       help='Motor speed in RPM')
    parser.add_argument('-l', '--load', type=float, default=75.0,
                       help='Load percentage (default: 75)')
    parser.add_argument('--cooling', type=float, default=0.85,
                       help='Cooling efficiency 0.0-1.0 (default: 0.85)')
    parser.add_argument('--hour', type=int, default=12,
                       help='Hour of day 0-23 (default: 12)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.ambient < 0 or args.ambient > 50:
        print("❌ Ambient temperature must be between 0-50°C")
        sys.exit(1)
    if args.current < 0 or args.current > 200:
        print("❌ Current must be between 0-200A")
        sys.exit(1)
    if args.voltage < 0 or args.voltage > 500:
        print("❌ Voltage must be between 0-500V")
        sys.exit(1)
    if args.rpm < 0 or args.rpm > 4000:
        print("❌ RPM must be between 0-4000")
        sys.exit(1)
    if args.load < 0 or args.load > 100:
        print("❌ Load must be between 0-100%")
        sys.exit(1)
    if args.cooling < 0 or args.cooling > 1:
        print("❌ Cooling efficiency must be between 0.0-1.0")
        sys.exit(1)
    if args.hour < 0 or args.hour > 23:
        print("❌ Hour must be between 0-23")
        sys.exit(1)
    
    predict_temperature(args)

if __name__ == "__main__":
    main()