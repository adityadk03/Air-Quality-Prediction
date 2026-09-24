"""
Air Quality Prediction and Environmental Analytics System
Flask Backend REST API
Academic Internship Program - IBM SkillsBuild & BharatCares / AICTE
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'AirQualityUCI.csv')

# Load trained artifacts
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
IMPUTER_PATH = os.path.join(MODEL_DIR, 'imputer.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)
    print("[Flask Backend] Successfully loaded model, scaler, imputer, and metrics.")
except Exception as e:
    print(f"[Flask Backend] Warning: Artifact loading error: {e}")
    model, scaler, imputer, metrics_data = None, None, None, None

EXPECTED_FEATURES = [
    'PT08.S1(CO)',
    'C6H6(GT)',
    'PT08.S2(NMHC)',
    'NOx(GT)',
    'PT08.S3(NOx)',
    'NO2(GT)',
    'PT08.S4(NO2)',
    'PT08.S5(O3)',
    'T',
    'RH',
    'AH'
]

# Cache cleaned dataset summary for /dataset endpoint
_dataset_cache = None

def get_dataset_summary():
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    try:
        df_raw = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        df = df_raw.iloc[:, :15].copy()
        df = df.dropna(how='all')
        df = df[df['Date'].notna()].copy()
        df = df.replace(-200, np.nan).replace(-200.0, np.nan)

        # Compute summary metrics
        numeric_cols = [c for c in df.columns if c not in ['Date', 'Time']]
        missing_counts = {col: int(df[col].isna().sum()) for col in numeric_cols}
        missing_pct = {col: round(float(df[col].isna().mean() * 100), 2) for col in numeric_cols}

        sample_rows = df.head(5).to_dict(orient='records')
        # Clean NaNs in sample_rows for JSON serialization
        clean_samples = []
        for row in sample_rows:
            clean_samples.append({k: (None if pd.isna(v) else v) for k, v in row.items()})

        _dataset_cache = {
            "total_raw_rows": int(len(df_raw)),
            "valid_observation_rows": int(len(df)),
            "columns": list(df.columns),
            "features_used": EXPECTED_FEATURES,
            "target": "CO(GT)",
            "averages": {
                "avg_co": round(float(df['CO(GT)'].mean()), 2) if 'CO(GT)' in df else None,
                "avg_temperature": round(float(df['T'].mean()), 2) if 'T' in df else None,
                "avg_relative_humidity": round(float(df['RH'].mean()), 2) if 'RH' in df else None,
                "avg_absolute_humidity": round(float(df['AH'].mean()), 4) if 'AH' in df else None,
            },
            "missing_summary": {
                "counts": missing_counts,
                "percentages": missing_pct
            },
            "sample_rows": clean_samples
        }
    except Exception as e:
        _dataset_cache = {"error": f"Failed to compute dataset summary: {str(e)}"}

    return _dataset_cache


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify backend status and artifact availability."""
    is_ready = all(v is not None for v in [model, scaler, imputer])
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "service": "Air Quality Prediction Backend",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """Provides architectural and evaluation information about the trained model."""
    if metrics_data is None:
        return jsonify({"error": "Model metrics not available. Ensure train_model.py has been run."}), 500
    
    return jsonify({
        "status": "success",
        "model_architecture": metrics_data.get("model_name", "Linear Regression"),
        "target": metrics_data.get("target", "CO(GT)"),
        "unit": "mg/m3",
        "features": EXPECTED_FEATURES,
        "train_samples": metrics_data.get("train_samples"),
        "test_samples": metrics_data.get("test_samples"),
        "evaluation_metrics": {
            "R2_Score": metrics_data.get("r2_test"),
            "MAE": metrics_data.get("mae_test"),
            "RMSE": metrics_data.get("rmse_test")
        },
        "coefficients": metrics_data.get("coefficients"),
        "intercept": metrics_data.get("intercept"),
        "comparison_model": {
            "model_name": "Random Forest Regressor",
            "evaluation_metrics": {
                "R2_Score": metrics_data.get("rf_comparison", {}).get("r2_test"),
                "MAE": metrics_data.get("rf_comparison", {}).get("mae_test"),
                "RMSE": metrics_data.get("rf_comparison", {}).get("rmse_test")
            }
        }
    }), 200


@app.route('/dataset', methods=['GET'])
def dataset():
    """Returns dataset summary statistics, column descriptors, and sample observations."""
    summary = get_dataset_summary()
    return jsonify(summary), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts 11 feature inputs, validates values, performs scaling,
    and returns predicted Carbon Monoxide CO(GT) concentration in mg/m3.
    """
    if model is None or scaler is None or imputer is None:
        return jsonify({
            "error": "Model artifacts are not loaded. Please verify train_model.py has been executed."
        }), 503

    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type. Request body must be JSON."
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Malformed or empty JSON payload."}), 400

    # Validate feature presence
    missing_features = [feat for feat in EXPECTED_FEATURES if feat not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
            "expected_features": EXPECTED_FEATURES
        }), 400

    # Validate numeric types and convert
    feature_values = []
    invalid_values = {}
    for feat in EXPECTED_FEATURES:
        val = data[feat]
        if val is None or val == "":
            invalid_values[feat] = "Value cannot be null or empty string."
            continue
        try:
            num_val = float(val)
            if np.isnan(num_val) or np.isinf(num_val):
                invalid_values[feat] = "Value cannot be NaN or Infinite."
            else:
                feature_values.append(num_val)
        except (ValueError, TypeError):
            invalid_values[feat] = f"Invalid numeric input '{val}'. Must be a valid float or integer."

    if invalid_values:
        return jsonify({
            "error": "Validation failed for one or more features.",
            "invalid_features": invalid_values
        }), 400

    try:
        # Construct DataFrame with proper feature names
        input_df = pd.DataFrame([feature_values], columns=EXPECTED_FEATURES)
        
        # Apply trained Imputer and Scaler
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Predict
        prediction = model.predict(input_scaled)
        pred_value = float(prediction[0])
        # CO concentration cannot be negative physically; clamp floor to 0.0 if slightly negative
        pred_clamped = max(0.0, pred_value)

        return jsonify({
            "status": "success",
            "prediction": round(pred_clamped, 3),
            "unit": "mg/m3",
            "model": "Linear Regression",
            "input_features": {feat: val for feat, val in zip(EXPECTED_FEATURES, feature_values)}
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Prediction computation failed: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Run Flask on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=False)
