"""
Air Quality Prediction and Environmental Analytics System
Model Training and Diagnostics Pipeline
Academic Internship Program - IBM SkillsBuild & BharatCares / AICTE
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def train():
    print("=" * 60)
    print("Air Quality Prediction - Model Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    data_path = os.path.join('data', 'AirQualityUCI.csv')
    print(f"[1/8] Loading dataset from {data_path}...")
    df_raw = pd.read_csv(data_path, sep=';', decimal=',')
    print(f"      Raw shape: {df_raw.shape}")

    # 2. Clean Data & Handle Missing Structure
    print("[2/8] Cleaning structure and dropping invalid rows/columns...")
    # Keep only first 15 substantive columns
    df = df_raw.iloc[:, :15].copy()
    
    # Drop rows that are completely empty / missing Date
    df = df.dropna(how='all')
    df = df[df['Date'].notna()].copy()
    print(f"      Valid observations with Date: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Replace -200 with NaN
    print("[3/8] Replacing missing value indicator (-200) with NaN...")
    df = df.replace(-200, np.nan)
    df = df.replace(-200.0, np.nan)

    # 4. Feature and Target Setup
    # NMHC(GT) is excluded due to ~90% missing values
    features = [
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
    target = 'CO(GT)'
    
    print(f"      Target: {target}")
    print(f"      Features ({len(features)}): {', '.join(features)}")

    # For supervised learning, filter rows with non-null target CO(GT)
    labeled_df = df.dropna(subset=[target]).copy()
    print(f"      Rows with ground-truth target '{target}': {len(labeled_df)}")

    X = labeled_df[features].copy()
    y = labeled_df[target].copy()

    # 5. Train / Test Split
    print("[4/8] Splitting dataset into train (80%) and test (20%) sets (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Train set size: {X_train.shape[0]} samples")
    print(f"      Test set size:  {X_test.shape[0]} samples")

    # 6. Fit Imputer and Scaler ONLY on Training Data (Zero Data Leakage)
    print("[5/8] Fitting SimpleImputer (median) and StandardScaler ONLY on training data...")
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 7. Model Training & Evaluation
    print("[6/8] Training primary LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate actual metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n" + "-" * 50)
    print("PRIMARY MODEL (LinearRegression) PERFORMANCE:")
    print(f"  Train R^2 Score : {r2_train:.4f}")
    print(f"  Test R^2 Score  : {r2_test:.4f}")
    print(f"  Test MAE        : {mae_test:.4f} mg/m3")
    print(f"  Test RMSE       : {rmse_test:.4f} mg/m3")
    print("-" * 50 + "\n")

    # Comparison Model: RandomForestRegressor
    print("Training comparison model (RandomForestRegressor)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_rf_pred = rf.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, y_rf_pred)
    rf_mae = mean_absolute_error(y_test, y_rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_rf_pred))
    print(f"  RandomForest Test R^2  : {rf_r2:.4f}")
    print(f"  RandomForest Test MAE  : {rf_mae:.4f} mg/m3")
    print(f"  RandomForest Test RMSE : {rf_rmse:.4f} mg/m3\n")

    # 8. Save Artifacts
    os.makedirs('model', exist_ok=True)
    print("[7/8] Saving artifacts to 'model/' directory...")
    joblib.dump(model, os.path.join('model', 'model.pkl'))
    joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
    joblib.dump(imputer, os.path.join('model', 'imputer.pkl'))
    print("      Saved model.pkl, scaler.pkl, imputer.pkl")

    # Save metrics metadata for API and UI consumption
    metrics_info = {
        "model_name": "Linear Regression",
        "target": target,
        "features": features,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "r2_train": round(float(r2_train), 4),
        "r2_test": round(float(r2_test), 4),
        "mae_test": round(float(mae_test), 4),
        "rmse_test": round(float(rmse_test), 4),
        "coefficients": {feat: round(float(coef), 4) for feat, coef in zip(features, model.coef_)},
        "intercept": round(float(model.intercept_), 4),
        "rf_comparison": {
            "r2_test": round(float(rf_r2), 4),
            "mae_test": round(float(rf_mae), 4),
            "rmse_test": round(float(rf_rmse), 4)
        }
    }
    with open(os.path.join('model', 'metrics.json'), 'w') as f:
        json.dump(metrics_info, f, indent=2)
    print("      Saved metrics.json")

    # 9. Diagnostics Visualization
    print("[8/8] Generating diagnostic plots...")
    residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.45, color='#2563eb', edgecolors='none', s=30)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
    ax1.set_title(f'Actual vs Predicted CO(GT) (R² = {r2_test:.4f})', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Actual CO(GT) [mg/m³]', fontsize=10)
    ax1.set_ylabel('Predicted CO(GT) [mg/m³]', fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Residuals vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(y_test_pred, residuals, alpha=0.45, color='#059669', edgecolors='none', s=30)
    ax2.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax2.set_title('Residuals vs. Predicted Values', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Predicted CO(GT) [mg/m³]', fontsize=10)
    ax2.set_ylabel('Residuals (Actual - Predicted) [mg/m³]', fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Residual Distribution
    ax3 = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax3, color='#6366f1', bins=35)
    ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Residual Error [mg/m³]', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Subplot 4: Feature Coefficients (Linear Regression)
    ax4 = axes[1, 1]
    coef_series = pd.Series(model.coef_, index=features).sort_values()
    colors = ['#dc2626' if c < 0 else '#2563eb' for c in coef_series.values]
    coef_series.plot(kind='barh', ax=ax4, color=colors)
    ax4.axvline(0, color='black', linestyle='-', lw=0.8)
    ax4.set_title('Standardized Feature Coefficients (Impact on CO)', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Standardized Coefficient Weight', fontsize=10)
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(pad=3.0)
    diag_path = os.path.join('model', 'diagnostics.png')
    fig.savefig(diag_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved diagnostic plots to {diag_path}")
    print("\nTraining and evaluation completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    train()
