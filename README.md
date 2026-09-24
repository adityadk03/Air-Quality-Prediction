# Air Quality Prediction and Environmental Analytics System

### IBM SkillsBuild Data Analytics with AI Academic Internship Program
**Conducted by BharatCares in association with All India Council for Technical Education (AICTE)**

---

## 1. Project Overview

The **Air Quality Prediction and Environmental Analytics System** is an end-to-end machine learning, REST API, and interactive dashboard platform developed to predict ground-truth ambient **Carbon Monoxide (CO)** concentrations from solid-state chemical sensor array responses and meteorological variables.

By combining rigorous statistical data preprocessing, leakage-free feature scaling, an interpretable Ordinary Least Squares (OLS) Linear Regression model, a production-ready Flask REST API, and a light-themed Streamlit user interface, the system demonstrates how low-cost multi-sensor telemetry can reliably estimate certified reference-grade environmental measurements.

---

## 2. Problem Statement

Air quality degradation caused by rapid urbanization and vehicular congestion presents severe public health challenges. Carbon Monoxide ($\text{CO}$), a toxic byproduct of incomplete combustion, significantly impairs oxygen transport in the human body. 

While certified reference monitoring stations deliver highly precise pollution measurements, their extreme capital expenditure and maintenance requirements prohibit dense geographic deployment. Low-cost solid-state metal oxide chemical sensors offer a scalable alternative; however, their signals exhibit non-linearities, cross-sensitivity to other gases, and environmental drift caused by changing temperature and humidity. 

This project solves this challenge by formulating a machine learning calibration pipeline that accurately maps 11 chemical and meteorological features to true hourly $\text{CO(GT)}$ concentrations in $\text{mg/m}^3$.

---

## 3. Objectives

1. Ingest and validate the benchmark UCI Air Quality dataset (handling European delimiter and decimal formats).
2. Clean structural anomalies and replace domain-specific error flags (`-200`) with `NaN`.
3. Perform missingness analysis and exclude unviable features (such as `NMHC(GT)` with >90% missingness).
4. Conduct Exploratory Data Analysis (EDA) on pollutant distributions, correlations, and diurnal cycles.
5. Establish a strict, leakage-free preprocessing pipeline fitting `SimpleImputer` and `StandardScaler` solely on training data.
6. Train and evaluate an interpretable `LinearRegression` model alongside a benchmark `RandomForestRegressor`.
7. Deploy an enterprise-ready Flask REST API (`http://localhost:5000`) with input validation and CORS support.
8. Deliver an interactive, light-themed Streamlit web interface with real-time predictions, data exploration, and model insights.

---

## 4. Dataset Description & Official Source

- **Dataset Name:** Air Quality Dataset
- **Official Repository Link:** [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Official Citation:**
  > Vito, S. (2008). Air Quality. UCI Machine Learning Repository. [https://doi.org/10.24432/C59K5F](https://doi.org/10.24432/C59K5F)
- **Deployment Period:** March 2004 to April 2005 (1 full year, 9,357 valid hourly observations).
- **Physical Context:** Co-located chemical sensor array and certified analyzer deployed at road level in an Italian city.

### Substantive Columns (15 Variables):
| Column | Role | Unit | Description |
| :--- | :--- | :--- | :--- |
| `Date` | Metadata | DD/MM/YYYY | Observation date |
| `Time` | Metadata | HH.mm.ss | Observation time |
| `CO(GT)` | **Target ($y$)** | $\text{mg/m}^3$ | True hourly averaged Carbon Monoxide concentration |
| `PT08.S1(CO)` | Feature ($X$) | Sensor resistance | Tin oxide sensor response nominally targeted to CO |
| `NMHC(GT)` | Excluded | $\mu\text{g/m}^3$ | Non-Metanic HydroCarbons (90.23% missing; excluded) |
| `C6H6(GT)` | Feature ($X$) | $\mu\text{g/m}^3$ | True hourly averaged Benzene concentration |
| `PT08.S2(NMHC)`| Feature ($X$) | Sensor resistance | Titania sensor response nominally targeted to NMHC |
| `NOx(GT)` | Feature ($X$) | ppb | True hourly averaged Nitrogen Oxides concentration |
| `PT08.S3(NOx)` | Feature ($X$) | Sensor resistance | Tungsten oxide sensor response nominally targeted to NOx |
| `NO2(GT)` | Feature ($X$) | $\mu\text{g/m}^3$ | True hourly averaged Nitrogen Dioxide concentration |
| `PT08.S4(NO2)` | Feature ($X$) | Sensor resistance | Tungsten oxide sensor response nominally targeted to NO2 |
| `PT08.S5(O3)` | Feature ($X$) | Sensor resistance | Indium oxide sensor response nominally targeted to O3 |
| `T` | Feature ($X$) | $^\circ\text{C}$ | Ambient Temperature |
| `RH` | Feature ($X$) | $\%$ | Relative Humidity |
| `AH` | Feature ($X$) | Absolute scale | Absolute Humidity |

---

## 5. Technologies Used

- **Programming Language:** Python 3.11+
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`LinearRegression`, `RandomForestRegressor`, `SimpleImputer`, `StandardScaler`)
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib, Seaborn
- **Backend API:** Flask, Flask-CORS
- **Frontend Dashboard:** Streamlit
- **Reporting & Documentation:** Python-Docx, Jupyter Notebook, Markdown

---

## 6. Project Structure

```text
Air-Quality-Prediction/
│
├── data/
│   └── AirQualityUCI.csv                 # Official benchmark dataset
│
├── model/
│   ├── model.pkl                         # Trained Linear Regression model
│   ├── scaler.pkl                        # Fitted StandardScaler
│   ├── imputer.pkl                       # Fitted SimpleImputer (median)
│   ├── metrics.json                      # Actual calculated performance metrics
│   └── diagnostics.png                   # 4-panel diagnostic visual suite
│
├── backend/
│   └── app.py                            # Flask REST API server (Port 5000)
│
├── frontend/
│   └── ui.py                             # Streamlit multi-page dashboard
│
├── train_model.py                        # Complete offline training & evaluation pipeline
│
├── YOURNAME_AirQualityPrediction.ipynb    # Executed 24-step internship notebook
│
├── YOURNAME_ProjectReport.docx           # Formatted academic project report
│
├── requirements.txt                      # Project dependency specification
│
├── README.md                             # Comprehensive project documentation
│
└── .gitignore                            # Python, IDE, and checkpoint ignore rules
```

---

## 7. Machine Learning Methodology & Preprocessing

### Data Cleaning & Hygiene:
1. Ingest raw CSV with semicolon delimiter (`;`) and decimal comma (`,`).
2. Truncate trailing empty columns (`Unnamed: 15` and `Unnamed: 16`) and 114 blank trailing rows.
3. Replace all `-200` domain missingness flags with `NaN`.
4. Filter 7,674 rows having non-null ground-truth target `CO(GT)`.
5. Exclude `NMHC(GT)` due to extreme missingness (90.23%).

### Leakage-Free Preprocessing Pipeline:
- **Partitioning:** $80\%$ training ($6,139$ samples), $20\%$ testing ($1,535$ samples) using `random_state=42`.
- **Imputation:** `SimpleImputer(strategy='median')` fit **only** on $X_{\text{train}}$ and transformed on $X_{\text{train}}$ and $X_{\text{test}}$.
- **Standardization:** `StandardScaler()` fit **only** on $X_{\text{train\_imputed}}$ and transformed on $X_{\text{train\_imputed}}$ and $X_{\text{test\_imputed}}$.

---

## 8. Actual Model Evaluation Metrics

All metrics reflect **actual calculated test performance** on the untouched 20% holdout set:

| Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE ($\text{mg/m}^3$) | Test RMSE ($\text{mg/m}^3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Primary)** | **0.8866** | **0.8888** | **0.3080** | **0.4813** |
| Random Forest Regressor (Benchmark) | 0.9654 | 0.9239 | 0.2516 | 0.3982 |

### Standardized Linear Regression Coefficients:
- **Intercept ($\beta_0$):** $2.1663\text{ mg/m}^3$
- **$\text{C}_6\text{H}_6(\text{GT})$ (Benzene):** $+0.5639$
- **$\text{NO}_x(\text{GT})$ (Nitrogen Oxides):** $+0.5545$
- **$\text{PT08.S4(NO2)}$:** $+0.4528$
- **$\text{PT08.S1(CO)}$:** $+0.2627$
- **$\text{NO}_2(\text{GT})$:** $+0.1116$
- **$\text{PT08.S3(NOx)}$:** $+0.0171$
- **$\text{PT08.S2(NMHC)}$:** $-0.0247$
- **$\text{AH}$ (Absolute Humidity):** $-0.0484$
- **$\text{RH}$ (Relative Humidity):** $-0.1396$
- **$\text{PT08.S5(O3)}$:** $-0.2153$
- **$\text{T}$ (Temperature):** $-0.2237$

---

## 9. API Endpoints (Flask Backend)

The backend server runs on `http://localhost:5000` and provides the following endpoints:

| Method | Endpoint | Description | Sample Response Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service and model readiness status | `{"status": "healthy", "model_loaded": true}` |
| `POST` | `/predict` | Predicts CO concentration from 11 inputs | `{"prediction": 1.842, "unit": "mg/m3", "status": "success"}` |
| `GET` | `/dataset` | Cleaned dataset metrics, averages, missing stats | `{"total_raw_rows": 9471, "valid_observation_rows": 9357, ...}` |
| `GET` | `/model_info` | Architecture, evaluation scores, coefficients | `{"evaluation_metrics": {"R2_Score": 0.8888, "MAE": 0.308, ...}}` |

### Sample POST `/predict` Request:
```json
{
  "PT08.S1(CO)": 1063.0,
  "C6H6(GT)": 8.2,
  "PT08.S2(NMHC)": 909.0,
  "NOx(GT)": 180.0,
  "PT08.S3(NOx)": 795.0,
  "NO2(GT)": 109.0,
  "PT08.S4(NO2)": 1463.0,
  "PT08.S5(O3)": 963.0,
  "T": 17.8,
  "RH": 49.2,
  "AH": 1.01
}
```

---

## 10. Installation & Execution Guide

### Step 1: Install Dependencies
Open a terminal in the project root directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run Offline Training Pipeline
To retrain the models, export artifacts, and generate the diagnostic suite:
```bash
python train_model.py
```

### Step 3: Launch the Flask Backend API
In a terminal window, start the Flask REST API server:
```bash
python backend/app.py
```
*The server will start listening on `http://localhost:5000`.*

### Step 4: Launch the Streamlit Frontend Dashboard
In a separate terminal window, launch the interactive UI:
```bash
streamlit run frontend/ui.py
```
*The web browser will automatically open `http://localhost:8501`.*

### Step 5: Execute the Jupyter Notebook
To run or inspect the literate analysis notebook:
```bash
jupyter notebook YOURNAME_AirQualityPrediction.ipynb
```
*Or execute headlessly:*
```bash
python -m nbconvert --to notebook --execute --inplace YOURNAME_AirQualityPrediction.ipynb
```

---

## 11. Limitations

1. **Physical Sensor Drift:** Solid-state metal oxide sensors experience chemical degradation over prolonged atmospheric exposure, requiring periodic physical recalibration.
2. **Missing Reference Values:** 17.99% of timestamps lacked reference analyzer CO(GT) measurements, creating disjointed intervals.
3. **Cross-Gas Interferences:** Metal oxide sensors exhibit cross-reactions to competing oxidizing and reducing gases, especially during extreme humidity shifts.
4. **Static Train/Test Partitioning:** Standard 80/20 random sampling assumes independent observations; future work can leverage sequential block splits.

---

## 12. Future Scope

1. **Deep Temporal Modeling:** Implement LSTM or Temporal Convolutional Networks (TCN) to exploit multi-hour temporal inertia.
2. **Edge IoT Deployment:** Quantize the trained Linear Regression pipeline via ONNX or TinyML for direct microcontroller execution (e.g. ESP32, Raspberry Pi).
3. **Adaptive Drift Compensation:** Incorporate online recursive calibration algorithms to dynamically correct for sensor aging.
4. **Multi-Pollutant Joint Prediction:** Extend model to simultaneously predict $\text{NO}_2$, $\text{O}_3$, and $\text{PM}_{2.5}$.

---

## 13. References

1. Vito, S. (2008). Air Quality. UCI Machine Learning Repository. [https://doi.org/10.24432/C59K5F](https://doi.org/10.24432/C59K5F)
2. UCI Machine Learning Repository. Official Dataset Catalog. [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)
