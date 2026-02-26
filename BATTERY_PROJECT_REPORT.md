# Battery SOH Data Engineering + MLOps Project Report

## 1) Executive Summary
This project implements an end-to-end battery analytics workflow from raw telemetry ingestion to machine learning inference and dashboard reporting. The system combines data engineering reliability checks, physics-informed feature engineering, model comparison (XGBoost, 2-layer LSTM, 2-layer GRU), and a Streamlit monitoring dashboard.

Primary objective:
- Predict next-step battery SOH proxy (SOC) and operationalize the result with reproducible pipeline outputs and dashboard visibility.

Current status:
- Pipeline: **SUCCESS**
- Data quality: **672,211 → 672,210 rows** (1 row removed)
- Safety monitor: **NORMAL** (no low-confidence sensor-noise flag)

---

## 2) Problem Context
Battery systems require reliable health tracking to support safe operation and maintenance planning. This project focuses on:
- Building a robust data pipeline for battery telemetry (`time_s`, `Amps`, `Volts`, `SOC`)
- Generating health-indicator features inspired by battery behavior
- Training and comparing multiple model families
- Serving interpretable outputs via dashboard and optional SQL publish

---

## 3) Data Pipeline Architecture
### Source
- `data/battery_data_with_soc.csv`

### Pipeline Stages
1. **Extract**
   - Supports file/folder CSV ingestion and schema validation.
2. **Transform**
   - Numeric coercion, missing-value filtering, deduplication on `time_s`
   - Feature derivation: `Power_W`, rolling features, differential signals.
3. **Feature Engineering (Physics-informed proxies)**
   - `Q_Ah`, `dQ`
   - `dVdQ` (DVA proxy), `dQdV` (ICA proxy)
   - `R_int_ohm` (internal resistance proxy)
4. **Model Training**
   - Main pipeline model: XGBoost next-step SOC predictor
   - Uncertainty via residual quantiles (P05/P95)
5. **Monitoring + Artifacts**
   - Metrics JSON, prediction JSON, pipeline summary JSON
   - Optional SQL publish path for analytics table serving

---

## 4) Implementation Highlights
### Core Pipeline
- `src/battery_pipeline.py`
- Supports:
  - configurable input path/glob
  - row capping for fast iteration
  - optional SQL publish and strict publish mode
  - safety monitor (sensor-noise based confidence flag)

### Model Comparison Track
- `src/compare_battery_models.py`
- Model variants:
  - **v1**: XGBoost
  - **v2**: 2-layer LSTM
  - **v3**: 2-layer GRU

### Dashboard
- `dashboard.py`
- Includes:
  - pipeline reliability + data governance
  - SOC/current/voltage trends with separate y-scales
  - model leaderboard
  - per-model actual vs predicted SOH charts with fixed `time_s` domain
  - uncertainty band metrics (P05/P50/P95)
  - physics-indicator visualizations

---

## 5) Latest Measured Results
### Main Pipeline Metrics (`models/battery_soc_metrics.json`)
- Train rows: **160,000**
- Test rows: **40,000**
- MAE: **0.7685**
- R²: **0.9000**
- Interval coverage: **14.7425%**
- Mean interval width: **0.1859**
- Predicted SOC (P50): **74.1801**
- Predicted SOC (P05 / P95): **74.0849 / 74.2708**

### Model Comparison (`models/battery_model_comparison.json`)
Leader by MAE: **v1 (XGBoost)**

- v1 XGBoost: MAE **0.000073**
- v3 2-layer GRU: MAE **0.002195**
- v2 2-layer LSTM: MAE **0.002430**

Note: In this dataset slice, SOC variance in some windows is very low, which can make R² less informative.

---

## 6) Data Quality and Safety Monitoring
### Quality Report
- Rows before transform: **672,211**
- Rows after transform: **672,210**
- Rows removed: **1**

### Safety Monitor
- Low-confidence flag: **False**
- Voltage noise latest/baseline: **0.000148 / 0.000166**
- Current noise latest/baseline: **0.0 / 0.0**

Interpretation:
- No elevated sensor-noise condition detected at the latest run.

---

## 7) MLOps Artifacts Produced
- `models/battery_soc_model.joblib`
- `models/battery_soc_metrics.json`
- `models/battery_soc_latest_prediction.json`
- `models/battery_pipeline_summary.json`
- `models/battery_model_comparison.json`
- `models/battery_model_v1_metrics.json`
- `models/battery_model_v2_metrics.json`
- `models/battery_model_v3_metrics.json`

---

## 8) Operational Commands
### Run main battery pipeline
```powershell
.venv\Scripts\python.exe src\battery_pipeline.py --data-path data/battery_data_with_soc.csv
```

### Run model comparison (v1/v2/v3)
```powershell
.venv\Scripts\python.exe src\compare_battery_models.py --data-path data/battery_data_with_soc.csv --max-rows 80000 --sequence-length 20 --epochs 4
```

### Launch dashboard
```powershell
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

---

## 9) Strengths for Portfolio
- End-to-end DE + ML integration
- Clear reproducible pipeline outputs
- Multi-model benchmark workflow
- Physics-informed feature engineering direction
- Safety-first operational checks in dashboard

---

## 10) Limitations and Next Steps
Current limitations:
- No explicit temperature channel in source data, so electro-thermal features are limited.
- Transfer learning across battery IDs/lifecycles is not yet applied.
- Uncertainty currently uses residual quantiles rather than fully conditional quantile models.

High-value next steps:
1. Add battery/cycle identifiers and temperature telemetry.
2. Add SHAP-based explanation panel in dashboard.
3. Implement ECM + residual hybrid model track.
4. Add scheduled retraining + drift alerts with threshold governance.

---

## 11) Conclusion
The project demonstrates a production-minded battery analytics pipeline: reliable ingestion, quality controls, physics-inspired features, model benchmarking, uncertainty reporting, and dashboard observability. It is a strong foundation for scaling toward industrial battery management use-cases.
