# Battery Intelligence: End-to-End MLOps & Data Engineering Pipeline

A production-grade asynchronous pipeline for battery analytics, moving beyond notebooks into a structured, reproducible system. This project demonstrates the full lifecycle: telemetry ingestion, automated data governance, physics-informed feature engineering, and multi-model benchmarking.

<img width="4454" height="1790" alt="ARCHITECTURE_DIAGRAM_HD" src="https://github.com/user-attachments/assets/a6a2f8c0-d28a-460a-8dfb-eefa7377fd8f" />


##  Project Philosophy
In industrial battery management, a model is only as good as the data feeding it. This project prioritizes **Data Engineering reliability** and **MLOps observability** over simple model fitting. 

### Key Engineering Pillars:
* **Data Engineering (DE):** Automated schema validation, strict deduplication, and signal noise monitoring.
* **MLOps:** Champion/Challenger model versioning (XGBoost vs. RNNs), automated artifact generation (JSON/Joblib), and uncertainty quantification.
* **Observability:** A Streamlit-based "Intelligence Dashboard" for real-time monitoring of pipeline health and drift.

---

##  System Architecture

### 1. Robust Data Pipeline (`src/battery_pipeline.py`)
The ingestion engine handles raw telemetry (Amps, Volts, Temp) with built-in quality gates:
* **Data Governance:** Automated filtering of non-numeric values and duplicate timestamps. In the latest run, the pipeline achieved a **100% retention rate** (672,210/672,211 rows).
* **Safety Monitor:** Comparison of current sensor noise against historical baselines to flag low-confidence data before it reaches the model.
* **Physics-Informed Features:** Calculation of $dV/dQ$ (Differential Voltage) and $dQ/dV$ (Incremental Capacity) to provide physical context to the ML models.

### 2. MLOps Workflow & Experiment Tracking
We treat model selection as a dynamic benchmark:
* **Model Comparison:** The system automatically benchmarks **v1 (XGBoost)**, **v2 (LSTM)**, and **v3 (GRU)**.
* **Standardized Artifacts:** Every run generates a `metrics.json` and `pipeline_summary.json` for full auditability.
* **Uncertainty Estimation:** Uses residual quantiles (P05/P95) to provide a confidence interval for every SOH prediction.

---

##  Performance & Results

### Model Leaderboard
The pipeline identifies the "Champion" model based on Mean Absolute Error (MAE).

| Model Version | Architecture | MAE | R² Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | **XGBoost Regressor** | **0.000073** | **0.9000** | **Champion** |
| v3 | 2-Layer GRU | 0.002195 | N/A | Challenger |
| v2 | 2-Layer LSTM | 0.002430 | N/A | Challenger |


*Figure 1: Feature correlation matrix and model performance comparison.*<img width="3344" height="964" alt="Screenshot 2026-02-26 153629" src="https://github.com/user-attachments/assets/08c03f96-1215-45b6-9d40-2bf7235c578a" />


### Pipeline Reliability
Our automated "Safety-First Monitor" ensures that sensor noise remains within normal bounds (Confidence Flag: **NORMAL**).


*Figure 2: Data quality report showing 100% pipeline success and row retention.*<img width="3330" height="1030" alt="Screenshot 2026-02-26 153616" src="https://github.com/user-attachments/assets/05999e35-bb33-42f9-97b3-48a8b5389077" />


---

##  Tech Stack
* **Language:** Python 3.10+
* **Orchestration:** Custom ETL Pipeline (Pandas/NumPy)
* **ML Frameworks:** XGBoost, PyTorch (LSTM/GRU), Scikit-Learn
* **Metadata & Storage:** Joblib (Models), JSON (Experiment Tracking), SQL (Analytics Publishing)
* **Dashboard:** Streamlit & Plotly

---

##  Dashboard Observability
The integrated Intelligence Dashboard provides a single source of truth for engineering teams:

* **Telemetry Trends:** Separate y-scaling for SOC, Current, and Voltage to preserve signal morphology.
* **Actual vs. Predicted SOH:** Visual validation of model performance across the full timeline.
* **Physics-Informed Indicators:** Real-time tracking of internal resistance and differential capacity.

![Prediction Uncertainty and Physics Indicators]
*Figure 3: Prediction intervals (P05/P50/P95) and physics-informed health indicators.*<img width="3322" height="711" alt="Screenshot 2026-02-26 153702" src="https://github.com/user-attachments/assets/c6d479fe-7205-4d26-b497-3b60fc37b6c0" />
<img width="3334" height="998" alt="Screenshot 2026-02-26 153531" src="https://github.com/user-attachments/assets/f521eb08-0905-4514-a470-48ccf84d3ceb" />
<img width="3346" height="1170" alt="Screenshot 2026-02-26 153644" src="https://github.com/user-attachments/assets/aa387768-5f04-4b0d-90cf-798136301842" />


