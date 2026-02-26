from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:
    raise RuntimeError("xgboost is required. Install with: pip install xgboost") from exc

try:
    from src.extract import extract_data
    from src.battery_transform import transform_battery_data, BATTERY_REQUIRED_COLUMNS
    from src.publish_battery_soc import publish_battery_soc_from_artifacts
except ModuleNotFoundError:
    from extract import extract_data
    from battery_transform import transform_battery_data, BATTERY_REQUIRED_COLUMNS
    from publish_battery_soc import publish_battery_soc_from_artifacts


FEATURE_COLS = [
    "time_s",
    "Amps",
    "Volts",
    "Power_W",
    "Q_Ah",
    "dQ",
    "dV",
    "dI",
    "dVdQ",
    "dQdV",
    "R_int_ohm",
    "DVA_Rolling_20",
    "ICA_Rolling_20",
    "Amps_RollingMean_20",
    "Volts_RollingMean_20",
    "Power_RollingMean_20",
    "SOC_Lag1",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["SOC_next"] = frame["SOC"].shift(-1)
    frame["SOC_Lag1"] = frame["SOC"].shift(1)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def train_soc_model(
    feature_df: pd.DataFrame,
) -> tuple[XGBRegressor, dict, dict]:
    split_idx = int(len(feature_df) * 0.8)
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["SOC_next"])

    train_preds = model.predict(train_df[FEATURE_COLS])
    train_residuals = train_df["SOC_next"].to_numpy(dtype=float) - np.asarray(train_preds, dtype=float)
    residual_q05 = float(np.quantile(train_residuals, 0.05))
    residual_q95 = float(np.quantile(train_residuals, 0.95))
    interval_offsets = {
        "residual_q05": residual_q05,
        "residual_q95": residual_q95,
    }

    if test_df.empty:
        metrics = {
            "train_rows": int(len(train_df)),
            "test_rows": 0,
            "mae": None,
            "r2": None,
            "model_name": "XGBoostRegressor (SOC next-step)",
        }
        return model, interval_offsets, metrics

    preds = model.predict(test_df[FEATURE_COLS])
    preds_arr = np.asarray(preds, dtype=float)
    y_test = test_df["SOC_next"].to_numpy(dtype=float)
    low_preds = preds_arr + residual_q05
    high_preds = preds_arr + residual_q95
    mae = _mae(y_test, preds_arr)
    r2 = _r2(y_test, preds_arr)
    interval_coverage = float(((test_df["SOC_next"] >= low_preds) & (test_df["SOC_next"] <= high_preds)).mean() * 100)
    mean_interval_width = float((high_preds - low_preds).mean())

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": mae,
        "r2": r2,
        "interval_coverage_pct": interval_coverage,
        "mean_interval_width": mean_interval_width,
        "model_name": "XGBoostRegressor (SOC next-step)",
    }
    return model, interval_offsets, metrics


def predict_next_soc(
    model: XGBRegressor,
    interval_offsets: dict,
    clean_df: pd.DataFrame,
) -> dict:
    if len(clean_df) < 2:
        raise ValueError("Need at least 2 rows to compute next-step SOC prediction")

    step_candidates = clean_df["time_s"].diff().dropna()
    positive_steps = step_candidates[step_candidates > 0]
    step_size = float(positive_steps.median()) if not positive_steps.empty else 1.0

    last_row = clean_df.iloc[-1]
    feature_row = pd.DataFrame(
        [
            {
                "time_s": float(last_row["time_s"] + step_size),
                "Amps": float(last_row["Amps"]),
                "Volts": float(last_row["Volts"]),
                "Power_W": float(last_row["Power_W"]),
                "Q_Ah": float(last_row["Q_Ah"] + ((last_row["Amps"] * step_size) / 3600.0)),
                "dQ": float((last_row["Amps"] * step_size) / 3600.0),
                "dV": 0.0,
                "dI": 0.0,
                "dVdQ": float(last_row["dVdQ"]),
                "dQdV": float(last_row["dQdV"]),
                "R_int_ohm": float(last_row["R_int_ohm"]),
                "DVA_Rolling_20": float(last_row["DVA_Rolling_20"]),
                "ICA_Rolling_20": float(last_row["ICA_Rolling_20"]),
                "Amps_RollingMean_20": float(last_row["Amps_RollingMean_20"]),
                "Volts_RollingMean_20": float(last_row["Volts_RollingMean_20"]),
                "Power_RollingMean_20": float(last_row["Power_RollingMean_20"]),
                "SOC_Lag1": float(last_row["SOC"]),
            }
        ]
    )

    predicted_soc = float(model.predict(feature_row[FEATURE_COLS])[0])
    predicted_soc_p05 = predicted_soc + float(interval_offsets.get("residual_q05", 0.0))
    predicted_soc_p95 = predicted_soc + float(interval_offsets.get("residual_q95", 0.0))
    return {
        "prediction_time_s": float(feature_row.iloc[0]["time_s"]),
        "predicted_soc": max(predicted_soc, 0.0),
        "predicted_soc_p05": max(predicted_soc_p05, 0.0),
        "predicted_soc_p95": max(predicted_soc_p95, 0.0),
        "step_size_seconds": step_size,
    }


def detect_sensor_noise(clean_df: pd.DataFrame) -> dict:
    volts_noise = clean_df["Volts"].rolling(25, min_periods=10).std().dropna()
    amps_noise = clean_df["Amps"].rolling(25, min_periods=10).std().dropna()

    if volts_noise.empty or amps_noise.empty:
        return {
            "low_confidence": False,
            "volts_noise_latest": None,
            "volts_noise_baseline": None,
            "amps_noise_latest": None,
            "amps_noise_baseline": None,
        }

    baseline_cut = max(int(len(volts_noise) * 0.4), 1)
    volts_baseline = float(volts_noise.iloc[:baseline_cut].median())
    amps_baseline = float(amps_noise.iloc[:baseline_cut].median())
    volts_latest = float(volts_noise.iloc[-200:].mean())
    amps_latest = float(amps_noise.iloc[-200:].mean())

    volts_alert = volts_latest > (volts_baseline * 2.0 if volts_baseline > 0 else float("inf"))
    amps_alert = amps_latest > (amps_baseline * 2.0 if amps_baseline > 0 else float("inf"))

    return {
        "low_confidence": bool(volts_alert or amps_alert),
        "volts_noise_latest": volts_latest,
        "volts_noise_baseline": volts_baseline,
        "amps_noise_latest": amps_latest,
        "amps_noise_baseline": amps_baseline,
    }


def run_battery_pipeline(
    data_path: str,
    data_glob: str = "*.csv",
    max_rows: int | None = 200000,
    publish_sql: bool = False,
    strict_publish: bool = False,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_df = extract_data(
        data_path,
        pattern=data_glob,
        required_columns=BATTERY_REQUIRED_COLUMNS,
    )
    clean_df = transform_battery_data(raw_df)
    feature_df = build_features(clean_df)

    sampled_rows = 0
    if max_rows is not None and len(feature_df) > max_rows:
        sampled_rows = int(len(feature_df) - max_rows)
        feature_df = feature_df.sample(n=max_rows, random_state=42).sort_values("time_s").reset_index(drop=True)

    model, interval_offsets, metrics = train_soc_model(feature_df)

    model_path = models_dir / "battery_soc_model.joblib"
    metrics_path = models_dir / "battery_soc_metrics.json"
    prediction_path = models_dir / "battery_soc_latest_prediction.json"
    summary_path = models_dir / "battery_pipeline_summary.json"

    joblib.dump(model, model_path)
    next_soc = predict_next_soc(model, interval_offsets, clean_df)
    safety_monitor = detect_sensor_noise(clean_df)

    metrics_payload = {
        **metrics,
        "rows_raw": int(len(raw_df)),
        "rows_clean": int(len(clean_df)),
        "rows_features": int(len(feature_df)),
        "rows_sampled_out": sampled_rows,
        "source_file_count": int(raw_df.attrs.get("source_file_count", 1)),
        "model_path": str(model_path),
        "residual_q05": float(interval_offsets.get("residual_q05", 0.0)),
        "residual_q95": float(interval_offsets.get("residual_q95", 0.0)),
        "predicted_soc": float(next_soc["predicted_soc"]),
        "predicted_soc_p05": float(next_soc["predicted_soc_p05"]),
        "predicted_soc_p95": float(next_soc["predicted_soc_p95"]),
        "generated_at_utc": _utc_now_iso(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    prediction_path.write_text(json.dumps(next_soc, indent=2), encoding="utf-8")

    published = False
    if publish_sql:
        published = publish_battery_soc_from_artifacts(
            metrics_path=metrics_path,
            prediction_path=prediction_path,
            strict=strict_publish,
        )

    summary_payload = {
        "status": "SUCCESS",
        "source_path": str(data_path),
        "source_glob": data_glob,
        "max_rows": max_rows,
        "publish_sql": publish_sql,
        "battery_prediction_published": bool(published),
        "safety_monitor": safety_monitor,
        "quality_report": clean_df.attrs.get("quality_report", {}),
        "metrics_path": str(metrics_path),
        "prediction_path": str(prediction_path),
        "generated_at_utc": _utc_now_iso(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Battery pipeline SUCCESS")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Latest prediction saved to: {prediction_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run battery data pipeline and SOC model training")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/battery_data_with_soc.csv",
        help="Input CSV file path or directory containing battery CSV files",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default="*.csv",
        help="Glob pattern used when --data-path points to a directory",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200000,
        help="Maximum feature rows used for training (set 0 to use all rows)",
    )
    parser.add_argument(
        "--publish-sql",
        action="store_true",
        help="Publish latest SOC prediction to analytics.battery_soc_predictions",
    )
    parser.add_argument(
        "--strict-publish",
        action="store_true",
        help="Fail pipeline if SQL publish fails",
    )
    args = parser.parse_args()

    run_battery_pipeline(
        data_path=args.data_path,
        data_glob=args.data_glob,
        max_rows=(None if args.max_rows == 0 else args.max_rows),
        publish_sql=args.publish_sql,
        strict_publish=args.strict_publish,
    )
