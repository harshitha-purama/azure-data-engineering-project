from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
else:
    tf.get_logger().setLevel("ERROR")

from src.battery_model_data import (
    SEQUENCE_FEATURE_COLS,
    TABULAR_FEATURE_COLS,
    default_data_path,
)
from src.battery_transform import BATTERY_REQUIRED_COLUMNS
from src.extract import extract_data
from src.battery_transform import transform_battery_data


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def resolve_rows_used(comparison: dict) -> int | None:
    results = comparison.get("results", []) if comparison else []
    row_values = [item.get("rows_used") for item in results if isinstance(item.get("rows_used"), int)]
    if not row_values:
        return 80000
    return int(min(row_values))


def resolve_sequence_length(comparison: dict) -> int:
    results = comparison.get("results", []) if comparison else []
    for item in results:
        value = item.get("sequence_length")
        if isinstance(value, int) and value > 1:
            return value
    return 30


def format_metric(value: float | int | None, currency: bool = False) -> str:
    if value is None:
        return "N/A"
    if currency:
        return f"${value:,.2f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return f"{value:,}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def build_quality_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    missing_values = int(raw_df[["time_s", "Amps", "Volts", "SOC"]].isna().any(axis=1).sum())
    duplicate_time = int(raw_df.duplicated(subset=["time_s"]).sum())
    non_numeric_rows = int(
        raw_df[["time_s", "Amps", "Volts", "SOC"]]
        .apply(lambda col: pd.to_numeric(col, errors="coerce"))
        .isna()
        .any(axis=1)
        .sum()
    )

    summary = pd.DataFrame(
        [
            {"Reason": "Missing Required Fields", "Rows": missing_values},
            {"Reason": "Duplicate time_s", "Rows": duplicate_time},
            {"Reason": "Non-numeric Sensor Rows", "Rows": non_numeric_rows},
        ]
    )
    summary = summary.sort_values("Rows", ascending=False).reset_index(drop=True)
    summary["Rows"] = summary["Rows"].astype(int)
    return summary


def build_quality_status(quality_df: pd.DataFrame) -> pd.DataFrame:
    status_df = quality_df.copy()
    status_df["Status"] = status_df["Rows"].apply(lambda value: "Passed" if value == 0 else "Failed")
    return status_df


def build_sequence_all(frame: pd.DataFrame, sequence_length: int):
    values = frame[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = frame["SOC_next"].to_numpy(dtype=np.float32)
    times = frame["time_s"].to_numpy(dtype=np.float32)

    x_seq = []
    y_seq = []
    t_seq = []
    for end in range(sequence_length, len(values) + 1):
        start = end - sequence_length
        x_seq.append(values[start:end])
        y_seq.append(targets[end - 1])
        t_seq.append(times[end - 1])

    if not x_seq:
        raise ValueError("Not enough rows for sequence model evaluation")

    x_all = np.asarray(x_seq, dtype=np.float32)
    y_all = np.asarray(y_seq, dtype=np.float32)
    t_all = np.asarray(t_seq, dtype=np.float32)

    split_idx = int(len(x_all) * 0.8)
    x_train = x_all[:split_idx]

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    x_all_2d = x_all.reshape(-1, x_all.shape[-1])
    scaler.fit(x_train_2d)
    x_all_scaled = scaler.transform(x_all_2d).reshape(x_all.shape)
    return x_all_scaled, y_all, t_all


def load_model_comparison(comparison_path: Path) -> dict:
    if not comparison_path.exists():
        return {}
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def get_model_paths(comparison: dict, models_dir: Path) -> dict:
    paths = {
        "v1": models_dir / "battery_model_v1_xgboost.joblib",
        "v2": models_dir / "battery_model_v2_lstm.keras",
        "v3": models_dir / "battery_model_v3_gru.keras",
    }

    for item in comparison.get("results", []):
        version = item.get("model_version")
        model_path = item.get("model_path")
        if version in paths and model_path:
            paths[version] = Path(model_path)
    return paths


def get_model_meta(comparison: dict) -> dict:
    meta = {
        "v2": {"target_mean": 0.0, "target_std": 1.0},
        "v3": {"target_mean": 0.0, "target_std": 1.0},
    }
    for item in comparison.get("results", []):
        version = item.get("model_version")
        if version in meta:
            mean_value = item.get("target_mean")
            std_value = item.get("target_std")
            if isinstance(mean_value, (int, float)):
                meta[version]["target_mean"] = float(mean_value)
            if isinstance(std_value, (int, float)) and float(std_value) > 0:
                meta[version]["target_std"] = float(std_value)
    return meta


def build_prediction_frames(frame: pd.DataFrame, model_paths: dict, model_meta: dict, sequence_length: int) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []

    all_tab = frame.copy()
    if model_paths["v1"].exists():
        xgb_model = joblib.load(model_paths["v1"])
        xgb_preds = xgb_model.predict(all_tab[TABULAR_FEATURE_COLS])
        prediction_frames.append(
            pd.DataFrame(
                {
                    "time_s": all_tab["time_s"].to_numpy(),
                    "actual_soh": all_tab["SOC_next"].to_numpy(),
                    "predicted_soh": xgb_preds,
                    "model": "v1 - XGBoost",
                }
            )
        )

    if tf is not None and model_paths["v2"].exists():
        x_all, y_all, t_all = build_sequence_all(frame, sequence_length=sequence_length)
        lstm_model = tf.keras.models.load_model(model_paths["v2"])
        lstm_preds_scaled = lstm_model.predict(x_all, verbose=0).reshape(-1)
        v2_mean = float(model_meta.get("v2", {}).get("target_mean", 0.0))
        v2_std = float(model_meta.get("v2", {}).get("target_std", 1.0))
        lstm_preds = (lstm_preds_scaled * v2_std) + v2_mean
        prediction_frames.append(
            pd.DataFrame(
                {
                    "time_s": t_all,
                    "actual_soh": y_all,
                    "predicted_soh": lstm_preds,
                    "model": "v2 - 2-Layer LSTM",
                }
            )
        )

    if tf is not None and model_paths["v3"].exists():
        x_all, y_all, t_all = build_sequence_all(frame, sequence_length=sequence_length)
        gru_model = tf.keras.models.load_model(model_paths["v3"])
        gru_preds_scaled = gru_model.predict(x_all, verbose=0).reshape(-1)
        v3_mean = float(model_meta.get("v3", {}).get("target_mean", 0.0))
        v3_std = float(model_meta.get("v3", {}).get("target_std", 1.0))
        gru_preds = (gru_preds_scaled * v3_std) + v3_mean
        prediction_frames.append(
            pd.DataFrame(
                {
                    "time_s": t_all,
                    "actual_soh": y_all,
                    "predicted_soh": gru_preds,
                    "model": "v3 - 2-Layer GRU",
                }
            )
        )

    if not prediction_frames:
        return pd.DataFrame(columns=["time_s", "actual_soh", "predicted_soh", "model"])

    output = pd.concat(prediction_frames, ignore_index=True)
    output["actual_soh"] = output["actual_soh"].astype(float)
    output["predicted_soh"] = output["predicted_soh"].astype(float)
    output["time_s"] = output["time_s"].astype(float)
    return output


def main() -> None:
    st.set_page_config(page_title="Battery Analytics + MLOps Dashboard", layout="wide")
    st.title("Battery SOH Intelligence Dashboard")

    project_root = Path(__file__).resolve().parent
    data_path = Path(default_data_path())
    summary_path = project_root / "models" / "battery_pipeline_summary.json"
    metrics_path = project_root / "models" / "battery_soc_metrics.json"
    comparison_path = project_root / "models" / "battery_model_comparison.json"

    metrics = load_metrics(metrics_path)
    pipeline_summary = load_metrics(summary_path)
    comparison = load_model_comparison(comparison_path)
    sequence_length = resolve_sequence_length(comparison)

    with st.spinner("Loading and preparing data..."):
        raw_df = extract_data(data_path, required_columns=BATTERY_REQUIRED_COLUMNS, verbose=False)
        clean_df = transform_battery_data(raw_df)
        model_frame = clean_df.copy()
        model_frame["SOC_next"] = model_frame["SOC"].shift(-1)
        model_frame["SOC_Lag1"] = model_frame["SOC"].shift(1)
        model_frame = model_frame.dropna().sort_values("time_s").reset_index(drop=True)
        quality_df = build_quality_summary(raw_df, clean_df)
        model_paths = get_model_paths(comparison, project_root / "models")
        model_meta = get_model_meta(comparison)
        prediction_df = build_prediction_frames(model_frame, model_paths, model_meta, sequence_length=sequence_length)

    last_run = pipeline_summary.get("completed_at_utc")
    if last_run:
        last_run = pd.to_datetime(last_run).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        last_run = (
            pd.to_datetime(metrics_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S UTC")
            if metrics_path.exists()
            else "Not available"
        )

    rows_raw = int(pipeline_summary.get("raw_rows", len(raw_df)))
    rows_clean = int(pipeline_summary.get("clean_rows", len(clean_df)))
    rows_removed = max(rows_raw - rows_clean, 0)
    retention_pct = (rows_clean / rows_raw * 100) if rows_raw > 0 else None
    pipeline_duration = pipeline_summary.get("duration_seconds")
    pipeline_status = pipeline_summary.get("status")
    safety_monitor = pipeline_summary.get("safety_monitor", {})

    st.subheader("Pipeline Reliability")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", pipeline_status if pipeline_status else ("Success" if metrics else "No output"))
    c2.metric("Raw Rows", format_metric(rows_raw))
    c3.metric("Clean Rows", format_metric(rows_clean), delta=f"{retention_pct:.2f}% retained" if retention_pct is not None else None)
    c4.metric("Rows Removed", format_metric(rows_removed))

    c5, c6 = st.columns(2)
    c5.metric("Last Model Run", last_run)
    if pipeline_duration is None:
        c6.metric("Pipeline Duration", "N/A")
    else:
        c6.metric("Pipeline Duration", f"{float(pipeline_duration):.2f}s")

    st.subheader("Safety-First Monitor")
    s1, s2, s3 = st.columns(3)
    low_conf = bool(safety_monitor.get("low_confidence", False))
    s1.metric("Confidence Flag", "LOW" if low_conf else "NORMAL")
    s2.metric(
        "Voltage Noise (latest/baseline)",
        f"{format_metric(safety_monitor.get('volts_noise_latest'))} / {format_metric(safety_monitor.get('volts_noise_baseline'))}",
    )
    s3.metric(
        "Current Noise (latest/baseline)",
        f"{format_metric(safety_monitor.get('amps_noise_latest'))} / {format_metric(safety_monitor.get('amps_noise_baseline'))}",
    )
    if low_conf:
        st.warning("Sensor noise is elevated. Marking predictions as low confidence.")

    if pipeline_status == "FAILED":
        st.error(f"Last pipeline run failed: {pipeline_summary.get('error', 'Unknown error')}")

    st.subheader("Data Governance Checks")
    total_issues = int(quality_df["Rows"].sum())
    if total_issues == 0:
        st.success("No rows were removed by quality rules. This source file appears to already be cleaned.")
    else:
        dq_bar = (
            alt.Chart(quality_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Reason:N", sort="-y", title="Filter Reason"),
                y=alt.Y("Rows:Q", title="Rows Removed"),
                color=alt.Color("Reason:N", scale=alt.Scale(scheme="set2"), legend=None),
                tooltip=["Reason:N", alt.Tooltip("Rows:Q", format=",")],
            )
        )
        st.altair_chart(dq_bar, width="stretch")

    status_df = build_quality_status(quality_df)
    status_chart = (
        alt.Chart(status_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Reason:N", title="Quality Check"),
            y=alt.Y("Rows:Q", title="Issue Rows"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Passed", "Failed"], range=["#22C55E", "#EF4444"]),
                legend=alt.Legend(title="Check Status"),
            ),
            tooltip=["Reason:N", "Status:N", alt.Tooltip("Rows:Q", format=",")],
        )
    )
    st.altair_chart(status_chart, width="stretch")

    st.subheader("Battery Signal Overview")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avg Voltage", f"{clean_df['Volts'].mean():.4f} V")
    b2.metric("Avg Current", f"{clean_df['Amps'].mean():.4f} A")
    b3.metric("Avg SOC", f"{clean_df['SOC'].mean():.2f}")
    b4.metric("Avg Power", f"{clean_df['Power_W'].mean():.4f} W")

    st.subheader("SOC / Current / Voltage Trend")
    trend_df = clean_df[["time_s", "SOC", "Amps", "Volts"]].copy()
    trend_sample = trend_df.iloc[:: max(len(trend_df) // 3000, 1)].copy()
    st.caption("Each signal uses its own y-scale to preserve shape and magnitude differences.")

    soc_chart = (
        alt.Chart(trend_sample)
        .mark_line(color="#2563EB")
        .encode(
            x=alt.X("time_s:Q", title="time_s"),
            y=alt.Y("SOC:Q", title="SOC"),
            tooltip=["time_s:Q", alt.Tooltip("SOC:Q", format=".4f")],
        )
        .properties(height=170)
    )

    current_chart = (
        alt.Chart(trend_sample)
        .mark_line(color="#DC2626")
        .encode(
            x=alt.X("time_s:Q", title="time_s"),
            y=alt.Y("Amps:Q", title="Current (A)"),
            tooltip=["time_s:Q", alt.Tooltip("Amps:Q", format=".4f")],
        )
        .properties(height=170)
    )

    voltage_chart = (
        alt.Chart(trend_sample)
        .mark_line(color="#059669")
        .encode(
            x=alt.X("time_s:Q", title="time_s"),
            y=alt.Y("Volts:Q", title="Voltage (V)"),
            tooltip=["time_s:Q", alt.Tooltip("Volts:Q", format=".4f")],
        )
        .properties(height=170)
    )

    stacked_trend = alt.vconcat(soc_chart, current_chart, voltage_chart).resolve_scale(y="independent", x="shared")
    st.altair_chart(stacked_trend, width="stretch")

    st.subheader("SOC Distribution")
    soc_hist = (
        alt.Chart(clean_df)
        .mark_bar(opacity=0.8)
        .encode(
            alt.X("SOC:Q", bin=alt.Bin(maxbins=40), title="SOC (SOH proxy)"),
            alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Rows")],
        )
    )
    st.altair_chart(soc_hist, width="stretch")

    st.subheader("Feature Correlation")
    corr_cols = ["Amps", "Volts", "SOC", "Power_W", "dV", "dI"]
    corr_matrix = clean_df[corr_cols].corr().reset_index().melt(id_vars="index", var_name="Feature", value_name="Corr")
    corr_matrix = corr_matrix.rename(columns={"index": "BaseFeature"})
    heatmap = (
        alt.Chart(corr_matrix)
        .mark_rect()
        .encode(
            x=alt.X("Feature:N", title="Feature"),
            y=alt.Y("BaseFeature:N", title="Feature"),
            color=alt.Color("Corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
            tooltip=["BaseFeature:N", "Feature:N", alt.Tooltip("Corr:Q", format=".3f")],
        )
    )
    st.altair_chart(heatmap, width="stretch")

    st.subheader("Model Leaderboard")
    leaderboard = pd.DataFrame(comparison.get("results", []))
    if leaderboard.empty:
        st.info("Run compare_battery_models.py to populate v1/v2/v3 leaderboard.")
    else:
        display_cols = ["model_version", "model_name", "mae", "r2", "rows_used", "test_rows"]
        display_cols = [col for col in display_cols if col in leaderboard.columns]
        st.dataframe(leaderboard[display_cols], width="stretch")

    st.subheader("Actual SOH vs Predicted SOH (All Models)")
    st.caption("SOH is visualized using SOC as the target proxy in this dataset.")
    if prediction_df.empty:
        st.warning("No model predictions available. Train models v1/v2/v3 first.")
    else:
        chart_df = prediction_df.sort_values(["model", "time_s"]).copy()
        max_points_per_model = 5000
        sampled_frames = []
        for model_name, frame in chart_df.groupby("model", sort=False):
            step = max(len(frame) // max_points_per_model, 1)
            sampled = frame.iloc[::step].copy()
            sampled["model"] = model_name
            sampled_frames.append(sampled)
        chart_df = pd.concat(sampled_frames, ignore_index=True)

        line_df = pd.concat(
            [
                chart_df.assign(series="Actual", value=chart_df["actual_soh"]),
                chart_df.assign(series="Predicted", value=chart_df["predicted_soh"]),
            ],
            ignore_index=True,
        )

        st.caption("Each model is plotted separately with fixed x-axis range 0–66,000 seconds.")
        ordered_models = ["v1 - XGBoost", "v2 - 2-Layer LSTM", "v3 - 2-Layer GRU"]
        available_models = [model_name for model_name in ordered_models if model_name in set(line_df["model"])]

        for model_name in available_models:
            model_df = line_df[line_df["model"] == model_name].copy()
            single_model_chart = (
                alt.Chart(model_df)
                .mark_line()
                .encode(
                    x=alt.X("time_s:Q", title="time_s", scale=alt.Scale(domain=[0, 66000])),
                    y=alt.Y("value:Q", title="SOH (SOC proxy)", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(domain=["Actual", "Predicted"], range=["#2563EB", "#DC2626"]),
                        title="Series",
                    ),
                    strokeDash=alt.StrokeDash(
                        "series:N",
                        scale=alt.Scale(domain=["Actual", "Predicted"], range=[[1, 0], [6, 4]]),
                    ),
                    tooltip=["series:N", "time_s:Q", alt.Tooltip("value:Q", format=".4f")],
                )
                .properties(height=230, title=model_name)
            )
            st.altair_chart(single_model_chart, width="stretch")

        comparison_bar = (
            alt.Chart(chart_df)
            .transform_calculate(abs_error="abs(datum.actual_soh - datum.predicted_soh)")
            .mark_bar()
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("mean(abs_error):Q", title="Mean Absolute Error (chart subset)"),
                color=alt.Color("model:N", legend=None),
            )
        )
        st.altair_chart(comparison_bar, width="stretch")
        st.caption("Prediction chart spans the full available timeline with even sampling for performance.")

    st.subheader("Prediction Uncertainty (Quantile Range)")
    q_low = metrics.get("predicted_soc_p05")
    q_mid = metrics.get("predicted_soc")
    q_high = metrics.get("predicted_soc_p95")
    q1, q2, q3 = st.columns(3)
    q1.metric("SOH P05", format_metric(q_low))
    q2.metric("SOH P50", format_metric(q_mid))
    q3.metric("SOH P95", format_metric(q_high))
    if q_low is not None and q_high is not None:
        st.caption(f"Prediction interval width: {float(q_high) - float(q_low):.4f}")

    st.subheader("Physics-Informed Health Indicators")
    hi_cols = ["time_s", "DVA_Rolling_20", "ICA_Rolling_20", "R_int_ohm"]
    hi_df = clean_df[hi_cols].copy()
    hi_sample = hi_df.iloc[:: max(len(hi_df) // 2500, 1)].copy()
    hi_long = hi_sample.melt(id_vars=["time_s"], var_name="indicator", value_name="value")
    hi_chart = (
        alt.Chart(hi_long)
        .mark_line()
        .encode(
            x=alt.X("time_s:Q", title="time_s"),
            y=alt.Y("value:Q", title="Indicator Value"),
            color=alt.Color("indicator:N", title="Health Indicator"),
            tooltip=["time_s:Q", "indicator:N", alt.Tooltip("value:Q", format=".6f")],
        )
    )
    st.altair_chart(hi_chart, width="stretch")


if __name__ == "__main__":
    main()