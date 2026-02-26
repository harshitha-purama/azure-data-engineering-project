from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyodbc

try:
    from src.load import get_connection
except ModuleNotFoundError:
    from load import get_connection


def publish_battery_soc_from_artifacts(
    metrics_path: Path | None = None,
    prediction_path: Path | None = None,
    strict: bool = False,
) -> bool:
    project_root = Path(__file__).resolve().parents[1]
    default_metrics_path = project_root / "models" / "battery_soc_metrics.json"
    default_prediction_path = project_root / "models" / "battery_soc_latest_prediction.json"

    final_metrics_path = metrics_path or default_metrics_path
    final_prediction_path = prediction_path or default_prediction_path

    if not final_metrics_path.exists() or not final_prediction_path.exists():
        message = (
            "Battery publish skipped: required artifact missing. "
            f"metrics={final_metrics_path.exists()}, prediction={final_prediction_path.exists()}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return False

    metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    prediction = json.loads(final_prediction_path.read_text(encoding="utf-8"))

    prediction_time_s = prediction.get("prediction_time_s")
    predicted_soc = prediction.get("predicted_soc")

    if prediction_time_s is None or predicted_soc is None:
        message = "Battery publish skipped: prediction payload missing prediction_time_s or predicted_soc"
        if strict:
            raise ValueError(message)
        print(message)
        return False

    params = (
        float(prediction_time_s),
        float(predicted_soc),
        prediction.get("step_size_seconds"),
        metrics.get("model_name"),
        metrics.get("mae"),
        metrics.get("r2"),
    )

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            EXEC analytics.sp_upsert_battery_soc_prediction
                @PredictionTimeS = ?,
                @PredictedSOC = ?,
                @StepSizeSeconds = ?,
                @ModelName = ?,
                @MAE = ?,
                @R2 = ?
            """,
            params,
        )
        conn.commit()
        print(f"Battery SOC publish complete: analytics.battery_soc_predictions (time_s={prediction_time_s})")
        return True
    except pyodbc.Error as exc:
        message = (
            "Battery SOC publish failed. Ensure SQL object exists by running "
            "sql/31_upsert_battery_soc_predictions.sql"
        )
        if strict:
            raise RuntimeError(message) from exc
        print(message)
        print(f"Underlying SQL error: {exc}")
        return False
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish latest battery SOC prediction to analytics.battery_soc_predictions")
    parser.add_argument("--strict", action="store_true", help="Fail with a non-zero exit on publish errors")
    args = parser.parse_args()

    publish_battery_soc_from_artifacts(strict=args.strict)
