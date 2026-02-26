from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:
    raise RuntimeError("xgboost is required. Install with: pip install xgboost") from exc

try:
    from src.battery_model_data import load_battery_frame, build_tabular_split, default_data_path
except ModuleNotFoundError:
    from battery_model_data import load_battery_frame, build_tabular_split, default_data_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def run_model_v1(data_path: str, data_glob: str = "*.csv", max_rows: int | None = 200000):
    frame = load_battery_frame(data_path=data_path, data_glob=data_glob, max_rows=max_rows)
    x_train, y_train, x_test, y_test = build_tabular_split(frame)

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
    model.fit(x_train, y_train)

    preds = np.asarray(model.predict(x_test), dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)
    mae = _mae(y_test_arr, preds)
    r2 = _r2_safe(y_test_arr, preds)

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "battery_model_v1_xgboost.joblib"
    metrics_path = models_dir / "battery_model_v1_metrics.json"

    joblib.dump(model, model_path)

    metrics = {
        "model_version": "v1",
        "model_name": "XGBoostRegressor",
        "mae": mae,
        "r2": r2,
        "rows_used": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "generated_at_utc": _utc_now_iso(),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model v1 complete: MAE={mae:.6f}, R2={r2:.6f}")
    print(f"Saved: {model_path}")
    print(f"Saved: {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battery model v1 - XGBoost")
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--data-glob", type=str, default="*.csv")
    parser.add_argument("--max-rows", type=int, default=200000, help="Set 0 to use all rows")
    args = parser.parse_args()

    run_model_v1(
        data_path=args.data_path,
        data_glob=args.data_glob,
        max_rows=(None if args.max_rows == 0 else args.max_rows),
    )
