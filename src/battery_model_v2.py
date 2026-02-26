from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise RuntimeError("tensorflow is required. Install with: pip install tensorflow") from exc

try:
    from src.battery_model_data import load_battery_frame, build_sequence_split, default_data_path
except ModuleNotFoundError:
    from battery_model_data import load_battery_frame, build_sequence_split, default_data_path


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


def run_model_v2(
    data_path: str,
    data_glob: str = "*.csv",
    max_rows: int | None = 200000,
    sequence_length: int = 30,
    epochs: int = 6,
):
    tf.random.set_seed(42)

    frame = load_battery_frame(data_path=data_path, data_glob=data_glob, max_rows=max_rows)
    x_train, y_train, x_test, y_test = build_sequence_split(frame, sequence_length=sequence_length)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x_train, y_train_scaled, epochs=epochs, batch_size=256, verbose=0)

    preds_scaled = model.predict(x_test, verbose=0).reshape(-1)
    preds = (preds_scaled * y_std) + y_mean
    y_test_arr = np.asarray(y_test, dtype=float)
    mae = _mae(y_test_arr, preds)
    r2 = _r2_safe(y_test_arr, preds)

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "battery_model_v2_lstm.keras"
    metrics_path = models_dir / "battery_model_v2_metrics.json"

    model.save(model_path)

    metrics = {
        "model_version": "v2",
        "model_name": "TwoLayerLSTM",
        "mae": mae,
        "r2": r2,
        "target_mean": y_mean,
        "target_std": y_std,
        "rows_used": int(len(frame)),
        "sequence_length": int(sequence_length),
        "epochs": int(epochs),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "generated_at_utc": _utc_now_iso(),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model v2 complete: MAE={mae:.6f}, R2={r2:.6f}")
    print(f"Saved: {model_path}")
    print(f"Saved: {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battery model v2 - two-layer LSTM")
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--data-glob", type=str, default="*.csv")
    parser.add_argument("--max-rows", type=int, default=200000, help="Set 0 to use all rows")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    run_model_v2(
        data_path=args.data_path,
        data_glob=args.data_glob,
        max_rows=(None if args.max_rows == 0 else args.max_rows),
        sequence_length=args.sequence_length,
        epochs=args.epochs,
    )
