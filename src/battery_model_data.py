from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.extract import extract_data
    from src.battery_transform import transform_battery_data, BATTERY_REQUIRED_COLUMNS
except ModuleNotFoundError:
    from extract import extract_data
    from battery_transform import transform_battery_data, BATTERY_REQUIRED_COLUMNS

TABULAR_FEATURE_COLS = [
    "time_s",
    "Amps",
    "Volts",
    "Power_W",
    "dV",
    "dI",
    "Amps_RollingMean_20",
    "Volts_RollingMean_20",
    "Power_RollingMean_20",
    "SOC_Lag1",
]

SEQUENCE_FEATURE_COLS = [
    "Amps",
    "Volts",
    "SOC",
    "Power_W",
    "dV",
    "dI",
    "Amps_RollingMean_20",
    "Volts_RollingMean_20",
    "Power_RollingMean_20",
]


def load_battery_frame(
    data_path: str,
    data_glob: str = "*.csv",
    max_rows: int | None = 200000,
    verbose: bool = True,
) -> pd.DataFrame:
    raw_df = extract_data(
        data_path,
        pattern=data_glob,
        required_columns=BATTERY_REQUIRED_COLUMNS,
        verbose=verbose,
    )
    clean_df = transform_battery_data(raw_df)

    frame = clean_df.copy()
    frame["SOC_next"] = frame["SOC"].shift(-1)
    frame["SOC_Lag1"] = frame["SOC"].shift(1)
    frame = frame.dropna().sort_values("time_s").reset_index(drop=True)

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)

    return frame


def build_tabular_split(frame: pd.DataFrame):
    split_idx = int(len(frame) * 0.8)
    train_df = frame.iloc[:split_idx].copy()
    test_df = frame.iloc[split_idx:].copy()
    x_train = train_df[TABULAR_FEATURE_COLS]
    y_train = train_df["SOC_next"]
    x_test = test_df[TABULAR_FEATURE_COLS]
    y_test = test_df["SOC_next"]
    return x_train, y_train, x_test, y_test


def build_sequence_split(frame: pd.DataFrame, sequence_length: int = 30):
    values = frame[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
    targets = frame["SOC_next"].to_numpy(dtype=np.float32)

    x_seq = []
    y_seq = []
    for end in range(sequence_length, len(values) + 1):
        start = end - sequence_length
        x_seq.append(values[start:end])
        y_seq.append(targets[end - 1])

    if not x_seq:
        raise ValueError("Not enough rows to build sequence dataset")

    x_all = np.asarray(x_seq, dtype=np.float32)
    y_all = np.asarray(y_seq, dtype=np.float32)

    split_idx = int(len(x_all) * 0.8)
    x_train = x_all[:split_idx]
    y_train = y_all[:split_idx]
    x_test = x_all[split_idx:]
    y_test = y_all[split_idx:]

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    x_test_2d = x_test.reshape(-1, x_test.shape[-1])

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)

    return x_train_scaled, y_train, x_test_scaled, y_test


def default_data_path() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "data" / "battery_data_with_soc.csv")
