from __future__ import annotations

import numpy as np
import pandas as pd

BATTERY_REQUIRED_COLUMNS = {
    "time_s",
    "Amps",
    "Volts",
    "SOC",
}


def transform_battery_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = BATTERY_REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Battery dataset is missing required columns: {missing_str}")

    transformed = df.copy()
    transformed["time_s"] = pd.to_numeric(transformed["time_s"], errors="coerce")
    transformed["Amps"] = pd.to_numeric(transformed["Amps"], errors="coerce")
    transformed["Volts"] = pd.to_numeric(transformed["Volts"], errors="coerce")
    transformed["SOC"] = pd.to_numeric(transformed["SOC"], errors="coerce")

    rows_before = int(len(transformed))
    transformed = transformed.dropna(subset=["time_s", "Amps", "Volts", "SOC"]).copy()
    transformed = transformed.sort_values("time_s").drop_duplicates(subset=["time_s"], keep="last").copy()

    transformed["dt_s"] = transformed["time_s"].diff().fillna(0.0)
    transformed["dt_s"] = transformed["dt_s"].clip(lower=0.0)

    transformed["dQ_Ah"] = (transformed["Amps"] * transformed["dt_s"]) / 3600.0
    transformed["Q_Ah"] = transformed["dQ_Ah"].cumsum()

    transformed["Power_W"] = transformed["Amps"] * transformed["Volts"]
    transformed["dV"] = transformed["Volts"].diff().fillna(0.0)
    transformed["dI"] = transformed["Amps"].diff().fillna(0.0)
    transformed["dQ"] = transformed["Q_Ah"].diff().fillna(0.0)

    transformed["dVdQ"] = transformed["dV"] / transformed["dQ"].replace(0, np.nan)
    transformed["dQdV"] = transformed["dQ"] / transformed["dV"].replace(0, np.nan)
    transformed["dVdQ"] = transformed["dVdQ"].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    transformed["dQdV"] = transformed["dQdV"].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    resistance_raw = transformed["dV"].abs() / transformed["dI"].abs().replace(0, np.nan)
    transformed["R_int_ohm"] = resistance_raw.replace([float("inf"), float("-inf")], pd.NA)
    transformed["R_int_ohm"] = transformed["R_int_ohm"].rolling(30, min_periods=1).median()
    transformed["R_int_ohm"] = transformed["R_int_ohm"].bfill().fillna(0.0)

    transformed["Amps_RollingMean_20"] = transformed["Amps"].rolling(20, min_periods=1).mean()
    transformed["Volts_RollingMean_20"] = transformed["Volts"].rolling(20, min_periods=1).mean()
    transformed["Power_RollingMean_20"] = transformed["Power_W"].rolling(20, min_periods=1).mean()
    transformed["DVA_Rolling_20"] = transformed["dVdQ"].rolling(20, min_periods=1).mean()
    transformed["ICA_Rolling_20"] = transformed["dQdV"].rolling(20, min_periods=1).mean()

    rows_after = int(len(transformed))
    transformed.attrs["quality_report"] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed_total": rows_before - rows_after,
    }

    return transformed.reset_index(drop=True)
