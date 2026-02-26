from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from src.battery_model_data import default_data_path
    from src.battery_model_v1 import run_model_v1
    from src.battery_model_v2 import run_model_v2
    from src.battery_model_v3 import run_model_v3
except ModuleNotFoundError:
    from battery_model_data import default_data_path
    from battery_model_v1 import run_model_v1
    from battery_model_v2 import run_model_v2
    from battery_model_v3 import run_model_v3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compare_models(
    data_path: str,
    data_glob: str = "*.csv",
    max_rows: int | None = 200000,
    sequence_length: int = 30,
    epochs: int = 6,
) -> dict:
    results = []

    results.append(run_model_v1(data_path=data_path, data_glob=data_glob, max_rows=max_rows))
    results.append(
        run_model_v2(
            data_path=data_path,
            data_glob=data_glob,
            max_rows=max_rows,
            sequence_length=sequence_length,
            epochs=epochs,
        )
    )
    results.append(
        run_model_v3(
            data_path=data_path,
            data_glob=data_glob,
            max_rows=max_rows,
            sequence_length=sequence_length,
            epochs=epochs,
        )
    )

    ranked = sorted(results, key=lambda item: item["mae"])

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "leader_by_mae": ranked[0]["model_version"],
        "results": ranked,
    }

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "models" / "battery_model_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Comparison saved: {out_path}")
    print(f"Best model by MAE: {payload['leader_by_mae']}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare battery models v1/v2/v3")
    parser.add_argument("--data-path", type=str, default=default_data_path())
    parser.add_argument("--data-glob", type=str, default="*.csv")
    parser.add_argument("--max-rows", type=int, default=200000, help="Set 0 to use all rows")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    compare_models(
        data_path=args.data_path,
        data_glob=args.data_glob,
        max_rows=(None if args.max_rows == 0 else args.max_rows),
        sequence_length=args.sequence_length,
        epochs=args.epochs,
    )
