from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}

def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def extract_data(path, pattern: str = "*.csv", required_columns=None, verbose: bool = True):
    source_path = Path(path)

    if source_path.is_dir():
        files = sorted(source_path.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matched pattern '{pattern}' in directory: {source_path}"
            )
        frames = [_read_csv_with_fallback(file_path) for file_path in files]
        df = pd.concat(frames, ignore_index=True)
        source_count = len(files)
    else:
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        df = _read_csv_with_fallback(source_path)
        source_count = 1

    required_set = set(required_columns) if required_columns is not None else REQUIRED_COLUMNS
    missing_columns = required_set.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        preview_columns = ", ".join(sorted(str(column) for column in df.columns[:10]))
        raise ValueError(
            "Source dataset is missing required columns: "
            f"{missing_str}. Sample detected columns: {preview_columns}"
        )

    df.attrs["source_file_count"] = source_count

    if verbose:
        print(
            f"Extracted {len(df):,} rows and {len(df.columns)} columns "
            f"from {source_count} file(s)"
        )

    return df