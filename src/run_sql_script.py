from __future__ import annotations

import argparse
from pathlib import Path

try:
    from src.load import get_connection
except ModuleNotFoundError:
    from load import get_connection


def split_batches(sql_text: str) -> list[str]:
    lines = sql_text.splitlines()
    batches: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.strip().upper() == "GO":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)

    last_batch = "\n".join(current).strip()
    if last_batch:
        batches.append(last_batch)

    return batches


def run_sql_file(sql_file: Path) -> None:
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    sql_text = sql_file.read_text(encoding="utf-8")
    batches = split_batches(sql_text)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for idx, batch in enumerate(batches, start=1):
            cursor.execute(batch)
            conn.commit()
            print(f"Batch {idx}/{len(batches)} executed")

        print(f"SQL script completed: {sql_file}")
    finally:
        cursor.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SQL file against Azure SQL using project connection settings")
    parser.add_argument("sql_file", help="Path to .sql file")
    args = parser.parse_args()

    run_sql_file(Path(args.sql_file).resolve())


if __name__ == "__main__":
    main()
