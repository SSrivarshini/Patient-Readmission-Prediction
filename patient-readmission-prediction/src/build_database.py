"""Load the raw CSV into a SQLite database so features can be pulled with SQL.

Run: python -m src.build_database
"""
import sqlite3

import pandas as pd

from src.config import DB_PATH, RAW_CSV


def build_db(csv_path=RAW_CSV, db_path=DB_PATH) -> int:
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("patients", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_id ON patients(patient_id)")
    return len(df)


if __name__ == "__main__":
    print(f"Loaded {build_db():,} records into {DB_PATH}")
