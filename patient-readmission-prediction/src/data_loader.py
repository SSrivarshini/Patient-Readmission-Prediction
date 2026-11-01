import sqlite3

import pandas as pd

from src.config import DB_PATH, FEATURES_SQL


def load_features(db_path=DB_PATH) -> pd.DataFrame:
    """Run sql/features.sql against the patients database and return the modelling table."""
    query = FEATURES_SQL.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
