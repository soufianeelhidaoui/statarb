from __future__ import annotations
from pathlib import Path
import duckdb

def write_scored_to_duckdb(parquet_path: str | Path, db_path: str | Path = "reports/analytics.duckdb", table: str = "pairs_scored"):
    db_path = Path(db_path)
    con = duckdb.connect(str(db_path))
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM read_parquet('{parquet_path}') LIMIT 0")
    con.execute(f"DELETE FROM {table}")
    con.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{parquet_path}')")
    con.close()
    return db_path
