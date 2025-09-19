from __future__ import annotations
from pathlib import Path
import polars as pl

def _compute_is_ex_div(df: pl.DataFrame, tol_bp: int = 1) -> pl.DataFrame:
    if "adj_close" not in df.columns or "close" not in df.columns:
        return df.with_columns(pl.lit(False).alias("is_ex_div"))
    f = (pl.col("adj_close") / pl.col("close")).alias("factor")
    df2 = df.with_columns(f)
    # changement de facteur ⇒ jour ex-div/split
    # tolérance en basis points sur la variation relative
    change = (pl.col("factor") / pl.col("factor").shift(1) - 1.0).abs()
    is_ex = (change * 10_000 > tol_bp)
    return df2.with_columns(is_ex.alias("is_ex_div")).drop("factor")

def ensure_is_ex_div(parquet_path: Path, tol_bp: int = 1) -> bool:
    if not parquet_path.exists():
        return False
    df = pl.read_parquet(parquet_path)
    if "is_ex_div" in df.columns:
        return True
    out = _compute_is_ex_div(df, tol_bp=tol_bp)
    out.write_parquet(parquet_path)
    return True

def ensure_folder_has_is_ex_div(root_dir: Path, tickers: list[str], tol_bp: int = 1) -> dict:
    root = Path(root_dir)
    results = {}
    for t in tickers:
        p = root / f"{t}.parquet"
        ok = ensure_is_ex_div(p, tol_bp=tol_bp)
        results[t] = ok
    return results
