from __future__ import annotations
from pathlib import Path
import polars as pl

def _ensure_date(df: pl.DataFrame) -> pl.DataFrame:
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df

def load_price_df(root_dir: str | Path, ticker: str, prefer_adj: bool = True, px_policy: str = "best") -> pl.DataFrame:
    fp = Path(root_dir) / f"{ticker}.parquet"
    df = pl.read_parquet(fp)
    df = _ensure_date(df)
    cols_keep = [c for c in ["open","high","low","close","adj_close","volume","is_ex_div"] if c in df.columns]
    df = df.select(["date", *cols_keep])

    has_close = "close" in df.columns
    has_adj   = "adj_close" in df.columns

    if px_policy == "close_only":
        if not has_close:
            raise ValueError(f"{ticker}: 'close' missing for close_only policy")
        df = df.with_columns([
            pl.col("close").alias("px"),
            pl.lit("close").alias("px_kind")
        ])
        return df

    # px_policy == "best": adj si possible, sinon close, mais en **coalesce ligne à ligne**
    if prefer_adj and has_adj and has_close:
        df = df.with_columns([
            pl.coalesce([pl.col("adj_close"), pl.col("close")]).alias("px"),
            pl.lit("adj_or_close").alias("px_kind")
        ])
    elif has_adj:
        df = df.with_columns([
            pl.col("adj_close").alias("px"),
            pl.lit("adj").alias("px_kind")
        ])
    elif has_close:
        df = df.with_columns([
            pl.col("close").alias("px"),
            pl.lit("close").alias("px_kind")
        ])
    else:
        raise ValueError(f"{ticker}: neither 'adj_close' nor 'close' present")

    return df
