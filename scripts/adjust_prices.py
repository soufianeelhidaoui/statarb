#!/usr/bin/env python3
import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

def load_params(path="config/params.yaml"):
    import yaml
    with open(path,"r") as f:
        return yaml.safe_load(f)

# 1) dans yahoo_series() : utiliser period="max" si pas de bornes
def yahoo_series(ticker: str, start=None, end=None):
    import yfinance as yf
    y = yf.Ticker(ticker)
    if start is not None or end is not None:
        hist = y.history(start=start, end=end, auto_adjust=False)
    else:
        hist = y.history(period="max", auto_adjust=False)  # <- important
    if hist is None or hist.empty:
        raise RuntimeError(f"Yahoo empty history for {ticker}")
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    df = pd.DataFrame({
        "Close": hist["Close"].astype(float),
        "Adj Close": hist["Adj Close"].astype(float)
    }, index=hist.index)
    div = y.dividends
    try: div = div.tz_localize(None)
    except Exception: pass
    return df, div

def process_ticker(root_dir: Path, ticker: str):
    fp = root_dir / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[SKIP] {ticker} parquet not found: {fp}")
        return False

    df = pl.read_parquet(fp).with_columns(pl.col("date").cast(pl.Date))
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf.set_index("date", inplace=True)

    if "close" not in pdf.columns:
        print(f"[ERR] {ticker} missing 'close' in parquet; cannot compute adj_close.")
        return False

    first_date = pdf.index.min()
    last_date = pdf.index.max()
    start = (first_date - pd.Timedelta(days=1)).date()
    end   = (last_date + pd.Timedelta(days=1)).date()

    try:
        yh, div = yahoo_series(ticker, start=start, end=end)
    except Exception as e:
        print(f"[WARN] Yahoo fetch failed for {ticker}: {e}")
        return False

    factor = (yh["Adj Close"] / yh["Close"]).replace([np.inf, -np.inf], np.nan).dropna()
    factor = factor.to_frame("factor").sort_index()
    aligned = pdf.join(factor, how="left")
    aligned["factor"] = aligned["factor"].ffill()
    aligned["adj_close"] = aligned["close"] * aligned["factor"]

    is_ex = pd.Series(False, index=aligned.index)
    if div is not None and len(div) > 0:
        ex_dates = set(pd.to_datetime(div.index).normalize())
        is_ex = aligned.index.normalize().isin(ex_dates)
    aligned["is_ex_div"] = is_ex.astype(bool)

    keep = [c for c in ["open","high","low","close","volume"] if c in aligned.columns]
    out = pd.DataFrame(index=aligned.index)
    out["date"] = out.index.date
    for c in keep:
        out[c] = aligned[c]
    out["adj_close"] = aligned["adj_close"]
    out["is_ex_div"] = aligned["is_ex_div"]
    pl.from_pandas(out.reset_index(drop=True)).write_parquet(fp)
    print(f"[OK] {ticker}: adj_close & is_ex_div updated.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.yaml")
    args = ap.parse_args()
    params = load_params(args.config)
    mode = params["env"]["mode"]
    root_dir = Path(params["data"].get("root_dir_prod" if mode=="prod" else "root_dir_dev", params["data"]["root_dir"]))
    for t in params["universe"]["tickers"]:
        process_ticker(root_dir, t)

if __name__ == "__main__":
    main()
