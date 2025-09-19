#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

def _load_params(path="config/params.yaml") -> dict:
    import yaml
    with open(path,"r") as f:
        return yaml.safe_load(f)

def _yahoo_series(ticker: str, start=None, end=None):
    import yfinance as yf
    y = yf.Ticker(ticker)
    if start is not None or end is not None:
        hist = y.history(start=start, end=end, auto_adjust=False)
    else:
        hist = y.history(period="max", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"Yahoo empty history for {ticker}")
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    df = pd.DataFrame({"Close": hist["Close"].astype(float), "AdjClose": hist["Adj Close"].astype(float)}, index=hist.index)
    div = y.dividends
    try: div = div.tz_localize(None)
    except Exception: pass
    return df, div

def _should_rebuild(pdf: pd.DataFrame, force: bool) -> bool:
    if force: return True
    have_adj = "adj_close" in pdf.columns
    have_exd = "is_ex_div" in pdf.columns
    if not have_adj or not have_exd:
        return True
    # adj_close présent mais trop de NaN → refaire
    nan_ratio = float(pd.isna(pdf["adj_close"]).mean()) if "adj_close" in pdf.columns else 1.0
    return nan_ratio > 0.01

def _process_one(root_dir: Path, ticker: str, force: bool) -> bool:
    fp = root_dir / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[SKIP] {ticker} parquet absent: {fp}")
        return False

    df = pl.read_parquet(fp).with_columns(pl.col("date").cast(pl.Date))
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.set_index("date")

    if "close" not in pdf.columns:
        print(f"[ERR] {ticker} sans colonne 'close'.")
        return False

    if not _should_rebuild(pdf, force):
        print(f"[OK] {ticker}: rien à faire")
        return True

    start = (pdf.index.min() - pd.Timedelta(days=1)).date()
    end   = (pdf.index.max() + pd.Timedelta(days=1)).date()

    yh, div = _yahoo_series(ticker, start=start, end=end)
    factor = (yh["AdjClose"] / yh["Close"]).replace([np.inf, -np.inf], np.nan).dropna()
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
        out[c] = aligned[c].astype(float)
    out["adj_close"] = aligned["adj_close"].astype(float)
    out["is_ex_div"] = aligned["is_ex_div"].astype(bool)

    pl.from_pandas(out.reset_index(drop=True)).write_parquet(fp)
    print(f"[FIX] {ticker}: adj_close & is_ex_div reconstruits")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    params = _load_params(args.config)
    src = (params.get("data",{}).get("source","yahoo") or "yahoo").lower()
    root_dir = Path(params["data"].get("root_dir_ibkr" if src=="ibkr" else "root_dir_yahoo"))

    print(f"[adjust_prices] root={root_dir}")
    tickers = params.get("universe",{}).get("tickers",[])
    ok = 0
    for i,t in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {t} …")
        if _process_one(root_dir, t, args.force):
            ok += 1
    print(f"[adjust_prices] terminé: ok={ok}/{len(tickers)} → {root_dir}")

if __name__ == "__main__":
    main()
