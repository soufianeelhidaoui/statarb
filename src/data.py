from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import pandas as pd
import polars as pl
import numpy as np

def _root_dir_for_source(params: dict) -> Path:
    data = params.get("data", {})
    src = data.get("source", "yahoo").lower()
    if data.get("separate_roots", True):
        return Path(data.get(f"root_dir_{src}", f"data/eod/ETFs_{src}"))
    return Path(data.get("root_dir", "data/eod/ETFs"))

def _write_provenance(root: Path, source: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Path(root, "_PROVENANCE.json").write_text(
        pd.Series({"source": source, "updated_at": pd.Timestamp.utcnow().isoformat()}).to_json(),
        encoding="utf-8"
    )

def _write_parquet(root: Path, ticker: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date","open","high","low","close","adj_close","volume"]
    pl.from_pandas(df[cols]).write_parquet(root / f"{ticker}.parquet")

def _fmt_prog(i: int, n: int, t: str) -> str:
    w = len(str(n))
    return f"[{str(i).rjust(w)}/{n}] {t}"

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance peut retourner MultiIndex; on aplati proprement
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns.to_list()]
    return df

def _to_1d(s: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            return s.iloc[:, 0]
        raise ValueError(f"Colonne inattendue 2D: shape={s.shape}")
    return s

def _ingest_yahoo(params: dict, tickers: Iterable[str]) -> Path:
    import yfinance as yf
    root = _root_dir_for_source(params)
    root.mkdir(parents=True, exist_ok=True)

    tickers = list(tickers)
    n = len(tickers)
    ok = 0
    for i, t in enumerate(tickers, start=1):
        print(f"[Yahoo] {_fmt_prog(i,n,t)} …")
        df = yf.download(t, period="max", interval="1d", auto_adjust=False, progress=False, group_by="column", threads=False)
        if df is None or df.empty:
            print(f"   [ERR] {t}: vide")
            continue
        df = _flatten_yf_columns(df)
        # uniformise noms
        ren = {"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}
        for k in list(ren.keys()):
            if k not in df.columns and k.lower() in df.columns:
                df.rename(columns={k.lower(): ren[k]}, inplace=True)
        df.rename(columns=ren, inplace=True)

        # force 1D
        for c in ["open","high","low","close","adj_close","volume"]:
            if c in df.columns:
                df[c] = _to_1d(df[c])

        df = df.reset_index().rename(columns={"Date":"date"})
        out = pd.DataFrame({
            "date": pd.to_datetime(df["date"]),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "adj_close": df["adj_close"].astype(float) if "adj_close" in df.columns else df["close"].astype(float),
            "volume": df["volume"].astype(float),
        })
        _write_parquet(root, t, out)
        ok += 1

    _write_provenance(root, "yahoo")
    print(f"[Yahoo] Terminé. OK={ok}/{n} → {root}")
    return root

def _ingest_ibkr(params: dict, tickers: Iterable[str]) -> Path:
    from ib_insync import IB, util, Stock
    root = _root_dir_for_source(params)
    root.mkdir(parents=True, exist_ok=True)

    tickers = list(tickers)
    n = len(tickers)
    mode = params.get("trading", {}).get("mode", "paper").lower()
    port = 7497 if mode == "paper" else 7496

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=117, timeout=8)

    ok = 0
    for i, t in enumerate(tickers, start=1):
        print(f"[IBKR] {_fmt_prog(i,n,t)} …")
        try:
            ct = ib.qualifyContracts(Stock(t, "SMART", "USD"))[0]
            bars = ib.reqHistoricalData(
                ct, endDateTime="", durationStr="30 Y", barSizeSetting="1 day",
                whatToShow="ADJUSTED_LAST", useRTH=True, formatDate=1
            )
            if not bars:
                bars = ib.reqHistoricalData(
                    ct, endDateTime="", durationStr="30 Y", barSizeSetting="1 day",
                    whatToShow="TRADES", useRTH=True, formatDate=1
                )
            df = util.df(bars)
            df = df.rename(columns={"date":"date","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
            if "date" not in df.columns:
                # parfois ib_insync renvoie index = date
                df = df.reset_index().rename(columns={"index":"date"})
            out = pd.DataFrame({
                "date": pd.to_datetime(df["date"]),
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "adj_close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
            })
            _write_parquet(root, t, out)
            print(f"   [OK] {t}: {len(out)} barres")
            ok += 1
        except Exception as e:
            print(f"   [ERR] {t}: {e}")

    ib.disconnect()
    _write_provenance(root, "ibkr")
    print(f"[IBKR] Terminé. OK={ok}/{n} → {root}")
    return root

def ensure_universe(params: dict, tickers: Iterable[str]) -> Path:
    src = params.get("data", {}).get("source", "yahoo").lower()
    root = _root_dir_for_source(params)
    root.mkdir(parents=True, exist_ok=True)

    tickers = list(tickers)
    missing = [t for t in tickers if not (root / f"{t}.parquet").exists()]

    if src == "yahoo":
        if missing:
            return _ingest_yahoo(params, missing)
        _write_provenance(root, "yahoo")
        return root

    if src == "ibkr":
        if missing:
            return _ingest_ibkr(params, missing)
        _write_provenance(root, "ibkr")
        return root

    raise ValueError(f"Source inconnue: {src}")

def get_price_series(root_dir: Path, ticker: str) -> pl.DataFrame:
    path = Path(root_dir) / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet manquant: {path}")
    return pl.read_parquet(path)
