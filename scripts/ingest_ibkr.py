#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from src.config import load_params
from src.universe import load_universe
from src.provenance import save_provenance, enforce_provenance

def _connect_ib(host: str = "127.0.0.1", port: int = 7497, client_id: int = 17):
    try:
        from ib_insync import IB
    except Exception as e:
        raise SystemExit("ib_insync non installé. Fais: `pip install ib_insync`") from e
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

def _ib_stock(symbol: str):
    from ib_insync import Stock
    return Stock(symbol, "SMART", "USD")

def _fetch_daily_history(ib, symbol: str, duration: str = "10 Y", barSize: str = "1 day", whatToShow: str = "TRADES"):
    from ib_insync import util
    contract = _ib_stock(symbol)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=barSize,
        whatToShow=whatToShow,
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    out = pd.DataFrame({
        "date": df["date"],
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    }).drop_duplicates(subset=["date"]).sort_values("date")
    return out

def _parquet_path(root_dir: Path, ticker: str) -> Path:
    return root_dir / f"{ticker}.parquet"

def _write_parquet_merge(fp: Path, new_df: pd.DataFrame, overwrite: bool = False) -> None:
    """
    Fusionne en garantissant un schéma commun pour éviter les erreurs de largeur:
    - colonnes cibles: date, open, high, low, close, volume, adj_close, is_ex_div
    - si adj_close / is_ex_div absents dans new -> on les crée à NULL
    - si adj_close / is_ex_div absents dans old -> idem
    """
    fp.parent.mkdir(parents=True, exist_ok=True)

    target_cols = ["date","open","high","low","close","volume","adj_close","is_ex_div"]

    def _as_pl(df: pd.DataFrame) -> pl.DataFrame:
        # force types canoniques
        base = pl.from_pandas(df).with_columns(pl.col("date").cast(pl.Date))
        # ajoute colonnes manquantes avec NULL au bon type
        add_exprs = []
        if "adj_close" not in base.columns:
            add_exprs.append(pl.lit(None, dtype=pl.Float64).alias("adj_close"))
        if "is_ex_div" not in base.columns:
            add_exprs.append(pl.lit(None, dtype=pl.Boolean).alias("is_ex_div"))
        if add_exprs:
            base = base.with_columns(add_exprs)
        # si d’autres colonnes manquent (ex: open/high/low/volume), on les crée à NULL float
        for c in ["open","high","low","close","volume"]:
            if c not in base.columns:
                base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        # veille à ne garder que les colonnes cibles dans l’ordre
        return base.select([c for c in target_cols if c in base.columns])

    new_pl = _as_pl(new_df)

    if overwrite or (not fp.exists()):
        new_pl.write_parquet(fp)
        return

    old = pl.read_parquet(fp)
    if "date" not in old.columns:
        raise ValueError(f"{fp} invalide: colonne 'date' absente")
    old = old.with_columns(pl.col("date").cast(pl.Date))
    for c in ["open","high","low","close","volume"]:
        if c not in old.columns:
            old = old.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    if "adj_close" not in old.columns:
        old = old.with_columns(pl.lit(None, dtype=pl.Float64).alias("adj_close"))
    if "is_ex_div" not in old.columns:
        old = old.with_columns(pl.lit(None, dtype=pl.Boolean).alias("is_ex_div"))
    old = old.select([c for c in target_cols if c in old.columns])

    merged = (
        pl.concat([old, new_pl], how="vertical")
        .sort("date")
        .unique(subset=["date"], keep="last")
    )
    merged.write_parquet(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", default="10 Y")
    ap.add_argument("--barSize", default="1 day")
    ap.add_argument("--what", default="TRADES")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientId", type=int, default=17)
    ap.add_argument("--throttleSec", type=float, default=0.8)
    args = ap.parse_args()

    params = load_params()
    mode = params.get("env", {}).get("mode", "dev")
    assert mode == "prod", "Passe env.mode=prod pour ingérer via IBKR."
    data = params["data"]
    root_dir = Path(data.get("root_dir_prod", data.get("root_dir", "data/eod/ETFs")))
    root_dir.mkdir(parents=True, exist_ok=True)

    try:
        enforce_provenance(root_dir, expected_source="ibkr", allow_unknown=True)
    except Exception as e:
        raise SystemExit(f"[SECURITY] {e}")

    tickers = load_universe()
    ib = _connect_ib(host=args.host, port=args.port, client_id=args.clientId)
    print(f"[IBKR] Connecté: {ib.isConnected()} — Ingestion {len(tickers)} tickers vers {root_dir}")

    ok = 0
    for i, t in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] {t} …")
            df = _fetch_daily_history(ib, t, duration=args.duration, barSize=args.barSize, whatToShow=args.what)
            if df.empty:
                print(f"   [WARN] Pas de données pour {t}")
            else:
                _write_parquet_merge(_parquet_path(root_dir, t), df, overwrite=args.overwrite)
                ok += 1
        except Exception as e:
            print(f"   [ERR] {t}: {e}")
        time.sleep(args.throttleSec)

    ib.disconnect()
    save_provenance(root_dir, "ibkr")
    print(f"[IBKR] Terminé. Tickers OK: {ok}/{len(tickers)}. Provenance='ibkr' écrite.")

if __name__ == "__main__":
    main()
