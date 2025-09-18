#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import yfinance as yf

def load_params(path="config/params.yaml"):
    import yaml
    with open(path,"r") as f:
        return yaml.safe_load(f)

# en tête du fichier
import pandas as pd
import polars as pl
from pathlib import Path
import yfinance as yf
import json
from datetime import datetime

def fetch_ticker(t: str) -> pd.DataFrame:
    df = yf.download(t, period="max", interval="1d",
                     auto_adjust=False, actions=True,
                     group_by="column", progress=False, threads=True)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo empty for {t}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    def _s(name):
        s = df[name]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        return pd.to_numeric(s, errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None)

    out = pd.DataFrame({
        "date": df.index.date,
        "open": _s("Open"),
        "high": _s("High"),
        "low":  _s("Low"),
        "close": _s("Close"),
        "adj_close": _s("Adj Close"),
        "volume": _s("Volume"),
    })

    # Nouveau: flag ex-div
    div = df.get("Dividends")
    if div is not None:
        is_ex = div.fillna(0).astype(float) > 0
        out["is_ex_div"] = is_ex.values
    else:
        out["is_ex_div"] = False

    return out.dropna(subset=["close"])



def main():
    from src.config import load_params
    from src.universe import load_universe

    params = load_params()
    tickers = load_universe()

    data = params.get("data", {})
    root = Path(data.get("root_dir_dev", data.get("root_dir", "data/eod/ETFs_dev")))
    root.mkdir(parents=True, exist_ok=True)

    ok = 0
    for t in tickers:
        print(f"[Yahoo] {t}…")
        try:
            pdf = fetch_ticker(t)
            pl.from_pandas(pdf).write_parquet(root / f"{t}.parquet")   # overwrite
            ok += 1
        except Exception as e:
            print(f"   [ERR] {t}: {e}")

    (root / "_PROVENANCE.json").write_text(json.dumps({
        "source": "yahoo",
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }, indent=2))
    print(f"[Yahoo] Terminé. Tickers OK: {ok}/{len(tickers)} → {root}")


if __name__ == "__main__":
    main()
