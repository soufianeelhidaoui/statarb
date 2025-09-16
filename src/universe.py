from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import load_params

def load_universe() -> list[str]:
    params = load_params()
    ticks = params['universe']['tickers']
    return list(dict.fromkeys(ticks))

def load_universe_from_csv(path: str | Path = 'data/metadata/universe.csv') -> list[str]:
    df = pd.read_csv(path)
    return df['symbol'].dropna().astype(str).unique().tolist()
