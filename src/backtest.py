from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple
from .stats import ols_hedge_ratio, spread_series, zscore
from .risk import position_size

def merge_close_series(a: pl.DataFrame, b: pl.DataFrame) -> pd.DataFrame:
    da = a.select(['date','close']).to_pandas().set_index('date')
    db = b.select(['date','close']).to_pandas().set_index('date')
    df = da.join(db, how='inner', lsuffix='_a', rsuffix='_b')
    df.columns = ['close_a','close_b']
    return df

def simulate_pair(df: pd.DataFrame, entry_z: float, exit_z: float, stop_z: float, z_win: int,
                  per_trade_pct: float, capital: float, costs_bp: float = 2.0) -> Tuple[float, pd.DataFrame]:
    alpha, beta = ols_hedge_ratio(df['close_a'], df['close_b'])
    spr = spread_series(df['close_a'], df['close_b'], alpha, beta)
    z = zscore(spr, z_win).reindex(df.index)
    signal = []
    state = 0
    for val in z:
        if np.isnan(val):
            signal.append(state); continue
        if abs(val) >= stop_z: state = 0
        elif val >= entry_z:   state = -1
        elif val <= -entry_z:  state = 1
        elif abs(val) <= exit_z: state = 0
        signal.append(state)
    sig = pd.Series(signal, index=df.index, name='signal')
    n_y, n_x = position_size(capital, per_trade_pct, df['close_a'].iloc[-1], df['close_b'].iloc[-1], beta)
    if n_y == 0 or n_x == 0:
        return 0.0, pd.DataFrame()

    pnl = [0.0]
    prev_price_y, prev_price_x = df['close_a'].iloc[0], df['close_b'].iloc[0]
    prev_sig = 0
    cost_mult = costs_bp / 10000.0
    rows = []
    for t, (py, px, s) in enumerate(zip(df['close_a'], df['close_b'], sig)):
        if t == 0:
            rows.append((df.index[t], s, 0.0, 0.0))
            prev_price_y, prev_price_x, prev_sig = py, px, s
            continue
        dy = py - prev_price_y
        dx = px - prev_price_x
        step_pnl = n_y * dy * s - n_x * dx * s
        if s != prev_sig:
            step_pnl -= (abs(n_y)*py + abs(n_x)*px) * cost_mult * 2
        pnl.append(pnl[-1] + step_pnl)
        rows.append((df.index[t], s, step_pnl, pnl[-1]))
        prev_price_y, prev_price_x, prev_sig = py, px, s
    journal = pd.DataFrame(rows, columns=['date','signal','step_pnl','cum_pnl']).set_index('date')
    total = float(journal['cum_pnl'].iloc[-1])
    return total, journal
