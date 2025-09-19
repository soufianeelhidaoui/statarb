from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import polars as pl

def merge_close_series(a_pl: pl.DataFrame, b_pl: pl.DataFrame) -> pd.DataFrame:
    da = a_pl.select(['date','close']).to_pandas()
    db = b_pl.select(['date','close']).to_pandas()
    da["date"] = pd.to_datetime(da["date"]); db["date"] = pd.to_datetime(db["date"])
    j = pd.merge(da, db, on="date", how="inner", suffixes=("_a","_b")).sort_values("date")
    j = j.rename(columns={"close_a":"ya","close_b":"xb"})
    return j[["date","ya","xb"]]

def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=window).mean()
    s = x.rolling(window, min_periods=window).std(ddof=1)
    z = (x - m) / s.replace(0.0, np.nan)
    return z

def simulate_pair(
    dfpl_merged,                   # pl.DataFrame|pd.DataFrame avec colonnes ['date','ya','xb', ('is_ex_div' opt.)]
    entry_z: float,
    exit_z: float,
    stop_z: float,
    z_window: int,
    risk_pct: float | None = None,
    capital: float = 100_000.0,
    costs_bp: int = 2,
    cool_off_bars: int = 0,
    min_bars_between_entries: int = 0,
    notional_per_trade: float | None = None,   # <-- nouveau
):
    import numpy as np
    import pandas as pd
    import polars as pl

    if isinstance(dfpl_merged, pl.DataFrame):
        cols = ["date","ya","xb"]
        if "is_ex_div" in dfpl_merged.columns:
            cols.append("is_ex_div")
        df = dfpl_merged.select(cols).to_pandas()
    else:
        cols = [c for c in ["date","ya","xb","is_ex_div"] if c in dfpl_merged.columns]
        df = dfpl_merged[cols].copy()

    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    if "is_ex_div" in df.columns:
        ma = df["is_ex_div"].astype(bool)
        mb = df["is_ex_div"].astype(bool)  # même colonne côté B si seule dispo
        mask = ~(ma.fillna(False).astype(bool) | ma.shift(1).fillna(False).astype(bool) |
                 mb.fillna(False).astype(bool) | mb.shift(1).fillna(False).astype(bool))
        df = df.loc[mask].copy()

    if len(df) < max(30, z_window + 1):
        out = df.copy()
        out["z"] = np.nan; out["pos"] = 0; out["signal"] = 0
        out["step_pnl"] = 0.0; out["cum_pnl"] = 0.0
        return 0.0, out.set_index("date")[["z","pos","signal","step_pnl","cum_pnl"]]

    X = np.vstack([np.ones(len(df)), df["xb"].values]).T
    b = np.linalg.lstsq(X, df["ya"].values, rcond=None)[0]
    alpha, beta = float(b[0]), float(b[1])
    spread = df["ya"] - (alpha + beta * df["xb"])

    roll_mean = spread.rolling(z_window).mean()
    roll_std  = spread.rolling(z_window).std(ddof=1)
    z = (spread - roll_mean) / roll_std
    df["z"] = z

    pos = 0
    last_entry_idx = -10**9
    cool_until_idx = -10**9
    pnl = 0.0
    step = []

    if (notional_per_trade is not None) and (notional_per_trade > 0):
        notional = float(notional_per_trade)
    else:
        notional = float(capital) * float(risk_pct or 0.0)

    cost_per_leg = abs(notional) * (costs_bp / 10_000.0)
    prev_spread = float(spread.iloc[0])

    # échelle pnL: lisse la std pour éviter NaN/0
    scale = roll_std.copy()
    if scale.isna().any():
        seed = scale.iloc[z_window:z_window+5].mean()
        scale = scale.fillna(seed)
        scale = scale.replace({0.0: np.nan}).bfill().ffill()
    scale = scale.replace({0.0: np.nan}).fillna(scale.mean())

    df["pos"] = 0; df["signal"] = 0; df["step_pnl"] = 0.0; df["cum_pnl"] = 0.0

    for i, (dt, zi, sp_i) in enumerate(zip(df["date"].values, df["z"].values, spread.values)):
        sig = 0
        can_enter = (i >= cool_until_idx) and (i - last_entry_idx >= int(min_bars_between_entries))

        if pos == 0 and can_enter and pd.notna(zi):
            if zi >= float(entry_z):
                pos = -1; sig = -1; pnl -= cost_per_leg; last_entry_idx = i
            elif zi <= -float(entry_z):
                pos = +1; sig = +1; pnl -= cost_per_leg; last_entry_idx = i

        if pos != 0 and pd.notna(zi):
            if (abs(zi) <= float(exit_z)) or (abs(zi) >= float(stop_z)):
                pos = 0; sig = 0; pnl -= cost_per_leg; cool_until_idx = i + int(cool_off_bars)

        d_spread = sp_i - prev_spread
        prev_spread = sp_i
        denom = float(scale.iloc[i]) if np.isfinite(scale.iloc[i]) else 1.0
        pnl_step = float(pos) * (d_spread / denom) * (notional * 0.01)
        pnl += pnl_step; step.append(pnl_step)

        df.loc[df.index[i], "pos"] = float(pos)
        df.loc[df.index[i], "signal"] = float(sig)
        df.loc[df.index[i], "step_pnl"] = float(pnl_step)
        df.loc[df.index[i], "cum_pnl"] = float(pnl)

    journal = df[["date","z","pos","signal","step_pnl","cum_pnl"]].set_index("date")
    return float(journal["cum_pnl"].iloc[-1]), journal

