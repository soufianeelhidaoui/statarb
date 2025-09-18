from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np

def merge_close_series(a: pl.DataFrame, b: pl.DataFrame) -> pd.DataFrame:
    da = a.select(['date','close']).to_pandas().set_index('date')
    db = b.select(['date','close']).to_pandas().set_index('date')
    df = da.join(db, how='inner', lsuffix='_a', rsuffix='_b')
    df.columns = ['close_a','close_b']
    # On renomme pour rester conforme à simulate_pair (ya/xb)
    df = df.rename(columns={"close_a":"ya","close_b":"xb"})
    df = df.reset_index().rename(columns={"index":"date"})
    return df

def simulate_pair(
    dfpl_merged: "pl.DataFrame|pd.DataFrame",
    entry_z: float,
    exit_z: float,
    stop_z: float,
    z_window: int,
    risk_pct: float,
    capital: float = 100_000.0,
    costs_bp: int = 2,
    cool_off_bars: int = 0,
    min_bars_between_entries: int = 0,
):
    """
    Backtest pairs trading (daily) avec anti-sur-trading:
      - cool_off_bars: N barres interdites après chaque sortie
      - min_bars_between_entries: espacement minimal entre deux entrées
    Hypothèses:
      - dfpl_merged contient au minimum colonnes ['date','ya','xb'] et (optionnel) 'is_ex_div'
    Retour:
      total_pnl (float), journal (pd.DataFrame: date, z, pos, signal, step_pnl, cum_pnl)
    """
    # --- Normalise en pandas ---
    if isinstance(dfpl_merged, pl.DataFrame):
        cols = ["date","ya","xb"] + (["is_ex_div"] if "is_ex_div" in dfpl_merged.columns else [])
        df = dfpl_merged.select(cols).to_pandas()
    else:
        cols = [c for c in ["date","ya","xb","is_ex_div"] if c in dfpl_merged.columns]
        df = dfpl_merged[cols].copy()
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    # Masque ex-div (J0/J+1)
    if "is_ex_div" in df.columns:
        mask = ~(df["is_ex_div"].fillna(False) | df["is_ex_div"].shift(1).fillna(False))
        df = df.loc[mask].copy()

    if len(df) < max(30, z_window + 1):
        journal = pd.DataFrame({
            "date": df["date"],
            "z": np.nan, "pos": 0, "signal": 0,
            "step_pnl": 0.0, "cum_pnl": 0.0,
        }).set_index("date")
        return 0.0, journal

    # Hedge ratio (alpha/beta) par OLS statique (fenêtre = toute la série backtest)
    X = np.vstack([np.ones(len(df)), df["xb"].values]).T
    alpha, beta = np.linalg.lstsq(X, df["ya"].values, rcond=None)[0]
    spread = df["ya"] - (alpha + beta * df["xb"])

    # Z-score rolling
    mean_roll = spread.rolling(z_window).mean()
    std_roll  = spread.rolling(z_window).std(ddof=1)
    z = (spread - mean_roll) / std_roll
    df["z"] = z

    # Machine à états
    pos = 0
    last_entry_idx = -10**9
    cool_until_idx = -10**9

    notional = capital * float(risk_pct)
    cost_per_leg = abs(notional) * (costs_bp / 10_000.0)

    prev_spread = float(spread.iloc[0])
    pnl = 0.0
    step_pnl = []

    # buffers
    df["pos"] = 0
    df["signal"] = 0
    df["step_pnl"] = 0.0
    df["cum_pnl"] = 0.0

    for i in range(len(df)):
        zi = df["z"].iloc[i]
        sp_i = float(spread.iloc[i])

        # entrée ?
        can_enter = (i >= cool_until_idx) and (i - last_entry_idx >= int(min_bars_between_entries))
        if pos == 0 and can_enter and pd.notna(zi):
            if zi >= float(entry_z):
                pos = -1
                df.loc[df.index[i], "signal"] = -1
                pnl -= cost_per_leg
                last_entry_idx = i
            elif zi <= -float(entry_z):
                pos = +1
                df.loc[df.index[i], "signal"] = +1
                pnl -= cost_per_leg
                last_entry_idx = i

        # sortie ?
        if pos != 0 and pd.notna(zi):
            if (abs(zi) <= float(exit_z)) or (abs(zi) >= float(stop_z)):
                pos = 0
                pnl -= cost_per_leg
                cool_until_idx = i + int(cool_off_bars)

        # PnL incrémental (approximation)
        d_spread = sp_i - prev_spread
        prev_spread = sp_i
        scale = float(std_roll.iloc[i]) if pd.notna(std_roll.iloc[i]) and std_roll.iloc[i] != 0 else 1.0
        pnl_step = float(pos) * (d_spread / scale) * (notional * 0.01)

        pnl += pnl_step
        step_pnl.append(pnl_step)

        df.loc[df.index[i], "pos"] = pos
        df.loc[df.index[i], "step_pnl"] = pnl_step
        df.loc[df.index[i], "cum_pnl"] = pnl

    journal = df[["date","z","pos","signal","step_pnl","cum_pnl"]].copy().set_index("date")
    return float(journal["cum_pnl"].iloc[-1]), journal
