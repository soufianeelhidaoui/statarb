from __future__ import annotations
import polars as pl
import pandas as pd
import numpy as np
from .filters.stat_filters import slope_direction_ok

def merge_close_series(a: pl.DataFrame, b: pl.DataFrame) -> pd.DataFrame:
    da = a.select(["date", "close"]).to_pandas().set_index("date")
    db = b.select(["date", "close"]).to_pandas().set_index("date")
    df = da.join(db, how="inner", lsuffix="_a", rsuffix="_b")
    df.columns = ["ya", "xb"]
    df = df.reset_index().rename(columns={"index": "date"})
    return df

def _ols_beta(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    X = np.vstack([np.ones(len(x)), x.values]).T
    alpha, beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    return float(alpha), float(beta)

def simulate_pair(
    df_merged: "pl.DataFrame | pd.DataFrame",
    entry_z: float,
    exit_z: float,
    stop_z: float,
    z_window: int,
    risk_pct: float,
    capital: float = 100_000.0,
    costs_bp: int = 2,
    cool_off_bars: int = 0,
    min_bars_between_entries: int = 0,
    notional_per_trade: float = 0.0,
    require_cross: bool = True,
    slope_confirm: bool = True,
    slope_lookback: int = 3,
):
    if isinstance(df_merged, pl.DataFrame):
        cols = [c for c in ["date", "ya", "xb", "is_ex_div"] if c in df_merged.columns]
        df = df_merged.select(cols).to_pandas()
    else:
        cols = [c for c in ["date", "ya", "xb", "is_ex_div"] if c in df_merged.columns]
        df = df_merged[cols].copy()

    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    if "is_ex_div" in df.columns:
        ma = df["is_ex_div"].astype(bool)
        mask = ~(ma.fillna(False) | ma.shift(1).fillna(False))
        df = df.loc[mask].copy()

    if len(df) < max(30, z_window + 1):
        j = pd.DataFrame(
            {"date": df["date"], "z": np.nan, "pos": 0, "signal": 0, "step_pnl": 0.0, "cum_pnl": 0.0}
        ).set_index("date")
        return 0.0, j

    alpha, beta = _ols_beta(df["ya"], df["xb"])
    spread = df["ya"] - (alpha + beta * df["xb"])
    m = spread.rolling(z_window).mean()
    s = spread.rolling(z_window).std(ddof=1)
    z = (spread - m) / s.replace(0.0, np.nan)
    df["z"] = z

    pos = 0
    last_entry = -10**9
    cool_until = -10**9
    z_prev = np.nan

    N = float(notional_per_trade) if notional_per_trade > 0 else float(capital) * float(risk_pct)
    cost_leg = abs(N) * (costs_bp / 10_000.0)

    hold_a = 0.0
    hold_b = 0.0
    pnl = 0.0
    step = []

    ya_prev = float(df["ya"].iloc[0])
    xb_prev = float(df["xb"].iloc[0])

    for i, (dt, zi, ya_i, xb_i) in enumerate(zip(df["date"].values, df["z"].values, df["ya"].values, df["xb"].values)):
        sig = 0
        can_enter = (i >= cool_until) and (i - last_entry >= int(min_bars_between_entries))
        enter_short = False
        enter_long = False

        if pd.notna(zi) and can_enter and pos == 0:
            if require_cross:
                if pd.notna(z_prev):
                    if (z_prev > entry_z) and (zi <= entry_z):
                        if not slope_confirm or slope_direction_ok(df["z"].iloc[:i+1], slope_lookback, -1):
                            enter_short = True
                    if (z_prev < -entry_z) and (zi >= -entry_z):
                        if not slope_confirm or slope_direction_ok(df["z"].iloc[:i+1], slope_lookback, 1):
                            enter_long = True
            else:
                if zi >= entry_z:
                    if not slope_confirm or slope_direction_ok(df["z"].iloc[:i+1], slope_lookback, -1):
                        enter_short = True
                elif zi <= -entry_z:
                    if not slope_confirm or slope_direction_ok(df["z"].iloc[:i+1], slope_lookback, 1):
                        enter_long = True

        if enter_short:
            pos = -1
            last_entry = i
            qa = (N / 2.0) / max(ya_i, 1e-9)
            qb = (N / 2.0) / max(xb_i, 1e-9) * beta
            hold_a = -qa
            hold_b = +qb
            pnl -= 2.0 * cost_leg
            sig = -1

        elif enter_long:
            pos = +1
            last_entry = i
            qa = (N / 2.0) / max(ya_i, 1e-9)
            qb = (N / 2.0) / max(xb_i, 1e-9) * beta
            hold_a = +qa
            hold_b = -qb
            pnl -= 2.0 * cost_leg
            sig = +1

        exit_now = False
        if pos != 0 and pd.notna(zi):
            if (abs(zi) <= float(exit_z)) or (abs(zi) >= float(stop_z)):
                exit_now = True

        dya = float(ya_i) - ya_prev
        dxb = float(xb_i) - xb_prev
        ya_prev = float(ya_i)
        xb_prev = float(xb_i)

        pnl_step = hold_a * dya + hold_b * dxb
        pnl += pnl_step
        step.append(pnl_step)

        if exit_now and pos != 0:
            pos = 0
            hold_a = 0.0
            hold_b = 0.0
            pnl -= 2.0 * cost_leg
            cool_until = i + int(cool_off_bars)

        df.loc[df.index[i], "pos"] = pos
        df.loc[df.index[i], "signal"] = sig
        df.loc[df.index[i], "step_pnl"] = pnl_step
        df.loc[df.index[i], "cum_pnl"] = pnl

        z_prev = zi

    j = df[["date", "z", "pos", "signal", "step_pnl", "cum_pnl"]].copy().set_index("date")
    return float(j["cum_pnl"].iloc[-1]), j
