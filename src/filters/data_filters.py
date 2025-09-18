from __future__ import annotations
import pandas as pd

def liquidity_filter(df_a: pd.DataFrame, df_b: pd.DataFrame, min_volume: int = 0) -> bool:
    if min_volume <= 0:
        return True
    try:
        va = int(df_a['volume'].iloc[-1])
        vb = int(df_b['volume'].iloc[-1])
        return (va >= min_volume) and (vb >= min_volume)
    except Exception:
        return True

def bidask_spread_filter(spread_a_bp: float | None, spread_b_bp: float | None, max_bp: int = 9999) -> bool:
    if max_bp is None:
        return True
    ok_a = (spread_a_bp is None) or (spread_a_bp <= max_bp)
    ok_b = (spread_b_bp is None) or (spread_b_bp <= max_bp)
    return ok_a and ok_b
