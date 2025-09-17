from __future__ import annotations
import numpy as np
import pandas as pd
from .stats import adf_pvalue, half_life_of_mean_reversion, ols_hedge_ratio, spread_series
def rolling_coint_stability(y: pd.Series, x: pd.Series, subwindows: int, lookback_days: int,
                            adf_thr: float, hl_max: float) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < subwindows * lookback_days: 
        return {"pass_ratio": 0.0, "ok": False, "pvals": [], "hls": []}
    pvals, hls = [], []
    for i in range(subwindows):
        start = max(0, len(df) - (i+1)*lookback_days)
        end   = len(df) - i*lookback_days
        win = df.iloc[start:end]
        if len(win) < lookback_days: 
            pvals.append(1.0); hls.append(np.inf); continue
        a, b = ols_hedge_ratio(win.iloc[:,0], win.iloc[:,1])
        spr = spread_series(win.iloc[:,0], win.iloc[:,1], a, b)
        p = adf_pvalue(spr)
        hl = half_life_of_mean_reversion(spr)
        pvals.append(1.0 if p is None else p)
        hls.append(np.inf if np.isnan(hl) else hl)
    pvals = pvals[::-1]; hls = hls[::-1]
    checks = [(pvals[i] <= adf_thr) and (hls[i] <= hl_max) for i in range(subwindows)]
    pass_ratio = sum(checks) / subwindows
    return {"pass_ratio": pass_ratio, "ok": pass_ratio >= 2/3, "pvals": pvals, "hls": hls}
