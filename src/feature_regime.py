from __future__ import annotations
import numpy as np
import pandas as pd
def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    s = series.dropna().values
    if len(s) < max_lag*2: return np.nan
    lags = np.arange(min_lag, max_lag+1)
    tau = [np.sqrt(np.std(s[lag:] - s[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    H = poly[0]
    return float(H)
def variance_ratio(series: pd.Series, q: int = 5) -> float:
    s = series.dropna().pct_change().dropna()
    if len(s) < q*2: return np.nan
    var1 = np.var(s, ddof=1)
    summed = s.rolling(q).sum().dropna()
    varq = np.var(summed, ddof=1) / q
    if varq == 0: return np.nan
    return float(var1 / varq)
def regime_is_mr(y: pd.Series, lookback: int, hurst_max: float, vr_max: float) -> bool:
    sub = y.dropna().iloc[-lookback:]
    if len(sub) < lookback: return False
    H = hurst_exponent(sub)
    VR = variance_ratio(sub, q=5)
    okH = (not np.isnan(H)) and (H < hurst_max)
    okV = (not np.isnan(VR)) and (VR <= vr_max)
    return bool(okH and okV)
