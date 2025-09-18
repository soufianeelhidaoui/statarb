from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float,float]:
    x_ = sm.add_constant(x.values, has_constant='add')
    model = sm.OLS(y.values, x_).fit()
    alpha = float(model.params[0]); beta = float(model.params[1])
    return alpha, beta

def half_life(spread: pd.Series) -> float:
    y = spread.shift(1).dropna()
    x = spread.dropna().loc[y.index]
    x_ = sm.add_constant(y.values, has_constant='add')
    model = sm.OLS(x.values, x_).fit()
    phi = float(model.params[1])
    if phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2) / np.log(phi))

def coint_adf(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 30:
        return 1.0
    try:
        res = adfuller(spread.values, autolag='AIC')
        return float(res[1])
    except Exception:
        return 1.0

def zscore(series: pd.Series, win: int) -> pd.Series:
    r = series.rolling(win)
    return (series - r.mean()) / r.std(ddof=0)

def stable_half_life(y: pd.Series, x: pd.Series, hl_min: float, hl_max: float, tol: float) -> tuple[bool, float]:
    alpha, beta = hedge_ratio(y, x)
    spread = y - (alpha + beta * x)
    hl = half_life(spread)
    if not np.isfinite(hl):
        return False, np.inf
    if (hl < hl_min) or (hl > hl_max):
        return False, hl
    n = len(spread.dropna())
    if n < 80:
        return True, hl
    sub1 = spread.iloc[int(n*0.25):]
    sub2 = spread.iloc[int(n*0.5):]
    def stable(h): return np.isfinite(h) and abs(h - hl)/max(hl,1e-6) <= tol
    return (stable(half_life(sub1)) and stable(half_life(sub2))), hl

def beta_stable(y: pd.Series, x: pd.Series, tol: float) -> tuple[bool, float]:
    alpha, beta = hedge_ratio(y, x)
    n = len(y.dropna().index.intersection(x.dropna().index))
    if n < 80:
        return True, beta
    mid = int(n*0.5)
    a1,b1 = hedge_ratio(y.iloc[:mid], x.iloc[:mid])
    a2,b2 = hedge_ratio(y.iloc[mid:], x.iloc[mid:])
    ok = abs(b1 - b2)/max(abs(beta),1e-6) <= tol
    return ok, beta

def z_window_by_half_life(hl: float, zmin: int, mult: float) -> int:
    return int(max(zmin, np.ceil(mult * max(hl,1.0))))
