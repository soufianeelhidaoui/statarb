from __future__ import annotations
import numpy as np
import pandas as pd

def compute_corr(a: pd.Series, b: pd.Series, lookback: int) -> float:
    s = pd.concat([a, b], axis=1).dropna()
    if len(s) < lookback:
        return np.nan
    s = s.iloc[-lookback:]
    return float(s.corr(method='pearson').iloc[0,1])

def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    s = pd.concat([y, x], axis=1).dropna()
    if len(s) < 5:
        return 0.0, 1.0
    X = np.column_stack([np.ones(len(s)), s.iloc[:,1].values])
    yv = s.iloc[:,0].values
    beta_hat = np.linalg.lstsq(X, yv, rcond=None)[0]
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    return alpha, beta

def spread_series(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    s = pd.concat([y, x], axis=1).dropna()
    if s.empty:
        return s.iloc[:,0]
    spr = s.iloc[:,0] - (alpha + beta*s.iloc[:,1])
    spr.name = 'spread'
    return spr

def half_life_of_mean_reversion(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 20:
        return np.nan
    s_lag = s.shift(1).dropna()
    ds = (s - s_lag).dropna()
    X = s_lag.loc[ds.index].values.reshape(-1,1)
    y = ds.values
    phi = np.linalg.lstsq(X, y, rcond=None)[0][0]
    if 1 + phi <= 0:
        return np.nan
    hl = -np.log(2) / np.log(1 + phi)
    return float(abs(hl))

def adf_pvalue(series: pd.Series) -> float | None:
    try:
        from statsmodels.tsa.stattools import adfuller
        s = series.dropna().values
        if len(s) < 20:
            return None
        res = adfuller(s, maxlag=1, autolag='AIC')
        return float(res[1])
    except Exception:
        return None

def compute_coint_stats(y: pd.Series, x: pd.Series, lookback: int) -> tuple[float, float, float, float, float]:
    yx = pd.concat([y, x], axis=1).dropna()
    if len(yx) < lookback:
        return (np.nan, np.nan, np.nan, 0.0, 1.0)
    yx = yx.iloc[-lookback:]
    alpha, beta = ols_hedge_ratio(yx.iloc[:,0], yx.iloc[:,1])
    spr = spread_series(yx.iloc[:,0], yx.iloc[:,1], alpha, beta)
    sigma = float(spr.std(ddof=1))
    hl = half_life_of_mean_reversion(spr)
    pval = adf_pvalue(spr)
    if pval is None or np.isnan(pval):
        if np.isnan(hl):
            pval = 1.0
        elif hl <= 5:
            pval = 0.02
        elif hl <= 10:
            pval = 0.05
        elif hl <= 20:
            pval = 0.1
        else:
            pval = 0.2
    return (float(pval), float(hl) if not np.isnan(hl) else np.inf, sigma, alpha, beta)

def zscore(series: pd.Series, win: int) -> pd.Series:
    s = series.dropna()
    if len(s) < win:
        return pd.Series(index=s.index, dtype=float, name='z')
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=1)
    z = (s - mu) / sd
    z.name = 'z'
    return z

def combine_score(rho: float, pval: float, half_life: float, sigma_spread: float) -> float:
    if any([np.isnan(rho), np.isnan(pval), np.isnan(half_life), np.isnan(sigma_spread)]):
        return -np.inf
    s1 = max(0.0, rho)
    s2 = -np.log(max(pval, 1e-6))
    s3 = 1.0 / (1.0 + half_life)
    s4 = 1.0 / (1.0 + sigma_spread)
    return float(2.0*s1 + 1.5*s2 + 1.0*s3 + 0.5*s4)

def _beta_ols(y, x):
    x_ = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(x_, y.values, rcond=None)[0]
    alpha, beta = float(b[0]), float(b[1])
    return alpha, beta

def _halflife_ar1(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return np.nan
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    X = s_lag.values.reshape(-1, 1)
    Y = ds.values
    phi = np.linalg.lstsq(np.hstack([np.ones_like(X), X]), Y, rcond=None)[0][1]
    if phi <= -0.999 or phi >= 0.999:
        return np.nan
    hl = -np.log(2) / np.log(1 + phi)
    return float(hl) if hl > 0 else np.nan

def rolling_zscore_spread(spread: pd.Series, window: int) -> pd.Series:
    return zscore(spread, window)