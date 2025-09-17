from __future__ import annotations
import numpy as np
import pandas as pd
def rolling_beta(y: pd.Series, x: pd.Series, win: int = 60) -> pd.Series:
    s = pd.concat([y, x], axis=1).dropna()
    yv, xv = s.iloc[:,0], s.iloc[:,1]
    cov = yv.rolling(win).cov(xv)
    var = xv.rolling(win).var()
    beta = cov / var
    return beta.rename("beta")
def dynamic_spread(y: pd.Series, x: pd.Series, beta: pd.Series, alpha: float = 0.0) -> pd.Series:
    df = pd.concat([y, x, beta], axis=1).dropna()
    spr = df.iloc[:,0] - (alpha + df["beta"] * df.iloc[:,1])
    spr.name = "spread"
    return spr
def beta_instability(beta: pd.Series, subwindows: int = 3) -> float:
    b = beta.dropna()
    if len(b) < subwindows*10: return np.inf
    tail = b.iloc[-subwindows*30:] if len(b) >= subwindows*30 else b
    import numpy as np
    chunks = np.array_split(tail.values, subwindows) if len(tail) >= subwindows else [tail.values]
    meds = [np.nanmedian(c) for c in chunks if len(c)>0]
    if len(meds) < 2: return np.inf
    return float(np.max(meds) - np.min(meds)) / (1e-9 + np.mean(np.abs(meds)))
