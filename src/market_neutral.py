from __future__ import annotations
import numpy as np
import pandas as pd
def residualize_vs_market(y: pd.Series, mkt: pd.Series, win: int = 60) -> pd.Series:
    df = pd.concat([y, mkt], axis=1).dropna()
    if len(df) < win: 
        return y.reindex_like(df.iloc[:,0])
    yv, mv = df.iloc[:,0], df.iloc[:,1]
    cov = yv.rolling(win).cov(mv)
    var = mv.rolling(win).var().replace(0, np.nan)
    beta_m = cov / var
    alpha = yv.rolling(win).mean() - beta_m * mv.rolling(win).mean()
    res = yv - (alpha + beta_m * mv)
    res.name = (y.name or "y") + "_res"
    return res
