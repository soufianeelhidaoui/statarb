from __future__ import annotations
import numpy as np
import pandas as pd
from .filters.stat_filters import hedge_ratio, coint_adf, z_window_by_half_life, zscore
from .filters.data_filters import liquidity_filter
from .filters.market_filters import vix_ok, macro_ok

def _apply_ex_div_mask(df: pd.DataFrame, days_after: int = 1) -> pd.Series:
    if "is_ex_div" not in df.columns:
        return pd.Series(False, index=df.index)
    m = df["is_ex_div"].astype(bool).copy()
    for k in range(1, int(days_after)):
        m |= df["is_ex_div"].shift(k, fill_value=False)
    return m

def _ar1_half_life(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 40:
        return np.inf
    x_lag = x.shift(1).dropna()
    y = x.loc[x_lag.index]
    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    phi = float(b[1])
    if phi <= -0.999 or phi >= 0.999:
        return np.inf
    hl = -np.log(2.0) / np.log(abs(phi))
    if not np.isfinite(hl) or hl <= 0:
        return np.inf
    return float(hl)

def _stable_half_life_on_spread(spread: pd.Series, lookback_days: int, subwindows: int, hl_min: float, hl_max: float, tol_frac: float) -> tuple[bool, float]:
    s = pd.Series(spread).dropna()
    if len(s) < max(lookback_days, 60):
        return False, np.inf
    s = s.iloc[-lookback_days:]
    n = len(s)
    k = int(max(1, subwindows))
    win = n // k
    hls = []
    for i in range(k):
        part = s.iloc[i*win:(i+1)*win] if i < k-1 else s.iloc[i*win:]
        if len(part) < 40:
            continue
        hls.append(_ar1_half_life(part))
    if not hls:
        return False, np.inf
    hls = np.array(hls, dtype=float)
    ok_slice = (hls >= hl_min) & (hls <= hl_max) & np.isfinite(hls)
    pass_ratio = ok_slice.mean()
    ok = bool(pass_ratio >= float(tol_frac))
    hl_med = float(np.nanmedian(hls)) if np.isfinite(np.nanmedian(hls)) else np.inf
    return ok, hl_med

def _crossing_ok(prev_z: float, z: float, entry_z: float) -> bool:
    if not np.isfinite(prev_z) or not np.isfinite(z):
        return False
    if (prev_z < entry_z) and (z >= entry_z):
        return True
    if (prev_z > -entry_z) and (z <= -entry_z):
        return True
    return False

def _slope_ok(z_series: pd.Series, lookback: int = 3, direction: int = 0) -> bool:
    z_series = pd.Series(z_series).dropna()
    if len(z_series) < max(lookback, 3):
        return False
    tail = z_series.iloc[-lookback:]
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail.values, 1)[0]
    if direction == 0:
        direction = 1 if tail.iloc[-1] >= 0 else -1
    return (slope * direction) > 0.0

def decide_pair(ya: pd.Series, xb: pd.Series, spy: pd.Series | None, params: dict, meta_a: dict | None = None, meta_b: dict | None = None) -> dict:
    a = ya.name
    b = xb.name
    idx = ya.dropna().index.intersection(xb.dropna().index)
    if len(idx) < 180:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Insufficient overlap","z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
    qa = params.get("quality", {})
    sf = params.get("stats_filters", {})
    mf = params.get("market_filters", {})
    lb = params.get("lookbacks", {})
    zmin = int(lb.get("zscore_days_min", 30))
    logic = params.get("decision", {})
    require_cross = bool(logic.get("entry_require_cross", True))
    slope_confirm = bool(logic.get("entry_slope_confirm", True))

    if mf.get("enable", True):
        if not vix_ok(mf.get("vix_path",""), mf.get("vix_max", 1000)):
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"High VIX regime","z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
        now_utc = pd.Timestamp.utcnow()
        if not macro_ok(mf.get("macro_calendar_csv",""), now_utc, mf.get("cool_off_hours", 0)):
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Macro event day","z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
    if not liquidity_filter(meta_a.get("df", pd.DataFrame()), meta_b.get("df", pd.DataFrame()), qa.get("min_volume",0)):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Low liquidity","z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
    if qa.get("mask_ex_div", True):
        mA = _apply_ex_div_mask(meta_a.get("df", pd.DataFrame()), qa.get("mask_ex_div_days_after",1))
        mB = _apply_ex_div_mask(meta_b.get("df", pd.DataFrame()), qa.get("mask_ex_div_days_after",1))
        m = (mA.reindex(index=idx, fill_value=False)) | (mB.reindex(index=idx, fill_value=False))
    else:
        m = pd.Series(False, index=idx)
    y = ya.loc[idx]
    x = xb.loc[idx]
    y2 = y.loc[~m]
    x2 = x.loc[~m]
    if len(y2) < 160:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Masked overlap too small","z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
    alpha, beta = hedge_ratio(y2, x2)
    spread = y2 - (alpha + beta * x2)
    ok_hl, hl = _stable_half_life_on_spread(
        spread=spread,
        lookback_days=int(params.get("stability", {}).get("lookback_days", 120)),
        subwindows=int(params.get("stability", {}).get("subwindows", 3)),
        hl_min=float(sf.get("half_life_min_days", 2)),
        hl_max=float(sf.get("half_life_max_days", 20)),
        tol_frac=float(params.get("stability", {}).get("min_pass_ratio", 0.67)),
    )
    if not ok_hl:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Half-life unstable/out-of-range","z_last":np.nan,"hl":hl,"beta":beta,"pval":1.0}
    pval = coint_adf(spread)
    if sf.get("require_coint", True) and (pval > sf.get("coint_pval_max", 0.05)):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"No cointegration","z_last":np.nan,"hl":hl,"beta":beta,"pval":pval}
    zwin = z_window_by_half_life(hl, zmin, float(params.get("lookbacks", {}).get("zscore_mult_half_life", 3.0)))
    z = zscore(spread, zwin)
    if len(z) < 5:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Too few z-bars","z_last":np.nan,"hl":hl,"beta":beta,"pval":pval}
    z_last = float(z.iloc[-1])
    z_prev = float(z.iloc[-2])
    if abs(z_last) > float(sf.get("z_cap", 5.0)):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Extreme z outlier","z_last":z_last,"hl":hl,"beta":beta,"pval":pval}
    entry_z = float(params["thresholds"]["entry_z"])
    exit_z = float(params["thresholds"]["exit_z"])
    stop_z = float(params["thresholds"]["stop_z"])
    
    if require_cross:
        enter = _crossing_ok(z_prev, z_last, entry_z)
    else:
        enter = abs(z_last) >= entry_z
    
    if enter:
        direction = 1 if z_last >= 0 else -1
        
        if slope_confirm and not _slope_ok(z.tail(5), lookback=3, direction=direction):
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Slope check failed","z_last":z_last,"hl":hl,"beta":beta,"pval":pval}
        
        if z_last >= entry_z:
            verdict, action, reason = "ENTER", "ShortY_LongX", "Crossing up" if require_cross else "Above threshold"
        else:
            verdict, action, reason = "ENTER", "LongY_ShortX", "Crossing down" if require_cross else "Below threshold"
    else:
        verdict, action, reason = "HOLD", "None", "No crossing" if require_cross else "No threshold breach"
    return {"a":a,"b":b,"verdict":verdict,"action":action,"reason":reason,"z_last":z_last,"hl":hl,"beta":beta,"pval":pval}
