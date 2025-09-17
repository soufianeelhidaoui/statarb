from __future__ import annotations
import numpy as np
import pandas as pd
from .stats import zscore, ols_hedge_ratio, spread_series, half_life_of_mean_reversion
from .hedge import rolling_beta, dynamic_spread, beta_instability
from .stability import rolling_coint_stability
from .feature_regime import regime_is_mr
from .market_neutral import residualize_vs_market
def dynamic_z_window(half_life: float, min_win: int) -> int:
    if np.isinf(half_life) or np.isnan(half_life): return max(20, min_win)
    return int(max(min_win, round(3.0 * half_life)))
def decide_pair(y_close: pd.Series, x_close: pd.Series, mkt: pd.Series | None, params: dict) -> dict:
    y_use, x_use = (y_close, x_close)
    if mkt is not None:
        y_use = residualize_vs_market(y_close, mkt, win=60)
        x_use = residualize_vs_market(x_close, mkt, win=60)
    win_beta = params.get("beta", {}).get("lookback", 60)
    beta = rolling_beta(y_use, x_use, win=win_beta)
    beta_instab = beta_instability(beta, subwindows=3)
    a0, b0 = ols_hedge_ratio(y_use, x_use)
    spr0 = spread_series(y_use, x_use, a0, b0)
    hl = half_life_of_mean_reversion(spr0)
    z_min = params.get("zscore", {}).get("min_window", 20)
    z_win = dynamic_z_window(hl, z_min)
    spr = dynamic_spread(y_use, x_use, beta, alpha=0.0)
    z = zscore(spr, z_win)
    z_last = float(z.dropna().iloc[-1]) if len(z.dropna()) else np.nan
    stab_cfg = params.get("stability", {})
    stab = rolling_coint_stability(y_use, x_use,
                                   subwindows=stab_cfg.get("subwindows",3),
                                   lookback_days=stab_cfg.get("lookback_days",120),
                                   adf_thr=stab_cfg.get("adf_threshold",0.05),
                                   hl_max=stab_cfg.get("half_life_max",20))
    reg_cfg = params.get("regime", {})
    rg_ok = regime_is_mr(spr, reg_cfg.get("lookback_days",120),
                         reg_cfg.get("hurst_max",0.5),
                         reg_cfg.get("variance_ratio_max",1.0))
    zcfg = params.get("zscore", {})
    entry_z = zcfg.get("entry_z", 2.2)
    exit_z  = zcfg.get("exit_z", 0.5)
    stop_z  = zcfg.get("stop_z", 3.0)
    dec_cfg = params.get("decision", {})
    reasons = []
    ok_quality = True
    if dec_cfg.get("require_stable_coint", True) and not stab["ok"]:
        ok_quality = False; reasons.append("coint instable")
    if dec_cfg.get("require_regime_ok", True) and not rg_ok:
        ok_quality = False; reasons.append("régime non MR")
    if beta_instab > dec_cfg.get("max_beta_instability_pct", 0.3):
        ok_quality = False; reasons.append("β instable")
    if not ok_quality or np.isnan(z_last):
        verdict = "HOLD"; action = "Rien"
    else:
        if abs(z_last) >= stop_z:
            verdict, action = "EXIT", "Sortie_immédiate"
        elif abs(z_last) >= entry_z:
            verdict = "ENTER"
            action = "ShortY_LongX" if z_last > 0 else "LongY_ShortX"
        elif abs(z_last) <= exit_z:
            verdict, action = "EXIT", "Sortie_neutre"
        else:
            verdict, action = "HOLD", "Rien"
    return {
        "z_last": z_last, "z_win": z_win, "hl_initial": hl,
        "beta_instability": beta_instab, "stab_pass_ratio": stab["pass_ratio"],
        "stab_ok": stab["ok"], "regime_ok": rg_ok,
        "verdict": verdict, "action": action,
        "reasons": ", ".join(reasons) if reasons else "OK"
    }
