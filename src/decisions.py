from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

from .filters.stat_filters import (
    hedge_ratio,        # alpha, beta
    coint_adf,          # p-val ADF
    z_window_by_half_life,
    zscore,
    stable_half_life,   # (ok_hl, hl)
    beta_stable,        # (ok_beta, beta)
)
from .filters.data_filters import liquidity_filter
from .filters.market_filters import vix_ok, macro_ok

# --- état persistant pour anti-sur-trading (cooldown & spacing) ---
STATE_PATH = Path("reports/state/trade_state.json")

def _load_state() -> dict:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))

def _pair_key(a: str, b: str) -> str:
    return f"{min(a,b)}/{max(a,b)}"

def _apply_ex_div_mask(df: pd.DataFrame, days_after: int = 1) -> pd.Series:
    if "is_ex_div" not in df.columns:
        return pd.Series(False, index=df.index)
    mask = df["is_ex_div"].copy().astype(bool)
    if days_after <= 0:
        return mask
    for k in range(1, days_after+1):
        mask = mask | df["is_ex_div"].shift(k, fill_value=False)
    return mask

def _can_enter_now(pair_key: str, now_idx: int, state: dict,
                   cool_off_bars: int, min_bars_between_entries: int) -> bool:
    rec = state.get(pair_key, {})
    last_entry = int(rec.get("last_entry_idx", -10**9))
    cool_until = int(rec.get("cool_until_idx", -10**9))
    if now_idx < cool_until:
        return False
    if now_idx - last_entry < int(min_bars_between_entries):
        return False
    return True

def _mark_entry(pair_key: str, now_idx: int, state: dict):
    rec = state.get(pair_key, {})
    rec["last_entry_idx"] = int(now_idx)
    state[pair_key] = rec

def _mark_exit(pair_key: str, now_idx: int, state: dict, cool_off_bars: int):
    rec = state.get(pair_key, {})
    rec["last_exit_idx"] = int(now_idx)
    rec["cool_until_idx"] = int(now_idx) + int(cool_off_bars)
    state[pair_key] = rec

def decide_pair(ya: pd.Series, xb: pd.Series, spy: pd.Series | None, params: dict,
                meta_a: dict | None = None, meta_b: dict | None = None) -> dict:
    """
    Décision unique (daily) pour une paire :
      - applique filtres marché (VIX, macro), liquidité, masquage ex-div J0/J+1,
      - vérifie stabilité HL & beta (comme ton scoring),
      - ADF p-val (cointegration),
      - z dynamique = f(half-life),
      - garde-fous d’exécution: cool_off_bars & min_bars_between_entries (état persistant).
    Retour: dict {a,b, verdict, action, reason, z_last, hl, beta, pval, z_window}
    """
    a = ya.name; b = xb.name
    idx = ya.dropna().index.intersection(xb.dropna().index).sort_values()
    if len(idx) < 120:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Insufficient overlap",
                "z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}

    qa = params.get("quality", {})
    sf = params.get("stats_filters", {})
    mf = params.get("market_filters", {})
    dec = params.get("decision", {})
    cool_off_bars = int(dec.get("cool_off_bars", 0))
    min_bars_between_entries = int(dec.get("min_bars_between_entries", 0))

    # --- Filtres marché ---
    if mf.get("enable", True):
        if not vix_ok(mf.get("vix_path",""), mf.get("vix_max", 1000)):
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"High VIX regime",
                    "z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}
        # macro day cool-off
        today_now = pd.Timestamp.utcnow()  # tz-naive UTC
        if not macro_ok(mf.get("macro_calendar_csv",""), today_now, mf.get("cool_off_hours", 0)):
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Macro event day",
                    "z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}

    # --- Liquidité ---
    if not liquidity_filter(meta_a.get("df", pd.DataFrame()), meta_b.get("df", pd.DataFrame()),
                            qa.get("min_volume",0)):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Low liquidity",
                "z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}

    # --- Masque ex-div J0/J+1 (si activé) ---
    if qa.get("mask_ex_div", True):
        mA = _apply_ex_div_mask(meta_a.get("df", pd.DataFrame()), qa.get("mask_ex_div_days_after",1))
        mB = _apply_ex_div_mask(meta_b.get("df", pd.DataFrame()), qa.get("mask_ex_div_days_after",1))
        m = (mA.reindex(index=idx, fill_value=False)) | (mB.reindex(index=idx, fill_value=False))
    else:
        m = pd.Series(False, index=idx)

    y = ya.loc[idx]; x = xb.loc[idx]
    y2 = y.loc[~m];  x2 = x.loc[~m]
    if len(y2) < 120:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Masked overlap too small",
                "z_last":np.nan,"hl":np.nan,"beta":np.nan,"pval":1.0}

    # --- Stabilité de la half-life (comme ton scoring) ---
    ok_hl, hl = stable_half_life(y2, x2,
                                 sf.get("half_life_min_days",2),
                                 sf.get("half_life_max_days",20),
                                 sf.get("half_life_stability_tol",0.2))
    if not ok_hl:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None",
                "reason":"Half-life unstable/out-of-range",
                "z_last":np.nan,"hl":hl,"beta":np.nan,"pval":1.0}

    # --- Stabilité beta ---
    ok_beta, beta = beta_stable(y2, x2, sf.get("beta_stability_tol",0.2))
    if not ok_beta:
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Beta unstable",
                "z_last":np.nan,"hl":hl,"beta":beta,"pval":1.0}

    # --- Hedge ratio final & spread ---
    alpha, beta_full = hedge_ratio(y2, x2)
    spread = y2 - (alpha + beta_full * x2)

    # --- Coint (ADF pval) ---
    pval = coint_adf(spread)
    if sf.get("require_coint", True) and (pval > sf.get("coint_pval_max", 0.05)):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"No cointegration",
                "z_last":np.nan,"hl":hl,"beta":beta_full,"pval":pval}

    # --- z dynamique ---
    zwin = z_window_by_half_life(hl,
                                 params["lookbacks"].get("zscore_days_min",20),
                                 params["lookbacks"].get("zscore_mult_half_life",3.0))
    z = zscore(spread, zwin)
    if not np.isfinite(z.iloc[-1]):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"z_nan",
                "z_last":np.nan,"hl":hl,"beta":beta_full,"pval":pval}
    z_last = float(z.iloc[-1])

    if abs(z_last) > sf.get("z_cap", 5.0):
        return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"Extreme z outlier",
                "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval}

    entry_z = float(params["thresholds"]["entry_z"])
    exit_z  = float(params["thresholds"]["exit_z"])
    stop_z  = float(params["thresholds"]["stop_z"])

    # --- Garde-fous d’exécution (état persistant) ---
    # index numérique simple (jour julien) pour les règles de spacing/cooldown
    now_idx = int(pd.Timestamp(y2.index[-1]).to_julian_date())
    state = _load_state()
    pair_k = _pair_key(a, b)

    can_enter = _can_enter_now(pair_k, now_idx, state, cool_off_bars, min_bars_between_entries)

    if abs(z_last) >= stop_z:
        _mark_exit(pair_k, now_idx, state, cool_off_bars)
        _save_state(state)
        return {"a":a,"b":b,"verdict":"EXIT","action":"Sortie_immediate","reason":"Stop z reached",
                "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}

    if abs(z_last) <= exit_z:
        _mark_exit(pair_k, now_idx, state, cool_off_bars)
        _save_state(state)
        return {"a":a,"b":b,"verdict":"EXIT","action":"Sortie_neutre","reason":"Mean reversion completed",
                "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}

    if abs(z_last) >= entry_z:
        if not can_enter:
            return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"cooldown/spacing",
                    "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}
        if z_last > 0:
            _mark_entry(pair_k, now_idx, state)
            _save_state(state)
            return {"a":a,"b":b,"verdict":"ENTER","action":"ShortY_LongX","reason":"z>=entry",
                    "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}
        else:
            _mark_entry(pair_k, now_idx, state)
            _save_state(state)
            return {"a":a,"b":b,"verdict":"ENTER","action":"LongY_ShortX","reason":"z<=-entry",
                    "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}

    return {"a":a,"b":b,"verdict":"HOLD","action":"None","reason":"No signal",
            "z_last":z_last,"hl":hl,"beta":beta_full,"pval":pval,"z_window":zwin}
