import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.stats import _beta_ols, _halflife_ar1

def all_pairs_from_universe(tickers: list[str]) -> list[tuple[str,str]]:
    return [(a,b) for a,b in itertools.combinations(tickers, 2)]

def _corr_last_window(s1: pd.Series, s2: pd.Series, days: int) -> float:
    idx = s1.dropna().index.intersection(s2.dropna().index)
    if len(idx) < days: return float("nan")
    r1 = s1.loc[idx].pct_change().tail(days)
    r2 = s2.loc[idx].pct_change().tail(days)
    return float(r1.corr(r2))

def _coint_pval(s1: pd.Series, s2: pd.Series, days: int) -> float:
    idx = s1.dropna().index.intersection(s2.dropna().index)
    s1w = s1.loc[idx].tail(days).dropna()
    s2w = s2.loc[idx].tail(days).dropna()
    idx2 = s1w.index.intersection(s2w.index)
    if len(idx2) < 60: return 1.0
    try:
        _, pval, _ = coint(s1w.values, s2w.values)
        return float(pval)
    except Exception:
        return 1.0

def score_pairs(price_map: dict[str, pd.DataFrame], pairs: list[tuple[str,str]], corr_days: int, coint_days: int) -> pd.DataFrame:
    rows = []
    for a,b in pairs:
        s1 = price_map[a]["close"]; s2 = price_map[b]["close"]
        corr = _corr_last_window(s1, s2, corr_days)
        pval = _coint_pval(s1, s2, coint_days)
        score = (corr if pd.notna(corr) else 0.0) - pval
        alpha, beta = _beta_ols(price_map[a]['close'].iloc[-coint_days:], price_map[b]['close'].iloc[-coint_days:])
        spread = price_map[a]['close'].iloc[-coint_days:] - (alpha + beta*price_map[b]['close'].iloc[-coint_days:])
        half_life = _halflife_ar1(spread)
        rows.append({"a":a,"b":b,"corr":corr,"pval":pval,"score":score, "half_life":half_life})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df

def select_top_pairs(scored_df, params):
    """
    Backward-compat: applique params.selection puis trie par 'score' et coupe à exports.topk.
    """
    sel = params.get("selection", {})
    min_corr = float(sel.get("min_corr", 0.6))
    max_hl = float(sel.get("max_half_life_days", 20))
    max_pval = float(sel.get("pval_coint", 0.05))
    topk = int(params.get("exports", {}).get("topk", 10))

    need = {"a","b","corr","half_life","pval","score"}
    missing = need - set(scored_df.columns)
    if missing:
        raise ValueError(f"select_top_pairs: colonnes manquantes: {missing}")

    filt = (
        (scored_df["corr"] >= min_corr) &
        (scored_df["half_life"] <= max_hl) &
        (scored_df["pval"] <= max_pval)
    )
    return scored_df.loc[filt].sort_values("score", ascending=False).head(topk).copy()
