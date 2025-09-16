from __future__ import annotations
from itertools import combinations
import pandas as pd
from .stats import compute_corr, compute_coint_stats, combine_score

def all_pairs_from_universe(tickers: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(set(tickers)), 2))

def score_pairs(price_df_map: dict[str, pd.DataFrame], pairs: list[tuple[str, str]], corr_days: int, coint_days: int) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        if a not in price_df_map or b not in price_df_map:
            continue
        a_close = price_df_map[a]['close']
        b_close = price_df_map[b]['close']
        rho = compute_corr(a_close, b_close, corr_days)
        pval, half_life, sigma, alpha, beta = compute_coint_stats(a_close, b_close, coint_days)
        score = combine_score(rho, pval, half_life, sigma)
        rows.append({'a': a, 'b': b, 'rho': rho, 'pval': pval, 'half_life': half_life, 'sigma_spread': sigma, 'alpha': alpha, 'beta': beta, 'score': score})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df

def select_top_pairs(scored: pd.DataFrame, min_rho: float, max_half_life: float, pval_coint: float, max_pairs: int) -> pd.DataFrame:
    if scored.empty:
        return scored
    sel = scored.query('rho >= @min_rho and half_life <= @max_half_life and pval <= @pval_coint').copy()
    sel = sel.nlargest(max_pairs, 'score').reset_index(drop=True)
    return sel
