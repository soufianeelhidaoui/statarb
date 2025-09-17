from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas as pd
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs, select_top_pairs
from src.backtest import merge_close_series, simulate_pair
def main():
    params = load_params()
    tickers = load_universe()
    ensure_universe(params)
    root_dir = Path(params['data']['root_dir'])
    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).select(['date','adj_close','close']).sort('date').to_pandas().set_index('date')
        s = dfpl['adj_close'].fillna(dfpl['close'])
        price_map[t] = pd.DataFrame({'close': s})
    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])
    top = select_top_pairs(scored, params['selection']['min_corr'], params['selection']['max_half_life_days'], params['selection']['pval_coint'], params.get('exports', {}).get('topk', 10))
    out_dir = Path(params.get('exports', {}).get('journals_dir', 'reports/journals'))
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, r in top.iterrows():
        a, b = r['a'], r['b']
        dfa = get_price_series(root_dir, a)
        dfb = get_price_series(root_dir, b)
        df = merge_close_series(dfa, dfb)
        total, journal = simulate_pair(
            df,
            params['thresholds']['entry_z'],
            params['thresholds']['exit_z'],
            params['thresholds']['stop_z'],
            max(params.get('zscore', {}).get('min_window', 20), params['lookbacks']['zscore_days']),
            params['risk']['per_trade_pct'],
            capital=100_000.0,
            costs_bp=params['costs']['slippage_bp']
        )
        journal.to_csv(out_dir / f"journal_{a}_{b}.csv")
if __name__ == "__main__":
    main()
