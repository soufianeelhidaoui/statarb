from __future__ import annotations
from pathlib import Path
import polars as pl
from tabulate import tabulate
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs, select_top_pairs
from src.backtest import merge_close_series, simulate_pair

def main():
    params = load_params()
    tickers = load_universe()
    ensure_universe(params)

    price_map = {}
    root_dir = Path(params['data']['root_dir'])
    for t in tickers:
        dfpl = get_price_series(root_dir, t).select(['date','close']).sort('date')
        price_map[t] = dfpl.to_pandas().set_index('date')

    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])

    out_dir = Path(params['reports']['pairs_scored_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'latest_pairs_scored.parquet'
    pl.from_pandas(scored).write_parquet(out_path)

    top = select_top_pairs(scored, params['selection']['min_corr'], params['selection']['max_half_life_days'], params['selection']['pval_coint'], params['risk']['max_pairs_open']*3)

    print("\nTop paires sélectionnées:\n")
    print(tabulate(top.head(15), headers='keys', tablefmt='psql', floatfmt='.4f'))

    if not top.empty:
        a, b = top.loc[0, 'a'], top.loc[0, 'b']
        dfa = get_price_series(root_dir, a)
        dfb = get_price_series(root_dir, b)
        df = merge_close_series(dfa, dfb)
        total, journal = simulate_pair(df, params['thresholds']['entry_z'], params['thresholds']['exit_z'], params['thresholds']['stop_z'], params['lookbacks']['zscore_days'], params['risk']['per_trade_pct'], capital=100_000.0, costs_bp=params['costs']['slippage_bp'])
        print(f"\nBacktest rapide sur {a}/{b} — PnL total (approx $): {total:.2f}")
        print(journal.tail(10))

if __name__ == '__main__':
    main()
