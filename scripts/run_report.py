from __future__ import annotations
from pathlib import Path
import polars as pl
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs, select_top_pairs
from src.report import generate_pair_report
from src.duck_analytics import write_scored_to_duckdb

def main():
    params = load_params()
    tickers = load_universe()
    ensure_universe(params)

    root_dir = Path(params['data']['root_dir'])
    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).select(['date','close']).sort('date')
        price_map[t] = dfpl.to_pandas().set_index('date')

    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])

    out_dir = Path(params['reports']['pairs_scored_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "latest_pairs_scored.parquet"
    pl.from_pandas(scored).write_parquet(parquet_path)

    db_path = write_scored_to_duckdb(parquet_path)

    top = select_top_pairs(scored, params['selection']['min_corr'], params['selection']['max_half_life_days'], params['selection']['pval_coint'], params['risk']['max_pairs_open']*3)
    if top.empty:
        print("Aucune paire sélectionnée."); return

    a, b = top.loc[0, 'a'], top.loc[0, 'b']
    out_html = Path("reports") / f"report_{a}_{b}.html"
    path = generate_pair_report(a, b, out_html=str(out_html))
    print(f"Rapport HTML: {path}")
    print(f"DuckDB analytics: {db_path}")

if __name__ == '__main__':
    main()
