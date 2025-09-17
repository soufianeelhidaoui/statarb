from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas as pd
import subprocess, sys

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs, select_top_pairs
from src.duck_analytics import write_scored_to_duckdb
from src.report_plus import generate_reports_bundle
from src.provenance import read_provenance, save_provenance, enforce_provenance

def _maybe_auto_ingest(params: dict, root_dir: Path) -> None:
    mode = params.get("env", {}).get("mode", "dev")
    if mode == "prod":
        print("[run_report] PROD mode → running ingest_ibkr.py first…")
        subprocess.run([sys.executable, "scripts/ingest_ibkr.py"], check=True)
        print("[run_report] Ingestion done → stamping provenance = 'ibkr'")
        save_provenance(root_dir, "ibkr")
    else:
        # In DEV, if no provenance file, stamp as 'yahoo' (best-effort)
        if read_provenance(root_dir) is None:
            print("[run_report] DEV mode and no provenance → stamping 'yahoo'")
            save_provenance(root_dir, "yahoo")

def main():
    params = load_params()
    root_dir = Path(params['data']['root_dir'])

    # 1) Ensure correct ingestion and provenance
    _maybe_auto_ingest(params, root_dir)
    mode = params.get("env", {}).get("mode", "dev")
    if mode == "prod":
        # Hard enforce IBKR provenance in PROD
        enforce_provenance(root_dir, "ibkr", allow_unknown=False)

    # 2) Universe & data
    tickers = load_universe()
    ensure_universe(params)  # in DEV, this may fetch Yahoo for missing files (after stamping 'yahoo')

    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).select(['date','adj_close','close']).sort('date').to_pandas().set_index('date')
        s = dfpl['adj_close'].fillna(dfpl['close'])
        price_map[t] = pd.DataFrame({'close': s})

    # 3) Scoring & selection
    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])

    out_dir = Path(params['reports']['pairs_scored_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "latest_pairs_scored.parquet"
    pl.from_pandas(scored).write_parquet(parquet_path)

    db_path = write_scored_to_duckdb(parquet_path)

    topk = params.get('exports', {}).get('topk', 10)
    top = select_top_pairs(scored, params['selection']['min_corr'], params['selection']['max_half_life_days'], params['selection']['pval_coint'], topk)
    if top.empty:
        print("Aucune paire sélectionnée.")
        return

    # 4) Bundle propre & horodaté (decisions + orders)
    bundle = generate_reports_bundle(
        tickers=load_universe(),
        root_dir=root_dir,
        top_pairs=top,
        out_base_dir=Path("reports"),
        rebalance_id=None
    )
    print("Bundle export:", bundle)
    print(f"DuckDB analytics: {db_path}")

if __name__ == '__main__':
    main()
