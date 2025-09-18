from __future__ import annotations
import os
from pathlib import Path
import polars as pl
import pandas as pd
import subprocess, sys

from src.config import load_params
from src.universe import load_universe
from src.provenance import read_provenance, save_provenance, enforce_provenance
from src.data_quality import load_price_df
from src.pairs import all_pairs_from_universe, score_pairs
from src.duck_analytics import write_scored_to_duckdb
from src.report_plus import generate_reports_bundle
from src.notify_email import load_email_config, send_email

def _root_dir_for_env(params: dict) -> Path:
    d = params["data"]
    if d.get("separate_roots", False):
        return Path(d["root_dir_prod" if params["env"]["mode"]=="prod" else "root_dir_dev"])
    return Path(d["root_dir"])

def _maybe_auto_ingest(params: dict, root_dir: Path) -> None:
    mode = params.get("env", {}).get("mode", "dev")
    if mode == "prod":
        print("[run_report] PROD -> ingest_ibkr.py ...")
        subprocess.run([sys.executable, "scripts/ingest_ibkr.py"], check=True)
        save_provenance(root_dir, "ibkr")
    else:
        print("[run_report] DEV -> ingest_yahoo.py ...")
        subprocess.run([sys.executable, "scripts/ingest_yahoo.py"], check=True)
        save_provenance(root_dir, "yahoo")

def _maybe_adjust_prices(params: dict) -> None:
    if params.get("quality", {}).get("rebuild_adj_from_yahoo", True):
        print("[run_report] Adjusting (adj_close & is_ex_div)...")
        subprocess.run([sys.executable, "scripts/adjust_prices.py", "--config", "config/params.yaml"], check=True)

def _send_summary_email(bundle: dict, decisions_df: pd.DataFrame, orders_df: pd.DataFrame) -> None:
    cfg = load_email_config()
    if not cfg.get("enabled", False):
        return
    env = bundle["env"]; src = bundle["source"]; rid = bundle["rebalance_id"]
    nb_enter = int((decisions_df["verdict"]=="ENTER").sum()) if not decisions_df.empty else 0
    nb_exit  = int((decisions_df["verdict"]=="EXIT").sum()) if not decisions_df.empty else 0
    html = f"""
    <h3>StatArb - {env.upper()} - {rid} - source={src}</h3>
    <p><b>Entries:</b> {nb_enter} | <b>Exits:</b> {nb_exit}</p>
    <p><b>Bundle:</b> {bundle['bundle_dir']}</p>
    <p><b>Decisions:</b> {bundle['decisions_csv']}<br/>
       <b>Orders:</b> {bundle['orders_csv']}</p>
    """
    try:
        send_email(subject=f"{env.upper()} {rid}: {nb_enter} ENTER / {nb_exit} EXIT", html_body=html, cfg=cfg)
        print("[email] Summary sent.")
    except Exception as e:
        print(f"[email] Failed: {e}")

def main():
    params = load_params()
    root_dir = _root_dir_for_env(params)
    _maybe_auto_ingest(params, root_dir)

    mode = params.get("env", {}).get("mode", "dev")
    enforce_provenance(root_dir, "ibkr" if mode=="prod" else "yahoo", allow_unknown=False)

    _maybe_adjust_prices(params)

    tickers = load_universe()
    price_map = {}; meta_map = {}
    for t in tickers:
        dfpl = load_price_df(root_dir, t, prefer_adj=params["quality"].get("prefer_adj_close", True),
                             px_policy=params["quality"].get("px_policy","best")).sort("date")
        pdf = dfpl.to_pandas().set_index("date")
        price_map[t] = pd.DataFrame({"close": pdf["px"]})
        meta_map[t] = {"df": pdf}

    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])

    out_dir = Path(params['reports']['pairs_scored_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "latest_pairs_scored.parquet"
    pl.from_pandas(scored).write_parquet(parquet_path)
    db_path = write_scored_to_duckdb(parquet_path)

    topk = params.get('exports', {}).get('topk', 20)
    bundle = generate_reports_bundle(
        tickers=tickers,
        root_dir=root_dir,
        top_pairs=scored.head(topk),
        out_base_dir=Path(params['exports'].get('reports_dir','reports')),
        rebalance_id=None
    )
    dec_df = pd.read_csv(bundle["decisions_csv"])
    try:
        if os.path.getsize(bundle["orders_csv"]) == 0:
            ord_df = pd.DataFrame()  
        else:
            ord_df = pd.read_csv(bundle["orders_csv"])
    except pd.errors.EmptyDataError:
        ord_df = pd.DataFrame()
    _send_summary_email(bundle, dec_df, ord_df)

    print("Bundle export:", bundle)
    print(f"DuckDB analytics: {db_path}")

if __name__ == '__main__':
    main()
