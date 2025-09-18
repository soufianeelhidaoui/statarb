#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import polars as pl
from tabulate import tabulate

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs
from src.backtest import merge_close_series, simulate_pair
from src.quality import (
    assert_provenance, assert_price_series_ok, assert_pairs_scored_schema,
    check_overlap_len, write_qa_log
)

# -------------------- helpers --------------------

def _root_dir_for_env(params: dict) -> Path:
    data = params.get("data", {})
    mode = params.get("env", {}).get("mode", "dev")
    if data.get("separate_roots", True):
        if mode == "prod":
            return Path(data.get("root_dir_prod", "data/eod/ETFs_prod"))
        else:
            return Path(data.get("root_dir_dev", "data/eod/ETFs_dev"))
    return Path(data.get("root_dir", "data/eod/ETFs"))

def _get_selection_params(params: dict) -> dict:
    defaults = {"min_corr": 0.6, "max_half_life_days": 20.0, "pval_coint": 0.05}
    sel = params.get("selection", {})
    if sel:
        out = defaults.copy(); out.update(sel); return out
    sf = params.get("stats_filters", {})
    return {
        "min_corr": defaults["min_corr"],
        "max_half_life_days": float(sf.get("half_life_max_days", defaults["max_half_life_days"])),
        "pval_coint": float(sf.get("coint_pval_max", defaults["pval_coint"])),
    }

def _detect_hl_col(scored: pd.DataFrame) -> Optional[str]:
    for c in ["half_life", "half_life_days", "hl"]:
        if c in scored.columns:
            return c
    return None

def _select_top_pairs(scored: pd.DataFrame, params: dict) -> pd.DataFrame:
    th = _get_selection_params(params)
    need = {"a","b","corr","pval","score"}
    missing = need - set(scored.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes (min): {missing}")

    filt = (scored["corr"] >= float(th["min_corr"])) & (scored["pval"] <= float(th["pval_coint"]))
    hl_col = _detect_hl_col(scored)
    if hl_col is not None:
        filt &= (scored[hl_col] <= float(th["max_half_life_days"]))

    size_default = int(params.get("risk", {}).get("max_pairs_open", 5)) * 3
    size_cap = int(params.get("exports", {}).get("topk", size_default))
    topn = min(size_cap, size_default) if size_cap > 0 else size_default

    return scored.loc[filt].sort_values("score", ascending=False).head(topn).copy()

def _selection_diagnostics(scored: pd.DataFrame, params: dict) -> str:
    th_min_corr = float(params.get("selection", {}).get("min_corr",
                        params.get("stats_filters", {}).get("min_corr", 0.6)))
    th_pval = float(params.get("selection", {}).get("pval_coint",
                    params.get("stats_filters", {}).get("coint_pval_max", 0.05)))
    th_hl = float(params.get("selection", {}).get("max_half_life_days",
                 params.get("stats_filters", {}).get("half_life_max_days", 20.0)))

    have_hl = _detect_hl_col(scored)
    n_all = len(scored)
    n_corr = int((scored["corr"] >= th_min_corr).sum()) if "corr" in scored.columns else 0
    n_pval = int((scored["pval"] <= th_pval).sum()) if "pval" in scored.columns else 0
    n_hl = int((scored[have_hl] <= th_hl).sum()) if have_hl else None

    parts = [f"pairs={n_all}", f"pass_corr(≥{th_min_corr})={n_corr}", f"pass_pval(≤{th_pval})={n_pval}"]
    parts.append(f"pass_hl(≤{th_hl})={n_hl if n_hl is not None else 'N/A'}")
    if "score" in scored.columns:
        parts.append(f"score_min={scored['score'].min():.3f}, score_max={scored['score'].max():.3f}")
    return " | ".join(parts)

def _z_window_for_pair(row: pd.Series, params: dict) -> int:
    lb = params.get('lookbacks', {})
    z_min = int(lb.get('zscore_days_min', 20))
    mult = float(lb.get('zscore_mult_half_life', 3.0))
    hl_col = _detect_hl_col(pd.DataFrame([row]))
    hl = float(row.get(hl_col, float('nan'))) if hl_col else float('nan')
    if pd.notna(hl) and hl > 0:
        return max(z_min, int(round(mult * hl)))
    # garde-fou : si HL non dispo → 60 (conservateur)
    return max(z_min, 60)

def _count_entries(signal_series: pd.Series) -> int:
    """Nombre d'entrées = transitions 0 -> (±1)."""
    s = signal_series.fillna(0).astype(int)
    shifted = s.shift(1).fillna(0).astype(int)
    return int(((shifted == 0) & (s != 0)).sum())

# -------------------- main --------------------

def main():
    params = load_params()
    mode = params.get("env", {}).get("mode", "dev")
    qa_log = Path("reports/QA") / f"qa_{mode}.log"
    quality_cfg = params.get("quality", {})

    tickers = load_universe()
    ensure_universe(params)
    root_dir = _root_dir_for_env(params)

    # QA provenance
    expected_src = "ibkr" if mode == "prod" else {"yahoo", "yfinance"}
    assert_provenance(root_dir, expected_src, quality_cfg.get("require_provenance_match", True), qa_log)

    # Charger séries & QA par ticker
    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).sort('date')
        assert_price_series_ok(dfpl, t, quality_cfg, qa_log)
        pdf = dfpl.select(['date','adj_close','close']).to_pandas().set_index('date')
        s = pdf['adj_close'].fillna(pdf['close'])
        price_map[t] = pd.DataFrame({'close': s})

    # Scoring
    pairs = all_pairs_from_universe(tickers)
    lb = params.get('lookbacks', {})
    corr_days = int(lb.get('corr_days', 120))
    coint_days = int(lb.get('coint_days', 120))
    scored = score_pairs(price_map, pairs, corr_days, coint_days)
    assert_pairs_scored_schema(scored, quality_cfg, qa_log)

    # Sauvegarde snapshot des scores
    out_dir = Path(params.get('reports', {}).get('pairs_scored_dir', 'reports/pairs_scored'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'latest_pairs_scored.parquet'
    pl.from_pandas(scored).write_parquet(out_path)

    # Sélection
    top = _select_top_pairs(scored, params)
    print("\nTop paires sélectionnées:\n")
    if top.empty:
        print("(aucune paire après filtres)")
        print(f"[diagnostic sélection] {_selection_diagnostics(scored, params)}")
        return
    print(tabulate(top.head(15), headers='keys', tablefmt='psql', floatfmt='.4f'))

    # Mini backtest sur Top-5 avec z-window dynamique basé sur HL
    k = min(5, len(top))
    rows: List[Tuple[str,str,float,int,int,float]] = []  # (a,b,hl,zwin,entries,pnl)
    hl_col = _detect_hl_col(top)
    thr = params.get('thresholds', {})
    for i in range(k):
        a, b = str(top.iloc[i]["a"]), str(top.iloc[i]["b"])
        hl = float(top.iloc[i][hl_col]) if hl_col else float('nan')
        zwin = _z_window_for_pair(top.iloc[i], params)

        dfa = get_price_series(root_dir, a)
        dfb = get_price_series(root_dir, b)
        df = merge_close_series(dfa, dfb)

        total, journal = simulate_pair(
            df,
            thr.get('entry_z', 2.2),
            thr.get('exit_z', 0.5),
            thr.get('stop_z', 3.0),
            zwin,
            params.get('risk', {}).get('per_trade_pct', 0.003),
            capital=float(params.get('risk', {}).get('capital', 100_000.0)),
            costs_bp=int(params.get('costs', {}).get('slippage_bp', 2)),
        )
        entries = _count_entries(journal["signal"]) if "signal" in journal.columns else 0
        rows.append((a, b, hl, zwin, entries, float(total)))

    print("\nRésumé Top-5 (z dynamique = 3×HL, min_window={}):\n".format(int(params.get('lookbacks',{}).get('zscore_days_min',20))))
    print(tabulate(
        [
            {"pair": f"{a}/{b}", "HL(d)": (f"{hl:.1f}" if pd.notna(hl) else "NA"),
             "z_win": zwin, "entries": entries, "PnL($)": f"{pnl:,.2f}"}
            for (a,b,hl,zwin,entries,pnl) in rows
        ],
        headers="keys", tablefmt="psql"
    ))

    # Backtest détaillé sur la meilleure (1ère) pour trace rapide
    a, b, hl, zwin, entries, pnl = rows[0]
    print(f"\nBacktest détaillé sur {a}/{b} — HL≈{hl:.1f}j, z_window={zwin}, entrées={entries}, PnL≈${pnl:,.2f}")
    # (Affiche les 10 dernières lignes du journal)
    dfa = get_price_series(root_dir, a)
    dfb = get_price_series(root_dir, b)
    df = merge_close_series(dfa, dfb)
    total, journal = simulate_pair(
        df,
        thr.get('entry_z', 2.2),
        thr.get('exit_z', 0.5),
        thr.get('stop_z', 3.0),
        zwin,
        params.get('risk', {}).get('per_trade_pct', 0.003),
        capital=float(params.get('risk', {}).get('capital', 100_000.0)),
        costs_bp=int(params.get('costs', {}).get('slippage_bp', 2)),
    )
    print(journal.tail(10))


if __name__ == '__main__':
    main()
