#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional
import polars as pl
import pandas as pd
from tabulate import tabulate

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs
from src.backtest import merge_close_series, simulate_pair
from src.profile import merged_risk
from src.quality import (
    assert_provenance, assert_price_series_ok, assert_pairs_scored_schema, write_qa_log
)

def _root_dir(params: dict) -> Path:
    d = params.get("data", {})
    src = d.get("source", "yahoo").lower()
    if d.get("separate_roots", True):
        return Path(d.get(f"root_dir_{src}", f"data/eod/ETFs_{src}"))
    return Path(d.get("root_dir", "data/eod/ETFs"))

def _expected_prov(params: dict):
    src = params.get("data", {}).get("source", "yahoo").lower()
    return {"yahoo", "yfinance"} if src == "yahoo" else "ibkr"

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
        raise ValueError(f"missing columns: {missing}")
    filt = (scored["corr"] >= float(th["min_corr"])) & (scored["pval"] <= float(th["pval_coint"]))
    hl_col = _detect_hl_col(scored)
    if hl_col is not None:
        filt &= (scored[hl_col] <= float(th["max_half_life_days"]))
    size_default = int(params.get("risk", {}).get("max_pairs_open", 3)) * 3
    size_cap = int(params.get("exports", {}).get("topk", size_default))
    topn = min(size_cap, size_default) if size_cap > 0 else size_default
    return scored.loc[filt].sort_values("score", ascending=False).head(topn).copy()

def _selection_diagnostics(scored: pd.DataFrame, params: dict) -> str:
    th = _get_selection_params(params)
    have_hl = _detect_hl_col(scored)
    n_all = len(scored)
    n_corr = int((scored["corr"] >= th["min_corr"]).sum()) if "corr" in scored.columns else 0
    n_pval = int((scored["pval"] <= th["pval_coint"]).sum()) if "pval" in scored.columns else 0
    n_hl = int((scored[have_hl] <= th["max_half_life_days"]).sum()) if have_hl else None
    parts = [f"pairs={n_all}", f"pass_corr(≥{th['min_corr']})={n_corr}", f"pass_pval(≤{th['pval_coint']})={n_pval}"]
    parts.append(f"pass_hl(≤{th['max_half_life_days']})={n_hl if n_hl is not None else 'N/A'}")
    if "score" in scored.columns:
        parts.append(f"score_min={scored['score'].min():.3f}, score_max={scored['score'].max():.3f}")
    return " | ".join(parts)

def _z_window_for_row(row: pd.Series, params: dict) -> int:
    lb = params.get('lookbacks', {})
    z_min = int(lb.get('zscore_days_min', 12))
    mult = float(lb.get('zscore_mult_half_life', 3.0))
    hl_col = _detect_hl_col(pd.DataFrame([row]))
    hl = float(row.get(hl_col, float('nan'))) if hl_col else float('nan')
    if pd.notna(hl) and hl > 0:
        return max(z_min, int(round(mult * hl)))
    return max(z_min, 60)

def _count_entries(signal_series: pd.Series) -> int:
    s = signal_series.fillna(0).astype(int)
    shifted = s.shift(1).fillna(0).astype(int)
    return int(((shifted == 0) & (s != 0)).sum())

def main():
    params = load_params()
    risk = merged_risk(params)
    qa_log = Path("reports/QA") / f"qa_{params.get('data',{}).get('source','yahoo')}_{params.get('trading',{}).get('mode','paper')}.log"

    tickers = load_universe()
    ensure_universe(params, tickers)
    root_dir = _root_dir(params)
    assert_provenance(root_dir, _expected_prov(params), params.get("quality", {}).get("require_provenance_match", True), qa_log)

    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).sort('date')
        assert_price_series_ok(dfpl, t, params.get("quality", {}), qa_log)
        pdf = dfpl.select(['date','adj_close','close']).to_pandas().set_index('date')
        s = pdf['adj_close'].fillna(pdf['close'])
        price_map[t] = pd.DataFrame({'close': s})

    pairs = all_pairs_from_universe(tickers)
    lb = params.get('lookbacks', {})
    scored = score_pairs(price_map, pairs, int(lb.get('corr_days',120)), int(lb.get('coint_days',120)))
    assert_pairs_scored_schema(scored, params.get("quality", {}), qa_log)

    out_dir = Path(params.get('exports', {}).get('reports_dir', 'reports')) / "pairs_scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(scored).write_parquet(out_dir / 'latest_pairs_scored.parquet')

    top = _select_top_pairs(scored, params)
    print("\nTop paires sélectionnées:\n")
    if top.empty:
        print("(aucune paire après filtres)")
        print(f"[diagnostic sélection] {_selection_diagnostics(scored, params)}")
        return
    print(tabulate(top.head(15), headers='keys', tablefmt='psql', floatfmt='.4f'))

    k = min(5, len(top))
    rows: List[tuple] = []
    hl_col = _detect_hl_col(top)
    thr = params.get('thresholds', {})
    for i in range(k):
        a, b = str(top.iloc[i]["a"]), str(top.iloc[i]["b"])
        hl = float(top.iloc[i][hl_col]) if hl_col else float('nan')
        zwin = _z_window_for_row(top.iloc[i], params)
        dfa = get_price_series(root_dir, a)
        dfb = get_price_series(root_dir, b)
        df = merge_close_series(dfa, dfb)
        total, journal = simulate_pair(
            df,
            float(thr.get('entry_z', 2.2)),
            float(thr.get('exit_z', 0.5)),
            float(thr.get('stop_z', 3.0)),
            int(zwin),
            float(risk.get('per_trade_pct', 0.0) or 0.0),
            capital=float(risk.get('capital', 100000)),
            costs_bp=int(params.get('costs', {}).get('slippage_bp', 2)),
            cool_off_bars=2,
            min_bars_between_entries=2,
            notional_per_trade=float(risk.get('notional_per_trade', 0.0) or 0.0),
        )
        entries = _count_entries(journal["signal"]) if "signal" in journal.columns else 0
        rows.append((a, b, hl, zwin, entries, float(total)))

    print("\nRésumé Top-5 (z dynamique = 3×HL, min_window={}):\n".format(int(params.get('lookbacks',{}).get('zscore_days_min',12))))
    print(tabulate(
        [
            {"pair": f"{a}/{b}", "HL(d)": (f"{hl:.1f}" if pd.notna(hl) else "NA"),
             "z_win": zwin, "entries": entries, "PnL($)": f"{pnl:,.2f}"}
            for (a,b,hl,zwin,entries,pnl) in rows
        ],
        headers="keys", tablefmt="psql"
    ))

    a, b, hl, zwin, entries, pnl = rows[0]
    print(f"\nBacktest détaillé sur {a}/{b} — HL≈{(hl if pd.notna(hl) else 0):.1f}j, z_window={zwin}, entrées={entries}, PnL≈${pnl:,.2f}")
    dfa = get_price_series(root_dir, a)
    dfb = get_price_series(root_dir, b)
    df = merge_close_series(dfa, dfb)
    total, journal = simulate_pair(
        df,
        float(thr.get('entry_z', 2.2)),
        float(thr.get('exit_z', 0.5)),
        float(thr.get('stop_z', 3.0)),
        int(zwin),
        float(risk.get('per_trade_pct', 0.0) or 0.0),
        capital=float(risk.get('capital', 100000)),
        costs_bp=int(params.get('costs', {}).get('slippage_bp', 2)),
        cool_off_bars=2,
        min_bars_between_entries=2,
        notional_per_trade=float(risk.get('notional_per_trade', 0.0) or 0.0),
    )
    print(journal.tail(10))

if __name__ == '__main__':
    main()
