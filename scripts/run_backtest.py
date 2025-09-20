#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict
import polars as pl
import pandas as pd
from tabulate import tabulate

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series
from src.pairs import all_pairs_from_universe, score_pairs
from src.backtest import merge_close_series, simulate_pair
from src.profile import merged_risk
from src.quality import assert_provenance, assert_price_series_ok, assert_pairs_scored_schema

def _bundle_dir(params: dict) -> Path:
    from datetime import datetime
    src = params.get("data", {}).get("source", "yahoo").lower()
    mode = params.get("trading", {}).get("mode", "paper").lower()
    root = Path(params.get("exports", {}).get("reports_dir", "reports"))
    return root / "bundles" / src / mode / datetime.now().strftime("%Y-%m-%d")

def _root_dir(params: dict) -> Path:
    d = params.get("data", {})
    src = d.get("source", "yahoo").lower()
    if d.get("separate_roots", True):
        return Path(d.get(f"root_dir_{src}", f"data/eod/ETFs_{src}"))
    return Path(d.get("root_dir", "data/eod/ETFs"))

def _expected_prov(params: dict):
    src = params.get("data", {}).get("source", "yahoo").lower()
    return {"yahoo", "yfinance"} if src == "yahoo" else "ibkr"

def _detect_hl_col(scored: pd.DataFrame) -> Optional[str]:
    for c in ["half_life", "half_life_days", "hl"]:
        if c in scored.columns:
            return c
    return None

def _selection_thresholds(params: dict) -> Dict[str, float]:
    sel = params.get("selection", {})
    sf = params.get("stats_filters", {})
    return {
        "min_corr": float(sel.get("min_corr", sf.get("min_corr", 0.6))),
        "pval_coint": float(sel.get("pval_coint", sf.get("coint_pval_max", 0.05))),
        "max_hl": float(sel.get("max_half_life_days", sf.get("half_life_max_days", 20.0))),
    }

def _select_pairs(scored: pd.DataFrame, params: dict, topk: int) -> pd.DataFrame:
    th = _selection_thresholds(params)
    need = {"a", "b", "corr", "pval", "score"}
    miss = need - set(scored.columns)
    if miss:
        raise ValueError(f"pairs_scored missing columns: {miss}")
    flt = (scored["corr"] >= th["min_corr"]) & (scored["pval"] <= th["pval_coint"])
    hlcol = _detect_hl_col(scored)
    if hlcol is not None:
        flt &= (scored[hlcol] <= th["max_hl"])
    return scored.loc[flt].sort_values("score", ascending=False).head(topk).copy()

def _z_window(row: pd.Series, params: dict) -> int:
    lb = params.get("lookbacks", {})
    zmin = int(lb.get("zscore_days_min", 12))
    mult = float(lb.get("zscore_mult_half_life", 3.0))
    hlcol = _detect_hl_col(pd.DataFrame([row]))
    hl = float(row.get(hlcol, float("nan"))) if hlcol else float("nan")
    return max(zmin, int(round(mult * hl))) if pd.notna(hl) and hl > 0 else max(zmin, 60)

def _count_entries(sig: pd.Series) -> int:
    s = sig.fillna(0).astype(int)
    sh = s.shift(1).fillna(0).astype(int)
    return int(((sh == 0) & (s != 0)).sum())

def _extract_key_params(params: dict) -> dict:
    """Extract key parameters for logging/tracking"""
    from datetime import datetime, timezone
    key_sections = ["lookbacks", "thresholds", "selection", "decision", "stats_filters", "stability"]
    extracted = {}
    
    for section in key_sections:
        if section in params:
            extracted[section] = params[section]
    
    # Add timestamp and other metadata
    extracted["_meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": params.get("data", {}).get("source", "unknown"),
        "mode": params.get("trading", {}).get("mode", "unknown"),
        "capital": params.get("profiles", {}).get(params.get("trading", {}).get("mode", "paper"), {}).get("risk", {}).get("capital", params.get("risk", {}).get("capital", 0)),
        "universe_size": len(params.get("universe", {}).get("tickers", []))
    }
    
    return extracted

def _save_run_params(bundle_dir: Path, params: dict, script_name: str) -> None:
    """Save key parameters to bundle directory"""
    import json
    key_params = _extract_key_params(params)
    key_params["_meta"]["script"] = script_name
    
    bundle_dir.mkdir(parents=True, exist_ok=True)
    params_file = bundle_dir / "run_params.json"
    with open(params_file, 'w') as f:
        json.dump(key_params, f, indent=2, default=str)
    
    print(f"Parameters saved to: {params_file}")

def main():
    params = load_params()
    risk = merged_risk(params)
    tickers = load_universe()
    ensure_universe(params, tickers)
    root = _root_dir(params)
    qa_log = Path("reports/QA") / f"qa_{params.get('data',{}).get('source','yahoo')}_{params.get('trading',{}).get('mode','paper')}.log"
    
    assert_provenance(
        root,
        _expected_prov(params),
        params.get("quality", {}).get("require_provenance_match", True),
        qa_log,
    )

    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root, t).sort("date")
        assert_price_series_ok(dfpl, t, params.get("quality", {}), qa_log)
        pdf = dfpl.select(["date", "adj_close", "close"]).to_pandas().set_index("date")
        s = pdf["adj_close"].fillna(pdf["close"])
        price_map[t] = pd.DataFrame({"close": s})

    pairs = all_pairs_from_universe(tickers)
    lb = params.get("lookbacks", {})
    scored = score_pairs(price_map, pairs, int(lb.get("corr_days", 120)), int(lb.get("coint_days", 120)))
    assert_pairs_scored_schema(scored, params.get("quality", {}), qa_log)

    topk = 5
    top = _select_pairs(scored, params, topk)
    if top.empty:
        print("(aucune paire après filtres)"); return

    print("\nTop paires sélectionnées:\n")
    print(tabulate(top.head(15), headers="keys", tablefmt="psql", floatfmt=".4f"))

    bundle = _bundle_dir(params)
    out_dir = bundle / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save key parameters for tracking
    _save_run_params(out_dir, params, "run_backtest")

    rows: List[tuple] = []
    hlcol = _detect_hl_col(top)
    thr = params.get("thresholds", {})
    for _, r in top.iterrows():
        a, b = str(r["a"]), str(r["b"])
        hl = float(r[hlcol]) if hlcol and pd.notna(r[hlcol]) else float("nan")
        zwin = _z_window(r, params)

        dfa = get_price_series(root, a)
        dfb = get_price_series(root, b)
        df = merge_close_series(dfa, dfb)
        
        logic = params.get("decision", {})
        require_cross = bool(logic.get("entry_require_cross", True))
        slope_confirm = bool(logic.get("entry_slope_confirm", True))

        total, journal = simulate_pair(
            df,
            float(thr.get("entry_z", 2.2)),
            float(thr.get("exit_z", 0.5)),
            float(thr.get("stop_z", 3.0)),
            int(zwin),
            float(risk.get("per_trade_pct", 0.0) or 0.0),
            capital=float(risk.get("capital", 100000)),
            costs_bp=int(params.get("costs", {}).get("slippage_bp", 2)),
            cool_off_bars=int(params.get("decision", {}).get("cool_off_bars", 5)),
            min_bars_between_entries=int(params.get("decision", {}).get("min_bars_between_entries", 10)),
            notional_per_trade=float(risk.get("notional_per_trade", 0.0) or 0.0),
            require_cross=require_cross,
            slope_confirm=slope_confirm,
            slope_lookback=int(logic.get("slope_lookback", 3)),
        )
        journal.to_csv(out_dir / f"journal_{a}_{b}.csv")
        rows.append((a, b, hl, zwin, _count_entries(journal["signal"]) if "signal" in journal.columns else 0, float(total)))

    summary = pd.DataFrame([
        {"pair": f"{a}/{b}", "HL(d)": (f"{hl:.1f}" if pd.notna(hl) else "NA"), "z_win": zwin, "entries": entries, "PnL($)": pnl}
        for (a, b, hl, zwin, entries, pnl) in rows
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"\nRésumé Top-{len(top)} (z dynamique = 3×HL, min_window={int(lb.get('zscore_days_min', 12))}):\n")
    print(tabulate(summary, headers="keys", tablefmt="psql"))

if __name__ == "__main__":
    main()
