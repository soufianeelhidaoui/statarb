#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
import logging
import numpy as np, pandas as pd, polars as pl
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series, _root_dir_for_source
from src.pairs import all_pairs_from_universe, score_pairs
from src.profile import merged_risk
from src.quality import assert_provenance, assert_price_series_ok, assert_pairs_scored_schema, write_qa_log
from src.decisions import decide_pair

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _expected_prov(source: str):
    return {"yahoo","yfinance"} if (source or "yahoo").lower()=="yahoo" else "ibkr"

def _coalesce_meta(dfpl: pl.DataFrame) -> pd.DataFrame:
    cols = ["date","adj_close","close"]
    if "is_ex_div" in dfpl.columns: cols.append("is_ex_div")
    pdf = dfpl.select(cols).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    px = pdf["adj_close"].fillna(pdf["close"])
    out = pd.DataFrame({"px": px.values}, index=pdf["date"].values)
    if "is_ex_div" in pdf.columns: out["is_ex_div"] = pdf["is_ex_div"].astype(bool).values
    return out

def _coalesce_close(meta: pd.DataFrame, name: str) -> pd.Series:
    s = meta["px"].copy(); s.name = name; return s

def _sel_thresholds(params: dict) -> Dict[str,float]:
    sel = params.get("selection",{}); sf = params.get("stats_filters",{})
    return {
        "min_corr": float(sel.get("min_corr", sf.get("min_corr", 0.6))),
        "pval": float(sel.get("pval_coint", sf.get("coint_pval_max", 0.05))),
        "max_hl": float(sel.get("max_half_life_days", sf.get("half_life_max_days", 20.0))),
    }

def _hl_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("half_life","half_life_days","hl"):
        if c in df.columns: return c
    return None

def _extract_key_params(params: dict) -> dict:
    """Extract key parameters for logging/tracking"""
    import json
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
    
    params_file = bundle_dir / "run_params.json"
    with open(params_file, 'w') as f:
        json.dump(key_params, f, indent=2, default=str)
    
    logger.info(f"Parameters saved to: {params_file}")

def _select_pairs(scored: pd.DataFrame, params: dict, topk: int) -> pd.DataFrame:
    th = _sel_thresholds(params)
    need = {"a","b","corr","pval","score"}
    miss = need - set(scored.columns)
    if miss: raise ValueError(f"pairs_scored missing columns: {miss}")
    flt = (scored["corr"]>=th["min_corr"]) & (scored["pval"]<=th["pval"])
    h = _hl_col(scored)
    if h is not None: flt &= (scored[h]<=th["max_hl"])
    return scored.loc[flt].sort_values("score",ascending=False).head(topk).copy()

def _html_table(df: pd.DataFrame, title: str) -> str:
    style = """
    <style>
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:18px}
    h2{margin:6px 0 12px 0}
    table{border-collapse:collapse;width:100%;font-size:14px}
    th,td{padding:8px 10px;border-bottom:1px solid #eaecef;text-align:right}
    th{background:#f6f8fa;position:sticky;top:0}
    td:first-child,th:first-child{text-align:left}
    .enter{color:#0a7f2e;font-weight:600}.exit{color:#9c2b0e;font-weight:600}.hold{color:#667085}
    </style>
    """
    def cls(v):
        v=str(v).upper()
        if "ENTER" in v: return "enter"
        if "EXIT" in v: return "exit"
        return "hold"
    head = "<thead><tr>"+"".join(f"<th>{c}</th>" for c in df.columns)+"</tr></thead>"
    body = []
    for _,r in df.iterrows():
        t=[]
        for c in df.columns:
            v=r[c]
            if c in ("verdict","action"): t.append(f'<td class="{cls(v)}">{v}</td>')
            elif isinstance(v,float) and c in ("z_last","hl","beta","pval"): t.append(f"<td>{v:.4f}</td>")
            elif isinstance(v,float): t.append(f"<td>{v:.2f}</td>")
            else: t.append(f"<td>{v}</td>")
        body.append("<tr>"+"".join(t)+"</tr>")
    table = "<table>"+head+"<tbody>"+"".join(body)+"</tbody></table>"
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body><h2>{title}</h2>{table}</body></html>"

def main():
    logger.info("Starting report generation")
    start_time = datetime.now()
    
    try:
        params = load_params()
        source = params.get("data",{}).get("source","yahoo").lower()
        mode   = params.get("trading",{}).get("mode","paper").lower()
        risk   = merged_risk(params)

        logger.info(f"Source: {source}, mode: {mode}")

        qa_dir = Path("reports/QA"); qa_dir.mkdir(parents=True, exist_ok=True)
        qa_log = qa_dir / f"qa_{source}_{mode}.log"

        tickers = load_universe()
        ensure_universe(params, tickers)
        root = _root_dir_for_source(params)
        assert_provenance(root, _expected_prov(source), params.get("quality",{}).get("require_provenance_match", True), qa_log)

        logger.info(f"Processing {len(tickers)} tickers")

        meta, price_map = {}, {}
        for i, t in enumerate(tickers, start=1):
            try:
                dfpl = get_price_series(root, t).sort("date")
                assert_price_series_ok(dfpl, t, params.get("quality",{}), qa_log)
                m = _coalesce_meta(dfpl)
                meta[t] = m
                price_map[t] = pd.DataFrame({"close": m["px"]})
                logger.info(f"[{i}/{len(tickers)}] {t}: {len(m)} records")
            except Exception as e:
                logger.error(f"[{i}/{len(tickers)}] {t}: Failed - {e}")
                continue

        logger.info("Scoring pairs...")
        pairs = all_pairs_from_universe(tickers)
        lb = params.get("lookbacks",{})
        scored = score_pairs(price_map, pairs, int(lb.get("corr_days",120)), int(lb.get("coint_days",120)))
        assert_pairs_scored_schema(scored, params.get("quality",{}), qa_log)
        logger.info(f"Scored {len(scored)} pairs")

        bundle = Path(params.get("exports",{}).get("reports_dir","reports"))/"bundles"/source/mode/datetime.now().strftime("%Y-%m-%d")
        bundle.mkdir(parents=True, exist_ok=True)

        # Create decisions subfolder and save key parameters for tracking
        decisions_dir = bundle / "decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        _save_run_params(decisions_dir, params, "run_report")

        pl.from_pandas(scored).write_parquet(decisions_dir/"pairs_scored.parquet")

        topk = int(params.get("exports",{}).get("topk",20))
        cand = _select_pairs(scored, params, topk)
        logger.info(f"Selected {len(cand)} candidate pairs")

        decisions: List[Dict] = []
        spy_series = price_map.get("SPY", pd.DataFrame({"close": pd.Series(dtype=float)}))["close"] if "SPY" in price_map else None

        for i, (_, row) in enumerate(cand.reset_index(drop=True).iterrows(), start=1):
            a = str(row["a"]); b = str(row["b"])
            ya = _coalesce_close(meta[a], a); xb = _coalesce_close(meta[b], b)
            d = decide_pair(ya, xb, spy_series, params, meta_a={"df": meta[a].rename(columns={"px":"close"})}, meta_b={"df": meta[b].rename(columns={"px":"close"})})
            d["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            decisions.append(d)
            logger.info(f"[{i}/{len(cand)}] {a}-{b}: {d.get('verdict', 'N/A')}")

        if not decisions:
            logger.warning("No decisions generated")
            return

        dec_df = pd.DataFrame(decisions)
        dec_cols = ["ts","a","b","verdict","action","reason","z_last","hl","beta","pval"]
        dec_out = dec_df[dec_cols].copy()
        dec_out.to_csv(decisions_dir/"decisions.csv", index=False)

        html = _html_table(dec_out, f"Decisions — {source.upper()} / {mode} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        (decisions_dir/"decisions.html").write_text(html, encoding="utf-8")

        orders = []
        for _, r in dec_df.iterrows():
            if r["verdict"] not in ("ENTER","EXIT"): continue
            a = str(r["a"]); b = str(r["b"])
            pa = float(meta[a]["px"].iloc[-1]); pb = float(meta[b]["px"].iloc[-1])
            beta = float(r.get("beta", 1.0) if pd.notna(r.get("beta", np.nan)) else 1.0)
            notional = float(risk.get("notional_per_trade", 0.0) or 0.0)
            if notional<=0 and float(risk.get("per_trade_pct",0.0) or 0.0)>0:
                notional = float(risk.get("capital",0.0) or 0.0) * float(risk.get("per_trade_pct",0.0))
            qa = int(max(0, np.floor((notional/2.0)/max(pa,1e-9))))
            qb = int(max(0, np.floor((notional/2.0)/max(pb,1e-9))))
            action = str(r["action"])
            if action == "ShortY_LongX": side_a, side_b = "SELL_A", "BUY_B"
            elif action == "LongY_ShortX": side_a, side_b = "BUY_A", "SELL_B"
            else: side_a, side_b = "CLOSE_A", "CLOSE_B"
            orders.append({
                "ts": r["ts"], "a": a, "b": b, "verdict": r["verdict"], "action": action, "reason": r["reason"],
                "price_a": pa, "price_b": pb, "qty_a": qa, "qty_b": qb, "side_a": side_a, "side_b": side_b
            })
        
        pd.DataFrame(orders).to_csv(decisions_dir/"orders.csv", index=False)
        
        elapsed = datetime.now() - start_time
        enter_count = sum(1 for d in decisions if d.get("verdict") == "ENTER")
        exit_count = sum(1 for d in decisions if d.get("verdict") == "EXIT")
        
        logger.info(f"Completed in {elapsed.total_seconds():.1f}s")
        logger.info(f"Results: {len(decisions)} decisions ({enter_count} ENTER, {exit_count} EXIT), {len(orders)} orders")
        logger.info(f"Bundle: {bundle}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1

if __name__ == "__main__":
    main()