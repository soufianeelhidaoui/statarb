#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series, _root_dir_for_source
from src.pairs import all_pairs_from_universe, score_pairs
from src.profile import merged_risk
from src.quality import (
    assert_provenance, assert_price_series_ok, assert_pairs_scored_schema, write_qa_log
)
from src.decisions import decide_pair

def _expected_prov(source: str):
    s = (source or "yahoo").lower()
    return {"yahoo", "yfinance"} if s == "yahoo" else "ibkr"

def _coalesce_meta(dfpl: pl.DataFrame) -> pd.DataFrame:
    cols = ["date", "adj_close", "close"]
    if "is_ex_div" in dfpl.columns:
        cols.append("is_ex_div")
    pdf = dfpl.select(cols).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    px = pdf["adj_close"].fillna(pdf["close"])
    out = pd.DataFrame({"px": px.values}, index=pdf["date"].values)
    if "is_ex_div" in pdf.columns:
        out["is_ex_div"] = pdf["is_ex_div"].astype(bool).values
    return out

def _coalesce_close_series(meta: pd.DataFrame, name: str) -> pd.Series:
    s = meta["px"].copy()
    s.name = name
    return s

def _selection_thresholds(params: dict) -> Dict[str, float]:
    sel = params.get("selection", {})
    sf  = params.get("stats_filters", {})
    return {
        "min_corr": float(sel.get("min_corr", sf.get("min_corr", 0.6 if "min_corr" in sf else 0.6))),
        "pval_coint": float(sel.get("pval_coint", sf.get("coint_pval_max", 0.05))),
        "max_hl": float(sel.get("max_half_life_days", sf.get("half_life_max_days", 20.0))),
    }

def _detect_hl_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("half_life", "half_life_days", "hl"):
        if c in df.columns:
            return c
    return None

def _select_pairs_for_decision(scored: pd.DataFrame, params: dict, topk: int) -> pd.DataFrame:
    th = _selection_thresholds(params)
    need = {"a", "b", "corr", "pval", "score"}
    miss = need - set(scored.columns)
    if miss:
        raise ValueError(f"pairs_scored missing columns: {miss}")
    flt = (scored["corr"] >= th["min_corr"]) & (scored["pval"] <= th["pval_coint"])
    hl_col = _detect_hl_col(scored)
    if hl_col is not None:
        flt &= (scored[hl_col] <= th["max_hl"])
    return scored.loc[flt].sort_values("score", ascending=False).head(topk).copy()

def _size_legs(price_a: float, price_b: float, beta: float,
               notional_per_trade: float | None, per_trade_pct: float | None, capital: float) -> Tuple[int, int]:
    if notional_per_trade is not None and notional_per_trade > 0:
        N = float(notional_per_trade)
    elif per_trade_pct is not None and per_trade_pct > 0:
        N = float(capital) * float(per_trade_pct)
    else:
        N = 0.0
    if N <= 0:
        return (0, 0)
    qa = int(max(0, np.floor((N / 2.0) / max(price_a, 1e-9))))
    qb = int(max(0, np.floor((N / 2.0) / max(price_b, 1e-9))))
    return (qa, qb)

def _action_to_legs(action: str) -> Tuple[str, str]:
    if action == "ShortY_LongX":
        return ("SELL_A", "BUY_B")
    if action == "LongY_ShortX":
        return ("BUY_A", "SELL_B")
    if action.startswith("Sortie"):
        return ("CLOSE_A", "CLOSE_B")
    return ("NONE", "NONE")

def _html_table(df: pd.DataFrame, title: str) -> str:
    style = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 18px; }
    h2 { margin: 6px 0 12px 0; }
    table { border-collapse: collapse; width: 100%; font-size: 14px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #eaecef; text-align: right; }
    th { background: #f6f8fa; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    .enter { color: #0a7f2e; font-weight: 600; }
    .exit { color: #9c2b0e; font-weight: 600; }
    .hold { color: #667085; }
    </style>
    """
    def cls(v):
        v = str(v).upper()
        if "ENTER" in v: return "enter"
        if "EXIT" in v:  return "exit"
        return "hold"
    rows = []
    rows.append("<table>")
    rows.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead>")
    rows.append("<tbody>")
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            val = r[c]
            if c in ("verdict", "action"):
                cells.append(f'<td class="{cls(val)}">{val}</td>')
            elif isinstance(val, float):
                if c in ("z_last","hl","beta","pval"):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{val:.2f}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body><h2>{title}</h2>" + "\n".join(rows) + "</body></html>"

def main():
    params = load_params()
    source = params.get("data", {}).get("source", "yahoo").lower()
    mode   = params.get("trading", {}).get("mode", "paper").lower()
    risk   = merged_risk(params)

    qa_dir = Path("reports/QA"); qa_dir.mkdir(parents=True, exist_ok=True)
    qa_log = qa_dir / f"qa_{source}_{mode}.log"

    tickers = load_universe()
    print(f"[run_report] source={source} mode={mode} tickers={len(tickers)}")

    ensure_universe(params, tickers)
    root = _root_dir_for_source(params)
    assert_provenance(root, _expected_prov(source), params.get("quality",{}).get("require_provenance_match", True), qa_log)

    meta: Dict[str, pd.DataFrame] = {}
    price_map: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {t} …", flush=True)
        dfpl = get_price_series(root, t).sort("date")
        assert_price_series_ok(dfpl, t, params.get("quality",{}), qa_log)
        m = _coalesce_meta(dfpl)
        meta[t] = m
        price_map[t] = pd.DataFrame({"close": m["px"]})

    pairs = all_pairs_from_universe(tickers)
    lb = params.get("lookbacks", {})
    scored = score_pairs(price_map, pairs, int(lb.get("corr_days",120)), int(lb.get("coint_days",120)))
    assert_pairs_scored_schema(scored, params.get("quality",{}), qa_log)

    ts_day = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    bundle = Path("reports") / source / mode / ts_day
    bundle.mkdir(parents=True, exist_ok=True)

    pl.from_pandas(scored).write_parquet(bundle / "pairs_scored.parquet")

    topk = int(params.get("exports", {}).get("topk", 20))
    cand = _select_pairs_for_decision(scored, params, topk)

    decisions: List[Dict] = []
    spy_series = price_map.get("SPY", pd.DataFrame({"close": pd.Series(dtype=float)}))["close"] if "SPY" in price_map else None

    for _, row in cand.reset_index(drop=True).iterrows():
        a = str(row["a"]); b = str(row["b"])
        ya = _coalesce_close_series(meta[a], a)
        xb = _coalesce_close_series(meta[b], b)
        d = decide_pair(
            ya, xb, spy_series, params,
            meta_a={"df": meta[a].rename(columns={"px":"close"})},
            meta_b={"df": meta[b].rename(columns={"px":"close"})}
        )
        d["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        decisions.append(d)

    dec_df = pd.DataFrame(decisions)
    if dec_df.empty:
        print("[run_report] aucune décision")
        return

    dec_cols = ["ts","a","b","verdict","action","reason","z_last","hl","beta","pval"]
    dec_out = dec_df[dec_cols].copy()

    decisions_csv = bundle / "decisions.csv"
    orders_csv    = bundle / "orders.csv"
    decisions_html = bundle / "decisions.html"

    dec_out.to_csv(decisions_csv, index=False)
    html = _html_table(dec_out, f"Decisions — {source.upper()} / {mode} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    decisions_html.write_text(html, encoding="utf-8")

    orders = []
    for _, r in dec_df.iterrows():
        if r["verdict"] not in ("ENTER","EXIT"):
            continue
        a = str(r["a"]); b = str(r["b"])
        pa = float(meta[a]["px"].iloc[-1]); pb = float(meta[b]["px"].iloc[-1])
        beta = float(r.get("beta", 1.0) if pd.notna(r.get("beta", np.nan)) else 1.0)
        qty_a, qty_b = _size_legs(
            pa, pb, beta,
            notional_per_trade = float(risk.get("notional_per_trade", 0.0) or 0.0),
            per_trade_pct      = float(risk.get("per_trade_pct", 0.0) or 0.0),
            capital            = float(risk.get("capital", 0.0) or 0.0),
        )
        side_a, side_b = ("NONE", "NONE")
        act = str(r["action"])
        if act == "ShortY_LongX":
            side_a, side_b = ("SELL_A", "BUY_B")
        elif act == "LongY_ShortX":
            side_a, side_b = ("BUY_A", "SELL_B")
        elif act.startswith("Sortie"):
            side_a, side_b = ("CLOSE_A", "CLOSE_B")

        orders.append({
            "ts": r["ts"],
            "a": a, "b": b,
            "verdict": r["verdict"],
            "action": r["action"],
            "reason": r["reason"],
            "price_a": pa, "price_b": pb,
            "qty_a": int(qty_a), "qty_b": int(qty_b),
            "side_a": side_a, "side_b": side_b,
        })

    pd.DataFrame(orders).to_csv(orders_csv, index=False)

    print(f"[run_report] bundle → {bundle}")
    print(f"[run_report] decisions → {decisions_csv}")
    print(f"[run_report] orders    → {orders_csv}")
    print(f"[run_report] html      → {decisions_html}")

if __name__ == "__main__":
    main()
