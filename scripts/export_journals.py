#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import polars as pl

from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series, _root_dir_for_source
from src.pairs import all_pairs_from_universe, score_pairs
from src.quality import assert_provenance, assert_price_series_ok, assert_pairs_scored_schema, write_qa_log

def _coalesce(dfpl: pl.DataFrame) -> pd.DataFrame:
    cols = ["date","adj_close","close"]
    if "is_ex_div" in dfpl.columns:
        cols.append("is_ex_div")
    pdf = dfpl.select(cols).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    px = pdf["adj_close"].fillna(pdf["close"])
    out = pd.DataFrame({"px": px.values}, index=pdf["date"].values)
    if "is_ex_div" in pdf.columns:
        out["is_ex_div"] = pdf["is_ex_div"].astype(bool).values
    return out

def main():
    params = load_params()
    source = params.get("data",{}).get("source","yahoo").lower()
    mode   = params.get("trading",{}).get("mode","paper").lower()

    qa_log = Path("reports/QA") / f"qa_{source}_{mode}.log"
    tickers = load_universe()
    root = _root_dir_for_source(params)
    ensure_universe(params, tickers)

    expected = {"yahoo","yfinance"} if source=="yahoo" else "ibkr"
    assert_provenance(root, expected, params.get("quality",{}).get("require_provenance_match", True), qa_log)

    price_map = {}
    meta = {}
    for t in tickers:
        dfpl = get_price_series(root, t).sort("date")
        assert_price_series_ok(dfpl, t, params.get("quality",{}), qa_log)
        meta[t] = _coalesce(dfpl)
        price_map[t] = pd.DataFrame({"close": meta[t]["px"]})

    pairs = all_pairs_from_universe(tickers)
    lb = params.get("lookbacks",{})
    scored = score_pairs(price_map, pairs, int(lb.get("corr_days",120)), int(lb.get("coint_days",120)))
    assert_pairs_scored_schema(scored, params.get("quality",{}), qa_log)

    # Top N
    sel = params.get("selection",{})
    min_corr = float(sel.get("min_corr", 0.55))
    pmax = float(sel.get("pval_coint", 0.10))
    hlmax = float(sel.get("max_half_life_days", 30))
    flt = (scored["corr"]>=min_corr) & (scored["pval"]<=pmax)
    if "half_life" in scored.columns:
        flt &= (scored["half_life"]<=hlmax)
    topk = int(params.get("exports",{}).get("topk", 20))
    top = scored.loc[flt].sort_values("score", ascending=False).head(topk)

    # répertoire daté
    day = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(params.get("exports",{}).get("journals_dir","reports/journals"))/source/mode/day
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, r in top.iterrows():
        a, b = str(r["a"]), str(r["b"])
        dfa = meta[a]; dfb = meta[b]
        j = pd.DataFrame(index = dfa.index.intersection(dfb.index))
        j["ya"] = dfa["px"].reindex(j.index)
        j["xb"] = dfb["px"].reindex(j.index)
        if "is_ex_div" in dfa.columns:
            j["is_ex_div_a"] = dfa["is_ex_div"].reindex(j.index, fill_value=False).astype(bool)
        if "is_ex_div" in dfb.columns:
            j["is_ex_div_b"] = dfb["is_ex_div"].reindex(j.index, fill_value=False).astype(bool)

        if ("is_ex_div_a" in j.columns) or ("is_ex_div_b" in j.columns):
            ma = j.get("is_ex_div_a", pd.Series(False, index=j.index)).astype(bool)
            mb = j.get("is_ex_div_b", pd.Series(False, index=j.index)).astype(bool)
            m = ~(ma | ma.shift(1).fillna(False) | mb | mb.shift(1).fillna(False))
            j = j.loc[m.values].copy()

        # beta & spread & z
        import numpy as np
        X = np.vstack([np.ones(len(j)), j["xb"].values]).T
        bcoef = np.linalg.lstsq(X, j["ya"].values, rcond=None)[0]
        beta = float(bcoef[1])
        spread = j["ya"] - beta * j["xb"]
        zwin = max(int(lb.get("zscore_days_min",12)), 60 if "half_life" not in r or pd.isna(r["half_life"]) else int(round(3*max(2.0, float(r["half_life"])))))
        z = (spread - spread.rolling(zwin).mean()) / spread.rolling(zwin).std(ddof=1)

        out = pd.DataFrame({
            "date": j.index.date,
            "ya": j["ya"].astype(float),
            "xb": j["xb"].astype(float),
            "is_ex_div_a": j.get("is_ex_div_a", pd.Series(False, index=j.index)).astype(bool).values,
            "is_ex_div_b": j.get("is_ex_div_b", pd.Series(False, index=j.index)).astype(bool).values,
            "beta": beta,
            "spread": spread.astype(float),
            "z": z.astype(float),
        })
        path = out_dir / f"journal_{a}_{b}.csv"
        out.to_csv(path, index=False)
        print(f"[journal] {a}_{b} → {path}")

    print("[export_journals] Terminé.")
if __name__ == "__main__":
    main()
