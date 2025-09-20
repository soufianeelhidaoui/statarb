#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse, pandas as pd, polars as pl, numpy as np
import logging
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, get_price_series, _root_dir_for_source
from src.pairs import all_pairs_from_universe, score_pairs
from src.quality import assert_provenance, assert_price_series_ok, assert_pairs_scored_schema, write_qa_log

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _expected_prov(source: str):
    return {"yahoo","yfinance"} if (source or "yahoo").lower()=="yahoo" else "ibkr"

def _coalesce(dfpl: pl.DataFrame) -> pd.DataFrame:
    cols = ["date","adj_close","close"]
    if "is_ex_div" in dfpl.columns: cols.append("is_ex_div")
    pdf = dfpl.select(cols).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["px"] = pdf["adj_close"].fillna(pdf["close"])
    pdf = pdf.set_index("date")
    return pdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting journal export")
    start_time = datetime.now()

    try:
        params = load_params()
        src = params.get("data",{}).get("source","yahoo").lower()
        mode = params.get("trading",{}).get("mode","paper").lower()

        logger.info(f"Source: {src}, mode: {mode}")

        tickers = load_universe()
        ensure_universe(params, tickers)
        root = _root_dir_for_source(params)
        qa_log = Path("reports/QA")/f"qa_{src}_{mode}.log"
        assert_provenance(root, _expected_prov(src), params.get("quality",{}).get("require_provenance_match", True), qa_log)

        logger.info(f"Processing {len(tickers)} tickers")

        price_map, meta = {}, {}
        for i, t in enumerate(tickers, start=1):
            try:
                dfpl = get_price_series(root, t).sort("date")
                assert_price_series_ok(dfpl, t, params.get("quality",{}), qa_log)
                m = _coalesce(dfpl)
                meta[t] = m
                price_map[t] = pd.DataFrame({"close": m["px"]})
                logger.debug(f"[{i}/{len(tickers)}] {t}: {len(m)} records")
            except Exception as e:
                logger.error(f"[{i}/{len(tickers)}] {t}: Failed - {e}")
                continue

        logger.info("Scoring pairs...")
        lb = params.get("lookbacks",{})
        scored = score_pairs(price_map, all_pairs_from_universe(tickers), int(lb.get("corr_days",120)), int(lb.get("coint_days",120)))
        assert_pairs_scored_schema(scored, params.get("quality",{}), qa_log)
        logger.info(f"Scored {len(scored)} pairs")

        day = datetime.now().strftime("%Y-%m-%d")
        bundle = Path(args.bundle) if args.bundle else Path("reports/bundles")/src/mode/day
        (bundle/"journals").mkdir(parents=True, exist_ok=True)

        qual = params.get("quality",{})
        mask_flag = bool(qual.get("mask_ex_div", True))
        after = int(qual.get("mask_ex_div_days_after", 1))

        top = scored.sort_values("score",ascending=False).head(12)
        logger.info(f"Exporting journals for top {len(top)} pairs")

        for i, (_, r) in enumerate(top.iterrows(), start=1):
            a, b = str(r["a"]), str(r["b"])
            try:
                A = meta[a]; B = meta[b]
                j = pd.concat([A[["px", *([ "is_ex_div"] if "is_ex_div" in A.columns else [])]].rename(columns={"px":"ya"}),
                               B[["px", *([ "is_ex_div"] if "is_ex_div" in B.columns else [])]].rename(columns={"px":"xb"})],
                              axis=1, join="inner").dropna(subset=["ya","xb"])
                
                if mask_flag:
                    ma_flag = A["is_ex_div"] if "is_ex_div" in A.columns else pd.Series(False, index=A.index, dtype=bool)
                    mb_flag = B["is_ex_div"] if "is_ex_div" in B.columns else pd.Series(False, index=B.index, dtype=bool)
                    ma_flag = ma_flag.reindex(j.index, fill_value=False).astype(bool)
                    mb_flag = mb_flag.reindex(j.index, fill_value=False).astype(bool)
                    
                    # Build mask efficiently without triggering pandas warnings
                    mask = ~(ma_flag | mb_flag)
                    
                    # Apply shifts for additional days after ex-div
                    for k in range(1, after + 1):
                        ma_shifted = ma_flag.shift(k, fill_value=False)
                        mb_shifted = mb_flag.shift(k, fill_value=False)
                        mask = mask & ~(ma_shifted | mb_shifted)
                    
                    j = j.loc[mask]
                
                cov = j["ya"].cov(j["xb"]); var = j["xb"].var()
                beta = (cov/var) if (var and var!=0) else 1.0
                roll = max(int(lb.get("zscore_days_min",12)), 60) if pd.isna(r.get("half_life",np.nan)) else max(int(lb.get("zscore_days_min",12)), int(round(float(r["half_life"])*float(lb.get("zscore_mult_half_life",3.0)))))
                spread = j["ya"] - beta*j["xb"]
                m = spread.rolling(roll).mean(); s = spread.rolling(roll).std(ddof=1)
                z = (spread - m) / s.replace(0.0, np.nan)
                out = pd.DataFrame({"date": j.index.date, "ya": j["ya"].astype(float), "xb": j["xb"].astype(float), "beta": float(beta), "spread": spread.astype(float), "z": z.astype(float)})
                out.to_csv(bundle/"journals"/f"journal_{a}_{b}.csv", index=False)
                logger.info(f"[{i}/{len(top)}] {a}-{b}: Exported {len(out)} records")
            except Exception as e:
                logger.error(f"[{i}/{len(top)}] {a}-{b}: Failed - {e}")
                continue

        elapsed = datetime.now() - start_time
        logger.info(f"Completed in {elapsed.total_seconds():.1f}s")
        logger.info(f"Bundle: {bundle}")

    except Exception as e:
        logger.error(f"Journal export failed: {e}")
        return 1

if __name__ == "__main__":
    main()