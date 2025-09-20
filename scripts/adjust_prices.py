#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _load_params(path="config/params.yaml") -> dict:
    import yaml
    logger.info(f"Loading configuration from {path}")
    try:
        with open(path,"r") as f:
            params = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def _yahoo_series(ticker: str, start=None, end=None):
    import yfinance as yf
    logger.debug(f"Fetching Yahoo data for {ticker} from {start} to {end}")
    try:
        y = yf.Ticker(ticker)
        if start is not None or end is not None:
            hist = y.history(start=start, end=end, auto_adjust=False)
        else:
            hist = y.history(period="max", auto_adjust=False)
        
        if hist is None or hist.empty:
            raise RuntimeError(f"Yahoo returned empty history for {ticker}")
        
        logger.debug(f"Downloaded {len(hist)} records for {ticker}")
        
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        df = pd.DataFrame({
            "Close": hist["Close"].astype(float), 
            "AdjClose": hist["Adj Close"].astype(float)
        }, index=hist.index)
        
        div = y.dividends
        try: 
            div = div.tz_localize(None)
        except Exception: 
            pass
        
        logger.debug(f"Retrieved {len(div) if div is not None else 0} dividend records for {ticker}")
        return df, div
        
    except Exception as e:
        logger.error(f"Failed to fetch Yahoo data for {ticker}: {e}")
        raise

def _should_rebuild(pdf: pd.DataFrame, force: bool) -> bool:
    if force: 
        logger.debug("Force rebuild requested")
        return True
    
    have_adj = "adj_close" in pdf.columns
    have_exd = "is_ex_div" in pdf.columns
    
    if not have_adj or not have_exd:
        logger.debug(f"Missing columns: adj_close={have_adj}, is_ex_div={have_exd}")
        return True
    
    # Check for excessive NaN values in adj_close
    nan_ratio = float(pd.isna(pdf["adj_close"]).mean()) if "adj_close" in pdf.columns else 1.0
    if nan_ratio > 0.01:
        logger.debug(f"High NaN ratio in adj_close: {nan_ratio:.2%}")
        return True
    
    logger.debug("Data quality check passed, no rebuild needed")
    return False

def _process_one(root_dir: Path, ticker: str, force: bool) -> tuple[bool, str]:
    fp = root_dir / f"{ticker}.parquet"
    
    if not fp.exists():
        return False, "File not found"

    try:
        # Load existing data
        df = pl.read_parquet(fp).with_columns(pl.col("date").cast(pl.Date))
        pdf = df.to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.set_index("date")
        
        logger.debug(f"Loaded {len(pdf)} records for {ticker}")

        if "close" not in pdf.columns:
            return False, "Missing close column"

        if not _should_rebuild(pdf, force):
            return True, "Up-to-date"

        # Fetch Yahoo data for adjustment factors
        start = (pdf.index.min() - pd.Timedelta(days=1)).date()
        end   = (pdf.index.max() + pd.Timedelta(days=1)).date()

        yh, div = _yahoo_series(ticker, start=start, end=end)
        
        # Calculate adjustment factor
        factor = (yh["AdjClose"] / yh["Close"]).replace([np.inf, -np.inf], np.nan).dropna()
        factor = factor.to_frame("factor").sort_index()
        logger.debug(f"Calculated adjustment factors for {len(factor)} dates")

        # Apply adjustments
        aligned = pdf.join(factor, how="left")
        aligned["factor"] = aligned["factor"].ffill()
        aligned["adj_close"] = aligned["close"] * aligned["factor"]

        # Mark ex-dividend dates
        is_ex = pd.Series(False, index=aligned.index)
        if div is not None and len(div) > 0:
            ex_dates = set(pd.to_datetime(div.index).normalize())
            is_ex = aligned.index.normalize().isin(ex_dates)
            logger.debug(f"Marked {is_ex.sum()} ex-dividend dates for {ticker}")
        aligned["is_ex_div"] = is_ex.astype(bool)

        # Prepare output
        keep = [c for c in ["open","high","low","close","volume"] if c in aligned.columns]
        out = pd.DataFrame(index=aligned.index)
        out["date"] = out.index.date
        for c in keep:
            out[c] = aligned[c].astype(float)
        out["adj_close"] = aligned["adj_close"].astype(float)
        out["is_ex_div"] = aligned["is_ex_div"].astype(bool)

        # Write back to parquet
        pl.from_pandas(out.reset_index(drop=True)).write_parquet(fp)
        
        adj_count = (~pd.isna(out["adj_close"])).sum()
        ex_div_count = out["is_ex_div"].sum()
        
        return True, f"Updated ({adj_count} adj, {ex_div_count} ex-div)"
        
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser(description="Adjust ETF prices with dividends and splits")
    ap.add_argument("--config", default="config/params.yaml", help="Configuration file path")
    ap.add_argument("--force", action="store_true", help="Force rebuild even if data exists")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Starting price adjustment process")
    start_time = datetime.now()

    try:
        params = _load_params(args.config)
        src = (params.get("data",{}).get("source","yahoo") or "yahoo").lower()
        root_dir = Path(params["data"].get("root_dir_ibkr" if src=="ibkr" else "root_dir_yahoo"))

        logger.info(f"Data source: {src}, root: {root_dir}")
        
        if not root_dir.exists():
            logger.error(f"Root directory does not exist: {root_dir}")
            return 1

        tickers = params.get("universe",{}).get("tickers",[])
        if not tickers:
            logger.warning("No tickers found in configuration")
            return 0
            
        logger.info(f"Processing {len(tickers)} tickers")

        success_count = 0
        failed_tickers = []

        for i, ticker in enumerate(tickers, start=1):
            success, status = _process_one(root_dir, ticker, args.force)
            logger.info(f"[{i}/{len(tickers)}] {ticker}: {status}")
            
            if success:
                success_count += 1
            else:
                failed_tickers.append(ticker)

        # Summary
        elapsed = datetime.now() - start_time
        logger.info(f"Completed in {elapsed.total_seconds():.1f}s - {success_count}/{len(tickers)} successful")
        
        if failed_tickers:
            logger.warning(f"Failed: {failed_tickers}")
            return 1
        else:
            logger.info("All tickers processed successfully")
            return 0

    except Exception as e:
        logger.error(f"Price adjustment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())