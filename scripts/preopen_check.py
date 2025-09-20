#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, pandas as pd, yaml, logging
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_params(p="config/params.yaml")->dict:
    with open(p,"r") as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bdir = Path(args.bundle)
    logger.info(f"Pre-market check for bundle: {bdir}")

    dec = bdir/"decisions.csv"
    ords= bdir/"orders.csv"
    
    # Load and validate files
    if not dec.exists():
        logger.warning(f"Decisions file not found: {dec}")
        D = pd.DataFrame()
    else:
        D = pd.read_csv(dec)
        logger.debug(f"Loaded {len(D)} decisions from {dec}")
    
    if not ords.exists():
        logger.warning(f"Orders file not found: {ords}")
        O = pd.DataFrame()
    else:
        O = pd.read_csv(ords)
        logger.debug(f"Loaded {len(O)} orders from {ords}")

    # Filter relevant data
    D = D[D["verdict"].isin(["ENTER","EXIT","HOLD"])] if not D.empty else D
    O = O[O["verdict"].isin(["ENTER","EXIT"])] if not O.empty else O

    # Summary counts
    enter_count = len(D[D["verdict"] == "ENTER"]) if not D.empty else 0
    exit_count = len(D[D["verdict"] == "EXIT"]) if not D.empty else 0
    hold_count = len(D[D["verdict"] == "HOLD"]) if not D.empty else 0
    order_count = len(O)

    logger.info(f"Summary: {enter_count} ENTER, {exit_count} EXIT, {hold_count} HOLD decisions → {order_count} orders")

    # Display tables
    print(f"\n{'='*60}")
    print(f"PRE-MARKET CHECK - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Bundle: {bdir}")
    print(f"{'='*60}")
    
    print(f"\nDECISIONS ({len(D)} total):")
    if D.empty:
        print("  No decisions to review")
    else:
        print(D[["a","b","verdict","action","reason","z_last","hl","beta","pval"]].to_string(index=False))
    
    print(f"\nORDERS TO EXECUTE ({len(O)} total):")
    if O.empty:
        print("  No orders to execute")
        logger.info("All clear - no orders to execute")
    else:
        print(O[["a","b","side_a","qty_a","side_b","qty_b"]].to_string(index=False))
        
        # Risk warnings
        total_notional = (O["qty_a"] * O["price_a"] + O["qty_b"] * O["price_b"]).sum() if "price_a" in O.columns else 0
        if total_notional > 0:
            logger.info(f"Total notional to trade: ${total_notional:,.0f}")
        
        if len(O) > 5:
            logger.warning(f"High order count: {len(O)} orders (review carefully)")

    print(f"{'='*60}\n")

    # Final validation
    if not O.empty:
        logger.info("Ready for market execution - review orders above")
        return 0
    else:
        logger.info("No execution needed today")
        return 0

if __name__ == "__main__":
    exit(main())