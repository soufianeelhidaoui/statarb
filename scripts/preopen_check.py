#!/usr/bin/env python3
"""
Pre-open control script for StatArb orders.

Validates the latest (or specified) orders.csv before market open.

Usage:
  python scripts/preopen_check.py                 # auto-detect latest bundle
  python scripts/preopen_check.py --rebalance 2025-09-17
  python scripts/preopen_check.py --exec-date 2025-09-17    # expected open date (Toronto)
  python scripts/preopen_check.py --strict                 # fail on warnings

Exit codes:
  0 = OK (orders valid for execution today)
  1 = WARNING (non-fatal issues)
  2 = ERROR (do NOT execute; missing/invalid)
"""
import sys, argparse, re
from pathlib import Path
import pandas as pd
from datetime import datetime, date
from pandas.tseries.offsets import BDay

ROOT = Path(".")
REPORTS = ROOT / "reports"

REQUIRED_COLS = [
    "rebalance_id","pair","a","b","verdict","action",
    "side_y","side_x","entry_rule","entry_when",
    "target_notional_total","target_notional_leg_y","target_notional_leg_x"
]
VALID_ACTIONS = {"ShortY_LongX","LongY_ShortX"}
VALID_SIDES = {"SHORT_Y","LONG_X","LONG_Y","SHORT_X","HOLD"}

def parse_args():
    ap = argparse.ArgumentParser(description="Pre-open validator for orders.csv")
    ap.add_argument("--rebalance", help="rebalance_id (YYYY-MM-DD). If omitted, uses latest reports/<date>/")
    ap.add_argument("--exec-date", help="Expected execution date (YYYY-MM-DD). Defaults to today.")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    ap.add_argument("--reports-dir", default="reports", help="Reports base directory.")
    return ap.parse_args()

def latest_bundle_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    # find dirs like YYYY-MM-DD, sort desc
    cand = []
    for p in base.iterdir():
        if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name):
            cand.append(p)
    if not cand:
        return None
    return sorted(cand, key=lambda x: x.name, reverse=True)[0]

def next_business_day(d: date) -> date:
    return (pd.Timestamp(d) + BDay(1)).date()

def main():
    args = parse_args()
    reports_base = Path(args.reports_dir)
    # Determine bundle dir
    if args.rebalance:
        bundle = reports_base / args.rebalance
    else:
        bundle = latest_bundle_dir(reports_base)
        if bundle is None:
            print("ERROR: No reports/<YYYY-MM-DD>/ found.")
            sys.exit(2)
    orders_path = bundle / "orders.csv"
    decisions_path = bundle / "decisions.csv"
    print(f"[preopen] Bundle: {bundle}")
    # Check files
    if not orders_path.exists():
        print(f"WARNING: {orders_path} not found. If you expected ENTER signals, rerun analysis.")
        # fall back to decisions to understand why
        if decisions_path.exists():
            dec = pd.read_csv(decisions_path)
            vc = dec["verdict"].value_counts(dropna=False)
            print("[preopen] Decisions verdict counts:\n", vc.to_string())
        sys.exit(1)
    # Load orders
    try:
        df = pd.read_csv(orders_path)
    except Exception as e:
        print(f"ERROR: Cannot read {orders_path}: {e}")
        sys.exit(2)
    if df.empty:
        print("WARNING: orders.csv is empty (no ENTER). Nothing to execute today.")
        sys.exit(1)
    # Column checks
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:", missing)
        sys.exit(2)
    # Action & sides validity
    bad_action = df.loc[~df["action"].isin(VALID_ACTIONS)]
    if not bad_action.empty:
        print("ERROR: Invalid action values:\n", bad_action[["pair","action"]].to_string(index=False))
        sys.exit(2)
    bad_side = df.loc[~df["side_y"].isin(VALID_SIDES) | ~df["side_x"].isin(VALID_SIDES)]
    if not bad_side.empty:
        print("ERROR: Invalid side values:\n", bad_side[["pair","side_y","side_x"]].to_string(index=False))
        sys.exit(2)
    # entry_rule must be next_open
    if not (df["entry_rule"] == "next_open").all():
        print("ERROR: entry_rule must be 'next_open' for all rows.")
        sys.exit(2)
    # Expected execution date
    if args.exec_date:
        exp_date_str = args.exec_date
    else:
        exp_date_str = date.today().strftime("%Y-%m-%d")
    try:
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
    except Exception:
        print("ERROR: --exec-date must be YYYY-MM-DD")
        sys.exit(2)
    # Validate entry_when matches expected date 09:30:00
    bad_when = []
    for i, r in df.iterrows():
        try:
            ew = datetime.strptime(str(r["entry_when"]).strip(), "%Y-%m-%d %H:%M:%S")
            if (ew.date() != exp_date) or (ew.strftime("%H:%M:%S") != "09:30:00"):
                bad_when.append((r["pair"], r["entry_when"]))
        except Exception:
            bad_when.append((r.get("pair","?"), r.get("entry_when","<parse error>")))
    if bad_when:
        print("WARNING: entry_when mismatch for rows (pair, entry_when):")
        for p, w in bad_when:
            print(" -", p, "→", w, "(expected", exp_date.strftime("%Y-%m-%d 09:30:00") + ")")
        if args.strict:
            sys.exit(2)
    # Basic NaN checks on notionals
    num_cols = ["target_notional_total","target_notional_leg_y","target_notional_leg_x"]
    if df[num_cols].isna().any().any():
        print("ERROR: NaNs in notional fields.")
        sys.exit(2)
    # Summary
    pairs = df["pair"].tolist()
    total_notional = df["target_notional_total"].sum()
    print(f"OK: {len(df)} orders ready for execution today {exp_date_str} 09:30.")
    print("Pairs:", ", ".join(pairs))
    print(f"Total target notional: {total_notional:,.2f}")
    sys.exit(0)

if __name__ == "__main__":
    main()
