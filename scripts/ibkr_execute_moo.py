#!/usr/bin/env python3
"""
IBKR Paper Execution for StatArb orders (Market-on-Open)

- Reads the latest (or specified) orders.csv from reports/<rebalance_id>/
- Computes quantities from target notionals using a reference price
  (IBKR snapshot market data when available; fallback to last close from Parquet)
- Places MOO (OPG) orders for both legs of each ENTER signal
- Default is PAPER (TWS paper port 7497). Live is blocked unless --live is explicitly passed.

Usage:
  python scripts/ibkr_execute_moo.py
  python scripts/ibkr_execute_moo.py --rebalance 2025-09-17
  python scripts/ibkr_execute_moo.py --dry-run           # Dry-run (ne soumet pas, affiche le plan)
  python scripts/ibkr_execute_moo.py --live --port 7496  # YOU TAKE RESPONSIBILITY

Requires:
  - ib-insync
  - TWS/Gateway running (Paper: 7497, Live: 7496)
"""

import sys, argparse, re, math, asyncio
from pathlib import Path
from datetime import datetime, date
import pandas as pd

from ib_insync import IB, Stock, util, MarketOrder, Order, Contract

ROOT = Path(".")
REPORTS = ROOT / "reports"

def latest_bundle_dir(base: Path) -> Path | None:
    cand = []
    for p in base.iterdir():
        if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name):
            cand.append(p)
    if not cand:
        return None
    return sorted(cand, key=lambda x: x.name, reverse=True)[0]

def load_orders(bundle: Path) -> pd.DataFrame:
    path = bundle / "orders.csv"
    if not path.exists():
        raise FileNotFoundError(f"orders.csv not found in {bundle}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("orders.csv is empty (no ENTER).")
    required = [
        "rebalance_id","pair","a","b","verdict","action",
        "side_y","side_x","entry_rule","entry_when",
        "target_notional_total","target_notional_leg_y","target_notional_leg_x"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in orders.csv: {missing}")
    # Only ENTER
    df = df[df["verdict"] == "ENTER"].copy()
    if df.empty:
        raise RuntimeError("orders.csv has no ENTER rows.")
    return df

async def snapshot_price(ib: IB, symbol: str) -> float | None:
    """Request a quick snapshot price from IBKR; return last or close if available."""
    try:
        c = Stock(symbol, 'SMART', 'USD')
        # qualify to resolve conId
        qc = await ib.qualifyContractsAsync(c)
        if not qc:
            return None
        t = ib.reqMktData(c, '', snapshot=True, regulatorySnapshot=False)
        # Wait briefly for snapshot
        await util.sleepAsync(2.0)
        # Prefer last price; fallback to close
        price_candidates = [t.last, t.close, t.marketPrice()]
        for p in price_candidates:
            if p and p > 0:
                return float(p)
        return None
    except Exception:
        return None

def parquet_close_price(root_dir: Path, ticker: str) -> float | None:
    try:
        import polars as pl
        fp = Path(root_dir) / f"{ticker}.parquet"
        if not fp.exists():
            return None
        df = pl.read_parquet(fp).select(['date','adj_close','close']).sort('date')
        s = df['adj_close'].fill_null(df['close'])
        return float(s[-1])
    except Exception:
        return None

def compute_qty(notional: float, ref_price: float) -> int:
    if notional <= 0 or not ref_price or ref_price <= 0:
        return 0
    return max(1, int(math.floor(notional / ref_price)))

def build_opg_order(action: str, qty: int) -> Order:
    # Market-On-Open order in IBKR: MKT with TIF='OPG'
    side = 'BUY' if action.upper() in ('BUY','LONG') else 'SELL'
    o = MarketOrder(side, qty)
    o.tif = 'OPG'
    return o

async def place_pair_orders(ib: IB, row: pd.Series, root_dir: Path, dry_run: bool):
    """Place both legs according to sides and target notionals."""
    a = str(row['a'])
    b = str(row['b'])
    side_y = str(row['side_y'])
    side_x = str(row['side_x'])
    notional_y = float(row['target_notional_leg_y'])
    notional_x = float(row['target_notional_leg_x'])

    # Map Y -> a, X -> b
    sy = a
    sx = b

    # Reference prices (prefer IB snapshot)
    py = await snapshot_price(ib, sy)
    px = await snapshot_price(ib, sx)

    # Fallback to Parquet close
    root_dir = Path(root_dir)
    if py is None:
        py = parquet_close_price(root_dir, sy)
    if px is None:
        px = parquet_close_price(root_dir, sx)

    if py is None or px is None:
        print(f"[WARN] Missing reference price (py={py}, px={px}) for {sy}/{sx}; skipping.")
        return

    qty_y = compute_qty(notional_y, py)
    qty_x = compute_qty(notional_x, px)
    if qty_y == 0 or qty_x == 0:
        print(f"[WARN] Zero qty after sizing for {sy}/{sx}; skipping.")
        return

    # Contracts
    cy = Stock(sy, 'SMART', 'USD')
    cx = Stock(sx, 'SMART', 'USD')
    await ib.qualifyContractsAsync(cy, cx)

    # Build orders
    oy = build_opg_order('BUY' if 'LONG' in side_y else 'SELL', qty_y)
    ox = build_opg_order('BUY' if 'LONG' in side_x else 'SELL', qty_x)

    # Optional: OCA group to avoid orphan legs (best-effort; OPG may not fully respect OCA at open)
    oca_group = f"OCA_{a}_{b}_{datetime.now().strftime('%H%M%S')}"
    oy.ocaGroup = oca_group
    ox.ocaGroup = oca_group
    oy.ocaType = 1
    ox.ocaType = 1

    print(f"""[PLAN] {a}/{b}:
  Y={sy} {side_y} qty={qty_y} @ref≈{py:.4f}
  X={sx} {side_x} qty={qty_x} @ref≈{px:.4f}
  tif=OPG (MOO) OCA={oca_group}
""")

    if dry_run:
        return

    ty = ib.placeOrder(cy, oy)
    tx = ib.placeOrder(cx, ox)
    # Wait briefly to get orderIds
    await util.sleepAsync(0.5)
    print(f"[SUBMIT] Orders submitted: Y(orderId={ty.order.orderId}), X(orderId={tx.order.orderId})")

async def main_async(args):
    # Determine bundle
    reports_base = Path(args.reports_dir)
    if args.rebalance:
        bundle = reports_base / args.rebalance
    else:
        bundle = latest_bundle_dir(reports_base)
        if bundle is None:
            print("ERROR: No reports/<YYYY-MM-DD>/ found.")
            sys.exit(2)

    # Load orders
    orders = load_orders(bundle)
    print(f"[exec] Loaded {len(orders)} ENTER rows from {bundle/'orders.csv'}")

    # Sanity date (entry_when)
    today = date.today().strftime("%Y-%m-%d")
    mismatch = orders[~orders['entry_when'].astype(str).str.contains(today)]
    if not mismatch.empty:
        print(f"WARNING: Some orders have entry_when not matching today {today}:")
        print(mismatch[['pair','entry_when']].to_string(index=False))

    # Connect to IBKR
    ib = IB()
    host = args.host
    port = args.port
    clientId = args.client_id
    if not args.live:
        # paper trading default port
        port = args.port or 7497
    else:
        # user explicitly chose live; default to 7496 if not provided
        port = args.port or 7496

    print(f"[exec] Connecting to IBKR at {host}:{port} (clientId={clientId}) {'LIVE' if args.live else 'PAPER'}")
    await ib.connectAsync(host, port, clientId=clientId, readonly=False)
    if not ib.isConnected():
        print("ERROR: Could not connect to IBKR.")
        sys.exit(2)

    try:
        for _, row in orders.iterrows():
            await place_pair_orders(ib, row, args.root_dir, dry_run=args.dry_run)
    finally:
        ib.disconnect()

def parse_args():
    ap = argparse.ArgumentParser(description="IBKR Paper Execution (MOO) for StatArb orders")
    ap.add_argument("--rebalance", help="rebalance_id YYYY-MM-DD; default: latest bundle")
    ap.add_argument("--reports-dir", default="reports", help="Reports base directory")
    ap.add_argument("--root-dir", default="data/eod/ETFs", help="Parquet root for fallback prices")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--client-id", type=int, default=123)
    ap.add_argument("--dry-run", action="store_true", help="Do not submit orders; print plan only")
    ap.add_argument("--live", action="store_true", help="Allow LIVE trading (otherwise PAPER only)")
    return ap.parse_args()

def main():
    args = parse_args()
    if not args.live:
        print("[safety] LIVE disabled. Running in PAPER mode. Use --live to override (at your own risk).")
    util.startLoop()  # ensure event loop
    asyncio.get_event_loop().run_until_complete(main_async(args))

if __name__ == "__main__":
    main()
