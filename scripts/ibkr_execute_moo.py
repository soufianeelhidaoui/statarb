#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, time, yaml, pandas as pd
import logging

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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    params = load_params()
    mode = params.get("trading", {}).get("mode", "paper").upper()
    
    logger.info(f"IBKR execution [{mode}] {'(DRY-RUN)' if args.dry_run else '(LIVE)'} for bundle: {args.bundle}")

    try:
        df = pd.read_csv(Path(args.bundle)/"orders.csv") if (Path(args.bundle)/"orders.csv").exists() else pd.DataFrame()
        df = df[df["verdict"].isin(["ENTER","EXIT"])].copy() if not df.empty else df

        if df.empty:
            logger.info("No orders to execute")
            return 0

        host = params.get("execution",{}).get("ib",{}).get("host","127.0.0.1")
        port = int(params.get("execution",{}).get("ib",{}).get(
            "port_paper", 7497 if params.get("trading",{}).get("mode","paper")=="paper" else params.get("execution",{}).get("ib",{}).get("port_live",7496)
        ))
        client_id = int(params.get("execution",{}).get("ib",{}).get("client_id", 23))
        allow_fractional = bool(params.get("execution",{}).get("ib",{}).get("allow_fractional", True))

        logger.info(f"IBKR connection: {host}:{port} (client_id={client_id})")
        logger.info(f"Loaded {len(df)} orders for execution")

        if args.dry_run:
            logger.info("DRY-RUN mode - orders will NOT be executed")
            for i, r in enumerate(df.iterrows(), start=1):
                _, row = r
                a, b = str(row["a"]), str(row["b"])
                sa, sb = str(row["side_a"]), str(row["side_b"])
                qa, qb = float(row["qty_a"]), float(row["qty_b"])
                
                # For CLOSE orders, show that we would check positions
                order_info = []
                if sa in ("BUY_A", "SELL_A"):
                    order_info.append(f"{sa.split('_')[0]} {int(qa)} {a}")
                elif sa == "CLOSE_A":
                    order_info.append(f"CLOSE {a} (would check current position)")
                
                if sb in ("BUY_B", "SELL_B"):
                    order_info.append(f"{sb.split('_')[0]} {int(qb)} {b}")
                elif sb == "CLOSE_B":
                    order_info.append(f"CLOSE {b} (would check current position)")
                
                logger.info(f"[{i}/{len(df)}] {a}-{b}: {row['action']} → {' | '.join(order_info)} @ MOO")
            logger.info("DRY-RUN completed - no orders sent")
            return 0

        from ib_insync import IB, Stock, MarketOrder
        ib = IB()
        
        try:
            ib.connect(host, port, clientId=client_id)
            logger.info("Connected to IBKR TWS/Gateway")
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return 1

        def sym(t): return Stock(t, 'SMART', 'USD', primaryExchange='ARCA')
        
        def get_position_size(ticker):
            """Get current position size for a ticker. Returns 0 if no position."""
            try:
                positions = ib.positions()
                for pos in positions:
                    if pos.contract.symbol == ticker and pos.contract.secType == 'STK':
                        return pos.position
                return 0.0
            except Exception as e:
                logger.warning(f"Failed to get position for {ticker}: {e}")
                return 0.0
        
        orders_sent = 0
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            try:
                a,b = str(r["a"]), str(r["b"])
                qa,qb = float(r["qty_a"]), float(r["qty_b"])
                if not allow_fractional:
                    qa,qb = int(round(qa)), int(round(qb))
                sa,sb = str(r["side_a"]), str(r["side_b"])

                orders_this_pair = 0
                if sa in ("BUY_A","SELL_A"):
                    ib.placeOrder(sym(a), MarketOrder("BUY" if sa=="BUY_A" else "SELL", qa, tif='OPG'))
                    orders_this_pair += 1
                if sb in ("BUY_B","SELL_B"):
                    ib.placeOrder(sym(b), MarketOrder("BUY" if sb=="BUY_B" else "SELL", qb, tif='OPG'))
                    orders_this_pair += 1
                if sa=="CLOSE_A":
                    current_pos_a = get_position_size(a)
                    if current_pos_a != 0:
                        close_side = "SELL" if current_pos_a > 0 else "BUY"
                        close_qty = abs(current_pos_a)
                        if not allow_fractional:
                            close_qty = int(round(close_qty))
                        ib.placeOrder(sym(a), MarketOrder(close_side, close_qty, tif='OPG'))
                        orders_this_pair += 1
                        logger.info(f"CLOSE_A for {a}: {close_side} {close_qty} (current position: {current_pos_a})")
                    else:
                        logger.info(f"CLOSE_A for {a}: No position to close")
                
                if sb=="CLOSE_B":
                    current_pos_b = get_position_size(b)
                    if current_pos_b != 0:
                        close_side = "SELL" if current_pos_b > 0 else "BUY"
                        close_qty = abs(current_pos_b)
                        if not allow_fractional:
                            close_qty = int(round(close_qty))
                        ib.placeOrder(sym(b), MarketOrder(close_side, close_qty, tif='OPG'))
                        orders_this_pair += 1
                        logger.info(f"CLOSE_B for {b}: {close_side} {close_qty} (current position: {current_pos_b})")
                    else:
                        logger.info(f"CLOSE_B for {b}: No position to close")

                orders_sent += orders_this_pair
                logger.info(f"[{i}/{len(df)}] {a}-{b}: Sent {orders_this_pair} orders [{mode}]")
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"[{i}/{len(df)}] {a}-{b}: Failed to place orders - {e}")
                continue

        ib.disconnect()
        logger.info(f"Execution complete [{mode}]: {orders_sent} orders sent to IBKR")
        return 0

    except Exception as e:
        logger.error(f"IBKR execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())