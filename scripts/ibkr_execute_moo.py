#!/usr/bin/env python3
"""
Exécute les ordres MOO/MKT à l'ouverture (paper/live via IBKR).
- Lit le dernier bundle reports/<env>/<YYYY-MM-DD>/orders.csv
- --dry-run (par défaut) affiche les ordres sans exécuter
- Nécessite ib_insync et TWS/IBG connecté en paper/live selon ton compte
"""
import argparse
from pathlib import Path
import pandas as pd
from src.config import load_params

def _find_latest_bundle(reports_dir: Path, env: str) -> Path | None:
    env_dir = reports_dir / env
    if not env_dir.exists():
        return None
    days = sorted([d for d in env_dir.iterdir() if d.is_dir()], reverse=True)
    return days[0] if days else None

def _root_dir_for_env(params: dict) -> Path:
    d = params["data"]
    if d.get("separate_roots", False):
        return Path(d["root_dir_prod" if params["env"]["mode"]=="prod" else "root_dir_dev"])
    return Path(d["root_dir"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["dev","prod"])
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()

    params = load_params()
    env_mode = args.env or params["env"]["mode"]
    bundle_dir = _find_latest_bundle(Path(params["exports"].get("reports_dir","reports")), env_mode)
    if bundle_dir is None: raise SystemExit("No bundle found")

    orders = pd.read_csv(bundle_dir / "orders.csv")
    if orders.empty:
        print("No orders to execute."); return

    print(f"[EXEC] env={env_mode} dry_run={args.dry_run} orders={len(orders)}")
    for _, r in orders.iterrows():
        print(f" - {r['pair']}: {r['action']} notional_total={r['target_notional_total']} entry_when={r['entry_when']} rule={r['entry_rule']}")

    if args.dry_run:
        print("[EXEC] Dry-run: no orders sent.")
        return

    # --- live/paper via ib_insync (squelette) ---
    try:
        from ib_insync import IB, Stock, MarketOrder, util
    except Exception:
        raise SystemExit("ib_insync non installé. `pip install ib_insync`")

    ib = IB()
    # adapte host/port/clientId à ton TWS/IBG (paper)
    ib.connect(host='127.0.0.1', port=7497, clientId=7)  # 7497 = paper TWS par défaut

    for _, r in orders.iterrows():
        a,b = r["a"], r["b"]
        # Simplification: envoie deux legs à notional/2 en Market (à affiner en shares exacts)
        # IB exige des quantités en shares -> il faut convertir notional en qty selon prix courant.
        for sym, side in [(a, r["side_y"]), (b, r["side_x"])]:
            if side == "HOLD": continue
            contract = Stock(sym, 'SMART', 'USD')
            # récup prix pour estimer qty
            mkt = ib.reqMktData(contract, '', False, False)
            ib.sleep(1.0)
            last = mkt.last if mkt.last else (mkt.close if mkt.close else None)
            if not last:
                print(f"[EXEC] skip {sym}: pas de prix")
                continue
            notional = float(r["target_notional_leg_y"] if side in ("LONG_Y","SHORT_Y") else r["target_notional_leg_x"])
            qty = max(1, int(round(notional / float(last))))
            action = "BUY" if side in ("LONG_Y","LONG_X") else "SELL"
            order = MarketOrder(action, qty)
            trade = ib.placeOrder(contract, order)
            print(f"[EXEC] {sym} {action} {qty} @MKT (est ~{last}) -> {trade.orderStatus.status}")

    ib.disconnect()

if __name__ == "__main__":
    main()
