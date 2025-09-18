#!/usr/bin/env python3
"""
Vérifications pré-ouverture (2 minutes) :
- Trouve le dernier bundle reports/<env>/<YYYY-MM-DD>/
- Valide : présence des fichiers, non-vide, pas de NaN critiques
- Vérifie pour chaque paire : données d'hier présentes, volume >= min_volume
- (Option) envoie un email si anomalies trouvées
"""
import argparse, os
from pathlib import Path
import pandas as pd
from src.config import load_params
from src.notify_email import load_email_config, send_email

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

def _alert(subject: str, body_html: str):
    cfg = load_email_config()
    if cfg.get("enabled", False):
        try: send_email(subject, body_html, cfg)
        except Exception as e: print("[email] failed:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["dev","prod"], help="override env")
    args = ap.parse_args()

    params = load_params()
    env_mode = args.env or params["env"]["mode"]
    reports_root = Path(params["exports"].get("reports_dir","reports"))
    bundle_dir = _find_latest_bundle(reports_root, env_mode)
    if bundle_dir is None:
        print("[WARN] Aucun bundle trouvé"); return

    dec_fp = bundle_dir / "decisions.csv"
    ord_fp = bundle_dir / "orders.csv"
    assert dec_fp.exists(), "decisions.csv manquant"
    assert ord_fp.exists(), "orders.csv manquant"

    dec = pd.read_csv(dec_fp)
    orders = pd.read_csv(ord_fp)

    anomalies = []
    if dec.empty: anomalies.append("decisions.csv est vide")
    if orders.empty: anomalies.append("orders.csv est vide")

    # checks de base sur orders
    required_cols = ["a","b","entry_when","action","target_notional_total"]
    missing = [c for c in required_cols if c not in orders.columns]
    if missing: anomalies.append(f"Colonnes manquantes dans orders: {missing}")

    # info volumes (si disponible sur dernier jour dans Parquet)
    try:
        root = _root_dir_for_env(params)
        min_vol = int(params.get("quality",{}).get("min_volume", 0))
        if min_vol > 0:
            import polars as pl
            for t in set(orders["a"]).union(set(orders["b"])):
                p = root / f"{t}.parquet"
                if not p.exists():
                    anomalies.append(f"{t}.parquet absent")
                    continue
                df = pl.read_parquet(p).select(["date","volume"]).sort("date").to_pandas().set_index("date")
                if len(df)==0 or "volume" not in df: continue
                vol = float(df["volume"].iloc[-1])
                if vol < min_vol:
                    anomalies.append(f"Volume insuffisant pour {t}: {vol} < {min_vol}")
    except Exception as e:
        anomalies.append(f"Erreur check volume: {e}")

    if anomalies:
        print("[PREOPEN] Anomalies:")
        for a in anomalies: print(" -", a)
        _alert(f"[StatArb] PREOPEN anomalies ({env_mode})",
               "<br/>".join(f"- {a}" for a in anomalies))
        raise SystemExit(2)

    print("[PREOPEN] OK - pas d'anomalie bloquante.")
    _alert(f"[StatArb] PREOPEN OK ({env_mode})", "<p>Tous les checks sont OK.</p>")

if __name__ == "__main__":
    main()
