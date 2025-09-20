#!/usr/bin/env python3
from __future__ import annotations
import sys, subprocess, shlex, argparse
from pathlib import Path
from datetime import datetime, timezone
import yaml

def load_params(path: str="config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def bundle_dir(params: dict, day: str|None=None) -> Path:
    src = params.get("data",{}).get("source","yahoo").lower()
    mode= params.get("trading",{}).get("mode","paper").lower()
    day = day or datetime.now().strftime("%Y-%m-%d")
    return Path("reports/bundles")/src/mode/day

def run(cmd: str) -> int:
    print(f"[run] {cmd}")
    return subprocess.call(shlex.split(cmd))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["evening","preopen","summary"])
    ap.add_argument("--day", default=None)
    args = ap.parse_args()

    params = load_params()
    bdir = bundle_dir(params, args.day)
    bdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if args.phase == "evening":
        # 1) Ingestion (fait les 2 sources via params)
        rc = run(f"{py} scripts/ingest_data.py")
        if rc != 0: sys.exit(rc)
        # 2) Ajustement des prix/adjs + ex-div
        rc = run(f"{py} scripts/adjust_prices.py")
        if rc != 0: sys.exit(rc)
        # 3) Décisions + Orders + HTML → bundle
        rc = run(f"{py} scripts/run_report.py --bundle {bdir}")
        if rc != 0: sys.exit(rc)
        # 4) Journaux z-score → bundle
        rc = run(f"{py} scripts/export_journals.py --bundle {bdir}")
        sys.exit(rc)

    if args.phase == "preopen":
        # 1) Revue et garde-fous
        rc = run(f"{py} scripts/preopen_check.py --bundle {bdir}")
        if rc != 0: sys.exit(rc)
        # 2) Exécution IBKR réelle (Market-On-Open, pas de dry-run ici)
        rc = run(f"{py} scripts/ibkr_execute_moo.py --bundle {bdir} --dry-run")
        sys.exit(rc)

    if args.phase == "summary":
        # Email HTML riche inline à partir du bundle
        rc = run(f"{py} scripts/notify_email.py --bundle {bdir}")
        if rc != 0:
            print("[WARN] email non envoyé")
        sys.exit(rc)

if __name__ == "__main__":
    main()
