#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe, _root_dir_for_source

def main():
    params = load_params()
    tickers = load_universe()
    source = params.get("data", {}).get("source", "yahoo").lower()
    mode = params.get("trading", {}).get("mode", "paper").lower()

    print(f"[ingest_data] source={source} mode={mode}")
    root = ensure_universe(params, tickers)
    print(f"[ingest_data] ok → {root} (provenance: {(Path(root)/'_PROVENANCE.json').as_posix()})")

if __name__ == "__main__":
    main()
