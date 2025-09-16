from __future__ import annotations
from src.config import load_params
from src.universe import load_universe
from src.data import ensure_universe

def main():
    params = load_params()
    assert params.get('env',{}).get('mode','dev') != 'dev', "Passez env.mode=prod dans config/params.yaml pour IBKR."
    tickers = load_universe()
    ensure_universe(params)
    print("Ingestion IBKR terminée (ou fichiers déjà présents).")

if __name__ == "__main__":
    main()
