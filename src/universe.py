from .config import load_params

def load_universe() -> list[str]:
    p = load_params()
    return list(p["universe"]["tickers"])
