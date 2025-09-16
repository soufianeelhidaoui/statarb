from __future__ import annotations
def position_size(capital: float, per_trade_pct: float, price_y: float, price_x: float, beta: float) -> tuple[int, int]:
    risk_budget = capital * per_trade_pct
    if price_y <= 0 or price_x <= 0:
        return 0, 0
    n_y = max(1, int(risk_budget / price_y))
    n_x = max(1, int(abs(beta) * n_y * price_y / price_x))
    return n_y, n_x

def clamp_open_pairs(current_open: int, max_pairs_open: int) -> bool:
    return current_open < max_pairs_open
