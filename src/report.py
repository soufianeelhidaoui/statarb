from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from .config import load_params
from .data import get_price_series
from .backtest import merge_close_series, simulate_pair

def generate_pair_report(a: str, b: str, out_html: str | Path = "reports/report.html") -> Path:
    params = load_params()
    root_dir = Path(params['data']['root_dir'])
    dfa = get_price_series(root_dir, a)
    dfb = get_price_series(root_dir, b)
    df = merge_close_series(dfa, dfb)

    total, journal = simulate_pair(
        df,
        params['thresholds']['entry_z'],
        params['thresholds']['exit_z'],
        params['thresholds']['stop_z'],
        params['lookbacks']['zscore_days'],
        params['risk']['per_trade_pct'],
        capital=100_000.0,
        costs_bp=params['costs']['slippage_bp']
    )

    out_dir = Path(out_html).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = plt.figure()
    journal['cum_pnl'].plot()
    plt.title(f"Cumulative PnL {a}/{b}")
    pnl_png = out_dir / f"pnl_{a}_{b}.png"
    fig1.savefig(pnl_png, bbox_inches="tight")
    plt.close(fig1)

    html = f"""
<html><head><meta charset='utf-8'><title>Pair Report {a}/{b}</title></head>
<body>
<h2>Pair Report {a}/{b}</h2>
<p><b>Total PnL (approx $):</b> {total:.2f}</p>
<img src="{pnl_png.name}" alt="PnL plot"/>
<h3>Journal (tail)</h3>
{journal.tail(50).to_html()}
</body></html>
"""

    out_path = Path(out_html)
    out_path.write_text(html, encoding="utf-8")
    return out_path
