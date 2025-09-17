
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from .config import load_params
from .data import get_price_series
from .decisions import decide_pair

def _infer_rebalance_id(dt: pd.Timestamp | None = None) -> str:
    """
    Default rebalance_id = today's date (YYYY-MM-DD) if not provided.
    If a reference timestamp is passed, use its date.
    """
    if dt is None:
        return datetime.now().strftime("%Y-%m-%d")
    return pd.Timestamp(dt).strftime("%Y-%m-%d")

def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    # Approximation using pandas BDay
    from pandas.tseries.offsets import BDay
    return (pd.Timestamp(d) + BDay(1)).normalize()

def _orders_from_decisions(dec: pd.DataFrame, last_price_date: pd.Timestamp, params: dict) -> pd.DataFrame:
    """
    Build a simple 'next-open' orders CSV from decisions.
    We keep sizing placeholders (to be filled by execution layer), but we compute notionals.
    """
    per_trade_pct = params.get("risk", {}).get("per_trade_pct", 0.003)
    capital = params.get("risk", {}).get("capital", 100_000.0)  # optional; fallback 100k
    notional = float(capital) * float(per_trade_pct)

    # Map action -> leg sides
    def legs(action: str) -> tuple[str, str]:
        if action == "ShortY_LongX":
            return ("SHORT_Y", "LONG_X")
        elif action == "LongY_ShortX":
            return ("LONG_Y", "SHORT_X")
        return ("HOLD", "HOLD")

    # When to execute: next business day open
    entry_when = _next_business_day(pd.Timestamp(last_price_date))

    rows = []
    for _, r in dec.iterrows():
        a, b = r["a"], r["b"]
        verdict = r["verdict"]
        action = r["action"]
        side_y, side_x = legs(action)

        if verdict != "ENTER":
            # Only produce orders for ENTER signals; EXIT/HOLD managed by position monitor
            continue

        rows.append({
            "rebalance_id": r["rebalance_id"],
            "pair": f"{a}/{b}",
            "a": a,
            "b": b,
            "verdict": verdict,
            "action": action,
            "side_y": side_y,
            "side_x": side_x,
            "entry_rule": "next_open",
            "entry_when": entry_when.strftime("%Y-%m-%d 09:30:00"),
            "target_notional_total": round(notional, 2),
            "target_notional_leg_y": round(notional/2, 2),
            "target_notional_leg_x": round(notional/2, 2),
            # Placeholders for the execution layer:
            "expected_price_rule": "OPEN[J+1]",
            "qty_y": np.nan,
            "qty_x": np.nan,
        })

    return pd.DataFrame(rows)

def generate_reports_bundle(tickers: list[str],
                            root_dir: Path,
                            top_pairs: pd.DataFrame,
                            out_base_dir: Path | None = None,
                            rebalance_id: str | None = None) -> dict:
    """
    Single entry point producing a clean, timestamped export bundle:
      reports/<rebalance_id>/{decisions.html, decisions.csv, orders.csv}

    Returns dict with paths.
    """
    params = load_params()
    out_base = Path(out_base_dir) if out_base_dir else Path("reports")
    rbid = rebalance_id or _infer_rebalance_id(None)
    bundle_dir = out_base / rbid
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Optional SPY for market residualization in decide_pair()
    spy = None
    try:
        if "SPY" in tickers:
            df_spy = get_price_series(root_dir, "SPY").select(["date","adj_close","close"]).to_pandas().set_index("date")
            spy = df_spy["adj_close"].fillna(df_spy["close"])
    except Exception:
        spy = None

    # Compute decisions once
    rows = []
    last_price_date = None
    for _, row in top_pairs.iterrows():
        a, b = row["a"], row["b"]
        dfa = get_price_series(root_dir, a).select(["date","adj_close","close"]).to_pandas().set_index("date")
        dfb = get_price_series(root_dir, b).select(["date","adj_close","close"]).to_pandas().set_index("date")
        ya = dfa["adj_close"].fillna(dfa["close"]).rename(a)
        xb = dfb["adj_close"].fillna(dfb["close"]).rename(b)

        # Track last common price date for the pair
        common_idx = ya.dropna().index.intersection(xb.dropna().index)
        if len(common_idx) > 0:
            last_price_date = max(last_price_date or common_idx[-1], common_idx[-1])

        verdict = decide_pair(ya, xb, spy, params)
        rows.append({"a": a, "b": b, **verdict})

    dec = pd.DataFrame(rows)
    dec.insert(0, "rebalance_id", rbid)

    # Write decisions CSV & HTML
    decisions_csv = bundle_dir / "decisions.csv"
    decisions_html = bundle_dir / "decisions.html"
    dec.to_csv(decisions_csv, index=False)

    html = [
        "<html><head><meta charset='utf-8'><title>StatArb Decisions</title></head><body>",
        f"<h2>Décisions Top-K paires — Rebalance {rbid}</h2>",
        dec.to_html(index=False, float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else x),
        "</body></html>"
    ]
    decisions_html.write_text("\n".join(html), encoding="utf-8")

    # Build orders CSV (ENTER only), using last_price_date
    if last_price_date is None and len(dec) > 0:
        # Fallback: use today if we somehow couldn't infer
        last_price_date = pd.Timestamp(datetime.now().date())

    orders_df = _orders_from_decisions(dec, last_price_date, params)
    orders_csv = bundle_dir / "orders.csv"
    orders_df.to_csv(orders_csv, index=False)

    return {
        "rebalance_id": rbid,
        "bundle_dir": str(bundle_dir),
        "decisions_csv": str(decisions_csv),
        "decisions_html": str(decisions_html),
        "orders_csv": str(orders_csv),
    }
