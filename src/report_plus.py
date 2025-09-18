from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from .config import load_params
from .data_quality import load_price_df
from .decisions import decide_pair
from .provenance import read_provenance

def _infer_rebalance_id(dt: pd.Timestamp | None = None) -> str:
    if dt is None:
        return datetime.now().strftime("%Y-%m-%d")
    return pd.Timestamp(dt).strftime("%Y-%m-%d")

def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    from pandas.tseries.offsets import BDay
    return (pd.Timestamp(d) + BDay(1)).normalize()

def _orders_from_decisions(dec: pd.DataFrame, last_price_date: pd.Timestamp, params: dict) -> pd.DataFrame:
    per_trade_pct = params.get("risk", {}).get("per_trade_pct", 0.003)
    capital = params.get("risk", {}).get("capital", 100_000.0)
    notional = float(capital) * float(per_trade_pct)

    cols = [
        "env","source","rebalance_id","pair","a","b","verdict","action",
        "side_y","side_x","entry_rule","entry_when",
        "target_notional_total","target_notional_leg_y","target_notional_leg_x",
        "expected_price_rule","qty_y","qty_x"
    ]
    rows = []

    def legs(action: str) -> tuple[str, str]:
        if action == "ShortY_LongX": return ("SHORT_Y","LONG_X")
        if action == "LongY_ShortX": return ("LONG_Y","SHORT_X")
        return ("HOLD","HOLD")

    from pandas.tseries.offsets import BDay
    entry_when = (pd.Timestamp(last_price_date) + BDay(1)).normalize().strftime("%Y-%m-%d 09:30:00")

    if not dec.empty:
        for _, r in dec.iterrows():
            if r.get("verdict") != "ENTER":
                continue
            side_y, side_x = legs(r.get("action","None"))
            rows.append({
                "env": r.get("env"), "source": r.get("source"), "rebalance_id": r.get("rebalance_id"),
                "pair": f"{r['a']}/{r['b']}", "a": r["a"], "b": r["b"],
                "verdict": r["verdict"], "action": r["action"],
                "side_y": side_y, "side_x": side_x,
                "entry_rule": "next_open", "entry_when": entry_when,
                "target_notional_total": round(notional,2),
                "target_notional_leg_y": round(notional/2,2),
                "target_notional_leg_x": round(notional/2,2),
                "expected_price_rule": "OPEN[J+1]",
                "qty_y": float("nan"), "qty_x": float("nan"),
            })

    return pd.DataFrame(rows, columns=cols)


def generate_reports_bundle(tickers: list[str], root_dir: Path, top_pairs: pd.DataFrame,
                            out_base_dir: Path | None = None, rebalance_id: str | None = None) -> dict:
    params = load_params()
    env_mode = params.get("env", {}).get("mode", "dev")
    prov = read_provenance(root_dir) or {}; source = prov.get("source","unknown")
    out_base = Path(out_base_dir) if out_base_dir else Path("reports")
    rbid = rebalance_id or _infer_rebalance_id(None)
    bundle_dir = out_base / env_mode / rbid
    bundle_dir.mkdir(parents=True, exist_ok=True)

    spy = None
    try:
        if "SPY" in tickers:
            df_spy = load_price_df(root_dir, "SPY").select(["date","px"]).to_pandas().set_index("date")
            spy = df_spy["px"]
    except Exception:
        spy = None

    rows = []; last_price_date = None
    for _, row in top_pairs.iterrows():
        a, b = row["a"], row["b"]
        dfa = load_price_df(root_dir, a).to_pandas().set_index("date")
        dfb = load_price_df(root_dir, b).to_pandas().set_index("date")
        ya = dfa["px"].rename(a); xb = dfb["px"].rename(b)
        common_idx = ya.dropna().index.intersection(xb.dropna().index)
        if len(common_idx) == 0:
            continue
        last_price_date = max(last_price_date or common_idx[-1], common_idx[-1])
        verdict = decide_pair(ya, xb, spy, params, meta_a={"df":dfa}, meta_b={"df":dfb})
        rows.append({"env": env_mode, "source": source, "rebalance_id": rbid, **verdict})

    dec = pd.DataFrame(rows)
    decisions_csv = bundle_dir / "decisions.csv"
    decisions_html = bundle_dir / "decisions.html"
    dec.to_csv(decisions_csv, index=False)
    html = [
        "<html><head><meta charset='utf-8'><title>StatArb Decisions</title></head><body>",
        f"<h2>Décisions Top-K paires — {env_mode.upper()} — Rebalance {rbid} (source={source})</h2>",
        dec.to_html(index=False, float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else x),
        "</body></html>"
    ]
    decisions_html.write_text("\n".join(html), encoding="utf-8")

    if last_price_date is None and len(dec) > 0:
        from datetime import datetime as _dt
        last_price_date = pd.Timestamp(_dt.now().date())

    orders_df = _orders_from_decisions(dec, last_price_date, params)
    orders_csv = bundle_dir / "orders.csv"
    orders_df.to_csv(orders_csv, index=False)

    return {"env": env_mode, "source": source, "rebalance_id": rbid,
            "bundle_dir": str(bundle_dir),
            "decisions_csv": str(decisions_csv),
            "decisions_html": str(decisions_html),
            "orders_csv": str(orders_csv)}
