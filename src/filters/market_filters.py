from __future__ import annotations
from pathlib import Path
import polars as pl
import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

def vix_ok(vix_path: str | Path, vix_max: float) -> bool:
    try:
        df = pl.read_parquet(vix_path).select(['date','close']).sort('date').to_pandas().set_index('date')
        last = float(df['close'].iloc[-1])
        return last <= vix_max
    except Exception:
        return True

def _default_end_time_for_event(event: str) -> str:
    e = event.lower()
    
    if "cpi" in e or "employment" in e or "nfp" in e:
        return "10:30"
    if "fomc" in e:
        return "16:30"
    if "boc" in e:
        return "11:30"
    return "16:00"

def macro_ok(macro_csv: str | Path, now: pd.Timestamp | None = None, cool_off_hours: int = 0) -> bool:
    """
    Bloque seulement jusqu’à l’heure de l’évènement (+ cool_off_hours).
    Tolère un 'now' naïf (sans timezone) ou tz-aware.
    - Si 'time' (HH:MM, ET) est présent dans le CSV, on l'utilise.
    - Sinon, on applique des bornes par défaut selon le type d'évènement.
    - "NYSE Holiday" => blocage toute la journée.
    """
    if now is None:
        now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now_utc = now.tz_localize("UTC")
    else:
        now_utc = now.tz_convert("UTC")
    now_et = now_utc.astimezone(ET)
    today = now_et.date()

    # Charger le calendrier
    try:
        df = pd.read_csv(macro_csv)
    except Exception:
        return True
    if df.empty or "date" not in df.columns or "event" not in df.columns:
        return True

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    except Exception:
        return True
    todays = df[df["date"] == today]
    if todays.empty:
        return True

    if todays["event"].str.contains("NYSE Holiday", case=False, na=False).any():
        return False

    has_time = "time" in todays.columns

    for _, r in todays.iterrows():
        ev = str(r.get("event", ""))
        if has_time and pd.notna(r.get("time", "")) and str(r["time"]).strip():
            hhmm = str(r["time"]).strip()
            try:
                t_et = pd.Timestamp(f"{today} {hhmm}:00", tz=ET)
            except Exception:
                t_et = pd.Timestamp(f"{today} 09:30:00", tz=ET)
            end_t = t_et + pd.Timedelta(hours=cool_off_hours)
        else:
            hhmm = _default_end_time_for_event(ev)
            t_et = pd.Timestamp(f"{today} {hhmm}:00", tz=ET)
            end_t = t_et + pd.Timedelta(hours=cool_off_hours)

        if now_et <= end_t:
            return False 

    return True
