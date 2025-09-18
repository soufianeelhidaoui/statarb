from __future__ import annotations
from pathlib import Path
import pandas as pd
import polars as pl
import json
from datetime import datetime

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_qa_log(path, lines: list[str]):
    """Tolérant: accepte str/Path et crée le dossier si besoin."""
    p = Path(path)  
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for L in lines:
            f.write((L.rstrip() if isinstance(L, str) else str(L)).encode("utf-8", "ignore").decode("utf-8") + "\n")

def assert_provenance(root_dir: Path, expected, enabled: bool, qa_log):
    """
    Vérifie _PROVENANCE.json['source'] ∈ expected.
    - expected peut être str ou une collection (set/list/tuple).
    """
    if not enabled:
        return
    p = Path(root_dir) / "_PROVENANCE.json"
    if not p.exists():
        msg = f"[FATAL] {_now_str()} provenance manquante: {p}"
        write_qa_log(qa_log, [msg])
        raise RuntimeError(msg)

    data = json.loads(p.read_text())
    got = data.get("source", "")

    if isinstance(expected, (set, list, tuple)):
        exp = set(expected)
    else:
        exp = {str(expected)}

    if got not in exp:
        msg = f"[FATAL] {_now_str()} provenance mismatch: expected∈{sorted(exp)} got={got}"
        write_qa_log(qa_log, [msg])
        raise RuntimeError(msg)

    write_qa_log(qa_log, [f"[OK] {_now_str()} provenance source={got} acceptée."])

def assert_price_series_ok(dfpl: pl.DataFrame, ticker: str, cfg: dict, qa_log: Path):
    strict = cfg.get("strict", False)
    min_overlap = int(cfg.get("min_overlap_days", 120))
    max_nan_ratio = float(cfg.get("max_nan_ratio_px", 0.01))
    require_adj = bool(cfg.get("require_adj_close", True))
    require_ex_div = bool(cfg.get("require_ex_div_mask", True))

    # colonnes
    need = {"date","close"}
    missing = need - set(dfpl.columns)
    if missing:
        msg = f"[FATAL] {_now_str()} {ticker}: colonnes manquantes {missing}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

    # coalesce px
    has_adj = "adj_close" in dfpl.columns
    if require_adj and not has_adj:
        msg = f"[FATAL] {_now_str()} {ticker}: adj_close absent en mode strict"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

    dfpl = dfpl.select([
        "date",
        (pl.coalesce([pl.col("adj_close"), pl.col("close")]).alias("px")) if has_adj else pl.col("close").alias("px"),
        *([pl.col("is_ex_div")] if "is_ex_div" in dfpl.columns else [])
    ]).sort("date")

    pdf = dfpl.to_pandas().sort_values("date")
    n = len(pdf)
    min_len = int(max(
        cfg.get("min_history_days", 0),     
        120                                  
    ))
    if n < min_len:
        msg = f"[FATAL] {_now_str()} {ticker}: historique trop court n={n} < {min_len}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

    n_nan = int(pdf["px"].isna().sum())
    ratio = 0.0 if n == 0 else n_nan / n
    n_nan = int(pdf["px"].isna().sum())
    ratio = 0.0 if n == 0 else n_nan / n

    if strict and ratio > max_nan_ratio:
        msg = f"[FATAL] {_now_str()} {ticker}: ratio NaN px={ratio:.3%} > {max_nan_ratio:.3%}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

    # overlap (seulement vérifiable au moment du merge, donc on log ici la longueur)
    write_qa_log(qa_log, [f"[OK] {_now_str()} {ticker}: n={n}, nan_px={n_nan} ({ratio:.2%})"])

    if require_ex_div and "is_ex_div" not in pdf.columns:
        msg = f"[FATAL] {_now_str()} {ticker}: is_ex_div absent en mode strict"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

def assert_pairs_scored_schema(scored: pd.DataFrame, cfg: dict, qa_log: Path):
    strict = cfg.get("strict", False)
    need_min = {"a","b","corr","pval","score"}
    missing = need_min - set(scored.columns)
    if missing:
        msg = f"[FATAL] {_now_str()} pairs_scored: colonnes minimales manquantes {missing}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)

    if cfg.get("require_half_life", True):
        if not any(c in scored.columns for c in ["half_life","half_life_days","hl"]):
            msg = f"[FATAL] {_now_str()} pairs_scored: half_life requise mais absente"
            write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    write_qa_log(qa_log, [f"[OK] {_now_str()} pairs_scored schema validé."])

def check_overlap_len(ya: pd.Series, xb: pd.Series, cfg: dict, qa_log: Path):
    min_overlap = int(cfg.get("min_overlap_days", 120))
    j = pd.concat([ya, xb], axis=1).dropna()
    if len(j) < min_overlap:
        msg = f"[FATAL] {_now_str()} overlap insuffisant: {len(j)} < {min_overlap}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    write_qa_log(qa_log, [f"[OK] {_now_str()} overlap={len(j)} jours ≥ {min_overlap}"])
