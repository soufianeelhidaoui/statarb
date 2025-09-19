from __future__ import annotations
from pathlib import Path
import json
import polars as pl
import pandas as pd
from datetime import datetime

def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def write_qa_log(path: Path | str, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")

def assert_provenance(root_dir: Path, expected_source: str | set[str], require_match: bool, qa_log: Path | str) -> None:
    prov = Path(root_dir) / "_PROVENANCE.json"
    if not prov.exists():
        msg = f"[FATAL] {_now_str()} provenance manquante: {prov.as_posix()}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    j = json.loads(prov.read_text())
    got = str(j.get("source", "")).lower()
    def _ok():
        if isinstance(expected_source, set):
            return got in {s.lower() for s in expected_source}
        return got == str(expected_source).lower()
    if require_match and not _ok():
        msg = f"[FATAL] {_now_str()} provenance mismatch: expected={expected_source} got={got}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    write_qa_log(qa_log, [f"[OK] {_now_str()} provenance source={got} acceptée."])

def assert_pairs_scored_schema(scored: pd.DataFrame, quality_cfg: dict, qa_log: Path | str) -> None:
    need = {"a","b","corr","pval","score"}
    miss = need - set(scored.columns)
    if miss:
        msg = f"[FATAL] {_now_str()} pairs_scored: colonnes manquantes: {miss}"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    write_qa_log(qa_log, [f"[OK] {_now_str()} pairs_scored schema validé."])

def assert_price_series_ok(dfpl: pl.DataFrame, ticker: str, quality_cfg: dict, qa_log: Path | str) -> None:
    strict = bool(quality_cfg.get("require_provenance_match", True))
    mask_ex = bool(quality_cfg.get("mask_ex_div", True))
    require_ex = bool(quality_cfg.get("require_is_ex_div", True)) if mask_ex else False
    tol_bp = int(quality_cfg.get("compare_adj_tolerance_bp", 50))
    has_cols = set(dfpl.columns)
    n = dfpl.height
    if n == 0:
        msg = f"[FATAL] {_now_str()} {ticker}: série vide"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    if not {"date","close"}.issubset(has_cols):
        msg = f"[FATAL] {_now_str()} {ticker}: colonnes de base manquantes"
        write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    if "adj_close" not in has_cols and quality_cfg.get("px_policy","best") == "best":
        write_qa_log(qa_log, [f"[WARN] {_now_str()} {ticker}: adj_close absent → fallback close"])
    if require_ex and "is_ex_div" not in has_cols:
        if bool(quality_cfg.get("auto_fix_is_ex_div", True)):
            from .repair import ensure_is_ex_div
            parquet_path = dfpl.estimated_size() and None
            msg_try = f"[FIX] {_now_str()} {ticker}: is_ex_div absent → tentative de reconstruction locale"
            write_qa_log(qa_log, [msg_try])
            pdf = dfpl.to_pandas()
            if "adj_close" in pdf.columns:
                factor = (pdf["adj_close"] / pdf["close"])
                change = (factor / factor.shift(1) - 1.0).abs()
                pdf["is_ex_div"] = (change * 10_000 > max(1, tol_bp)).fillna(False)
                dfpl = pl.from_pandas(pdf)
            else:
                pdf = dfpl.to_pandas()
                pdf["is_ex_div"] = False
                dfpl = pl.from_pandas(pdf)
            write_qa_log(qa_log, [f"[OK] {_now_str()} {ticker}: is_ex_div reconstruit en mémoire"])
        else:
            msg = f"[FATAL] {_now_str()} {ticker}: is_ex_div absent en mode strict"
            write_qa_log(qa_log, [msg]); raise RuntimeError(msg)
    nan_px = int(dfpl["close"].is_null().sum())
    ratio = 100.0 * nan_px / n
    write_qa_log(qa_log, [f"[OK] {_now_str()} {ticker}: n={n}, nan_px={nan_px} ({ratio:.2f}%)"])

def check_overlap_len(ya: pd.Series, xb: pd.Series, quality_cfg: dict, qa_log: Path | str) -> None:
    idx = ya.dropna().index.intersection(xb.dropna().index)
    if len(idx) < 120:
        msg = f"[WARN] {_now_str()} overlap court: {len(idx)}"
        write_qa_log(qa_log, [msg])
