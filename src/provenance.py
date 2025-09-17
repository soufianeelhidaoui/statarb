from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

PROV_FILE = "_PROVENANCE.json"

def prov_path(root_dir: Path | str) -> Path:
    return Path(root_dir) / PROV_FILE

def read_provenance(root_dir: Path | str) -> dict | None:
    p = prov_path(root_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_provenance(root_dir: Path | str, source: str) -> dict:
    p = prov_path(root_dir)
    data = {"source": source, "updated_at": datetime.utcnow().isoformat() + "Z"}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def enforce_provenance(root_dir: Path | str, expected_source: str, allow_unknown: bool = False) -> None:
    info = read_provenance(root_dir)
    if info is None:
        if allow_unknown:
            return
        raise RuntimeError(f"Provenance missing in {root_dir}. Expected '{expected_source}'. Run proper ingestion first.")
    got = info.get("source")
    if got != expected_source:
        raise RuntimeError(f"Data provenance mismatch in {root_dir}: found '{got}', expected '{expected_source}'. Avoid mixing DEV/PROD sources.")
