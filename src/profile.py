from __future__ import annotations

def merged_risk(params: dict) -> dict:
    base = dict(params.get("risk", {}))
    mode = params.get("trading", {}).get("mode", "paper")
    prof = params.get("profiles", {}).get(mode, {})
    if "risk" in prof:
        base.update({k: v for k, v in prof["risk"].items() if v is not None})
    return base
