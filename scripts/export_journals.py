#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import polars as pl
from datetime import datetime

from src.config import load_params
from src.data import get_price_series, ensure_universe
from src.universe import load_universe
from src.pairs import all_pairs_from_universe, score_pairs
from src.quality import (
    assert_provenance, assert_price_series_ok, assert_pairs_scored_schema,
    check_overlap_len, write_qa_log
)
from src.stats import zscore  # Utilisez zscore au lieu de rolling_zscore_spread


def _root_dir_for_env(params: dict) -> Path:
    data = params.get("data", {})
    mode = params.get("env", {}).get("mode", "dev")
    if data.get("separate_roots", True):
        if mode == "prod":
            return Path(data.get("root_dir_prod", "data/eod/ETFs_prod"))
        else:
            return Path(data.get("root_dir_dev", "data/eod/ETFs_dev"))
    return Path(data.get("root_dir", "data/eod/ETFs"))


def _get_selection_params(params: dict) -> dict:
    defaults = {"min_corr": 0.6, "max_half_life_days": 20.0, "pval_coint": 0.05}
    sel = params.get("selection", {})
    if sel:
        out = defaults.copy()
        out.update(sel)
        return out
    sf = params.get("stats_filters", {})
    return {
        "min_corr": defaults["min_corr"],
        "max_half_life_days": float(sf.get("half_life_max_days", defaults["max_half_life_days"])),
        "pval_coint": float(sf.get("coint_pval_max", defaults["pval_coint"])),
    }


def _select_pairs(scored: pd.DataFrame, params: dict) -> pd.DataFrame:
    th = _get_selection_params(params)
    need = {"a","b","corr","half_life","pval","score"}
    missing = need - set(scored.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans pairs_scored: {missing}")

    filt = (
        (scored["corr"] >= float(th["min_corr"])) &
        (scored["half_life"] <= float(th["max_half_life_days"])) &
        (scored["pval"] <= float(th["pval_coint"]))
    )
    topk = int(params.get('exports', {}).get('topk', 10))
    return scored.loc[filt].sort_values("score", ascending=False).head(topk).copy()


def main():
    params = load_params()
    mode = params.get("env", {}).get("mode", "dev")
    
    # Créer le dossier QA s'il n'existe pas
    qa_dir = Path("reports/QA")
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    # Définir qa_log comme un chemin (Path) et ne plus le modifier par la suite
    qa_log = qa_dir / f"qa_{mode}.log"
    quality_cfg = params.get("quality", {})

    ensure_universe(params)
    tickers = load_universe()
    root_dir = _root_dir_for_env(params)
    expected_src = "ibkr" if mode == "prod" else {"yahoo", "yfinance"}
    
    # Appel unique à assert_provenance
    assert_provenance(root_dir, expected_src, quality_cfg.get("require_provenance_match", True), qa_log)

    # price_map coalesce adj_close -> close
    price_map = {}
    for t in tickers:
        dfpl = get_price_series(root_dir, t).sort('date')
        assert_price_series_ok(dfpl, t, quality_cfg, qa_log)
        pdf = dfpl.select(['date','adj_close','close']).to_pandas().set_index('date')
        s = pdf['adj_close'].fillna(pdf['close'])
        price_map[t] = pd.DataFrame({'close': s})


    # scoring
    lookbacks = params.get('lookbacks', {})
    corr_days = int(lookbacks.get('corr_days', 120))
    coint_days = int(lookbacks.get('coint_days', 120))
    pairs = all_pairs_from_universe(tickers)
    scored = score_pairs(price_map, pairs, corr_days, coint_days)
    assert_pairs_scored_schema(scored, quality_cfg, qa_log)


    top = _select_pairs(scored, params)
    
    # Si aucune paire ne passe les filtres, on affiche un message et on termine
    if len(top) == 0:
        print("Aucune paire ne correspond aux critères de sélection.")
        return

    # Dossier horodaté
    run_day = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(params.get('exports', {}).get('journals_dir', 'reports/journals')) / run_day
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate journals for each pair
    for _, r in top.iterrows():
        a, b = str(r['a']), str(r['b'])

        dfa = get_price_series(root_dir, a).select(['date','adj_close','close']).sort('date').to_pandas()
        dfb = get_price_series(root_dir, b).select(['date','adj_close','close']).sort('date').to_pandas()

        dfa['px'] = dfa['adj_close'].fillna(dfa['close'])
        dfb['px'] = dfb['adj_close'].fillna(dfb['close'])

        ja = dfa[['date','px']].rename(columns={'px':'ya'})
        jb = dfb[['date','px']].rename(columns={'px':'xb'})
        j = pd.merge(ja, jb, on='date', how='inner').sort_values('date')
        j['date'] = pd.to_datetime(j['date'])
        
        # QA check après création de j
        check_overlap_len(j['ya'], j['xb'], quality_cfg, qa_log)

        cov = j['ya'].cov(j['xb'])
        var = j['xb'].var()
        beta = cov/var if var and var != 0 else 1.0

        spread = j['ya'] - beta*j['xb']
        z_window = int(lookbacks.get('zscore_days', 60))
        z = zscore(spread, z_window)

        out = pd.DataFrame({
            'date': j['date'].dt.date,
            'ya': j['ya'].astype(float),
            'xb': j['xb'].astype(float),
            'beta': beta,
            'spread': spread.astype(float),
            'z': z.astype(float),
        })
        out.to_csv(out_dir / f"journal_{a}_{b}.csv", index=False)
        print(f"[journal] {a}_{b} → {out_dir}/journal_{a}_{b}.csv")

    print("[export_journals] Terminé.")


if __name__ == "__main__":
    main()