---
applyTo: '**'
---
# StatArb-ETFs — AI Coding Instructions (Enriched)

## 0) TL;DR

* **Mode**: `env.mode: dev` (Yahoo) for prototyping, `prod` (IBKR) for live/paper.
* **Flow**: Universe → Pairs → Scoring → Selection → Backtest → Report → (Paper/Prod).
* **Golden rules**: everything **config-driven**, **Parquet + Polars** for I/O, **Pandas** for math, **no hardcoded tickers**.

```mermaid
flowchart LR
  U[Universe (config/CSV)] --> D[Data I/O (Polars/Parquet)]
  D --> S[Stats (Pandas)]
  S --> P[Pairs: score & select]
  P --> B[Backtest (costs)]
  B --> R[Report (HTML/PNG, DuckDB)]
  R --> X[(Paper/Prod IBKR)]
```

---

## 1) Overview

Statistical arbitrage on **ETF pairs** (market-neutral) with dual-mode execution:

* **dev** → yfinance data (fast prototyping)
* **prod** → Interactive Brokers (ib\_insync) for ingestion & execution

---

## 2) Architecture & Data Flow

* **Configuration**: `config/params.yaml` drives all parameters (univers, fenêtres, seuils, coûts, risques, chemins, env).
* **Universe**: 15 liquid ETFs in `universe.tickers` (also mirrored in `data/metadata/universe.csv`).
* **Data Pipeline**: `src/data.py` reads/writes **Parquet** under `data/eod/ETFs/`.
* **Core Flow**: Universe → **generate 105 pairs** → **score** (corr, coint, half-life) → **select top-K** → **backtest** (z-score rules) → **report** (HTML + DuckDB).

### Directory Layout

```
statarb/
  config/params.yaml         # commenté, source de vérité
  data/
    eod/ETFs/*.parquet       # EOD par ticker
    metadata/universe.csv    # alternative à YAML pour l'univers
  notebooks/exploration.ipynb
  reports/
    pairs_scored/latest_pairs_scored.parquet
    analytics.duckdb
    report_<A>_<B>.html
  scripts/
    run_backtest.py          # backtest rapide top pair
    run_report.py            # scoring + DuckDB + HTML
    ingest_ibkr.py           # ingestion PROD IBKR → Parquet
  src/
    config.py, data.py, stats.py, pairs.py,
    backtest.py, signals.py, risk.py,
    report.py, duck_analytics.py, execution_ib.py, universe.py
```

---

## 3) Key Entry Points

* **`scripts/run_backtest.py`** → Backtest rapide sur la meilleure paire, PnL console.
* **`scripts/run_report.py`** → Scoring complet, export DuckDB, HTML + PNG.
* **`scripts/ingest_ibkr.py`** → Ingestion PROD via IBKR (ib\_insync) en Parquet.

---

## 4) Critical Patterns & Conventions

### 4.1 Data Handling

* **Polars for I/O**, **Pandas for analytics**:

  * Polars lit/écrit Parquet vite et proprement.
  * Stats (ADF, rolling, OLS) tournent sur Pandas (compatibles `statsmodels`).
* **Schema attendu** (Parquet par ticker):

  ```
  ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
  ```

  `date` = `Datetime[ns]` (UTC). Index Pandas = `date`.
* **Merging**: toujours joindre sur **intersection** de dates (inner join).

### 4.2 Configuration-Driven Design

```python
from src.config import load_params
params = load_params()  # Ne jamais hardcoder des valeurs et ne jamais modifier params en place ou mettre des valeurs par défaut dans le code ou les scripts ainsi que maintenir toute la config dans params.yaml, pas de mock data.
root_dir = params['data']['root_dir']
min_corr = params['selection']['min_corr']
```

### 4.3 Mode Switching

* **Dev**: `env.mode: dev` → `yfinance` (auto-download si manquant)
* **Prod**: `env.mode: prod` → **IBKR ingestion** (via `scripts/ingest_ibkr.py`), le code lit ensuite les Parquet.
* **Routage** (déjà implémenté): `src/data.py::ensure_universe(params)`.

### 4.4 Pairs Trading Pipeline

1. **Generate** → `pairs.all_pairs_from_universe()`
2. **Score** → `pairs.score_pairs()` (corr, coint (ADF+fallback), half-life, sigma → score composite)
3. **Select** → `pairs.select_top_pairs()` via `min_corr`, `max_half_life`, `pval_coint`, `max_pairs`
4. **Backtest** → `backtest.simulate_pair()` (règles z-score, costos/slippage, sizing)

---

## 5) Statistical Components (src/stats.py)

* **Hedge ratio (β)**: OLS `y = α + βx` → `ols_hedge_ratio()`
* **Spread**: `spread = y − (α + βx)`
* **Stationnarité**:

  * ADF sur résidus (si `statsmodels` dispo)
  * **Fallback** cohérent basé sur half-life si ADF manquant
* **Réversion**: `half_life_of_mean_reversion()` par AR(1)
* **Signal**: `zscore(spread, win)` (rolling μ/σ)

> Tip: Toute statistique doit **dropna()**, et chaque fenêtre doit **contrôler sa taille**.

---

## 6) Risk & Trading Rules (defaults)

* **Entrée**: `|z| ≥ entry_z`
* **Sortie**: `|z| ≤ exit_z` (ou croisement 0)
* **Stop**: `|z| ≥ stop_z` **ou** perte notionnelle > `k × ATR` (si tu l’ajoutes)
* **Time-stop**: `time_stop_days` jours (daily EOD)
* **Sizing**: risquer \~`0.25–0.5%` du capital par paire, hedge via β
* **Cap exposition**: max `3–5` paires ouvertes, net exposure ≈ 0

Tous ces paramètres sont dans `config/params.yaml`.

---

## 7) Development Workflow

### 7.1 Adding New Features

1. **Paramètre d’abord** → ajoute dans `config/params.yaml`
2. **Code ensuite** → adapte `src/*` en lisant la config (jamais l’inverse)
3. **Test local** → `python scripts/run_backtest.py`
4. **Validation** → `python scripts/run_report.py` (produit parquet + duckdb + html)

### 7.2 Data Management

* **Assurer la data**: `ensure_universe(params)` (dev → télécharge; prod → ingère via IBKR si manquant)
* **Charger**: `get_price_series(root_dir, ticker)` → Polars DF → convertis en Pandas pour stats
* **Nouveaux tickers**: mets-les dans `universe.tickers` (YAML) + relance ingestion

### 7.3 Testing Changes

```bash
# Scoring + backtest rapide
python scripts/run_backtest.py

# Scoring + DuckDB + HTML + PNG PnL
python scripts/run_report.py
```

---

## 8) Production Notes (IBKR / ib\_insync)

* **Ingestion** (historique EOD) via `scripts/ingest_ibkr.py`

  * **Pré-requis**: TWS/IB Gateway lancé, API activée (port 7497 par défaut), `ib_insync` installé.
  * Les Parquet sont écrits dans `data/eod/ETFs/`.
* **Exécution**: `src/execution_ib.py` expose un client à brancher avec `ib_insync` (place orders long/short symétriques, logs).
* **Sécurité**: commence en **paper** chez IBKR, journalise tout (signals, ordres, fills, PnL, erreurs), mets un **kill-switch** (ex: désactiver si VIX>seuil ou drawdown jour > 2%).

---

## 9) Quality Gates (what “good” looks like)

### 9.1 Code Quality

* **No hardcoding** (tickers, fenêtres, seuils)
* **Small pure functions** (stat/IO/selection séparés)
* **Fail fast** (asserts sur NaN, fenêtres trop courtes, fichiers manquants)

### 9.2 Backtest Sanity

* **Always include costs/slippage** (stress ×2)
* **Walk-forward** (mensuel mini) si tu étends la boucle
* **Metrics**: Sharpe/Sortino ≥ \~1.2, Max DD ≤ \~10%, Profit Factor ≥ \~1.3

### 9.3 Determinism

* Même entrée → même sortie (scoring et sélection persistés en Parquet/DuckDB + versionner la config)

---

## 10) Common Gotchas & Fixes

* **Empty DataFrames** → vérifier avant stats; log & skip.
* **Index misaligned** → toujours rejoindre sur `date` intersec; attention aux jours fériés.
* **ADF indisponible** → fallback half-life est prévu; installe `statsmodels` pour tests robustes.
* **Coûts dévorent la perf** → relever `entry_z`, exécuter à open/close, ordres limites, éviter ETFs à spread large.
* **Latence IBKR** → batcher requêtes historiques; limiter fréquence; cache local en Parquet.

---

## 11) Output Structure

* **Backtest**: console + `reports/pairs_scored/latest_pairs_scored.parquet`
* **Reports**: `reports/report_<A>_<B>.html` + `reports/pnl_<A>_<B>.png` + `reports/analytics.duckdb`
* **Data**: `data/eod/ETFs/<TICKER>.parquet` (1 fichier/ticker)

---

## 12) How to Extend Safely

### 12.1 Add regime filter (optional)

* Ex: n’exécuter que quand **mean-reversion** > **trend** (Hurst < 0.5, ou accélérer la pénalisation du half-life).
* Implémente un `regime.py` avec `is_reversion_regime()` et branche son résultat dans `signals` (garde les signaux si True).

### 12.2 Add portfolio-level risk

* Cap `gross exposure`, `sector caps`, **daily DD stop** global (déjà paramétré dans `risk` → ajoute la logique dans l’exécuteur/monitoring).

### 12.3 Rolling β & monthly re-selection

* Aujourd’hui: β global pour le backtest rapide.
* Prod: recalcul β **mensuel** (ou hebdo) et re-sélection **mensuelle** (déjà paramétrable via `pairs.rebalance_days`).

---

## 13) Minimal Examples (copy-paste)

### Load config & ensure data

```python
from src.config import load_params
from src.data import ensure_universe, get_price_series
from src.universe import load_universe
from pathlib import Path

params = load_params()
ensure_universe(params)
tickers = load_universe()
dfpl = get_price_series(Path(params['data']['root_dir']), tickers[0])  # Polars DF
```

### Score and select top-K

```python
import polars as pl
import pandas as pd
from src.pairs import all_pairs_from_universe, score_pairs, select_top_pairs

root = Path(params['data']['root_dir'])
price_map = {}
for t in tickers:
    price_map[t] = get_price_series(root, t).select(['date','close']).sort('date').to_pandas().set_index('date')

pairs = all_pairs_from_universe(tickers)
scored = score_pairs(price_map, pairs, params['lookbacks']['corr_days'], params['lookbacks']['coint_days'])
top = select_top_pairs(scored, params['selection']['min_corr'], params['selection']['max_half_life_days'], params['selection']['pval_coint'], 10)
```

### Backtest one pair

```python
from src.backtest import merge_close_series, simulate_pair

A, B = top.loc[0, 'a'], top.loc[0, 'b']
dfa = get_price_series(root, A)
dfb = get_price_series(root, B)
df = merge_close_series(dfa, dfb)
pnl, journal = simulate_pair(df, params['thresholds']['entry_z'], params['thresholds']['exit_z'], params['thresholds']['stop_z'], params['lookbacks']['zscore_days'], params['risk']['per_trade_pct'], capital=100_000.0, costs_bp=params['costs']['slippage_bp'])
```

---

## 14) Checklists

### Dev (Yahoo)

* [ ] `env.mode: dev`
* [ ] `ensure_universe(params)` → Parquet présents
* [ ] `run_report.py` OK (Parquet + DuckDB + HTML)

### Prod (IBKR)

* [ ] TWS/IBGW up, API on (port 7497)
* [ ] `env.mode: prod`
* [ ] `scripts/ingest_ibkr.py` → Parquet ok
* [ ] `run_report.py` OK
* [ ] Paper trading d’abord + logs exhaustifs

---

## 15) Troubleshooting Quick Hits

* **“File not found …/ETFs/<T>.parquet”** → lance `ensure_universe(params)` (dev) ou `ingest_ibkr.py` (prod).
* **“ADF not found”** → `pip install statsmodels`; sinon fallback half-life utilisé.
* **Peu de paires sélectionnées** → baisse `min_corr` ou augmente lookbacks; assure assez d’historique.
* **Trop de faux signaux** → relève `entry_z` (2.0→2.2/2.5), baisse `exit_z` (0.5→0.3), allonge `zscore_days` (60→90).
* **Perf mangée par coûts** → ordres limites, open/close, `entry_z↑`, ETFs plus liquides.

---

## 16) Governance

* **Every change** goes through `config/params.yaml` → commit & tag (data science is ops).
* **Reproducibility**: persist scoring outputs (`reports/pairs_scored/*.parquet`) & config versions.
* **Observability**: keep logs of signals, orders, fills, PnL, DD; alert on anomalies.

## 17) Utilisation de projet
```mermaid
sequenceDiagram
    autonumber
    actor U as Utilisateur (toi)
    participant CFG as Config/params.yaml
    participant CLI as Script (run_report.py / run_backtest.py)
    participant DATA as Data I/O (src/data.py)
    participant P as Pairs (score/select)
    participant S as Stats (src/stats.py)
    participant BT as Backtest (src/backtest.py)
    participant R as Report (src/report.py)
    participant DB as DuckDB (analytics.duckdb)
    participant IB as IBKR TWS/Gateway

    U->>CFG: Définir env.mode (dev|prod), univers, fenêtres, seuils
    U->>CLI: Lancer `python scripts/run_report.py`
    CLI->>CFG: Charger paramètres
    CLI->>DATA: ensure_universe(params)

    alt env.mode == dev (Yahoo)
        DATA->>DATA: Télécharger/MAJ EOD via yfinance si manquant
        DATA->>DATA: Écrire Parquet par ticker (Polars)
    else env.mode == prod (IBKR)
        DATA->>IB: Connexion ib_insync (localhost:7497)
        DATA->>IB: Requête historiques EOD par ticker (si manquants)
        IB-->>DATA: Barres historiques (OHLCV)
        DATA->>DATA: Écrire Parquet par ticker (Polars)
    end

    CLI->>DATA: Charger Parquet -> séries close (Polars→Pandas)
    CLI->>P: Générer toutes paires (N=15 → 105)
    P->>S: Corr, coint (ADF/fallback), half-life, sigma
    S-->>P: Scores composites par paire
    P-->>CLI: Top-K paires (filtres min_corr, max_half_life, pval_coint)

    CLI->>R: Générer rapport HTML (PNG PnL)
    R->>BT: Backtest rapide sur la meilleure paire
    BT-->>R: Journal, PnL cumulé
    R-->>CLI: Chemin report_<A>_<B>.html + PNG
    CLI->>DB: Écrire latest_pairs_scored.parquet → DuckDB
    CLI-->>U: Afficher Top-K / métriques + liens (rapport, Parquet, DB)

    note over U: Boucle jour 1: vérifier pairs, lancer backtest/rapport\nBoucle jour N: répéter, ajuster config si besoin
  ```

  18) Ameliorations iteratives
  ```mermaid
  sequenceDiagram
    autonumber
    actor U as Utilisateur (R&D)
    participant CFG as Config/params.yaml
    participant NB as Notebook d'exploration
    participant DATA as Data I/O
    participant P as Pairs (score/select)
    participant S as Stats (stationnarité, β, z)
    participant BT as Backtest (simulate_pair)
    participant REP as Reporting (HTML/PNG)
    participant QA as Validation (DuckDB/metrics)
    participant EXE as Exécution (IBKR paper/live)
    participant GIT as Git (versionnement)

    U->>CFG: Proposer changements (fenêtres, seuils, univers)
    U->>NB: Explorer spreads/z (top paires), sanity-check visuel
    NB->>DATA: Charger séries (Parquet)
    NB->>S: Tester nouvelles métriques (ex: rolling β, ADF strict)
    NB-->>U: Observations (bruit, half-life, stabilité)

    U->>P: Re-scoring complet (corr + coint + score)
    P->>BT: Backtests multi-paires (avec coûts/slippage)
    BT-->>QA: Résultats (Sharpe, DD, PF, turnover)

    alt Métriques OK ?
        QA-->>REP: Générer rapport consolidé + Top-K
        REP-->>U: Décision: garder paramètres
        U->>GIT: Commit config + outputs clés (Parquet/HTML)
        U->>EXE: (Option) Déployer en **paper trading** IBKR
        EXE-->>U: Logs ordres/fills, PnL live, alertes
    else Besoin d’améliorer
        QA-->>U: Identifier points faibles (faux signaux, coûts élevés)
        U->>CFG: Ajuster thresholds/lookbacks/filters
        U->>S: Ajouter features (regime filter, pénalisation half-life)
        U->>P: Re-run scoring/selection → boucle
    end

    loop Cadence (hebdo/mensuelle)
        U->>DATA: Rafraîchir données
        U->>P: Rebalance paires (pairs.rebalance_days)
        U->>BT: Backtests de continuité (walk-forward)
        U->>QA: Contrôles de dérive/robustesse
        U->>GIT: Versionner config + artefacts
    end
  ```