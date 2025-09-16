from __future__ import annotations
from pathlib import Path
import datetime as dt
import polars as pl
import pandas as pd

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch_eod_yfinance(ticker: str, start: str = '2015-01-01', end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception:
        raise RuntimeError("yfinance n'est pas installé.")
    if end is None:
        end = dt.date.today().isoformat()
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée pour {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
    df.index.name = 'date'
    return df[['open','high','low','close','adj_close','volume']].astype(float)

def to_parquet_polars(df: pd.DataFrame, out_path: str | Path):
    out_path = Path(out_path)
    _ensure_dir(out_path)
    tbl = pl.from_pandas(df.reset_index())
    tbl = tbl.with_columns([pl.col('date').cast(pl.Datetime)])
    tbl.write_parquet(out_path, compression='snappy')

def load_parquet_polars(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(str(path))

def get_price_series(root_dir: str | Path, ticker: str) -> pl.DataFrame:
    p = Path(root_dir) / f"{ticker}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {p}.")
    return load_parquet_polars(p)

def ensure_universe_parquet(tickers: list[str], root_dir: str | Path, start='2018-01-01'):
    for t in tickers:
        out = Path(root_dir) / f"{t}.parquet"
        if out.exists():
            continue
        df = fetch_eod_yfinance(t, start=start)
        to_parquet_polars(df, out)

# ==== IBKR ingestion (PROD) ====
def _ibkr_connect(host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
    from ib_insync import IB
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

def fetch_eod_ibkr(ib, ticker: str, currency: str = 'USD', exchange: str = 'SMART',
                   duration: str = '3 Y', bar_size: str = '1 day',
                   what_to_show: str = 'ADJUSTED_LAST', useRTH: bool = True):
    from ib_insync import Stock, util
    df = None
    contract = Stock(ticker, exchange, currency)
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=duration,
                                barSizeSetting=bar_size, whatToShow=what_to_show,
                                useRTH=useRTH, formatDate=1)
    df = util.df(bars)
    if df.empty:
        raise RuntimeError(f"Aucune donnée IBKR pour {ticker}")
    df = df.rename(columns={"date":"date","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    df = df[["date","open","high","low","close","volume"]]
    df['date'] = pd.to_datetime(df['date'])
    return df

def ensure_universe_parquet_ibkr(tickers: list[str], root_dir: str | Path,
                                 host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
    ib = _ibkr_connect(host, port, client_id)
    try:
        for t in tickers:
            out = Path(root_dir) / f"{t}.parquet"
            if out.exists():
                continue
            df = fetch_eod_ibkr(ib, t)
            to_parquet_polars(df.set_index('date'), out)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

def get_prices_by_env(ticker: str, params: dict, start: str = '2018-01-01'):
    mode = params.get('env', {}).get('mode', 'dev')
    if mode == 'dev':
        df = fetch_eod_yfinance(ticker, start=start)
        return df
    elif mode == 'prod':
        raise NotImplementedError("Mode prod/IBKR: ingestion via scripts/ingest_ibkr.py puis lecture Parquet.")
    else:
        raise ValueError(f"env.mode inconnu: {mode}")

def ensure_universe(params: dict):
    tickers = params['universe']['tickers']
    root_dir = Path(params['data']['root_dir'])
    mode = params.get('env',{}).get('mode','dev')
    if mode == 'dev':
        ensure_universe_parquet(tickers, root_dir)
    elif mode == 'prod':
        missing = [t for t in tickers if not (root_dir / f"{t}.parquet").exists()]
        if missing:
            ensure_universe_parquet_ibkr(missing, root_dir)
    else:
        raise ValueError(f"env.mode inconnu: {mode}")
