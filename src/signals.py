from __future__ import annotations
import pandas as pd
from .stats import zscore, spread_series

def pair_signals(y_close: pd.Series, x_close: pd.Series, alpha: float, beta: float, z_win: int,
                 entry_z: float, exit_z: float, stop_z: float) -> pd.DataFrame:
    spr = spread_series(y_close, x_close, alpha, beta)
    z = zscore(spr, z_win)
    df = pd.concat([spr.rename('spread'), z], axis=1).dropna()
    side = []
    for val in df['z']:
        if abs(val) >= stop_z:
            side.append(0)
        elif val >= entry_z:
            side.append(-1)
        elif val <= -entry_z:
            side.append(1)
        elif abs(val) <= exit_z:
            side.append(0)
        else:
            side.append(None)
    df['raw_signal'] = side
    df['signal'] = df['raw_signal'].ffill().fillna(0).astype(int)
    return df
