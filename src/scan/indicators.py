from __future__ import annotations
import numpy as np
import pandas as pd

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return pd.Series(tr, index=df.index).rolling(n).mean()
