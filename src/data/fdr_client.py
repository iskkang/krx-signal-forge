from __future__ import annotations

import time
import random
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr

def fetch_ohlcv(ticker: str, start: str | None = None, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLCV from FinanceDataReader.
    For KRX tickers, use 6-digit code (e.g., 005930).
    """
    last_err: Exception | None = None
    for i in range(retries):
        try:
            df = fdr.DataReader(ticker, start=start, end=end)
            if df is None or df.empty:
                return pd.DataFrame()
            # Standardize columns
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })
            # Some sources may include extra columns; keep required
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df = df[keep]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        except Exception as e:
            last_err = e
            # small jitter backoff
            time.sleep(0.7 * (2 ** i) + random.random() * 0.3)
    raise RuntimeError(f"FDR fetch failed for {ticker}: {last_err}")

def recent_start_for_lookback(lookback_bars: int) -> str:
    # crude calendar: fetch a bit more than bars to cover holidays
    days = int(lookback_bars * 1.8)
    dt = datetime.utcnow() - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
