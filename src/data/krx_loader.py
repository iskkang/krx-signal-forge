from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pykrx import stock


@dataclass
class KRXBars:
    ticker: str
    df: pd.DataFrame  # index: date, cols: open/high/low/close/volume/turnover


def fetch_ohlcv(
    ticker: str,
    end_yyyymmdd: str,
    lookback_days: int = 420,
) -> Optional[KRXBars]:
    """
    Fetch OHLCV+turnover (KRW) for a single ticker.

    Returns df with columns:
      open, high, low, close, volume, turnover
    """
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(lookback_days * 2.2))  # holiday buffer
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    try:
        raw = stock.get_market_ohlcv_by_date(start, end, ticker)
        if raw is None or len(raw) == 0:
            return None

        required = {"시가", "고가", "저가", "종가", "거래량"}
        if not required.issubset(set(raw.columns)):
            return None

        df = pd.DataFrame(index=raw.index.copy())
        df["open"] = raw["시가"].astype("float64")
        df["high"] = raw["고가"].astype("float64")
        df["low"] = raw["저가"].astype("float64")
        df["close"] = raw["종가"].astype("float64")
        df["volume"] = raw["거래량"].astype("float64")

        if "거래대금" in raw.columns:
            df["turnover"] = raw["거래대금"].astype("float64")
        else:
            df["turnover"] = df["close"] * df["volume"]

        df = df.dropna()
        df = df.tail(int(lookback_days))

        if len(df) < int(lookback_days * 0.7):
            return None

        return KRXBars(ticker=ticker, df=df)

    except Exception:
        return None
