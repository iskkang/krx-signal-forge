from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr


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
    FinanceDataReader 기반 OHLCV.
    - 컬럼: Open/High/Low/Close/Volume (일반적으로 제공)
    - turnover는 close*volume으로 근사 (거래대금 컬럼이 없기 때문)
    """
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(lookback_days * 2.2))
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    try:
        raw = fdr.DataReader(ticker, start, end)
        if raw is None or len(raw) == 0:
            return None

        # 표준화
        df = pd.DataFrame(index=raw.index.copy())
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in raw.columns:
                return None

        df["open"] = pd.to_numeric(raw["Open"], errors="coerce").astype("float64")
        df["high"] = pd.to_numeric(raw["High"], errors="coerce").astype("float64")
        df["low"] = pd.to_numeric(raw["Low"], errors="coerce").astype("float64")
        df["close"] = pd.to_numeric(raw["Close"], errors="coerce").astype("float64")
        df["volume"] = pd.to_numeric(raw["Volume"], errors="coerce").astype("float64")

        df = df.dropna()
        df["turnover"] = df["close"] * df["volume"]

        df = df.tail(int(lookback_days))
        if len(df) < int(lookback_days * 0.7):
            return None

        return KRXBars(ticker=ticker, df=df)

    except Exception:
        return None
