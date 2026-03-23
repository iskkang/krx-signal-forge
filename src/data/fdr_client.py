from __future__ import annotations

import time
import random
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr


def fetch_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """FinanceDataReader 로 OHLCV 조회.

    KRX 종목: 6자리 코드 (e.g., 005930)
    지수: KS11(KOSPI), KQ11(KOSDAQ)
    """
    last_err: Exception | None = None
    for i in range(retries):
        try:
            df = fdr.DataReader(ticker, start=start, end=end)
            if df is None or df.empty:
                return pd.DataFrame()
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # 지수는 Volume 이 없을 수 있음 — 0 으로 채움
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.7 * (2 ** i) + random.random() * 0.5)
    raise RuntimeError(f"FDR fetch failed for {ticker}: {last_err}")


def recent_start_for_lookback(lookback_bars: int) -> str:
    """lookback_bars 거래일을 커버하는 시작일 (달력일 기준 넉넉하게)."""
    days = int(lookback_bars * 1.85)
    dt = datetime.utcnow() - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
