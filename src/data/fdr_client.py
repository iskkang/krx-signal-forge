from __future__ import annotations

import time
import random
import threading
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr


def fetch_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    retries: int = 2,
    timeout_sec: float = 12.0,
) -> pd.DataFrame:
    """FinanceDataReader 로 OHLCV 조회.

    timeout_sec: 단일 요청 최대 대기 시간. 초과 시 빈 DataFrame 반환.
    retries: 타임아웃/에러 시 재시도 횟수 (기본 2 → 최대 3회 시도).
    """
    last_err: Exception | None = None

    for i in range(retries + 1):
        result: list[pd.DataFrame] = []
        exc: list[Exception] = []

        def _fetch():
            try:
                df = fdr.DataReader(ticker, start=start, end=end)
                result.append(df)
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        if t.is_alive():
            # 타임아웃 — 스레드는 daemon이라 메인 종료 시 자동 정리
            last_err = TimeoutError(f"FDR timeout after {timeout_sec}s for {ticker}")
            if i < retries:
                time.sleep(0.5 * (i + 1))
            continue

        if exc:
            last_err = exc[0]
            if i < retries:
                time.sleep(0.5 * (i + 1) + random.random() * 0.3)
            continue

        df = result[0] if result else None
        if df is None or df.empty:
            return pd.DataFrame()

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        return df

    # 모든 재시도 소진
    return pd.DataFrame()


def recent_start_for_lookback(lookback_bars: int) -> str:
    days = int(lookback_bars * 1.85)
    dt = datetime.utcnow() - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
