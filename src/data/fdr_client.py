from __future__ import annotations

import time
import random
import threading
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr


# 데이터 신선도 허용 최대 일수 (달력일 기준)
# 월~금 장 마감 후 실행 가정: 주말 포함 최대 4일(금→월) 허용
_MAX_STALE_DAYS = 6


def _validate_freshness(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """fetch된 df의 최신 날짜가 충분히 최근인지 검증.

    Returns:
        df 그대로 반환 (유효)
        None (데이터가 너무 오래됨 — stale 또는 FDR 오류)
    """
    if df is None or df.empty:
        return None
    latest = df.index[-1]
    stale_days = (pd.Timestamp.now() - latest).days
    if stale_days > _MAX_STALE_DAYS:
        print(f"  [STALE] {ticker}: 최신 날짜 {latest.date()} ({stale_days}일 전) — 스킵")
        return None
    return df


def fetch_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    retries: int = 2,
    timeout_sec: float = 12.0,
) -> pd.DataFrame:
    """FinanceDataReader OHLCV 조회 + 신선도 검증.

    신선도 검증 실패(너무 오래된 데이터) → 빈 DataFrame 반환.
    타임아웃 → 빈 DataFrame 반환 (무한 대기 없음).
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

        # ── 신선도 검증: 너무 오래된 데이터면 빈 DF 반환 ──
        validated = _validate_freshness(df, ticker)
        if validated is None:
            return pd.DataFrame()

        return validated

    return pd.DataFrame()


def recent_start_for_lookback(lookback_bars: int) -> str:
    days = int(lookback_bars * 1.85)
    dt = datetime.utcnow() - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
