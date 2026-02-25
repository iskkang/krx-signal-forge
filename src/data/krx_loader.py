from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pykrx import stock


@dataclass
class KRXBars:
    df: pd.DataFrame  # index: date, cols: open/high/low/close/volume/turnover
    ticker: str


def fetch_ohlcv(
    ticker: str,
    end_yyyymmdd: str,
    lookback_days: int = 400,
) -> Optional[KRXBars]:
    """
    Returns df with columns:
      open, high, low, close, volume, turnover
    """
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(lookback_days * 2.2))  # 휴일 감안 여유
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    try:
        raw = stock.get_market_ohlcv_by_date(start, end, ticker)
        if raw is None or len(raw) == 0:
            return None

        # 표준 컬럼명 매핑
        # raw columns typically: 시가 고가 저가 종가 거래량 거래대금
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
            # 거래대금이 없다면 close*volume 근사 (정확도는 떨어짐)
            df["turnover"] = df["close"] * df["volume"]

        # 실제로는 영업일 수 기준으로 tail
        df = df.dropna().tail(int(lookback_days))
        if len(df) < int(lookback_days * 0.7):
            return None

        return KRXBars(df=df, ticker=ticker)
    except Exception:
        return None
