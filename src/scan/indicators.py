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
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    return pd.Series(tr, index=df.index).rolling(n).mean()


def ema_slope_pct(s: pd.Series, n: int, lookback: int = 10) -> float:
    """EMA(n) 의 lookback 기간 기울기 (% 단위).
    양수 = 상승, 음수 = 하락."""
    e = ema(s, n)
    if len(e) < lookback + 2:
        return 0.0
    base = float(e.iloc[-(lookback + 1)])
    if base <= 0:
        return 0.0
    return float((e.iloc[-1] - base) / base * 100.0)


def rs_ratio(stock_close: pd.Series, index_close: pd.Series, period: int = 63) -> float:
    """12주(63거래일) 상대강도: 종목 수익률 - 지수 수익률 (% 단위).
    양수 = 지수 대비 아웃퍼폼."""
    if index_close is None or index_close.empty:
        return 0.0
    if len(stock_close) < period or len(index_close) < period:
        return 0.0
    s_ret = float(stock_close.iloc[-1] / stock_close.iloc[-period] - 1.0) * 100.0
    i_ret = float(index_close.iloc[-1] / index_close.iloc[-period] - 1.0) * 100.0
    return round(s_ret - i_ret, 2)


def count_ema20_dips(close: pd.Series, ema_n: int = 20, lookback: int = 120) -> int:
    """최근 lookback 봉 안에서 EMA(n) 아래로 새로 진입한 횟수.
    1st pullback=1, 3rd=3. 숫자가 작을수록 신선한 신호."""
    e = ema(close, ema_n)
    n = min(lookback, len(close) - 1)
    below = (close.iloc[-n:] < e.iloc[-n:]).values
    dips = 0
    prev = False
    for b in below:
        if b and not prev:
            dips += 1
        prev = b
    return dips


def pullback_metrics(df: pd.DataFrame, ema_n: int = 20, max_search: int = 35) -> dict:
    """EMA 재탈환 직전 눌림의 품질 지표.

    Returns dict with:
        pullback_days       : EMA 아래 머문 연속 봉 수
        pullback_vol_ratio  : 눌림 평균 거래량 / VMA20
        pullback_depth_pct  : 눌림 저점 낙폭 %
    """
    close = df["Close"]
    vol = df["Volume"]
    e = ema(close, ema_n)
    v20 = sma(vol, 20)

    pb_idx: list[int] = []
    for i in range(len(df) - 2, max(0, len(df) - 2 - max_search), -1):
        if close.iloc[i] <= e.iloc[i]:
            pb_idx.append(i)
        else:
            break

    n_days = len(pb_idx)
    vma_val = float(v20.iloc[-1]) if not pd.isna(v20.iloc[-1]) else 1.0

    pb_vol_ratio = 0.0
    pb_depth_pct = 0.0

    if pb_idx and vma_val > 0:
        pb_vol_mean = float(vol.iloc[pb_idx].mean())
        pb_vol_ratio = round(pb_vol_mean / vma_val, 3)
        earliest = min(pb_idx)
        if earliest >= 1:
            pre_slice = close.iloc[max(0, earliest - 20): earliest]
            if not pre_slice.empty:
                pre_high = float(pre_slice.max())
                pb_low = float(close.iloc[pb_idx].min())
                if pre_high > 0:
                    pb_depth_pct = round((1.0 - pb_low / pre_high) * 100.0, 2)

    return {
        "pullback_days": n_days,
        "pullback_vol_ratio": pb_vol_ratio,
        "pullback_depth_pct": pb_depth_pct,
    }
