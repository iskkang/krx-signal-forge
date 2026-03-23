from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from .indicators import sma, atr, ema, ema_slope_pct


@dataclass
class GateResult:
    ok: bool
    reason: str


# ─────────────────────────────────────────────
# 기본 게이트
# ─────────────────────────────────────────────

def gate_history(df: pd.DataFrame, min_bars: int) -> GateResult:
    if df is None or df.empty or len(df) < min_bars:
        return GateResult(False, "missing_history")
    if df.iloc[-1].isna().any():
        return GateResult(False, "nan_last_row")
    return GateResult(True, "ok")


def gate_liquidity(df: pd.DataFrame, min_price_krw: float, min_traded_value_20d_krw: float) -> GateResult:
    close = df["Close"]
    vol = df["Volume"]
    last_close = float(close.iloc[-1])
    if last_close < min_price_krw:
        return GateResult(False, "low_price")
    traded_value = (close * vol).rolling(20).mean().iloc[-1]
    if pd.isna(traded_value) or float(traded_value) < min_traded_value_20d_krw:
        return GateResult(False, "low_traded_value")
    return GateResult(True, "ok")


def gate_volatility(df: pd.DataFrame, min_atr_pct: float, max_atr_pct: float) -> GateResult:
    a = atr(df, 14).iloc[-1]
    c = df["Close"].iloc[-1]
    if pd.isna(a) or pd.isna(c) or float(c) <= 0:
        return GateResult(False, "atr_nan")
    atr_pct = float(a / c * 100.0)
    if atr_pct < min_atr_pct:
        return GateResult(False, "atr_too_low")
    if atr_pct > max_atr_pct:
        return GateResult(False, "atr_too_high")
    return GateResult(True, "ok")


def gate_trend_ma200(df: pd.DataFrame) -> GateResult:
    ma200 = sma(df["Close"], 200).iloc[-1]
    if pd.isna(ma200):
        return GateResult(False, "ma200_nan")
    if float(df["Close"].iloc[-1]) <= float(ma200):
        return GateResult(False, "below_ma200")
    return GateResult(True, "ok")


def gate_ema20_slope(
    df: pd.DataFrame,
    min_slope_pct: float = 0.0,
    lookback: int = 10,
) -> GateResult:
    """EMA20이 우상향 중인지 확인.
    min_slope_pct=0.0 이면 하락 EMA20만 제거 (수평은 허용).
    """
    slope = ema_slope_pct(df["Close"], 20, lookback=lookback)
    if slope < min_slope_pct:
        return GateResult(False, "ema20_not_rising")
    return GateResult(True, "ok")


def gate_setup_pullback(
    df: pd.DataFrame,
    max_day_ret_pct: float = 8.0,
    max_gap_pct: float = 4.0,
    max_extended_pct: float = 8.0,
    max_below_ema20_pct: float = 10.0,
    hh_lookback: int = 60,
    min_from_hh_pct: float = 80.0,
) -> GateResult:
    """사전 필터: 스파이크·갭업 제거 + EMA20 근방 확인 + 60일 고점 근처 확인.

    v3 핵심 수정 — ema20_band_pct(대칭) → 비대칭 구조로 변경:
      • 하락 방향: close가 EMA20 아래 max_below_ema20_pct(10%) 초과면 제거
                   (너무 멀리 떨어져 있어 당일 재탈환 불가)
      • 상승 방향: close가 EMA20 위 max_extended_pct(8%) 초과면 제거
                   (이미 많이 올라 추격매수 위험)
      이렇게 하면 강한 재탈환 캔들(종가 EMA20 +3~6%)이 살아남는다.
    """
    need = max(hh_lookback, 200) + 5
    if df is None or df.empty or len(df) < need:
        return GateResult(False, "setup_history_short")

    o = float(df["Open"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])

    day_ret = (c / prev_c - 1.0) * 100.0
    gap = (o / prev_c - 1.0) * 100.0

    if day_ret > max_day_ret_pct:
        return GateResult(False, "setup_day_spike")
    if gap > max_gap_pct:
        return GateResult(False, "setup_gap_up")

    ema20_v = float(ema(df["Close"], 20).iloc[-1])
    if ema20_v <= 0:
        return GateResult(False, "setup_ma_nan")

    deviation = (c - ema20_v) / ema20_v * 100.0  # 양수=위, 음수=아래

    if deviation < -max_below_ema20_pct:
        # EMA20 아래로 너무 멀리 — 당일 재탈환 불가능
        return GateResult(False, "setup_too_far_below_ema20")

    if deviation > max_extended_pct:
        # EMA20 위로 너무 높이 — 이미 과열, 추격 위험
        return GateResult(False, "setup_too_extended")

    hh = float(df["Close"].tail(hh_lookback).max())
    if hh <= 0:
        return GateResult(False, "setup_hh_bad")
    if c < hh * (min_from_hh_pct / 100.0):
        return GateResult(False, "setup_too_far_from_hh")

    return GateResult(True, "ok")


# ─────────────────────────────────────────────
# strategy eval 이후 게이트 (reclaim 발생 확인 후)
# ─────────────────────────────────────────────

def gate_reclaim_candle(
    df: pd.DataFrame,
    min_close_pct: float = 0.55,
    min_body_ratio: float = 0.35,
) -> GateResult:
    """재탈환 캔들 강도 검증 (strategy eval 이후에만 호출).

    조건:
      1) 양봉 (close > open)
      2) 종가가 당일 범위 상위 45% 이상에 위치
      3) 캔들 몸통이 전체 범위 35% 이상
    """
    o = float(df["Open"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])

    bar_range = h - l
    if bar_range <= 0:
        return GateResult(False, "zero_range_candle")
    if c <= o:
        return GateResult(False, "bearish_reclaim_candle")

    close_pct = (c - l) / bar_range
    if close_pct < min_close_pct:
        return GateResult(False, "weak_close_position")

    body_ratio = abs(c - o) / bar_range
    if body_ratio < min_body_ratio:
        return GateResult(False, "small_body_candle")

    return GateResult(True, "ok")


def gate_pullback_quality(
    df: pd.DataFrame,
    min_days: int = 2,
    max_days: int = 20,
    max_vol_ratio: float = 0.85,
    max_depth_pct: float = 15.0,
) -> GateResult:
    """눌림 품질 검증 (strategy eval 이후에만 호출).

    건강한 눌림 3조건:
      1) 기간 2~20일 (너무 짧으면 노이즈, 너무 길면 추세 훼손)
      2) 눌림 중 거래량 VMA20 대비 85% 이하 (조용한 매도)
      3) 직전 고점 대비 낙폭 15% 이내
    """
    close = df["Close"]
    vol = df["Volume"]
    e = ema(close, 20)
    v20 = sma(vol, 20)

    pb_idx: list[int] = []
    for i in range(len(df) - 2, max(0, len(df) - 2 - max_days - 5), -1):
        if close.iloc[i] <= e.iloc[i]:
            pb_idx.append(i)
        else:
            break

    n_days = len(pb_idx)
    if n_days < min_days:
        return GateResult(False, "pullback_too_short")
    if n_days > max_days:
        return GateResult(False, "pullback_too_long")

    vma_val = float(v20.iloc[-1]) if not pd.isna(v20.iloc[-1]) else 0.0
    if vma_val > 0:
        pb_vol_mean = float(vol.iloc[pb_idx].mean())
        if pb_vol_mean / vma_val > max_vol_ratio:
            return GateResult(False, "pullback_vol_high")

    earliest = min(pb_idx)
    if earliest >= 1:
        pre_slice = close.iloc[max(0, earliest - 20): earliest]
        if not pre_slice.empty:
            pre_high = float(pre_slice.max())
            pb_low = float(close.iloc[pb_idx].min())
            if pre_high > 0:
                depth = (1.0 - pb_low / pre_high) * 100.0
                if depth > max_depth_pct:
                    return GateResult(False, "pullback_too_deep")

    return GateResult(True, "ok")


def gate_market_regime(
    index_df: Optional[pd.DataFrame],
    ma_period: int = 50,
) -> GateResult:
    """시장 레짐 필터: KOSPI 지수가 MA50 위에 있어야 한다.
    index_df 가 없으면 통과 처리.
    """
    if index_df is None or index_df.empty or len(index_df) < ma_period + 5:
        return GateResult(True, "no_index_data")
    ma = sma(index_df["Close"], ma_period).iloc[-1]
    last = float(index_df["Close"].iloc[-1])
    if pd.isna(ma) or float(ma) <= 0:
        return GateResult(True, "ma_nan")
    if last < float(ma):
        return GateResult(False, f"market_below_ma{ma_period}")
    return GateResult(True, "ok")
