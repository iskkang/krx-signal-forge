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
# 기존 게이트 (유지 + 정리)
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


def gate_setup_pullback(
    df: pd.DataFrame,
    max_day_ret_pct: float = 8.0,
    max_gap_pct: float = 4.0,
    ema20_band_pct: float = 3.0,
    hh_lookback: int = 60,
    min_from_hh_pct: float = 80.0,
) -> GateResult:
    """당일 급등·갭업 스파이크 제거 + 60일 고점 근처 확인.

    v2 변경: 중복 추세 체크(MA200, EMA스택) 제거.
    max_extended_pct 제거 (ema20_band_pct 안에 이미 포함됨).
    min_from_hh_pct 85% → 80% 로 완화 (좋은 눌림목 포착).
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

    band = abs(c - ema20_v) / ema20_v * 100.0
    if band > ema20_band_pct:
        return GateResult(False, "setup_not_near_ema20")

    hh = float(df["Close"].tail(hh_lookback).max())
    if hh <= 0:
        return GateResult(False, "setup_hh_bad")
    if c < hh * (min_from_hh_pct / 100.0):
        return GateResult(False, "setup_too_far_from_hh")

    return GateResult(True, "ok")


# ─────────────────────────────────────────────
# 신규 게이트 (A+ 업그레이드)
# ─────────────────────────────────────────────

def gate_ema20_slope(
    df: pd.DataFrame,
    min_slope_pct: float = 0.5,
    lookback: int = 10,
) -> GateResult:
    """EMA20이 우상향 중인지 확인.

    수평·하락 EMA20 위에서의 재탈환은 가짜 신호 비율이 높다.
    min_slope_pct: 10거래일간 EMA20 상승률 최솟값(%).
    """
    slope = ema_slope_pct(df["Close"], 20, lookback=lookback)
    if slope < min_slope_pct:
        return GateResult(False, "ema20_not_rising")
    return GateResult(True, "ok")


def gate_pullback_quality(
    df: pd.DataFrame,
    min_days: int = 2,
    max_days: int = 20,
    max_vol_ratio: float = 0.85,
    max_depth_pct: float = 15.0,
) -> GateResult:
    """눌림 품질 검증.

    건강한 눌림의 3가지 조건:
      1) 기간: 너무 짧거나(1일 노이즈) 너무 길지 않아야 한다
      2) 거래량: 눌림 중 거래량이 VMA20 대비 줄어야 한다 (매도 압력 약함)
      3) 깊이: 직전 고점 대비 15% 이내 눌림 (추세 훼손 없음)
    """
    close = df["Close"]
    vol = df["Volume"]
    e = ema(close, 20)
    v20 = sma(vol, 20)

    # 어제(iloc[-2])부터 역방향으로 EMA 아래 연속 봉 수집
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

    # 눌림 깊이: 직전 20봉 고점 대비 낙폭
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


def gate_reclaim_candle(
    df: pd.DataFrame,
    min_close_pct: float = 0.60,
    min_body_ratio: float = 0.40,
) -> GateResult:
    """재탈환 캔들 강도 검증.

    진짜 돌파 캔들의 특성:
      1) 종가가 당일 범위의 상위 40% 이상에 위치 (close_pct >= 0.60)
      2) 캔들 몸통이 전체 범위의 40% 이상 (body_ratio >= 0.40)
      3) 양봉이어야 한다 (close > open)

    윗꼬리 긴 도지, 음봉 돌파 등을 자동 제거.
    """
    o = float(df["Open"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])

    bar_range = h - l
    if bar_range <= 0:
        return GateResult(False, "zero_range_candle")

    close_pct = (c - l) / bar_range
    if close_pct < min_close_pct:
        return GateResult(False, "weak_close_position")

    body_ratio = abs(c - o) / bar_range
    if body_ratio < min_body_ratio:
        return GateResult(False, "small_body_candle")

    if c <= o:
        return GateResult(False, "bearish_reclaim_candle")

    return GateResult(True, "ok")


def gate_market_regime(index_df: Optional[pd.DataFrame], ma_period: int = 50) -> GateResult:
    """시장 레짐 필터: KOSPI 지수가 MA50 위에 있어야 한다.

    지수가 MA50 아래일 때 개별주 매수 신호는 성공률이 급격히 낮아진다.
    index_df 가 없으면 통과 처리 (데이터 미확보 시 스캔 막지 않음).
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
