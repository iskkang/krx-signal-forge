from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .indicators import sma, atr, ema


@dataclass
class GateResult:
    ok: bool
    reason: str


def gate_history(df: pd.DataFrame, min_bars: int) -> GateResult:
    if df is None or df.empty or len(df) < min_bars:
        return GateResult(False, "missing_history")
    # allow NaNs early, but not in the last row
    if df.isna().any().any() and df.iloc[-1].isna().any():
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
    max_extended_pct: float = 6.0,
    hh_lookback: int = 60,
    min_from_hh_pct: float = 85.0,
) -> GateResult:
    """
    G2 setup filter to avoid "today spike" names and force a pullback/reclaim structure.
    - Reject big day spikes / big gaps
    - Require price near EMA20 (pullback zone)
    - Reject too extended vs EMA20 (no chasing)
    - Keep names not too far from recent highs (avoid dead-cat bounces)
    """
    need = max(hh_lookback, 200) + 5
    if df is None or df.empty or len(df) < need:
        return GateResult(False, "setup_history_short")

    o = float(df["Open"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])

    # A) reject big day spike and big gap-up
    day_ret = (c / prev_c - 1.0) * 100.0
    gap = (o / prev_c - 1.0) * 100.0
    if day_ret > max_day_ret_pct:
        return GateResult(False, "setup_day_spike")
    if gap > max_gap_pct:
        return GateResult(False, "setup_gap_up")

    ema20_v = ema(df["Close"], 20).iloc[-1]
    ema50_v = ema(df["Close"], 50).iloc[-1]
    ma200_v = sma(df["Close"], 200).iloc[-1]
    if pd.isna(ema20_v) or pd.isna(ema50_v) or pd.isna(ma200_v):
        return GateResult(False, "setup_ma_nan")

    # D) trend bias
    if not (float(ema20_v) >= float(ema50_v) and c > float(ma200_v)):
        return GateResult(False, "setup_trend_not_ok")

    # B) must be near EMA20 (pullback zone)
    band = abs(c - float(ema20_v)) / float(ema20_v) * 100.0
    if band > ema20_band_pct:
        return GateResult(False, "setup_not_near_ema20")

    # C) not too extended vs EMA20 (no chase)
    extended = (c / float(ema20_v) - 1.0) * 100.0
    if extended > max_extended_pct:
        return GateResult(False, "setup_extended")

    # recent high proximity (avoid deep bounces)
    hh = float(df["Close"].tail(hh_lookback).max())
    if hh <= 0:
        return GateResult(False, "setup_hh_bad")
    if c < hh * (min_from_hh_pct / 100.0):
        return GateResult(False, "setup_too_far_from_hh")

    return GateResult(True, "ok")
