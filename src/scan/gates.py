# src/scan/gates.py
from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional

from .indicators import sma, atr, ema

def gate_setup_pullback(
    df: pd.DataFrame,
    max_day_ret_pct: float = 8.0,
    max_gap_pct: float = 4.0,
    ema20_band_pct: float = 3.0,
    max_extended_pct: float = 6.0,
    hh_lookback: int = 60,
    min_from_hh_pct: float = 85.0,
) -> GateResult:
    if df is None or df.empty or len(df) < max(hh_lookback, 60, 200) + 5:
        return GateResult(False, "setup_history_short")

    o = float(df["Open"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    prev_c = float(df["Close"].iloc[-2])

    # A) 급등 컷: 당일 수익률, 갭
    day_ret = (c / prev_c - 1.0) * 100.0
    gap = (o / prev_c - 1.0) * 100.0
    if day_ret > max_day_ret_pct:
        return GateResult(False, "setup_day_spike")
    if gap > max_gap_pct:
        return GateResult(False, "setup_gap_up")

    # 추세선들
    ema20 = ema(df["Close"], 20).iloc[-1]
    ema50 = ema(df["Close"], 50).iloc[-1]
    ma200 = sma(df["Close"], 200).iloc[-1]
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ma200):
        return GateResult(False, "setup_ma_nan")

    # D) 상승 추세 기반 (기본)
    if not (ema20 >= ema50 and c > ma200):
        return GateResult(False, "setup_trend_not_ok")

    # B) EMA20 근처 회복(눌림)
    band = abs(c - float(ema20)) / float(ema20) * 100.0
    if band > ema20_band_pct:
        return GateResult(False, "setup_not_near_ema20")

    # C) 과열/확장 컷(추격 금지)
    extended = (c / float(ema20) - 1.0) * 100.0
    if extended > max_extended_pct:
        return GateResult(False, "setup_extended")

    # 60일 고점 대비 너무 멀면 제외(바닥 반등주 제거)
    hh = float(df["Close"].tail(hh_lookback).max())
    if hh <= 0:
        return GateResult(False, "setup_hh_bad")
    if c < hh * (min_from_hh_pct / 100.0):
        return GateResult(False, "setup_too_far_from_hh")

    return GateResult(True, "ok")
