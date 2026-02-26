from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from .indicators import sma, atr

@dataclass
class GateResult:
    ok: bool
    reason: str

def gate_history(df: pd.DataFrame, min_bars: int) -> GateResult:
    if df is None or df.empty or len(df) < min_bars:
        return GateResult(False, "missing_history")
    if df.isna().any().any():
        # allow minor NaNs early, but not last row
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
