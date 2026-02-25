from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd

from .indicators import sma, ema


class GateResult(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Candidate:
    ticker: str
    name: str | None
    price: float
    turnover_krw_20d: float
    score: float
    tag: str
    reason: str


def _mean_turnover(df: pd.DataFrame, n: int = 20) -> float:
    tv = df["turnover"].tail(n)
    if tv.isna().any():
        tv = tv.dropna()
    return float(tv.mean()) if len(tv) > 0 else float("nan")


def apply_gates(
    ticker: str,
    name: str | None,
    bars: pd.DataFrame,
    rules: dict,
    drop_counter: Dict[str, int],
    skip_counter: Dict[str, int],
) -> Tuple[GateResult, Candidate | None]:
    """
    Practical swing template gates:
      G1: data/liquidity/price
      G2: trend + EMA reclaim setup
      G3: volume spike + short breakout trigger
    """
    g1 = rules["g1"]
    g2 = rules["g2"]
    g3 = rules["g3"]

    if bars is None or len(bars) == 0:
        skip_counter["skip_empty_bars"] = skip_counter.get("skip_empty_bars", 0) + 1
        return GateResult.SKIP, None

    if len(bars) < int(g1["min_history_days"]):
        skip_counter["skip_short_history"] = skip_counter.get("skip_short_history", 0) + 1
        return GateResult.SKIP, None

    close = bars["close"]
    price = float(close.iloc[-1])

    if price < float(g1["min_price_krw"]):
        drop_counter["g1_low_price"] = drop_counter.get("g1_low_price", 0) + 1
        return GateResult.FAIL, None

    turnover_20d = _mean_turnover(bars, 20)
    if not (turnover_20d == turnover_20d):  # NaN
        skip_counter["skip_no_turnover"] = skip_counter.get("skip_no_turnover", 0) + 1
        return GateResult.SKIP, None

    if turnover_20d < float(g1["min_turnover_krw_20d"]):
        drop_counter["g1_low_turnover"] = drop_counter.get("g1_low_turnover", 0) + 1
        return GateResult.FAIL, None

    # --- G2: trend + reclaim setup
    ma200 = sma(close, int(g2["trend_ma_days"]))
    ema20 = ema(close, int(g2["ema_fast_days"]))
    ema50 = ema(close, int(g2["ema_mid_days"]))

    if g2.get("require_close_above_ma200", True):
        if float(close.iloc[-1]) <= float(ma200.iloc[-1]):
            drop_counter["g2_below_ma200"] = drop_counter.get("g2_below_ma200", 0) + 1
            return GateResult.FAIL, None

    # "EMA20 reclaim within N days": was below ema20, then close above ema20 recently
    reclaim_days = int(g2["reclaim_within_days"])
    recent = bars.tail(reclaim_days + 2)
    rec_close = recent["close"]
    rec_ema20 = ema20.loc[recent.index]

    was_below = (rec_close.shift(1) < rec_ema20.shift(1)).any()
    now_above = float(close.iloc[-1]) > float(ema20.iloc[-1])

    if not (was_below and now_above):
        drop_counter["g2_no_ema20_reclaim"] = drop_counter.get("g2_no_ema20_reclaim", 0) + 1
        # This is a good "SOFT" condition: trend OK but not reclaimed yet
        cand = Candidate(
            ticker=ticker,
            name=name,
            price=price,
            turnover_krw_20d=turnover_20d,
            score=0.0,
            tag="SOFT_SETUP",
            reason="Trend OK, waiting for EMA20 reclaim"
        )
        return GateResult.SOFT, cand

    # Additional setup: ema20 >= ema50 (trend alignment)
    if float(ema20.iloc[-1]) < float(ema50.iloc[-1]):
        drop_counter["g2_ema20_below_ema50"] = drop_counter.get("g2_ema20_below_ema50", 0) + 1
        cand = Candidate(
            ticker=ticker,
            name=name,
            price=price,
            turnover_krw_20d=turnover_20d,
            score=0.0,
            tag="SOFT_SETUP",
            reason="EMA20 reclaimed but EMA20<EMA50 (needs strength)"
        )
        return GateResult.SOFT, cand

    # --- G3: trigger
    vol = bars["volume"]
    vol_lookback = int(g3["vol_lookback"])
    vol20 = float(vol.tail(vol_lookback).mean())
    vol_today = float(vol.iloc[-1])
    vol_spike = vol_today >= vol20 * float(g3["vol_spike_mult"])

    # breakout: close above highest close of last N days (excluding today)
    brk_n = int(g3["breakout_lookback"])
    prev_max = float(close.iloc[-(brk_n+1):-1].max())
    breakout = float(close.iloc[-1]) > prev_max

    if not (vol_spike and breakout):
        drop_counter["g3_no_trigger"] = drop_counter.get("g3_no_trigger", 0) + 1
        cand = Candidate(
            ticker=ticker,
            name=name,
            price=price,
            turnover_krw_20d=turnover_20d,
            score=0.0,
            tag="SOFT_SETUP",
            reason="Setup OK, waiting for vol-spike + breakout trigger"
        )
        return GateResult.SOFT, cand

    # scoring: prefer higher turnover and bigger breakout margin
    breakout_margin = (float(close.iloc[-1]) / prev_max - 1.0) * 100.0
    score = (turnover_20d / 1e9) + breakout_margin  # simple

    cand = Candidate(
        ticker=ticker,
        name=name,
        price=price,
        turnover_krw_20d=turnover_20d,
        score=float(score),
        tag="HARD",
        reason=f"EMA20 reclaim + vol spike + {brk_n}D breakout"
    )
    return GateResult.HARD, cand
