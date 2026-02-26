from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

from .scan.gates import gate_history, gate_liquidity, gate_volatility, gate_trend_ma200
from .scan.strategy import eval_ema20_reclaim_with_volume, SignalEval

@dataclass
class Candidate:
    ticker: str
    name: str
    bucket: str  # HARD or SOFT
    score: float
    metrics: dict[str, float]
    checks: dict[str, bool]

def score_candidate(sig: SignalEval) -> float:
    # simple: prefer higher volume multiple and moderate ATR
    v_mult = (sig.metrics.get("vol", 0.0) / max(sig.metrics.get("vma20", 1.0), 1.0))
    atrp = sig.metrics.get("atr14_pct", 0.0)
    # penalize extreme ATR
    atr_pen = 0.0
    if atrp > 10:
        atr_pen = (atrp - 10) * 0.8
    return float(v_mult * 10.0 - atr_pen)

def scan_one(ticker: str, name: str, df: pd.DataFrame, settings: dict[str, Any],
             drop: Dict[str, int], skip: Dict[str, int]) -> list[Candidate]:
    out: list[Candidate] = []

    lb = int(settings.get("lookback_bars", 260))
    g = gate_history(df, min_bars=max(220, lb))
    if not g.ok:
        skip[g.reason] = skip.get(g.reason, 0) + 1
        return out

    lf = settings["liquidity_filter"]
    if lf.get("enabled", True):
        g = gate_liquidity(df, float(lf["min_price_krw"]), float(lf["min_traded_value_20d_krw"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    vf = settings["volatility_filter"]
    if vf.get("enabled", True):
        g = gate_volatility(df, float(vf["min_atr_pct"]), float(vf["max_atr_pct"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    tf = settings["trend_filter"]
    if tf.get("enabled", True) and tf.get("ma200_required", True):
        g = gate_trend_ma200(df)
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    st = settings["strategy"]
    sig = eval_ema20_reclaim_with_volume(
        df,
        ema_fast=int(st["ema_fast"]),
        ema_slow=int(st["ema_slow"]),
        vol_sma_n=int(st["volume_sma"]),
        vol_mult_hard=float(st["volume_mult_hard"]),
        vol_mult_soft=float(st["volume_mult_soft"]),
    )
    if not sig.reasons_true.get("history_ok", False):
        skip["strategy_history_short"] = skip.get("strategy_history_short", 0) + 1
        return out

    if not (sig.soft or sig.hard):
        drop["no_signal"] = drop.get("no_signal", 0) + 1
        return out

    score = score_candidate(sig)
    bucket = "HARD" if sig.hard else "SOFT"
    out.append(Candidate(
        ticker=ticker,
        name=name,
        bucket=bucket,
        score=score,
        metrics=sig.metrics,
        checks=sig.reasons_true
    ))
    return out
