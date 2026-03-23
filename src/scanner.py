from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from .scan.gates import (
    gate_history,
    gate_liquidity,
    gate_volatility,
    gate_trend_ma200,
    gate_ema20_slope,
    gate_setup_pullback,
    gate_pullback_quality,
    gate_reclaim_candle,
)
from .scan.strategy import eval_ema20_reclaim_with_volume, SignalEval
from .scan.indicators import rs_ratio, count_ema20_dips


@dataclass
class Candidate:
    ticker: str
    name: str
    bucket: str
    score: float
    metrics: dict[str, float]
    checks: dict[str, bool]


def score_candidate(
    sig: SignalEval,
    rs_12w: float = 0.0,
    dip_count: int = 1,
) -> float:
    m = sig.metrics
    vma = max(m.get("vma20", 1.0), 1.0)
    v_mult = m.get("vol", 0.0) / vma
    rs_bonus = min(max(rs_12w, 0.0) * 1.5, 12.0)
    pb_vol_r = m.get("pullback_vol_ratio", 1.0)
    pb_quality_bonus = max(0.0, (0.85 - pb_vol_r) * 15.0)
    atrp = m.get("atr14_pct", 0.0)
    atr_penalty = max(0.0, (atrp - 8.0) * 0.5)
    dip_penalty = max(0.0, (dip_count - 1) * 4.0)
    return float(v_mult * 10.0 + rs_bonus + pb_quality_bonus - atr_penalty - dip_penalty)


def scan_one(
    ticker: str,
    name: str,
    df: pd.DataFrame,
    settings: dict[str, Any],
    drop: Dict[str, int],
    skip: Dict[str, int],
    index_df: Optional[pd.DataFrame] = None,
) -> list[Candidate]:
    """게이트 순서:
      1) history → 2) liquidity → 3) volatility → 4) MA200
      5) ema20_slope → 6) setup_pullback
      ↓ strategy eval (reclaim 발생 확인) ↓
      7) reclaim_candle → 8) pullback_quality
    """
    out: list[Candidate] = []

    lb = int(settings.get("lookback_bars", 260))
    g = gate_history(df, min_bars=max(220, lb))
    if not g.ok:
        skip[g.reason] = skip.get(g.reason, 0) + 1
        return out

    lf = settings.get("liquidity_filter", {})
    if lf.get("enabled", True):
        g = gate_liquidity(df, float(lf["min_price_krw"]), float(lf["min_traded_value_20d_krw"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    vf = settings.get("volatility_filter", {})
    if vf.get("enabled", True):
        g = gate_volatility(df, float(vf["min_atr_pct"]), float(vf["max_atr_pct"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    tf = settings.get("trend_filter", {})
    if tf.get("enabled", True) and tf.get("ma200_required", True):
        g = gate_trend_ma200(df)
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    ef = settings.get("ema20_slope_filter", {})
    if ef.get("enabled", True):
        g = gate_ema20_slope(
            df,
            min_slope_pct=float(ef.get("min_slope_pct", 0.0)),
            lookback=int(ef.get("lookback", 10)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    sf = settings.get("setup_filter", {})
    if sf.get("enabled", True):
        g = gate_setup_pullback(
            df,
            max_day_ret_pct=float(sf.get("max_day_ret_pct", 8.0)),
            max_gap_pct=float(sf.get("max_gap_pct", 4.0)),
            max_extended_pct=float(sf.get("max_extended_pct", 8.0)),
            max_below_ema20_pct=float(sf.get("max_below_ema20_pct", 10.0)),
            hh_lookback=int(sf.get("hh_lookback", 60)),
            min_from_hh_pct=float(sf.get("min_from_hh_pct", 78.0)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── strategy eval: 여기서 reclaim 발생 여부 판단 ──
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

    # ── reclaim 확인 후에만 캔들·눌림 품질 체크 ──
    cf = settings.get("candle_filter", {})
    if cf.get("enabled", True):
        g = gate_reclaim_candle(
            df,
            min_close_pct=float(cf.get("min_close_pct", 0.55)),
            min_body_ratio=float(cf.get("min_body_ratio", 0.35)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    pf = settings.get("pullback_filter", {})
    if pf.get("enabled", True):
        g = gate_pullback_quality(
            df,
            min_days=int(pf.get("min_days", 2)),
            max_days=int(pf.get("max_days", 20)),
            max_vol_ratio=float(pf.get("max_vol_ratio", 0.85)),
            max_depth_pct=float(pf.get("max_depth_pct", 15.0)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    index_close = index_df["Close"] if (index_df is not None and not index_df.empty) else None
    rs_12w = rs_ratio(df["Close"], index_close, period=63)
    dip_count = count_ema20_dips(df["Close"], ema_n=int(st["ema_fast"]), lookback=120)
    sig.metrics["rs_12w"] = rs_12w
    sig.metrics["ema20_dip_count"] = float(dip_count)

    score = score_candidate(sig, rs_12w=rs_12w, dip_count=dip_count)
    bucket = "HARD" if sig.hard else "SOFT"
    out.append(Candidate(
        ticker=ticker, name=name, bucket=bucket,
        score=score, metrics=sig.metrics, checks=sig.reasons_true,
    ))
    return out
