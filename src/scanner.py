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
    bucket: str          # "HARD" or "SOFT"
    score: float
    metrics: dict[str, float]
    checks: dict[str, bool]


def score_candidate(
    sig: SignalEval,
    rs_12w: float = 0.0,
    dip_count: int = 1,
) -> float:
    """개선된 스코어링.

    구성:
      vol_mult × 10          : 거래량 배수 (핵심 모멘텀 지표)
      + rs_bonus             : 지수 대비 상대강도 보너스 (최대 +12)
      + pb_quality_bonus     : 눌림 품질 보너스 (조용한 눌림일수록 ↑)
      - atr_penalty          : 과도한 변동성 패널티
      - dip_penalty          : 반복 눌림 패널티 (1st pullback 우대)
    """
    m = sig.metrics
    vma = max(m.get("vma20", 1.0), 1.0)
    v_mult = m.get("vol", 0.0) / vma

    # 상대강도 보너스: 1% 아웃퍼폼당 +1.5점, 최대 +12점
    rs_bonus = min(max(rs_12w, 0.0) * 1.5, 12.0)

    # 눌림 품질 보너스: 거래량이 낮을수록(조용할수록) 가산
    pb_vol_r = m.get("pullback_vol_ratio", 1.0)
    pb_quality_bonus = max(0.0, (0.85 - pb_vol_r) * 15.0)  # 0.85 이하면 보너스 시작

    # ATR 패널티: 8% 초과분에 패널티
    atrp = m.get("atr14_pct", 0.0)
    atr_penalty = max(0.0, (atrp - 8.0) * 0.5)

    # 눌림 반복 패널티: 2nd pull부터 -4점씩
    dip_penalty = max(0.0, (dip_count - 1) * 4.0)

    return float(
        v_mult * 10.0
        + rs_bonus
        + pb_quality_bonus
        - atr_penalty
        - dip_penalty
    )


def scan_one(
    ticker: str,
    name: str,
    df: pd.DataFrame,
    settings: dict[str, Any],
    drop: Dict[str, int],
    skip: Dict[str, int],
    index_df: Optional[pd.DataFrame] = None,
) -> list[Candidate]:
    """단일 종목 스캔. 모든 게이트를 순서대로 통과해야 Candidate 반환."""
    out: list[Candidate] = []

    # ── 1. 히스토리 ──────────────────────────────────────────
    lb = int(settings.get("lookback_bars", 260))
    g = gate_history(df, min_bars=max(220, lb))
    if not g.ok:
        skip[g.reason] = skip.get(g.reason, 0) + 1
        return out

    # ── 2. 유동성 ────────────────────────────────────────────
    lf = settings.get("liquidity_filter", {})
    if lf.get("enabled", True):
        g = gate_liquidity(df, float(lf["min_price_krw"]), float(lf["min_traded_value_20d_krw"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 3. 변동성 ────────────────────────────────────────────
    vf = settings.get("volatility_filter", {})
    if vf.get("enabled", True):
        g = gate_volatility(df, float(vf["min_atr_pct"]), float(vf["max_atr_pct"]))
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 4. MA200 추세 ─────────────────────────────────────────
    tf = settings.get("trend_filter", {})
    if tf.get("enabled", True) and tf.get("ma200_required", True):
        g = gate_trend_ma200(df)
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 5. EMA20 기울기 (신규) ────────────────────────────────
    ef = settings.get("ema20_slope_filter", {})
    if ef.get("enabled", True):
        g = gate_ema20_slope(
            df,
            min_slope_pct=float(ef.get("min_slope_pct", 0.5)),
            lookback=int(ef.get("lookback", 10)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 6. 재탈환 캔들 강도 (신규, 빠른 체크) ─────────────────
    cf = settings.get("candle_filter", {})
    if cf.get("enabled", True):
        g = gate_reclaim_candle(
            df,
            min_close_pct=float(cf.get("min_close_pct", 0.60)),
            min_body_ratio=float(cf.get("min_body_ratio", 0.40)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 7. 셋업 필터 (간소화: 스파이크·갭·근접성 체크만) ─────
    sf = settings.get("setup_filter", {})
    if sf.get("enabled", True):
        g = gate_setup_pullback(
            df,
            max_day_ret_pct=float(sf.get("max_day_ret_pct", 8.0)),
            max_gap_pct=float(sf.get("max_gap_pct", 4.0)),
            ema20_band_pct=float(sf.get("ema20_band_pct", 3.0)),
            hh_lookback=int(sf.get("hh_lookback", 60)),
            min_from_hh_pct=float(sf.get("min_from_hh_pct", 80.0)),
        )
        if not g.ok:
            drop[g.reason] = drop.get(g.reason, 0) + 1
            return out

    # ── 8. 눌림 품질 (신규) ───────────────────────────────────
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

    # ── 9. 전략 평가 ─────────────────────────────────────────
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

    # ── 10. 보조 지표 계산 (스코어에 반영) ───────────────────
    index_close = index_df["Close"] if (index_df is not None and not index_df.empty) else None
    rs_12w = rs_ratio(df["Close"], index_close, period=63)
    dip_count = count_ema20_dips(df["Close"], ema_n=int(st["ema_fast"]), lookback=120)

    sig.metrics["rs_12w"] = rs_12w
    sig.metrics["ema20_dip_count"] = float(dip_count)

    score = score_candidate(sig, rs_12w=rs_12w, dip_count=dip_count)
    bucket = "HARD" if sig.hard else "SOFT"

    out.append(Candidate(
        ticker=ticker,
        name=name,
        bucket=bucket,
        score=score,
        metrics=sig.metrics,
        checks=sig.reasons_true,
    ))
    return out
