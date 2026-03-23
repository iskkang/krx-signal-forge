from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .indicators import ema, sma, atr, pullback_metrics


@dataclass
class SignalEval:
    hard: bool
    soft: bool
    reasons_true: dict[str, bool]
    metrics: dict[str, float]


def eval_ema20_reclaim_with_volume(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    vol_sma_n: int = 20,
    vol_mult_hard: float = 1.2,
    vol_mult_soft: float = 1.0,
) -> SignalEval:
    """EMA20 재탈환 + 거래량 확인 전략.

    v2 변경사항:
    - pullback_metrics() 결과를 metrics 에 추가 (눌림 품질 정보 포함)
    - SOFT 조건도 vol_mult_soft >= 1.0 이므로 의미 있는 거래량 필요
    """
    _empty = SignalEval(False, False, {"history_ok": False}, {})
    if df is None or df.empty or len(df) < max(ema_slow, 200, vol_sma_n, 30):
        return _empty

    close = df["Close"]
    vol = df["Volume"]

    ema20 = ema(close, ema_fast)
    ema50 = ema(close, ema_slow)
    vma = sma(vol, vol_sma_n)

    # 핵심 조건: 오늘 EMA20 위로 재탈환 (어제는 EMA20 이하)
    reclaim = (
        float(close.iloc[-1]) > float(ema20.iloc[-1])
        and float(close.iloc[-2]) <= float(ema20.iloc[-2])
    )

    # 추세 정렬: EMA20 >= EMA50 (상승 추세 유지)
    ema_stack = float(ema20.iloc[-1]) >= float(ema50.iloc[-1])

    vol_ok_hard = float(vol.iloc[-1]) >= float(vma.iloc[-1]) * vol_mult_hard
    vol_ok_soft = float(vol.iloc[-1]) >= float(vma.iloc[-1]) * vol_mult_soft

    atr14 = atr(df, 14).iloc[-1]
    atr_pct = float(atr14 / close.iloc[-1] * 100.0) if float(close.iloc[-1]) else 0.0

    soft = bool(reclaim and vol_ok_soft)
    hard = bool(reclaim and vol_ok_hard and ema_stack)

    # 눌림 품질 지표 (텔레그램 메시지 및 스코어에 활용)
    pb = pullback_metrics(df, ema_n=ema_fast)

    reasons = {
        "history_ok": True,
        "reclaim": bool(reclaim),
        "ema20_ge_ema50": bool(ema_stack),
        "vol_ge_soft": bool(vol_ok_soft),
        "vol_ge_hard": bool(vol_ok_hard),
    }
    metrics = {
        "close": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "vol": float(vol.iloc[-1]),
        "vma20": float(vma.iloc[-1]),
        "atr14_pct": round(atr_pct, 2),
        # 눌림 품질 지표
        "pullback_days": float(pb["pullback_days"]),
        "pullback_vol_ratio": float(pb["pullback_vol_ratio"]),
        "pullback_depth_pct": float(pb["pullback_depth_pct"]),
    }
    return SignalEval(hard=hard, soft=soft, reasons_true=reasons, metrics=metrics)
