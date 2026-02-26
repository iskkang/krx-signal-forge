from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .indicators import ema, sma, atr

@dataclass
class SignalEval:
    hard: bool
    soft: bool
    reasons_true: dict[str, bool]
    metrics: dict[str, float]

def eval_ema20_reclaim_with_volume(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
                                   vol_sma_n: int = 20, vol_mult_hard: float = 1.2, vol_mult_soft: float = 1.0) -> SignalEval:
    # Requires sufficient history
    if df is None or df.empty or len(df) < max(ema_slow, 200, vol_sma_n, 30):
        return SignalEval(False, False, {"history_ok": False}, {})

    close = df["Close"]
    vol = df["Volume"]

    ema20 = ema(close, ema_fast)
    ema50 = ema(close, ema_slow)
    vma = sma(vol, vol_sma_n)

    # Reclaim condition: today close above ema20 AND yesterday close <= ema20 (cross reclaim)
    reclaim = (close.iloc[-1] > ema20.iloc[-1]) and (close.iloc[-2] <= ema20.iloc[-2])

    # Trend bias: ema20 > ema50 (optional, used for ranking)
    ema_stack = ema20.iloc[-1] >= ema50.iloc[-1]

    vol_ok_hard = vol.iloc[-1] >= (vma.iloc[-1] * vol_mult_hard)
    vol_ok_soft = vol.iloc[-1] >= (vma.iloc[-1] * vol_mult_soft)

    atr14 = atr(df, 14).iloc[-1]
    atr_pct = float(atr14 / close.iloc[-1] * 100.0) if close.iloc[-1] else 0.0

    # Soft: reclaim but volume only meets soft level
    soft = bool(reclaim and vol_ok_soft)
    # Hard: reclaim and stronger volume + ema stack
    hard = bool(reclaim and vol_ok_hard and ema_stack)

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
        "atr14_pct": float(atr_pct),
    }
    return SignalEval(hard=hard, soft=soft, reasons_true=reasons, metrics=metrics)
