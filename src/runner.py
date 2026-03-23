from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .utils.timer import Timer
from .utils.io import read_text_lines, read_json, write_json
from .data.sqlite_cache import connect_sqlite, ensure_schema, upsert_prices, load_prices
from .data.fdr_client import fetch_ohlcv, recent_start_for_lookback
from .scanner import scan_one, Candidate
from .telegram_bot import send_message
from .scan.gates import gate_market_regime

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"
STATE_DIR = REPO_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

SQLITE_PATH = STATE_DIR / "market.sqlite"
LAST_SIGNALS_PATH = STATE_DIR / "last_signals.json"


def load_settings() -> dict[str, Any]:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def load_universe(universe_path: str) -> list[str]:
    return read_text_lines(REPO_ROOT / universe_path)


def load_name_map() -> dict[str, str]:
    p = REPO_ROOT / "config" / "name_map.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def fetch_index(ticker: str, lookback_bars: int = 120) -> Optional[pd.DataFrame]:
    """KOSPI(KS11) 또는 KOSDAQ(KQ11) 지수 데이터 가져오기."""
    try:
        start = recent_start_for_lookback(lookback_bars)
        df = fetch_ohlcv(ticker, start=start)
        return df if (df is not None and not df.empty) else None
    except Exception:
        return None


def update_cache_for_ticker(con, ticker: str, lookback_bars: int) -> None:
    start = recent_start_for_lookback(lookback_bars)
    df = fetch_ohlcv(ticker, start=start)
    if df is None or df.empty:
        return
    upsert_prices(con, ticker, df)


# ─────────────────────────────────────────────
# KRX 호가 단위
# ─────────────────────────────────────────────

def _krx_tick(price: float) -> int:
    p = float(price)
    if p < 1_000:     return 1
    if p < 5_000:     return 5
    if p < 10_000:    return 10
    if p < 50_000:    return 50
    if p < 100_000:   return 100
    if p < 500_000:   return 500
    return 1_000


def _round_up_to_tick(price: float) -> int:
    t = _krx_tick(price)
    return int(math.ceil(float(price) / t) * t)


def _round_down_to_tick(price: float) -> int:
    t = _krx_tick(price)
    return int(math.floor(float(price) / t) * t)


# ─────────────────────────────────────────────
# 매매 계획
# ─────────────────────────────────────────────

def compute_trade_plan(c: Candidate, settings: dict[str, Any]) -> dict[str, Any]:
    """진입·손절·TP1 계획 힌트 계산.

    v2 변경:
    - entry = close 기준이 아닌, 당일 고가 + 1틱 돌파매수 방식으로 변경
      (종가 버퍼 방식은 갭 리스크가 너무 크다)
    - 텔레그램 메시지에 "다음날 시가 확인 후 진입" 주의사항 추가
    """
    tp = settings.get("trade_plan", {})
    stop_atr_mult = float(tp.get("stop_atr_mult", 1.5))
    tp1_r_multiple = float(tp.get("tp1_r_multiple", 2.0))
    max_gap_pct = float(tp.get("max_gap_pct", 2.0))

    close = float(c.metrics.get("close", 0.0))
    atr_pct = float(c.metrics.get("atr14_pct", 0.0))
    atr_val = close * (atr_pct / 100.0)

    # 진입: 당일 종가보다 1 ATR 의 20% 위 (=전날 고가 근사값)
    # 실전에서는 다음날 시가를 확인 후 2% 이내 갭이면 진입
    entry_raw = close + atr_val * 0.2
    entry_i = _round_up_to_tick(entry_raw)

    stop_raw = entry_i - atr_val * stop_atr_mult
    stop_i = _round_down_to_tick(stop_raw)

    risk = max(1.0, float(entry_i - stop_i))
    tp1_i = _round_down_to_tick(float(entry_i) + tp1_r_multiple * risk)

    return {
        "entry": entry_i,
        "stop": stop_i,
        "tp1": tp1_i,
        "risk": int(risk),
        "atr14_pct": atr_pct,
        "stop_atr_mult": stop_atr_mult,
        "tp1_r_multiple": tp1_r_multiple,
        "max_gap_skip_pct": max_gap_pct,
    }


# ─────────────────────────────────────────────
# 텔레그램 메시지 포맷
# ─────────────────────────────────────────────

def _fmt_telegram(
    candidates: list[Candidate],
    settings: dict[str, Any],
    market_ok: bool,
    market_reason: str,
) -> str:
    regime_note = "" if market_ok else f" ⚠️ 시장 약세({market_reason})"
    lines = [f"🔔 KRX Signal Forge — 신규 HARD 시그널{regime_note}"]

    for c in candidates:
        plan = compute_trade_plan(c, settings)
        m = c.metrics
        vol_x = m.get("vol", 0) / max(m.get("vma20", 1), 1)
        pb_days = int(m.get("pullback_days", 0))
        pb_vol = m.get("pullback_vol_ratio", 0.0)
        pb_dep = m.get("pullback_depth_pct", 0.0)
        rs = m.get("rs_12w", 0.0)
        dip_n = int(m.get("ema20_dip_count", 1))
        dip_tag = f"{dip_n}차 눌림" if dip_n <= 3 else f"{dip_n}차 눌림(주의)"

        lines.append(
            f"\n● {c.ticker} {c.name}  score={c.score:.1f}\n"
            f"  현재가: {m.get('close',0):,.0f}원  |  {dip_tag}\n"
            f"  Entry:  {plan['entry']:,}원  (다음날 시가 확인 후 진입)\n"
            f"  Stop:   {plan['stop']:,}원  |  TP1: {plan['tp1']:,}원\n"
            f"  갭 >{plan['max_gap_skip_pct']:.1f}% 시 스킵  |  R={plan['risk']:,}원\n"
            f"  Vol: {vol_x:.1f}x  |  ATR: {m.get('atr14_pct',0):.1f}%  |  RS(12w): {rs:+.1f}%\n"
            f"  눌림: {pb_days}일  Vol비율: {pb_vol:.2f}x  낙폭: {pb_dep:.1f}%"
        )

    return "\n".join(lines)


def format_candidate(c: Candidate) -> str:
    m = c.metrics
    return (
        f"{c.ticker} {c.name} | {c.bucket} | score={c.score:.1f} "
        f"close={m.get('close',0):.0f} ema20={m.get('ema20',0):.0f} "
        f"vol={m.get('vol',0):.0f} vma20={m.get('vma20',0):.0f} "
        f"atr%={m.get('atr14_pct',0):.1f} rs={m.get('rs_12w',0):+.1f}% "
        f"pb_days={int(m.get('pullback_days',0))} pb_vol={m.get('pullback_vol_ratio',0):.2f}x"
    )


def _utc_day() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def run() -> int:
    timer = Timer()
    settings = load_settings()
    timer.lap("load_settings")

    universe = load_universe(settings["universe_path"])
    name_map = load_name_map()
    timer.lap("load_universe")

    # ── 시장 레짐 체크 ────────────────────────────────────────
    index_ticker = settings.get("index_ticker", "KS11")  # KOSPI 기본
    index_df = fetch_index(index_ticker, lookback_bars=120)
    regime = gate_market_regime(index_df)
    market_ok = regime.ok
    if not market_ok:
        print(f"[WARN] 시장 레짐: {regime.reason} — 신호 발생 시 보수적 접근 권장")
    timer.lap("market_regime")

    # ── SQLite 캐시 ───────────────────────────────────────────
    con = connect_sqlite(SQLITE_PATH)
    ensure_schema(con)
    timer.lap("open_sqlite")

    # ── 이전 HARD 목록 로드 (중복 알림 방지) ──────────────────
    last_state = read_json(LAST_SIGNALS_PATH, default={})
    last_hard_set: set[str] = set(
        last_state.get("summary", {}).get("hard_top_tickers", [])
    )

    drop: Dict[str, int] = {}
    skip: Dict[str, int] = {}
    candidates: list[Candidate] = []
    lookback_bars = int(settings.get("lookback_bars", 260))

    # ── 종목별 스캔 ───────────────────────────────────────────
    for ticker in universe:
        try:
            update_cache_for_ticker(con, ticker, lookback_bars)
            df = load_prices(con, ticker, limit=max(lookback_bars, 260))
            name = name_map.get(ticker, "")
            candidates.extend(
                scan_one(ticker, name, df, settings, drop, skip, index_df=index_df)
            )
        except Exception:
            skip["fetch_or_scan_error"] = skip.get("fetch_or_scan_error", 0) + 1

    timer.lap("scan_all")

    # ── 분류 및 정렬 ──────────────────────────────────────────
    hard_all = sorted(
        [c for c in candidates if c.bucket == "HARD"],
        key=lambda x: (-x.score, x.ticker),
    )
    soft_all = sorted(
        [c for c in candidates if c.bucket == "SOFT"],
        key=lambda x: (-x.score, x.ticker),
    )

    max_hard = int(settings["output"]["max_hard"])
    max_soft = int(settings["output"]["max_soft"])

    hard_top = hard_all[:max_hard]
    soft_top = soft_all[:max_soft]

    # ── 신규 HARD 시그널 탐지 (어제 목록에 없던 종목만) ────────
    new_hard = [c for c in hard_top if c.ticker not in last_hard_set]

    # ── 콘솔 출력 ─────────────────────────────────────────────
    print(
        f"Scanned: {len(universe)} | candidates: {len(candidates)} "
        f"| hard={len(hard_top)} soft={len(soft_top)} "
        f"| new_hard={len(new_hard)}"
    )
    if not market_ok:
        print(f"Market regime: {regime.reason}")
    if drop:
        top_drop = sorted(drop.items(), key=lambda x: -x[1])[:10]
        print("Drop:", ", ".join(f"{k}:{v}" for k, v in top_drop))
    if skip:
        top_skip = sorted(skip.items(), key=lambda x: -x[1])[:10]
        print("Skip:", ", ".join(f"{k}:{v}" for k, v in top_skip))

    print("\n── HARD candidates ──")
    for c in hard_top:
        marker = "[NEW]" if c.ticker in {x.ticker for x in new_hard} else "     "
        print(f"  {marker}", format_candidate(c))

    print("\n── SOFT watchlist ──")
    for c in soft_top:
        print("  -", format_candidate(c))

    # ── 텔레그램: 신규 HARD 시그널만 전송 ───────────────────────
    # (BUG FIX: 기존 SOFT→HARD 승격 방식 폐기.
    #  오늘 새로 HARD 조건을 충족한 종목을 즉시 전송.)
    tg = settings.get("telegram", {})
    if tg.get("enabled", False) and new_hard:
        msg = _fmt_telegram(new_hard, settings, market_ok, regime.reason)
        send_message(msg)
        print(f"\n[Telegram] {len(new_hard)}개 신규 HARD 시그널 전송 완료")
    elif tg.get("enabled", False) and not new_hard:
        print("[Telegram] 신규 HARD 시그널 없음 — 전송 생략")

    # ── 상태 저장 ─────────────────────────────────────────────
    write_json(LAST_SIGNALS_PATH, {
        "asof": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "scanned": len(universe),
            "candidates": len(candidates),
            "hard_top": len(hard_top),
            "hard_top_tickers": [c.ticker for c in hard_top],   # 다음날 중복 방지용
            "soft_top": len(soft_top),
            "new_hard": [c.ticker for c in new_hard],
            "market_ok": market_ok,
            "market_reason": regime.reason,
            "drop": drop,
            "skip": skip,
            "timing_ms": {lap.name: lap.ms for lap in timer.laps} | {"total": timer.total_ms()},
        },
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
