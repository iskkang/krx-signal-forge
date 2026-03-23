from __future__ import annotations

import json
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .utils.timer import Timer
from .utils.io import read_text_lines, read_json, write_json
from .data.sqlite_cache import connect_sqlite, ensure_schema, upsert_prices, load_prices, get_latest_date
from .data.fdr_client import fetch_ohlcv, recent_start_for_lookback
from .scanner import scan_one, Candidate
from .telegram_bot import send_message
from .scan.gates import gate_market_regime

REPO_ROOT   = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"
STATE_DIR   = REPO_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

SQLITE_PATH      = STATE_DIR / "market.sqlite"
LAST_SIGNALS_PATH = STATE_DIR / "last_signals.json"

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

def load_settings() -> dict[str, Any]:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

def load_universe(universe_path: str) -> list[str]:
    return read_text_lines(REPO_ROOT / universe_path)

def load_name_map() -> dict[str, str]:
    p = REPO_ROOT / "config" / "name_map.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

# ─────────────────────────────────────────────
# 캐시 업데이트 (병렬 + 스킵 로직)
# ─────────────────────────────────────────────

def _today_kst() -> str:
    """GitHub Actions 실행 시각 기준 KST 오늘 날짜 (UTC+9)."""
    return (pd.Timestamp.now("UTC") + pd.Timedelta(hours=9)).strftime("%Y-%m-%d")


def _needs_update(con, ticker: str, today: str) -> bool:
    """캐시에 오늘 데이터가 이미 있으면 fetch 스킵."""
    latest = get_latest_date(con, ticker)
    return latest is None or latest < today


def _fetch_and_cache(ticker: str, db_path: Path, lookback_bars: int, today: str) -> str:
    """워커 스레드에서 실행: 개별 종목 fetch → SQLite upsert.
    SQLite 는 스레드마다 별도 connection 사용 (thread-safety).
    반환: 'cached' | 'fetched' | 'skipped' | 'error'
    """
    try:
        con = connect_sqlite(db_path)
        if not _needs_update(con, ticker, today):
            con.close()
            return "cached"
        start = recent_start_for_lookback(lookback_bars)
        df = fetch_ohlcv(ticker, start=start)
        if df is not None and not df.empty:
            upsert_prices(con, ticker, df)
            con.close()
            return "fetched"
        con.close()
        return "skipped"
    except Exception:
        return "error"


def update_cache_parallel(
    universe: list[str],
    db_path: Path,
    lookback_bars: int,
    max_workers: int = 12,
    total_timeout_sec: float = 480.0,   # 8분 안에 끝내야 함
) -> dict[str, int]:
    """1,200종목 병렬 캐시 업데이트.

    - 이미 오늘 데이터가 있는 종목은 네트워크 요청 없이 스킵
    - ThreadPoolExecutor 12워커 병렬
    - total_timeout_sec 초 이내에 완료 안되면 남은 작업 포기
    """
    today = _today_kst()
    stats = {"cached": 0, "fetched": 0, "skipped": 0, "error": 0}
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_fetch_and_cache, t, db_path, lookback_bars, today): t
            for t in universe
        }
        done = 0
        for fut in as_completed(futs, timeout=total_timeout_sec):
            result = fut.result()
            stats[result] = stats.get(result, 0) + 1
            done += 1
            if done % 200 == 0:
                elapsed = time.monotonic() - start_time
                print(f"  [cache] {done}/{len(universe)} "
                      f"({elapsed:.0f}s) cached={stats['cached']} fetched={stats['fetched']}")

    return stats

# ─────────────────────────────────────────────
# 지수 데이터
# ─────────────────────────────────────────────

def fetch_index(ticker: str, lookback_bars: int = 120) -> Optional[pd.DataFrame]:
    try:
        start = recent_start_for_lookback(lookback_bars)
        df = fetch_ohlcv(ticker, start=start, timeout_sec=15.0)
        return df if (df is not None and not df.empty) else None
    except Exception:
        return None

# ─────────────────────────────────────────────
# KRX 호가 단위
# ─────────────────────────────────────────────

def _krx_tick(price: float) -> int:
    p = float(price)
    if p < 1_000:   return 1
    if p < 5_000:   return 5
    if p < 10_000:  return 10
    if p < 50_000:  return 50
    if p < 100_000: return 100
    if p < 500_000: return 500
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
    tp = settings.get("trade_plan", {})
    stop_atr_mult  = float(tp.get("stop_atr_mult", 1.5))
    tp1_r_multiple = float(tp.get("tp1_r_multiple", 2.0))
    max_gap_pct    = float(tp.get("max_gap_pct", 2.0))

    close   = float(c.metrics.get("close", 0.0))
    atr_pct = float(c.metrics.get("atr14_pct", 0.0))
    atr_val = close * (atr_pct / 100.0)

    entry_i = _round_up_to_tick(close + atr_val * 0.2)
    stop_i  = _round_down_to_tick(entry_i - atr_val * stop_atr_mult)
    risk    = max(1.0, float(entry_i - stop_i))
    tp1_i   = _round_down_to_tick(float(entry_i) + tp1_r_multiple * risk)

    return {
        "entry": entry_i, "stop": stop_i, "tp1": tp1_i,
        "risk": int(risk), "atr14_pct": atr_pct,
        "stop_atr_mult": stop_atr_mult, "tp1_r_multiple": tp1_r_multiple,
        "max_gap_skip_pct": max_gap_pct,
    }

# ─────────────────────────────────────────────
# 텔레그램 포맷
# ─────────────────────────────────────────────

def _fmt_telegram(
    candidates: list[Candidate],
    settings: dict[str, Any],
    market_ok: bool,
    market_reason: str,
) -> str:
    regime_note = "" if market_ok else f" ⚠️ 시장 약세({market_reason})"
    # 기준 날짜 명시: 항상 "언제 종가 기준인지" 표시
    scan_date = (pd.Timestamp.now("UTC") + pd.Timedelta(hours=9)).strftime("%Y-%m-%d")
    lines = [f"🔔 KRX Signal Forge — {scan_date} 종가 기준{regime_note}"]
    for c in candidates:
        plan = compute_trade_plan(c, settings)
        m    = c.metrics
        vol_x   = m.get("vol", 0) / max(m.get("vma20", 1), 1)
        pb_days = int(m.get("pullback_days", 0))
        pb_vol  = m.get("pullback_vol_ratio", 0.0)
        pb_dep  = m.get("pullback_depth_pct", 0.0)
        rs      = m.get("rs_12w", 0.0)
        dip_n   = int(m.get("ema20_dip_count", 1))
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

# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def run() -> int:
    RUNNER_START = time.monotonic()
    MAX_RUNNER_SEC = 540  # 9분 초과 시 강제 종료 (GitHub Actions timeout 전에)

    def _check_timeout(label: str):
        elapsed = time.monotonic() - RUNNER_START
        if elapsed > MAX_RUNNER_SEC:
            print(f"[TIMEOUT] {label} 에서 {elapsed:.0f}s 경과 — 강제 종료")
            raise SystemExit(1)

    timer = Timer()
    settings = load_settings()
    timer.lap("load_settings")

    universe = load_universe(settings["universe_path"])
    name_map = load_name_map()
    print(f"Universe: {len(universe)}종목")
    timer.lap("load_universe")

    # ── 시장 레짐 ───────────────────────────────────────────
    index_ticker = settings.get("index_ticker", "KS11")
    index_df = fetch_index(index_ticker, lookback_bars=120)
    regime = gate_market_regime(index_df)
    if not regime.ok:
        print(f"[WARN] 시장 레짐: {regime.reason}")
    timer.lap("market_regime")

    # ── 병렬 캐시 업데이트 ────────────────────────────────
    con = connect_sqlite(SQLITE_PATH)
    ensure_schema(con)
    con.close()   # 워커 스레드들이 각자 connection 사용

    cache_cfg  = settings.get("cache", {})
    max_workers = int(cache_cfg.get("max_workers", 12))
    cache_timeout = float(cache_cfg.get("fetch_timeout_sec", 480.0))

    print(f"[cache] 병렬 업데이트 시작 (workers={max_workers}, timeout={cache_timeout:.0f}s)")
    cache_stats = update_cache_parallel(
        universe, SQLITE_PATH,
        lookback_bars=int(settings.get("lookback_bars", 260)),
        max_workers=max_workers,
        total_timeout_sec=cache_timeout,
    )
    print(f"[cache] 완료 — {cache_stats}")
    timer.lap("cache_update")
    _check_timeout("cache_update 이후")

    # ── 스캔: 메인 connection 재사용 ─────────────────────
    con = connect_sqlite(SQLITE_PATH)
    last_state   = read_json(LAST_SIGNALS_PATH, default={})
    last_hard_set: set[str] = set(
        last_state.get("summary", {}).get("hard_top_tickers", [])
    )

    drop: Dict[str, int] = {}
    skip: Dict[str, int] = {}
    candidates: list[Candidate] = []
    lookback_bars = int(settings.get("lookback_bars", 260))

    for i, ticker in enumerate(universe):
        if i % 300 == 0:
            elapsed = time.monotonic() - RUNNER_START
            print(f"  [scan] {i}/{len(universe)} ({elapsed:.0f}s)")
        _check_timeout(f"scan {i}/{len(universe)}")

        try:
            df = load_prices(con, ticker, limit=max(lookback_bars, 260))
            name = name_map.get(ticker, "")
            candidates.extend(
                scan_one(ticker, name, df, settings, drop, skip, index_df=index_df)
            )
        except Exception:
            skip["scan_error"] = skip.get("scan_error", 0) + 1

    timer.lap("scan_all")

    # ── 정렬 및 분류 ──────────────────────────────────────
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
    new_hard  = [c for c in hard_top if c.ticker not in last_hard_set]

    # ── 콘솔 출력 ─────────────────────────────────────────
    elapsed_total = time.monotonic() - RUNNER_START
    scan_date = (pd.Timestamp.now("UTC") + pd.Timedelta(hours=9)).strftime("%Y-%m-%d")
    print(
        f"\n기준날짜: {scan_date} | Scanned: {len(universe)} "
        f"| candidates: {len(candidates)} "
        f"| hard={len(hard_top)} soft={len(soft_top)} "
        f"| new_hard={len(new_hard)} | elapsed={elapsed_total:.0f}s"
    )
    if not regime.ok:
        print(f"Market: {regime.reason}")
    if drop:
        top_drop = sorted(drop.items(), key=lambda x: -x[1])[:10]
        print("Drop:", ", ".join(f"{k}:{v}" for k, v in top_drop))
    if skip:
        top_skip = sorted(skip.items(), key=lambda x: -x[1])[:5]
        print("Skip:", ", ".join(f"{k}:{v}" for k, v in top_skip))

    print("\n── HARD candidates ──")
    for c in hard_top:
        marker = "[NEW]" if c.ticker in {x.ticker for x in new_hard} else "     "
        print(f"  {marker}", format_candidate(c))

    print("\n── SOFT watchlist ──")
    for c in soft_top:
        print("  -", format_candidate(c))

    # ── 텔레그램 ──────────────────────────────────────────
    tg = settings.get("telegram", {})
    if tg.get("enabled", False) and new_hard:
        msg = _fmt_telegram(new_hard, settings, regime.ok, regime.reason)
        send_message(msg)
        print(f"\n[Telegram] {len(new_hard)}개 전송 완료")
    elif tg.get("enabled", False):
        print("[Telegram] 신규 HARD 시그널 없음 — 전송 생략")

    # ── 상태 저장 ─────────────────────────────────────────
    write_json(LAST_SIGNALS_PATH, {
        "asof": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "scanned": len(universe),
            "candidates": len(candidates),
            "hard_top": len(hard_top),
            "hard_top_tickers": [c.ticker for c in hard_top],
            "soft_top": len(soft_top),
            "new_hard": [c.ticker for c in new_hard],
            "market_ok": regime.ok,
            "market_reason": regime.reason,
            "drop": drop,
            "skip": skip,
            "cache_stats": cache_stats,
            "timing_ms": {lap.name: lap.ms for lap in timer.laps} | {"total": timer.total_ms()},
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
