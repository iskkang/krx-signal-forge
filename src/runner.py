from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils.timer import Timer
from .utils.io import read_text_lines, read_json, write_json
from .data.sqlite_cache import connect_sqlite, ensure_schema, upsert_prices, load_prices
from .data.fdr_client import fetch_ohlcv, recent_start_for_lookback
from .scanner import scan_one, Candidate
from .telegram_bot import send_message

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"
STATE_DIR = REPO_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

SQLITE_PATH = STATE_DIR / "market.sqlite"
LAST_SIGNALS_PATH = STATE_DIR / "last_signals.json"
SOFT_WATCHLIST_PATH = STATE_DIR / "soft_watchlist.json"


def load_settings() -> dict[str, Any]:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def load_universe(universe_path: str) -> list[str]:
    return read_text_lines(REPO_ROOT / universe_path)


def load_name_map() -> dict[str, str]:
    p = REPO_ROOT / "config" / "name_map.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def update_cache_for_ticker(con, ticker: str, lookback_bars: int) -> None:
    # Fetch recent window and upsert
    start = recent_start_for_lookback(lookback_bars)
    df = fetch_ohlcv(ticker, start=start)
    if df is None or df.empty:
        return
    upsert_prices(con, ticker, df)


def _krx_tick(price: float) -> int:
    # KRX tick size rules (common board; simplified but practical)
    p = float(price)
    if p < 1000:
        return 1
    if p < 5000:
        return 5
    if p < 10000:
        return 10
    if p < 50000:
        return 50
    if p < 100000:
        return 100
    if p < 500000:
        return 500
    return 1000


def _round_up_to_tick(price: float) -> int:
    t = _krx_tick(price)
    return int(math.ceil(float(price) / t) * t)



def _round_down_to_tick(price: float) -> int:
    t = _krx_tick(price)
    return int(math.floor(float(price) / t) * t)



def compute_trade_plan(c: Candidate, settings: dict[str, Any]) -> dict[str, Any]:
    """
    Compute an actionable plan for HARD upgrades.
    This is a plan-hint (not an execution order).
    """
    tp = settings.get("trade_plan", {})
    entry_mode = str(tp.get("entry_mode", "close_buffer_pct"))
    entry_buffer_pct = float(tp.get("entry_buffer_pct", 0.3))
    stop_atr_mult = float(tp.get("stop_atr_mult", 1.5))
    tp1_r_multiple = float(tp.get("tp1_r_multiple", 2.0))

    close = float(c.metrics.get("close", 0.0))
    atr_pct = float(c.metrics.get("atr14_pct", 0.0))
    atr_val = close * (atr_pct / 100.0)

    if entry_mode == "close_buffer_pct":
        entry = close * (1.0 + entry_buffer_pct / 100.0)
    else:
        # Fallback/hint: assume entry near close
        entry = close

    entry_i = _round_up_to_tick(entry)
    stop = entry_i - (atr_val * stop_atr_mult)
    stop_i = _round_down_to_tick(stop)

    risk = max(1.0, float(entry_i - stop_i))
    tp1 = float(entry_i) + tp1_r_multiple * risk
    tp1_i = _round_down_to_tick(tp1)

    return {
        "entry": entry_i,
        "stop": stop_i,
        "tp1": tp1_i,
        "risk": int(risk),
        "atr14_pct": atr_pct,
        "entry_mode": entry_mode,
        "entry_buffer_pct": entry_buffer_pct,
        "stop_atr_mult": stop_atr_mult,
        "tp1_r_multiple": tp1_r_multiple,
    }


def format_candidate(c: Candidate) -> str:
    m = c.metrics
    return (
        f"{c.ticker} {c.name} | {c.bucket} | score={c.score:.1f} "
        f"close={m.get('close',0):.0f} ema20={m.get('ema20',0):.0f} "
        f"vol={m.get('vol',0):.0f} vma20={m.get('vma20',0):.0f} atr%={m.get('atr14_pct',0):.1f}"
    )


def _utc_day() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def run() -> int:
    timer = Timer()
    settings = load_settings()
    timer.lap("load_settings")

    universe = load_universe(settings["universe_path"])
    name_map = load_name_map()
    timer.lap("load_universe")

    con = connect_sqlite(SQLITE_PATH)
    ensure_schema(con)
    timer.lap("open_sqlite")

    # State: last signals (kept for reporting)
    _ = read_json(LAST_SIGNALS_PATH, default={})

    # State: soft watchlist
    watch_state = read_json(SOFT_WATCHLIST_PATH, default={"asof": None, "items": {}})
    watch_items: dict[str, Any] = watch_state.get("items", {}) or {}

    drop: Dict[str, int] = {}
    skip: Dict[str, int] = {}

    candidates: list[Candidate] = []
    lookback_bars = int(settings.get("lookback_bars", 260))

    for ticker in universe:
        try:
            update_cache_for_ticker(con, ticker, lookback_bars)
            df = load_prices(con, ticker, limit=max(lookback_bars, 260))
            name = name_map.get(ticker, "")
            candidates.extend(scan_one(ticker, name, df, settings, drop, skip))
        except Exception:
            skip["fetch_or_scan_error"] = skip.get("fetch_or_scan_error", 0) + 1

    timer.lap("scan_all")

    hard_all = [c for c in candidates if c.bucket == "HARD"]
    soft_all = [c for c in candidates if c.bucket == "SOFT"]

    hard_all.sort(key=lambda x: (-x.score, x.ticker))
    soft_all.sort(key=lambda x: (-x.score, x.ticker))

    max_hard = int(settings["output"]["max_hard"])
    max_soft = int(settings["output"]["max_soft"])

    # ---- Promotion detection: SOFT -> HARD
    upgrades = [c for c in hard_all if c.ticker in watch_items]
    upgrades.sort(key=lambda x: (-x.score, x.ticker))
    upgrades = upgrades[:max_hard]

    # ---- Update soft watchlist state
    today = _utc_day()

    # refresh/add today's soft items
    for c in soft_all:
        watch_items[c.ticker] = {
            "ticker": c.ticker,
            "name": c.name,
            "score": c.score,
            "metrics": c.metrics,
            "checks": c.checks,
            "first_seen": watch_items.get(c.ticker, {}).get("first_seen", today),
            "last_seen": today,
        }

    # remove upgraded items (we only care about promotion moment)
    for c in upgrades:
        watch_items.pop(c.ticker, None)

    # expire old items
    expire_days = int(settings.get("trade_plan", {}).get("soft_watch_expire_days", 14))
    to_del = []
    today_ts = pd.Timestamp(today)
    for t, it in watch_items.items():
        last_seen = str(it.get("last_seen", today))
        try:
            age = (today_ts - pd.Timestamp(last_seen)).days
        except Exception:
            age = 0
        if age > expire_days:
            to_del.append(t)
    for t in to_del:
        watch_items.pop(t, None)

    write_json(SOFT_WATCHLIST_PATH, {"asof": today, "items": watch_items})

    # ---- Print summary (top lists)
    hard_top = hard_all[:max_hard]
    soft_top = soft_all[:max_soft]

    print(f"Scanned: {len(universe)} | candidates: {len(candidates)} | hard={len(hard_top)} soft={len(soft_top)}")
    if drop:
        top_drop = sorted(drop.items(), key=lambda x: -x[1])[:10]
        print("Top drop reasons:", ", ".join([f"{k}:{v}" for k, v in top_drop]))
    if skip:
        top_skip = sorted(skip.items(), key=lambda x: -x[1])[:10]
        print("Top skip reasons:", ", ".join([f"{k}:{v}" for k, v in top_skip]))

    print("\nHARD candidates")
    for c in hard_top:
        print("-", format_candidate(c))

    print("\nSOFT watchlist")
    for c in soft_top:
        print("-", format_candidate(c))

    # ---- Telegram: ONLY send promotions
    if settings.get("telegram", {}).get("enabled", False) and upgrades:
        lines = ["KRX Signal Forge — SOFT → HARD promotion"]
        for c in upgrades:
            plan = compute_trade_plan(c, settings)
            lines.append(
                f"- {c.ticker} {c.name} | score={c.score:.1f} close={c.metrics.get('close',0):.0f} "
                f"Entry {plan['entry']:,} / Stop {plan['stop']:,} / TP1 {plan['tp1']:,} "
                f"(ATR% {plan['atr14_pct']:.1f}, R {plan['risk']:,}, rule: skip if next-day gap > {float(settings.get('trade_plan',{}).get('max_gap_pct',2.0)):.1f}%)"
            )
        send_message("\n".join(lines))

    # ---- Save last_signals for reporting
    write_json(LAST_SIGNALS_PATH, {
        "asof": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "scanned": len(universe),
            "candidates": len(candidates),
            "hard_top": len(hard_top),
            "soft_top": len(soft_top),
            "upgrades": [c.ticker for c in upgrades],
            "drop": drop,
            "skip": skip,
            "timing_ms": {lap.name: lap.ms for lap in timer.laps} | {"total": timer.total_ms()},
        },
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
