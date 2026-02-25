from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

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

def load_settings() -> dict[str, Any]:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))

def load_universe(universe_path: str) -> list[str]:
    return read_text_lines(REPO_ROOT / universe_path)

def load_name_map() -> dict[str, str]:
    # optional file: config/name_map.json (Code -> Name)
    p = REPO_ROOT / "config" / "name_map.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def update_cache_for_ticker(con, ticker: str, lookback_bars: int) -> None:
    # Fetch recent window and upsert (simple; incremental optimization can be added later)
    start = recent_start_for_lookback(lookback_bars)
    df = fetch_ohlcv(ticker, start=start)
    if df is None or df.empty:
        return
    upsert_prices(con, ticker, df)

def format_candidate(c: Candidate) -> str:
    m = c.metrics
    return (
        f"{c.ticker} {c.name} | {c.bucket} | score={c.score:.1f} "
        f"close={m.get('close',0):.0f} ema20={m.get('ema20',0):.0f} "
        f"vol={m.get('vol',0):.0f} vma20={m.get('vma20',0):.0f} atr%={m.get('atr14_pct',0):.1f}"
    )

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

    last_state = read_json(LAST_SIGNALS_PATH, default={})
    already_sent: dict[str, str] = last_state.get("sent", {})  # ticker -> last_bucket
    drop: Dict[str, int] = {}
    skip: Dict[str, int] = {}

    candidates: list[Candidate] = []
    lookback_bars = int(settings.get("lookback_bars", 260))

    # Update cache + scan
    for i, ticker in enumerate(universe, 1):
        try:
            update_cache_for_ticker(con, ticker, lookback_bars)
            df = load_prices(con, ticker, limit=max(lookback_bars, 260))
            name = name_map.get(ticker, "")
            candidates.extend(scan_one(ticker, name, df, settings, drop, skip))
        except Exception:
            skip["fetch_or_scan_error"] = skip.get("fetch_or_scan_error", 0) + 1

    timer.lap("scan_all")

    hard = [c for c in candidates if c.bucket == "HARD"]
    soft = [c for c in candidates if c.bucket == "SOFT"]

    hard.sort(key=lambda x: (-x.score, x.ticker))
    soft.sort(key=lambda x: (-x.score, x.ticker))

    hard = hard[: int(settings["output"]["max_hard"])]
    soft = soft[: int(settings["output"]["max_soft"])]

    # Dedupe alerts (don't spam same ticker+bucket)
    to_alert = []
    for c in hard + soft:
        prev = already_sent.get(c.ticker)
        if prev == c.bucket:
            continue
        to_alert.append(c)
        already_sent[c.ticker] = c.bucket

    # Print summary
    print(f"Scanned: {len(universe)} | candidates: {len(candidates)} | hard={len(hard)} soft={len(soft)}")
    if drop:
        top_drop = sorted(drop.items(), key=lambda x: -x[1])[:10]
        print("Top drop reasons:", ", ".join([f"{k}:{v}" for k,v in top_drop]))
    if skip:
        top_skip = sorted(skip.items(), key=lambda x: -x[1])[:10]
        print("Top skip reasons:", ", ".join([f"{k}:{v}" for k,v in top_skip]))

    print("\nHARD candidates")
    for c in hard:
        print("-", format_candidate(c))

    print("\nSOFT watchlist")
    for c in soft:
        print("-", format_candidate(c))

    # Telegram
    if settings.get("telegram", {}).get("enabled", False) and to_alert:
        msg = "KRX Signal Forge\n" + "\n".join([format_candidate(c) for c in to_alert])
        send_message(msg)

    # Save state
    write_json(LAST_SIGNALS_PATH, {
        "asof": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sent": already_sent,
        "summary": {
            "scanned": len(universe),
            "candidates": len(candidates),
            "hard": len(hard),
            "soft": len(soft),
            "drop": drop,
            "skip": skip,
            "timing_ms": {lap.name: lap.ms for lap in timer.laps} | {"total": timer.total_ms()}
        }
    })

    return 0

if __name__ == "__main__":
    raise SystemExit(run())
