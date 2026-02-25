# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data.krx_loader import fetch_ohlcv
from src.scan.gates import apply_gates, GateResult, Candidate
from src.utils import Timer, repo_root, load_json, save_json, pick_last_trading_day_kst
from src.telegram import maybe_send_from_env


def load_universe(universe_path: Path) -> List[str]:
    if not universe_path.exists():
        return []
    tickers = [x.strip() for x in universe_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    # de-dupe preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _format_candidate(c: Candidate) -> str:
    tv = int(c.turnover_krw_20d)
    name = c.name or ""
    return f"- {c.ticker} {name} | price={c.price:,.0f} | tv20={tv:,} | score={c.score:.2f} | {c.reason}"


def run_once() -> int:
    root = repo_root()
    rules = load_json(root / "config" / "scan_rules.json")
    universe_path = root / "config" / "universe.txt"
    state_path = root / "state" / "last_signals.json"

    asof = pick_last_trading_day_kst(rules.get("asof", "auto"))
    tickers = load_universe(universe_path)

    if not tickers:
        print("[FAIL] universe.txt is empty. Run scripts/update_universe.py first.", file=sys.stderr)
        return 2

    drop_counter: Dict[str, int] = {}
    skip_counter: Dict[str, int] = {}
    hard: List[Candidate] = []
    soft: List[Candidate] = []

    # load last signals to prevent duplicates
    last = {}
    if state_path.exists():
        try:
            last = load_json(state_path)
        except Exception:
            last = {}

    timings: Dict[str, float] = {}

    with Timer("total") as t_total:
        for i, ticker in enumerate(tickers, 1):
            with Timer("fetch") as t_fetch:
                bars_obj = fetch_ohlcv(ticker=ticker, end_yyyymmdd=asof, lookback_days=int(rules["lookback_days"]))
            timings["fetch_ms"] = timings.get("fetch_ms", 0.0) + t_fetch.elapsed_ms

            if bars_obj is None:
                skip_counter["skip_fetch_none"] = skip_counter.get("skip_fetch_none", 0) + 1
                continue

            bars = bars_obj.df

            # NOTE: Name lookup is optional; keep fast. (Can be added later via pykrx ticker name APIs.)
            name = None

            with Timer("gates") as t_g:
                res, cand = apply_gates(
                    ticker=ticker,
                    name=name,
                    bars=bars,
                    rules=rules,
                    drop_counter=drop_counter,
                    skip_counter=skip_counter,
                )
            timings["gates_ms"] = timings.get("gates_ms", 0.0) + t_g.elapsed_ms

            if cand is None:
                continue

            # dedupe: if hard signal already sent for this ticker on same asof, skip
            key = f"{ticker}"
            if res == GateResult.HARD:
                if last.get(key, {}).get("asof") == asof and last.get(key, {}).get("tag") == "HARD":
                    drop_counter["dup_hard_suppressed"] = drop_counter.get("dup_hard_suppressed", 0) + 1
                    continue
                hard.append(cand)
            elif res == GateResult.SOFT:
                soft.append(cand)

    timings["total_ms"] = t_total.elapsed_ms

    # sort
    hard.sort(key=lambda c: (-c.score, -c.turnover_krw_20d, c.ticker))
    soft.sort(key=lambda c: (-c.turnover_krw_20d, c.ticker))

    hard = hard[: int(rules["output"]["max_hard"])]
    soft = soft[: int(rules["output"]["max_soft"])]

    # render
    lines = []
    lines.append(f"📌 KRX Swing Scan (asof {asof})")
    lines.append(f"- Universe: {len(tickers)}")
    lines.append(f"- Hard: {len(hard)} | Soft: {len(soft)}")
    lines.append("")
    if hard:
        lines.append("✅ HARD candidates")
        lines.extend([_format_candidate(c) for c in hard])
        lines.append("")
    if soft:
        lines.append("🟡 SOFT watchlist")
        lines.extend([_format_candidate(c) for c in soft])
        lines.append("")
    # counters
    def _top(d: Dict[str,int], n=10):
        items = sorted(d.items(), key=lambda x: -x[1])[:n]
        return ", ".join([f"{k}:{v}" for k,v in items]) if items else "-"

    lines.append(f"Drop(top): {_top(drop_counter)}")
    lines.append(f"Skip(top): {_top(skip_counter)}")
    lines.append(f"Timing: total={timings.get('total_ms',0):.0f}ms, fetch={timings.get('fetch_ms',0):.0f}ms, gates={timings.get('gates_ms',0):.0f}ms")

    msg = "\n".join(lines)
    print(msg)

    # save last signals (only store hard + asof)
    out_last = dict(last) if isinstance(last, dict) else {}
    for c in hard:
        out_last[c.ticker] = {"asof": asof, "tag": "HARD", "price": c.price, "score": c.score}
    save_json(state_path, out_last)

    # telegram optional
    tg = rules.get("telegram", {})
    if tg.get("enabled", False):
        maybe_send_from_env(tg.get("env_token","TELEGRAM_TOKEN"), tg.get("env_chat_id","TELEGRAM_CHAT_ID"), msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(run_once())
