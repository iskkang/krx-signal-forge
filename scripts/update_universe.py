# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
from pykrx import stock
import pykrx

KST = timezone(timedelta(hours=9))


@dataclass
class Rules:
    market: str = "ALL"  # "KOSPI" | "KOSDAQ" | "ALL"
    asof: str = "auto"   # "YYYYMMDD" or "auto"
    top_n_mcap: int = 1200

    turnover_lookback: int = 20
    min_turnover_krw_20d: int = 5_000_000_000  # 50억

    min_price_krw: int = 2000
    min_count_valid: int = 700


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rules(path: Path) -> Rules:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Rules(**data)


def _today_yyyymmdd_kst() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def _is_trading_day(yyyymmdd: str) -> bool:
    """Fast trading-day check: KOSPI tickers non-empty."""
    t = stock.get_market_ticker_list(yyyymmdd, market="KOSPI")
    return isinstance(t, list) and len(t) > 0


def _pick_last_trading_day_kst(target: Optional[str] = None, max_back: int = 21) -> str:
    if target is None:
        base = datetime.now(KST).date()
    else:
        base = datetime.strptime(target, "%Y%m%d").date()

    last_exc: Optional[Exception] = None
    for i in range(max_back):
        d = base - timedelta(days=i)
        ymd = d.strftime("%Y%m%d")
        try:
            if _is_trading_day(ymd):
                return ymd
        except Exception as e:
            last_exc = e

    raise RuntimeError(f"Could not find a recent trading day (last_exc={last_exc}).")


def _get_last_n_trading_days(end_yyyymmdd: str, n: int, max_back: int = 120) -> List[str]:
    """
    Returns a list of last N trading days (ascending), ending at end_yyyymmdd (or earlier if end isn't trading day).
    Uses get_market_ohlcv_by_ticker which is much cheaper than per-ticker calls.
    """
    end_td = _pick_last_trading_day_kst(end_yyyymmdd)
    end_date = datetime.strptime(end_td, "%Y%m%d").date()

    days: List[str] = []
    checked = 0
    i = 0
    last_exc: Optional[Exception] = None
    while len(days) < n and checked < max_back:
        d = end_date - timedelta(days=i)
        i += 1
        checked += 1
        ymd = d.strftime("%Y%m%d")
        try:
            # This call fails/returns empty on non-trading days.
            df = stock.get_market_ohlcv_by_ticker(date=ymd, market="KOSPI")
            if df is None or len(df) == 0:
                continue
            days.append(ymd)
        except Exception as e:
            last_exc = e
            continue

    if len(days) < n:
        raise RuntimeError(f"Could not collect {n} trading days within lookback. got={len(days)} last_exc={last_exc}")

    return sorted(days)  # ascending


def _backup_universe(universe_path: Path, backup_dir: Path, ymd: str) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if universe_path.exists() and universe_path.read_text(encoding="utf-8").strip():
        backup_path = backup_dir / f"universe_{ymd}.txt"
        backup_path.write_text(universe_path.read_text(encoding="utf-8"), encoding="utf-8")


def _save_universe(universe_path: Path, tickers: list[str]) -> None:
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def build_universe(rules: Rules) -> tuple[str, pd.DataFrame]:
    # 1) asof
    asof = _pick_last_trading_day_kst(None if rules.asof == "auto" else rules.asof)

    # 2) market cap table (single call)
    mcap_df = stock.get_market_cap_by_ticker(asof, market=rules.market)
    if mcap_df is None or len(mcap_df) == 0:
        raise RuntimeError("market cap dataframe is empty")

    if "종가" not in mcap_df.columns or "시가총액" not in mcap_df.columns:
        raise RuntimeError(f"Unexpected columns in mcap_df: {list(mcap_df.columns)}")

    df = pd.DataFrame(index=mcap_df.index.copy())
    df["price"] = mcap_df["종가"].astype("float64")
    df["mcap"] = mcap_df["시가총액"].astype("float64")

    # 3) Top N by mcap
    df = df.sort_values("mcap", ascending=False).head(int(rules.top_n_mcap))
    tickers = df.index.tolist()

    # 4) Turnover 20D avg (bulk, 20 calls instead of 1200 calls)
    trading_days = _get_last_n_trading_days(asof, int(rules.turnover_lookback))
    tv_sum = pd.Series(0.0, index=tickers)

    for day in trading_days:
        # market-wide ohlcv for the day; includes 거래대금
        day_df = stock.get_market_ohlcv_by_ticker(date=day, market=rules.market)
        if day_df is None or len(day_df) == 0 or "거래대금" not in day_df.columns:
            continue
        sub = day_df.reindex(tickers)
        tv = sub["거래대금"].astype("float64").fillna(0.0)
        tv_sum = tv_sum.add(tv, fill_value=0.0)

    df["turnover_krw_20d"] = (tv_sum / float(rules.turnover_lookback)).astype("float64")

    # 5) filters
    df = df[df["price"] >= float(rules.min_price_krw)]
    df = df[df["turnover_krw_20d"] >= float(rules.min_turnover_krw_20d)]

    df = df.sort_values(["mcap", "turnover_krw_20d"], ascending=False)
    return asof, df


def main() -> int:
    root = _repo_root()
    rules_path = root / "config" / "universe_rules.json"
    universe_path = root / "config" / "universe.txt"
    backup_dir = root / "state" / "universe_backup"

    rules = _load_rules(rules_path)
    run_ymd = _today_yyyymmdd_kst()

    print(f"[INFO] pykrx_version={pykrx.__version__}")
    _backup_universe(universe_path, backup_dir, run_ymd)

    try:
        asof, df_final = build_universe(rules)
        tickers = df_final.index.tolist()

        if len(tickers) < int(rules.min_count_valid):
            raise RuntimeError(f"Universe too small ({len(tickers)}). Treat as failure and keep previous universe.")

        _save_universe(universe_path, tickers)

        print(f"[OK] asof={asof} market={rules.market}")
        print(
            f"[OK] universe_size={len(tickers)} "
            f"(top_n_mcap={rules.top_n_mcap}, lookback={rules.turnover_lookback}, "
            f"min_turnover_krw_20d={rules.min_turnover_krw_20d:,}, min_price_krw={rules.min_price_krw:,})"
        )

        preview = df_final.head(10).copy()
        preview["mcap"] = preview["mcap"].round(0).astype("int64")
        preview["turnover_krw_20d"] = preview["turnover_krw_20d"].round(0).astype("int64")
        print("[TOP10] ticker price mcap turnover_krw_20d")
        print(preview[["price", "mcap", "turnover_krw_20d"]].to_string())

        return 0

    except Exception as e:
        print(f"[FAIL] update_universe: {e}", file=sys.stderr)
        print("[FAIL] Keeping previous universe.txt (backup already created).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
