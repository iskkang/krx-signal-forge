# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# PyKRX
from pykrx import stock

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


def _pick_last_trading_day_kst(target: Optional[str] = None, max_back: int = 14) -> str:
    """
    PyKRX는 비거래일이면 데이터가 비거나 예외가 날 수 있어, 최근 거래일을 역으로 찾는다.
    target이 None이면 오늘 기준으로 찾는다.
    """
    if target is None:
        base = datetime.now(KST).date()
    else:
        base = datetime.strptime(target, "%Y%m%d").date()

    for i in range(max_back):
        d = base - timedelta(days=i)
        ymd = d.strftime("%Y%m%d")
        try:
            # 거래일이면 KOSPI tickers가 비지 않음(대부분)
            t = stock.get_market_ticker_list(ymd, market="KOSPI")
            if isinstance(t, list) and len(t) > 0:
                return ymd
        except Exception:
            pass

    raise RuntimeError("Could not find a recent trading day within lookback window.")


def _backup_universe(universe_path: Path, backup_dir: Path, ymd: str) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if universe_path.exists():
        backup_path = backup_dir / f"universe_{ymd}.txt"
        backup_path.write_text(universe_path.read_text(encoding="utf-8"), encoding="utf-8")


def _save_universe(universe_path: Path, tickers: list[str]) -> None:
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def build_universe(rules: Rules) -> tuple[str, pd.DataFrame]:
    """
    Returns: (asof_yyyymmdd, df_final)
    df_final index=ticker, columns include mcap, price, turnover_krw_20d
    """
    # 1) 결정일자(asof)
    if rules.asof == "auto":
        asof = _pick_last_trading_day_kst(None)
    else:
        asof = _pick_last_trading_day_kst(rules.asof)

    # 2) 시가총액 테이블
    # PyKRX: 시총/거래량/거래대금/종가 포함 DataFrame 반환
    mcap_df = stock.get_market_cap_by_ticker(asof, market=rules.market)
    if mcap_df is None or len(mcap_df) == 0:
        raise RuntimeError("market cap dataframe is empty")

    # 컬럼명은 보통: ['종가','시가총액','거래량','거래대금','상장주식수',...]
    # 안전하게 매핑
    col_price = "종가"
    col_mcap = "시가총액"
    if col_price not in mcap_df.columns or col_mcap not in mcap_df.columns:
        raise RuntimeError(f"Unexpected columns in mcap_df: {list(mcap_df.columns)}")

    df = pd.DataFrame(index=mcap_df.index.copy())
    df["price"] = mcap_df[col_price].astype("float64")
    df["mcap"] = mcap_df[col_mcap].astype("float64")

    # 3) 시총 Top N
    df = df.sort_values("mcap", ascending=False).head(int(rules.top_n_mcap))

    # 4) 거래대금 20일 평균(lookback) 계산
    # PyKRX는 일자범위로 OHLCV를 받으면 '거래대금' 컬럼이 포함됨
    # 개별 ticker loop이므로, 여기서 속도 최적화는 추후 캐시로 해결 (초기에는 안정 우선)
    end = asof
    # 대략 40영업일 정도 커버하도록 캘린더 여유를 둠(휴일/주말 고려)
    start_date = (datetime.strptime(asof, "%Y%m%d") - timedelta(days=90)).strftime("%Y%m%d")

    turnovers = {}
    for ticker in df.index.tolist():
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_date, end, ticker)
            if ohlcv is None or len(ohlcv) == 0:
                continue
            # 컬럼명: ['시가','고가','저가','종가','거래량','거래대금'] (대부분)
            if "거래대금" not in ohlcv.columns:
                continue
            tv = ohlcv["거래대금"].dropna().astype("float64").tail(int(rules.turnover_lookback))
            if len(tv) < max(5, int(rules.turnover_lookback * 0.7)):
                continue
            turnovers[ticker] = float(tv.mean())
        except Exception:
            continue

    df["turnover_krw_20d"] = pd.Series(turnovers)

    # 5) 필터: price & turnover
    df = df.dropna(subset=["turnover_krw_20d"])
    df = df[df["price"] >= float(rules.min_price_krw)]
    df = df[df["turnover_krw_20d"] >= float(rules.min_turnover_krw_20d)]

    # 6) 최종 정렬(시총 우선)
    df = df.sort_values(["mcap", "turnover_krw_20d"], ascending=False)

    return asof, df


def main() -> int:
    root = _repo_root()
    rules_path = root / "config" / "universe_rules.json"
    universe_path = root / "config" / "universe.txt"
    backup_dir = root / "state" / "universe_backup"

    rules = _load_rules(rules_path)

    # 실행 날짜(백업 네이밍)
    run_ymd = _today_yyyymmdd_kst()

    # 먼저 기존 universe 백업 (실패 대비)
    _backup_universe(universe_path, backup_dir, run_ymd)

    try:
        asof, df_final = build_universe(rules)
        tickers = df_final.index.tolist()

        # 검증
        if len(tickers) < int(rules.min_count_valid):
            raise RuntimeError(f"Universe too small ({len(tickers)}). Treat as failure and keep previous universe.")

        _save_universe(universe_path, tickers)

        print(f"[OK] asof={asof} market={rules.market}")
        print(f"[OK] universe_size={len(tickers)} (top_n_mcap={rules.top_n_mcap}, lookback={rules.turnover_lookback}, "
              f"min_turnover_krw_20d={rules.min_turnover_krw_20d:,}, min_price_krw={rules.min_price_krw:,})")

        # Top 10 preview
        preview = df_final.head(10).copy()
        preview["mcap"] = preview["mcap"].round(0).astype("int64")
        preview["turnover_krw_20d"] = preview["turnover_krw_20d"].round(0).astype("int64")
        print("[TOP10]")
        print(preview[["price", "mcap", "turnover_krw_20d"]].to_string())

        return 0

    except Exception as e:
        print(f"[FAIL] update_universe: {e}", file=sys.stderr)
        print("[FAIL] Keeping previous universe.txt (backup already created).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
