# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
import FinanceDataReader as fdr

KST = timezone(timedelta(hours=9))


@dataclass
class Rules:
    asof: str = "auto"  # "YYYYMMDD" or "auto"
    markets: List[str] = None  # ["KOSPI","KOSDAQ"]
    top_n_mcap: int = 1200
    min_price_krw: int = 2000
    # Universe 단계에서는 20D 평균 거래대금 대신, listing에 있는 'Amount'(당일 거래대금)로 1차 필터
    min_amount_krw_today: int = 5_000_000_000
    min_count_valid: int = 700
    listing_cache_days: int = 3

    def __post_init__(self):
        if self.markets is None:
            self.markets = ["KOSPI", "KOSDAQ"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rules(path: Path) -> Rules:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Rules(**data)


def _today_yyyymmdd_kst() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def _backup_universe(universe_path: Path, backup_dir: Path, ymd: str) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if universe_path.exists() and universe_path.read_text(encoding="utf-8").strip():
        (backup_dir / f"universe_{ymd}.txt").write_text(universe_path.read_text(encoding="utf-8"), encoding="utf-8")


def _save_universe(universe_path: Path, tickers: list[str]) -> None:
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def _asof_auto() -> str:
    """
    FDR은 '거래일 캘린더'를 직접 주지 않으니,
    오늘부터 역으로 14일 탐색하면서 삼성전자 데이터가 1개라도 나오는 날짜를 거래일로 간주.
    """
    base = datetime.now(KST).date()
    for i in range(14):
        d = base - timedelta(days=i)
        ymd = d.strftime("%Y%m%d")
        try:
            df = fdr.DataReader("005930", ymd, ymd)
            if df is not None and len(df) > 0:
                return ymd
        except Exception:
            continue
    # 최후: 그냥 어제로
    return (base - timedelta(days=1)).strftime("%Y%m%d")


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FinanceDataReader.StockListing('KRX') 컬럼은 버전/소스에 따라 달라질 수 있어
    가능한 키를 모두 흡수해 표준화한다.
    """
    out = df.copy()
    # code/symbol
    if "Code" in out.columns:
        out.rename(columns={"Code": "Ticker"}, inplace=True)
    elif "Symbol" in out.columns:
        out.rename(columns={"Symbol": "Ticker"}, inplace=True)
    elif "ticker" in out.columns:
        out.rename(columns={"ticker": "Ticker"}, inplace=True)

    # market
    if "Market" not in out.columns and "MarketId" in out.columns:
        out.rename(columns={"MarketId": "Market"}, inplace=True)

    # price (Close)
    if "Close" not in out.columns:
        for c in ["Price", "Last", "종가"]:
            if c in out.columns:
                out.rename(columns={c: "Close"}, inplace=True)
                break

    # market cap
    if "Marcap" not in out.columns:
        for c in ["MarketCap", "시가총액", "MarCap"]:
            if c in out.columns:
                out.rename(columns={c: "Marcap"}, inplace=True)
                break

    # amount (trading value)
    if "Amount" not in out.columns:
        for c in ["거래대금", "Value", "TrValue"]:
            if c in out.columns:
                out.rename(columns={c: "Amount"}, inplace=True)
                break

    return out


def _load_listing_cached(cache_path: Path, max_age_days: int) -> Optional[pd.DataFrame]:
    if not cache_path.exists():
        return None
    try:
        st = cache_path.stat()
        age = datetime.now(KST) - datetime.fromtimestamp(st.st_mtime, tz=KST)
        if age > timedelta(days=max_age_days):
            return None
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def _save_listing_cache(cache_path: Path, df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def build_universe(rules: Rules) -> tuple[str, pd.DataFrame]:
    asof = _asof_auto() if rules.asof == "auto" else rules.asof

    # listing cache
    root = _repo_root()
    cache_path = root / "state" / "cache" / "krx_listing.parquet"
    listing = _load_listing_cached(cache_path, rules.listing_cache_days)
    if listing is None:
        listing = fdr.StockListing("KRX")
        listing = _norm_columns(listing)
        _save_listing_cache(cache_path, listing)

    if listing is None or len(listing) == 0:
        raise RuntimeError("StockListing('KRX') returned empty.")

    needed = {"Ticker", "Market", "Close", "Marcap"}
    missing = [c for c in needed if c not in listing.columns]
    if missing:
        raise RuntimeError(f"Listing missing columns: {missing}. Available={list(listing.columns)}")

    df = listing.copy()
    df = df[df["Market"].isin(rules.markets)].copy()

    # numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Marcap"] = pd.to_numeric(df["Marcap"], errors="coerce")

    # optional Amount filter
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    else:
        df["Amount"] = pd.NA

    df = df.dropna(subset=["Ticker", "Close", "Marcap"])
    df = df[df["Close"] >= float(rules.min_price_krw)]

    # top by market cap
    df = df.sort_values("Marcap", ascending=False).head(int(rules.top_n_mcap))

    # liquidity prefilter by today's Amount if available
    if df["Amount"].notna().any():
        df = df[df["Amount"].fillna(0) >= float(rules.min_amount_krw_today)]

    df = df.sort_values(["Marcap"], ascending=False)

    return asof, df


def main() -> int:
    root = _repo_root()
    rules_path = root / "config" / "universe_rules.json"
    universe_path = root / "config" / "universe.txt"
    backup_dir = root / "state" / "universe_backup"

    rules = _load_rules(rules_path)
    run_ymd = _today_yyyymmdd_kst()

    print("[INFO] data_source=FinanceDataReader")
    _backup_universe(universe_path, backup_dir, run_ymd)

    try:
        asof, df_final = build_universe(rules)
        tickers = df_final["Ticker"].astype(str).tolist()

        if len(tickers) < int(rules.min_count_valid):
            raise RuntimeError(f"Universe too small ({len(tickers)}). Treat as failure and keep previous universe.")

        _save_universe(universe_path, tickers)

        print(f"[OK] asof={asof} markets={rules.markets}")
        print(f"[OK] universe_size={len(tickers)} (top_n_mcap={rules.top_n_mcap}, min_price_krw={rules.min_price_krw:,}, "
              f"min_amount_krw_today={rules.min_amount_krw_today:,})")

        preview = df_final.head(10).copy()
        cols = ["Ticker", "Name"] if "Name" in preview.columns else ["Ticker"]
        cols += ["Close", "Marcap"]
        if "Amount" in preview.columns:
            cols += ["Amount"]
        print("[TOP10]")
        print(preview[cols].to_string(index=False))

        return 0

    except Exception as e:
        print(f"[FAIL] update_universe: {e}", file=sys.stderr)
        print("[FAIL] Keeping previous universe.txt (backup already created).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
