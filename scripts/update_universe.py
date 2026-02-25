from __future__ import annotations

import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import FinanceDataReader as fdr

KST = timezone(timedelta(hours=9))

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
STATE_DIR = REPO_ROOT / "state"
CONFIG_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

UNIVERSE_PATH = CONFIG_DIR / "universe.txt"
NAME_MAP_PATH = CONFIG_DIR / "name_map.json"
BACKUP_DIR = STATE_DIR / "universe_backups"
BACKUP_DIR.mkdir(exist_ok=True)

# build cache to avoid recomputing everything when rerun locally
BUILD_CACHE = STATE_DIR / "universe_traded_value_cache.csv"

def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def _clean_krx_listing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("code") or cols.get("symbol") or cols.get("ticker")
    name_col = cols.get("name")
    mkt_col = cols.get("market")

    if code_col is None:
        raise RuntimeError(f"Cannot find code column in listing: {df.columns.tolist()}")

    df["Code"] = df[code_col].astype(str).str.zfill(6)
    df["Name"] = df[name_col].astype(str) if name_col else ""

    if mkt_col:
        mkt = df[mkt_col].astype(str).str.upper()
        df = df[(mkt == "KOSPI") | (mkt == "KOSDAQ")]

    # Filter common non-common-stock instruments by name
    bad = re.compile(r"(스팩|SPAC|리츠|REIT|ETN|ETF)", re.IGNORECASE)
    df = df[~df["Name"].str.contains(bad, na=False)]
    df = df[~df["Name"].str.contains(r"(우|우B|우C)$", na=False)]

    df = df.drop_duplicates(subset=["Code"]).reset_index(drop=True)
    return df[["Code", "Name"]]

def _fetch_traded_value_20d(code: str) -> tuple[str, float] | None:
    # traded value = mean(Close*Volume) over last 20 trading days
    # Use short window for speed.
    try:
        df = fdr.DataReader(code)  # full history; unavoidable in FDR for now
        if df is None or df.empty or len(df) < 25:
            return None
        close = df["Close"].tail(25)
        vol = df["Volume"].tail(25)
        tv = (close * vol).tail(20).mean()
        if pd.isna(tv):
            return None
        return code, float(tv)
    except Exception:
        return None

def main():
    # 1) listing
    listing = fdr.StockListing("KRX")
    listing = _clean_krx_listing(listing)

    # 2) build traded value cache (parallel)
    #    For GitHub Actions, keep workers moderate to reduce transient failures.
    max_workers = int(12)
    codes = listing["Code"].tolist()

    # If cache exists and is recent (same day), reuse it to speed reruns.
    cached = None
    if BUILD_CACHE.exists():
        try:
            cached = pd.read_csv(BUILD_CACHE)
        except Exception:
            cached = None

    tv_map: dict[str, float] = {}
    if cached is not None and {"Code","TradedValue20D"}.issubset(set(cached.columns)):
        for _, r in cached.iterrows():
            tv_map[str(r["Code"]).zfill(6)] = float(r["TradedValue20D"])

    missing = [c for c in codes if c not in tv_map]
    print(f"[INFO] listing={len(codes)} cached={len(tv_map)} missing={len(missing)}")

    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_traded_value_20d, c): c for c in missing}
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                if done % 250 == 0:
                    print(f"[PROGRESS] {done}/{len(missing)}")
                if res is None:
                    continue
                code, tv = res
                tv_map[code] = tv

    # 3) assemble final df
    listing["TradedValue20D"] = listing["Code"].map(tv_map)
    listing = listing.dropna(subset=["TradedValue20D"])
    listing["TradedValue20D"] = listing["TradedValue20D"].astype(float)

    # 4) select top 1200
    top = listing.sort_values("TradedValue20D", ascending=False).head(1200).reset_index(drop=True)

    # 5) backup existing universe
    if UNIVERSE_PATH.exists():
        backup_path = BACKUP_DIR / f"universe_{_now_kst()}.txt"
        backup_path.write_text(UNIVERSE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    # 6) write universe + name map
    UNIVERSE_PATH.write_text("\n".join(top["Code"].tolist()) + "\n", encoding="utf-8")

    # name map for nicer output
    import json
    name_map = {r["Code"]: r["Name"] for _, r in top.iterrows()}
    NAME_MAP_PATH.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # persist build cache (all codes we have)
    cache_df = pd.DataFrame([{"Code": c, "TradedValue20D": tv} for c, tv in tv_map.items()])
    cache_df.to_csv(BUILD_CACHE, index=False, encoding="utf-8")

    print(f"[OK] universe written: {UNIVERSE_PATH} (count={len(top)})")
    print(f"[OK] name_map written: {NAME_MAP_PATH} (count={len(name_map)})")

if __name__ == "__main__":
    main()
