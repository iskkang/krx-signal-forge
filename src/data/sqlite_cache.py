from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd


def connect_sqlite(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);")
    con.commit()


def upsert_prices(con: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    recs = []
    for idx, row in df.iterrows():
        d = pd.to_datetime(idx).strftime("%Y-%m-%d")
        recs.append((
            ticker, d,
            float(row.get("Open", float("nan"))),
            float(row.get("High", float("nan"))),
            float(row.get("Low", float("nan"))),
            float(row.get("Close", float("nan"))),
            float(row.get("Volume", float("nan"))),
        ))
    con.executemany(
        "INSERT OR REPLACE INTO prices(ticker,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
        recs,
    )
    con.commit()
    return len(recs)


def load_prices(
    con: sqlite3.Connection,
    ticker: str,
    limit: int = 400,
    max_stale_days: int = 6,
) -> pd.DataFrame:
    """가격 데이터 로드 + 캐시 신선도 검증.

    캐시의 최신 날짜가 max_stale_days 이상 오래됐으면 빈 DF 반환.
    → 스캐너가 stale 데이터로 잘못된 신호를 만드는 것을 방지.
    """
    q = """
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE ticker = ?
        ORDER BY date ASC
        LIMIT ?
    """
    rows = con.execute(q, (ticker, limit)).fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # ── 신선도 검증 ──────────────────────────────────────────
    latest = df.index[-1]
    stale_days = (pd.Timestamp.now() - latest).days
    if stale_days > max_stale_days:
        # 캐시에 오래된 데이터가 있으면 신호 계산에 사용하지 않음
        return pd.DataFrame()

    return df


def get_latest_date(con: sqlite3.Connection, ticker: str) -> str | None:
    """캐시에 저장된 ticker 의 가장 최근 날짜 반환. 없으면 None."""
    row = con.execute(
        "SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)
    ).fetchone()
    return row[0] if row and row[0] else None
