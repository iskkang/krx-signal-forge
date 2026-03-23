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


def load_prices(con: sqlite3.Connection, ticker: str, limit: int = 400) -> pd.DataFrame:
    """오름차순(오래된 → 최신) 정렬로 반환. indicators 계산에 적합한 순서."""
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
    return df
