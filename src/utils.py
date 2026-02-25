from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import FinanceDataReader as fdr

KST = timezone(timedelta(hours=9))


@dataclass
class Timer:
    name: str
    t0: float = 0.0
    elapsed_ms: float = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_last_trading_day_kst(target_yyyymmdd: Optional[str] = None, max_back: int = 21) -> str:
    """
    FDR 기반 거래일 탐색: 삼성전자(005930)가 1개라도 조회되면 거래일로 간주.
    """
    if target_yyyymmdd is None or target_yyyymmdd == "auto":
        base = datetime.now(KST).date()
    else:
        base = datetime.strptime(target_yyyymmdd, "%Y%m%d").date()

    last_exc: Optional[Exception] = None
    for i in range(max_back):
        d = base - timedelta(days=i)
        ymd = d.strftime("%Y%m%d")
        try:
            df = fdr.DataReader("005930", ymd, ymd)
            if df is not None and len(df) > 0:
                return ymd
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Could not find a recent trading day (last_exc={last_exc}).")
