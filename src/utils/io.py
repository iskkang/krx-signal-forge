from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def read_text_lines(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def write_text(path: str | Path, s: str) -> None:
    Path(path).write_text(s, encoding="utf-8")

def read_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
