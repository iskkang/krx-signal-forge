from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Lap:
    name: str
    ms: float

class Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.laps: list[Lap] = []

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps.append(Lap(name=name, ms=(now - self._last) * 1000.0))
        self._last = now

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
