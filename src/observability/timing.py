"""Low-overhead stage timing.

`perf_counter` deltas only - no I/O, no allocation beyond one dict entry per
stage, so instrumenting a call costs well under a microsecond.
"""
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field


@dataclass
class Stopwatch:
    """Accumulates named stage durations (milliseconds) for one operation."""

    stages: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            # accumulate if the same stage runs more than once
            self.stages[f"{name}_ms"] = round(
                self.stages.get(f"{name}_ms", 0.0) + elapsed, 2
            )

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)

    def snapshot(self) -> dict[str, float]:
        return {**self.stages, "total_ms": self.total_ms}


def stage(sw: Stopwatch | None, name: str):
    """Time `name` on `sw`, or do nothing when `sw` is None.

    Lets lower-level functions accept an optional Stopwatch without any
    behaviour change for callers that don't pass one.
    """
    return sw.stage(name) if sw is not None else nullcontext()
