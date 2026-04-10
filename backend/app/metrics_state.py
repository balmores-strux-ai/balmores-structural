"""In-process HTTP counters for optional Prometheus text export."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_by_class: dict[str, int] = {}


def record(status: int) -> None:
    global _by_class
    cls = f"{status // 100}xx"
    with _lock:
        _by_class[cls] = _by_class.get(cls, 0) + 1


def prometheus_text() -> str:
    with _lock:
        lines = [
            "# HELP balmores_http_requests_total HTTP responses by status class",
            "# TYPE balmores_http_requests_total counter",
        ]
        for k, v in sorted(_by_class.items()):
            lines.append(f'balmores_http_requests_total{{status_class="{k}"}} {v}')
    return "\n".join(lines) + "\n"
