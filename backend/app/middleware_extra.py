from __future__ import annotations

import json
import os
import time
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "-"


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{dt:.2f}"
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000
        rid = _rid(request)
        if os.getenv("ACCESS_LOG_JSON", "").lower() in ("1", "true", "yes"):
            line = {
                "event": "access",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "request_id": rid,
                "duration_ms": round(dt_ms, 3),
            }
            print(json.dumps(line, separators=(",", ":")), flush=True)
        elif os.getenv("ACCESS_LOG", "").lower() in ("1", "true", "yes"):
            print(
                f'access method={request.method} path={request.url.path} status={response.status_code} rid={rid}',
                flush=True,
            )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline security headers on API responses (opt out with SECURITY_HEADERS=0)."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        if os.getenv("SECURITY_HEADERS", "1").lower() not in ("0", "false", "no"):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
            response.headers.setdefault(
                "Permissions-Policy",
                "camera=(), microphone=(), geolocation=(), payment=()",
            )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Increment in-process counters for GET /metrics (Prometheus text)."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        try:
            from . import metrics_state

            metrics_state.record(response.status_code)
        except Exception:
            pass
        return response


class MaxBodyMiddleware(BaseHTTPMiddleware):
    """Reject oversized JSON bodies on chat routes (default 2 MiB)."""

    def __init__(self, app, max_bytes: int | None = None) -> None:
        super().__init__(app)
        self._max = max_bytes or int(os.getenv("MAX_BODY_BYTES", "2097152"))

    async def dispatch(self, request: Request, call_next: Callable):
        if request.method == "POST" and request.url.path in ("/chat", "/chat/stream"):
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    n = int(cl)
                except ValueError:
                    n = 0
                if n > self._max:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "payload_too_large",
                                "message": f"Request body exceeds {self._max} bytes",
                            },
                            "request_id": _rid(request),
                        },
                    )
        return await call_next(request)
