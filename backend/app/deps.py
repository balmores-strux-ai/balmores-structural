"""Optional shared dependencies (API key when configured)."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def require_api_key_if_configured(request: Request) -> None:
    """If API_KEY is set, require X-API-Key or Authorization: Bearer."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    provided = (request.headers.get("x-api-key") or "").strip()
    if not provided:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided or len(provided) != len(expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
