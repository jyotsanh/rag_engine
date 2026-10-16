"""Adds `X-Process-Time` (seconds) and writes one access-log line per request."""

import time

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_engine.config import logger

PROCESS_TIME_HEADER = "X-Process-Time"


class ProcessTimeMiddleware:
    """Pure ASGI; replaces uvicorn's access log (disabled in `dev()`/`server()`)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_with_timing(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                elapsed = time.perf_counter() - start
                MutableHeaders(scope=message)[PROCESS_TIME_HEADER] = f"{elapsed:.6f}"
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            logger.bind(
                method=scope["method"],
                path=scope["path"],
                status_code=status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            ).info("{} {} {}", scope["method"], scope["path"], status_code)
