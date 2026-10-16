"""Assigns every request an ID, binds it to the log context, and echoes it back."""

import re
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_engine.config import log_context

REQUEST_ID_HEADER = "X-Request-ID"
# accept a caller-supplied ID only if it is short and log-safe
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestIdMiddleware:
    """Pure ASGI so the context var spans the whole request, including streaming bodies.

    The ID is also stored on `scope["state"]`, so exception handlers running in
    Starlette's outermost `ServerErrorMiddleware` can still read it via
    `request.state.request_id`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = (
            dict(scope["headers"]).get(REQUEST_ID_HEADER.lower().encode(), b"").decode("latin-1")
        )
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with log_context(request_id=request_id):
            await self.app(scope, receive, send_with_header)
