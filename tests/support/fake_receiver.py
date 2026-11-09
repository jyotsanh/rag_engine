"""A local webhook receiver (Step 83): records every request's raw body and headers, and
answers from a script (status, headers, delay) so delivery tests drive the real HTTP path
through `SafeHttpClient` (with `SSRF_ALLOW_PRIVATE_NETWORKS` on, as for the fixture site)."""

import asyncio
import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


@dataclass(frozen=True)
class Received:
    method: str
    path: str
    headers: dict[str, str]  # lower-cased names
    body: bytes


@dataclass(frozen=True)
class Answer:
    status: int = 204
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0  # seconds before answering (a timeout test)


@dataclass
class _State:
    received: list[Received] = field(default_factory=list)
    script: list[Answer] = field(default_factory=list)  # consumed in order, then `default`
    default: Answer = field(default_factory=Answer)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _app(state: _State) -> Starlette:
    async def hook(request: Request) -> Response:
        body = await request.body()
        with state.lock:
            state.received.append(Received(request.method, request.url.path,
                                           {k.lower(): v for k, v in request.headers.items()},
                                           body))  # fmt: skip
            answer = state.script.pop(0) if state.script else state.default
        if answer.delay:
            await asyncio.sleep(answer.delay)
        return Response(b"ok", status_code=answer.status, headers=answer.headers)

    return Starlette(routes=[Route("/{path:path}", hook, methods=["POST", "GET"])])


class FakeReceiver:
    def __init__(self) -> None:
        self._state = _State()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        config = uvicorn.Config(_app(self._state), log_config=None, access_log=False,
                                lifespan="off", timeout_graceful_shutdown=1)  # fmt: skip
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True
        )

    def url(self, path: str = "/hook") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def received(self) -> list[Received]:
        with self._state.lock:
            return list(self._state.received)

    def answer(self, *script: Answer, default: Answer | None = None) -> None:
        """Answer the next requests with `script` in order, then `default` (204)."""
        with self._state.lock:
            self._state.script = list(script)
            self._state.default = default or Answer()

    def reset(self) -> None:
        with self._state.lock:
            self._state.received.clear()
            self._state.script.clear()
            self._state.default = Answer()

    def start(self) -> FakeReceiver:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("fake receiver failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._sock.close()
