"""A local stand-in for the OpenAI embeddings API (`POST /v1/embeddings`).

Point `OPENAI_BASE_URL` at `FakeOpenAI.base_url` and the real `OpenAIEmbedder` talks HTTP
to it: no production test hooks, the whole client path (batching, retries, parsing) is
exercised. Vectors are `fake_vector(text, dimensions)` (deterministic, unit length);
`fail_with` makes every request answer an error status until cleared.
"""

import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tests.support.fakes import fake_vector

_NATIVE = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


@dataclass
class _State:
    requests: list[dict] = field(default_factory=list)
    fail_with: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _app(state: _State) -> Starlette:
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        with state.lock:
            state.requests.append(body)
            failing = state.fail_with
        if failing is not None:
            code = "invalid_api_key" if failing == 401 else "server_error"
            return JSONResponse({"error": {"message": "fake failure", "code": code}}, failing)
        texts = body["input"]
        dimensions = body.get("dimensions") or _NATIVE[body["model"]]
        return JSONResponse(
            {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": fake_vector(t, dimensions)}
                    for i, t in enumerate(texts)
                ],
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts),
                },
            }
        )

    return Starlette(routes=[Route("/v1/embeddings", embeddings, methods=["POST"])])


class FakeOpenAI:
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

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def requests(self) -> list[dict]:
        with self._state.lock:
            return list(self._state.requests)

    def fail_with(self, status: int | None) -> None:
        with self._state.lock:
            self._state.fail_with = status

    def start(self) -> FakeOpenAI:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("fake OpenAI failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._sock.close()
