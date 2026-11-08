"""OpenAI embeddings over the REST API (Step 62).

`OpenAIEmbedder` (registered for `EmbeddingProvider.OPENAI`) calls `POST {base}/embeddings`
with a plain async `httpx` client: no SDK, so retries, backoff and concurrency are
entirely ours (Step 63).

**Batching.** The validated inputs are split, in order, into requests of at most
`min(EMBEDDING_MAX_BATCH_INPUTS, spec.max_batch_inputs)` inputs and
`min(EMBEDDING_MAX_BATCH_TOKENS, spec.max_batch_tokens)` summed tokens (counted with the
model's tokenizer; the base class already guarantees each input fits on its own).
Requests go out one after another here; Step 63 adds bounded concurrency.

**Request.** `{"model", "input": [...], "encoding_format": "float"}`, plus `dimensions`
only for a reduced space (a native-size space never sends it, so it also works with
models/proxies that don't support the parameter).

**Response.** Vectors are put back in input order by their `index` (OpenAI documents
order, but `index` is the contract); the indices must be exactly 0..n-1. Usage is the
billed `usage.prompt_tokens`, summed over requests, one request each.

**Failures** map to the Step 61 taxonomy (single attempt here; Step 63 retries the
retryable ones):
- timeouts, connection errors, 5xx → `EmbeddingUnavailableError` (retryable);
- 429 → `EmbeddingRateLimitedError` (retryable, `retry_after` from `retry-after-ms` /
  `Retry-After`), except `insufficient_quota` (billing) → `EmbeddingRequestError`;
- 401/403 → `EmbeddingAuthError`; other 4xx → `EmbeddingRequestError` (with OpenAI's
  error message and code, never the key);
- 2xx that isn't a well-formed embeddings payload → `EmbeddingResponseError`.
"""

import math
from collections.abc import Iterator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from rag_engine.config import EmbeddingProvider, Settings, settings
from rag_engine.domain import utc_now
from rag_engine.embeddings.base import (
    BaseEmbedder,
    EmbeddingUsageCounts,
    InputType,
    ProviderResult,
    TokenCounter,
    Vector,
)
from rag_engine.embeddings.dispatcher import register_embedder
from rag_engine.embeddings.exceptions import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingRateLimitedError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from rag_engine.embeddings.models import EmbeddingSpace

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_ERROR_MESSAGE = 300


@register_embedder(EmbeddingProvider.OPENAI)
class OpenAIEmbedder(BaseEmbedder):
    provider = EmbeddingProvider.OPENAI

    def __init__(
        self,
        space: EmbeddingSpace,
        *,
        config: Settings = settings,
        tokenizer: TokenCounter | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """`client`: a shared client (the caller owns and closes it); else one is created."""
        super().__init__(space, config=config, tokenizer=tokenizer)
        self.max_batch_inputs = min(config.EMBEDDING_MAX_BATCH_INPUTS, self.spec.max_batch_inputs)
        self.max_batch_tokens = min(config.EMBEDDING_MAX_BATCH_TOKENS, self.spec.max_batch_tokens)
        self._url = (config.OPENAI_BASE_URL or DEFAULT_BASE_URL).rstrip("/") + "/embeddings"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.OPENAI_TIMEOUT_SECONDS)
        self._headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- batching ---- #

    def batches(self, texts: list[str]) -> Iterator[tuple[int, list[str]]]:
        """`(offset, texts)` requests within the input-count and token budgets, in order."""
        start, tokens = 0, 0
        for index, text in enumerate(texts):
            count = self._tokenizer.count(text)
            full = index - start >= self.max_batch_inputs
            if index > start and (full or tokens + count > self.max_batch_tokens):
                yield start, texts[start:index]
                start, tokens = index, 0
            tokens += count
        if start < len(texts):
            yield start, texts[start:]

    async def _embed(self, texts: list[str], input_type: InputType) -> ProviderResult:
        vectors: list[Vector] = []
        usage = EmbeddingUsageCounts()
        for _, batch in self.batches(texts):
            batch_vectors, batch_usage = await self.request(batch)
            vectors.extend(batch_vectors)
            usage += batch_usage
        return ProviderResult(vectors, usage)

    # ---- one request ---- #

    def payload(self, texts: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.space.model.value,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions != self.spec.native_dimensions:
            body["dimensions"] = self.dimensions
        return body

    async def request(self, texts: list[str]) -> tuple[list[Vector], EmbeddingUsageCounts]:
        """One `POST /embeddings`; typed errors on failure (no retries here)."""
        try:
            response = await self._client.post(
                self._url, json=self.payload(texts), headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingUnavailableError(
                f"OpenAI embeddings request timed out ({type(exc).__name__})",
                details=self._details(),
            ) from exc
        except httpx.TransportError as exc:
            raise EmbeddingUnavailableError(
                f"OpenAI embeddings request failed ({type(exc).__name__})",
                details=self._details(),
            ) from exc
        if response.status_code != 200:
            raise self._error(response)
        return self._parse(response, len(texts))

    def _details(self, **extra: Any) -> dict[str, Any]:
        return {"provider": "openai", "model": self.space.model.value, **extra}

    def _error(self, response: httpx.Response) -> EmbeddingError:
        status = response.status_code
        message, code = _openai_error(response)
        details = self._details(status=status, **({"error_code": code} if code else {}))
        text = f"OpenAI returned {status}" + (f": {message}" if message else "")
        if status == 429 and code == "insufficient_quota":
            return EmbeddingRequestError(text, details=details)  # billing: retrying won't help
        if status == 429:
            return EmbeddingRateLimitedError(
                text, retry_after=retry_after_seconds(response.headers), details=details
            )
        if status >= 500:
            return EmbeddingUnavailableError(
                text, retry_after=retry_after_seconds(response.headers), details=details
            )
        if status in (401, 403):
            return EmbeddingAuthError(text, details=details)
        return EmbeddingRequestError(text, details=details)

    def _parse(
        self, response: httpx.Response, expected: int
    ) -> tuple[list[Vector], EmbeddingUsageCounts]:
        def invalid(reason: str) -> EmbeddingResponseError:
            return EmbeddingResponseError(
                f"OpenAI returned an invalid embeddings payload: {reason}",
                details=self._details(),
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise invalid("not JSON") from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise invalid("no data list")
        by_index: dict[int, Vector] = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise invalid("an item without index/embedding")
            if index in by_index:
                raise invalid(f"index {index} repeated")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in embedding):
                raise invalid(f"embedding {index} is not a list of numbers")
            by_index[index] = [float(v) for v in embedding]
        if sorted(by_index) != list(range(expected)):
            raise invalid(f"indices {sorted(by_index)[:5]}... are not 0..{expected - 1}")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        tokens = usage.get("prompt_tokens", usage.get("total_tokens", 0))
        tokens = tokens if isinstance(tokens, int) and tokens >= 0 else 0
        return [by_index[i] for i in range(expected)], EmbeddingUsageCounts(1, tokens)


def _openai_error(response: httpx.Response) -> tuple[str | None, str | None]:
    """OpenAI's `{"error": {"message", "code"|"type"}}`, cut to a safe length."""
    try:
        body = response.json()
    except ValueError:
        return None, None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None, None
    message = error.get("message")
    code = error.get("code") or error.get("type")
    message = " ".join(message.split())[:_MAX_ERROR_MESSAGE] if isinstance(message, str) else None
    return message, code if isinstance(code, str) else None


def retry_after_seconds(headers: httpx.Headers) -> float | None:
    """`retry-after-ms` (OpenAI) or `Retry-After` (seconds or an HTTP date), if present."""
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            value = float(ms) / 1000
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return value
    if (raw := headers.get("retry-after")) is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        try:
            value = (parsedate_to_datetime(raw) - utc_now()).total_seconds()
        except TypeError, ValueError:
            return None
    return max(value, 0.0) if math.isfinite(value) else None
